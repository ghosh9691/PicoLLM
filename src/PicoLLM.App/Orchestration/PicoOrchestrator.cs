using System.Text;
using PicoLLM.Browser;
using PicoLLM.Browser.Parsing;
using PicoLLM.Core.Model;
using PicoLLM.Gguf;
using PicoLLM.Tokenizer;
using PicoLLM.Training;

namespace PicoLLM.App.Orchestration;

/// <summary>
/// Coordinates the full PicoLLM pipeline: browse → tokenize → train → checkpoint → export.
/// </summary>
/// <remarks>
/// <para>
/// On the first session (no <c>tokenizer.json</c> in the data directory), the orchestrator
/// trains a BPE tokenizer from the collected corpus, then trains the model.
/// On subsequent sessions it reloads the saved tokenizer and resumes training from the
/// last checkpoint (if one exists).
/// </para>
/// <para>
/// Subscribe to <see cref="OnProgress"/> to receive pipeline events in real time.
/// </para>
/// </remarks>
public sealed class PicoOrchestrator
{
    private readonly OrchestratorConfig _config;
    private readonly Func<string, CancellationToken, Task<BrowseResult>> _fetchFunc;

    private BpeTokenizer? _tokenizer;
    private PicoLLMModel? _model;
    private AdamW? _optimizer;
    private TrainingLoop? _trainingLoop;
    private int _globalStep;

    // Accumulates all page text seen since the orchestrator was constructed.
    private readonly StringBuilder _corpusBuffer = new();

    // Tracks whether state (tokenizer + checkpoint) has been loaded from disk.
    private bool _stateLoaded;

    /// <summary>Subscribe to receive pipeline progress events.</summary>
    public event Action<OrchestratorEvent>? OnProgress;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a new <see cref="PicoOrchestrator"/>.
    /// </summary>
    /// <param name="config">Pipeline configuration (model, training, data directory).</param>
    /// <param name="httpOverride">
    ///   Optional delegate that replaces the real HTTP fetcher. Used for testing.
    ///   Receives (url, cancellationToken) and must return a <see cref="BrowseResult"/>.
    ///   When null, a real <see cref="HttpFetcher"/> is used.
    /// </param>
    public PicoOrchestrator(OrchestratorConfig config,
        Func<string, CancellationToken, Task<BrowseResult>>? httpOverride = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        if (httpOverride is not null)
        {
            _fetchFunc = httpOverride;
        }
        else
        {
            var fetcher = new HttpFetcher();
            _fetchFunc = (url, ct) => fetcher.FetchAsync(url, ct);
        }
    }

    // ── Public Pipeline API ──────────────────────────────────────────────────

    /// <summary>
    /// Fetches, parses, and trains on a single URL.
    /// Equivalent to calling <see cref="ProcessUrlsAsync"/> with a single-element list.
    /// </summary>
    public async Task ProcessUrlAsync(string url, CancellationToken cancellationToken = default) =>
        await ProcessUrlsAsync([url], cancellationToken).ConfigureAwait(false);

    /// <summary>
    /// Processes a batch of URLs in order: fetch → parse → accumulate → tokenize → train.
    /// Failed URLs are reported via <see cref="SessionErrorEvent"/> and skipped; remaining
    /// URLs continue processing.
    /// </summary>
    /// <param name="urls">URLs to fetch and train on.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task ProcessUrlsAsync(
        IEnumerable<string> urls,
        CancellationToken cancellationToken = default)
    {
        EnsureStateLoaded();

        var newPageTexts = new List<string>();

        foreach (var url in urls)
        {
            try
            {
                var result = await _fetchFunc(url, cancellationToken).ConfigureAwait(false);
                bool success = result.Status is BrowseStatus.Success or BrowseStatus.Redirect;
                Emit(new PageFetchedEvent(url, success, result.ErrorMessage));

                if (!success) continue;
                if (string.IsNullOrWhiteSpace(result.HtmlContent)) continue;

                var page = await HtmlParser
                    .ParseAsync(result.HtmlContent, url, cancellationToken)
                    .ConfigureAwait(false);
                Emit(new PageParsedEvent(url, page.CleanText.Length, page.Images.Count, page.Links.Count));

                if (string.IsNullOrWhiteSpace(page.CleanText)) continue;

                newPageTexts.Add(page.CleanText);
                _corpusBuffer.AppendLine(page.CleanText);
            }
            catch (Exception ex)
            {
                Emit(new SessionErrorEvent(url, ex.Message));
            }
        }

        if (newPageTexts.Count == 0) return;

        // ── First session: train tokenizer from accumulated corpus ───────────
        if (_tokenizer is null)
        {
            var corpus = _corpusBuffer.ToString();
            var trainer = new BpeTrainer();
            var tokConfig = trainer.Train(corpus, _config.Model.VocabSize);
            _tokenizer = BpeTokenizer.FromConfig(tokConfig);

            Directory.CreateDirectory(_config.DataDirectory);
            _tokenizer.Save(Path.Combine(_config.DataDirectory, "tokenizer.json"));
            Emit(new TokenizerTrainedEvent(_tokenizer.VocabSize));
        }

        // ── Ensure model is initialized (uses actual tokenizer VocabSize) ─────
        EnsureModelInitialized();

        // ── Run training steps on the full accumulated corpus ─────────────────
        var allTokens = _tokenizer.Encode(_corpusBuffer.ToString());
        int minTokens = _config.Training.SeqLen + 1;
        if (allTokens.Length < minTokens) return;

        int totalSteps = _config.StepsPerPage * newPageTexts.Count;
        for (int i = 0; i < totalSteps; i++)
        {
            var metrics = _trainingLoop!.Step(allTokens, _globalStep);
            _globalStep++;
            Emit(new TrainingStepEvent(
                metrics.Step, metrics.Loss, metrics.LearningRate,
                metrics.TokensPerSec, metrics.GradNorm));
        }
    }

    /// <summary>
    /// Ends the current session: saves a training checkpoint and exports the model to GGUF.
    /// Call this when the user finishes browsing.
    /// </summary>
    /// <param name="ggufOutputPath">Destination path for the exported .gguf file.</param>
    public void EndSession(string ggufOutputPath)
    {
        ArgumentNullException.ThrowIfNull(ggufOutputPath);
        if (_model is null || _tokenizer is null || _optimizer is null) return;

        Directory.CreateDirectory(_config.DataDirectory);
        var checkpointPath = Path.Combine(_config.DataDirectory, "checkpoint.pckp");
        CheckpointManager.Save(_model, _optimizer, _globalStep, checkpointPath);
        Emit(new CheckpointSavedEvent(checkpointPath));

        GgufExporter.Export(_model, _tokenizer, ggufOutputPath);
        var fileSize = new FileInfo(ggufOutputPath).Length;
        Emit(new GgufExportedEvent(ggufOutputPath, fileSize));
    }

    // ── Private: State Loading ───────────────────────────────────────────────

    private void EnsureStateLoaded()
    {
        if (_stateLoaded) return;
        _stateLoaded = true;

        var tokenizerPath  = Path.Combine(_config.DataDirectory, "tokenizer.json");
        var checkpointPath = Path.Combine(_config.DataDirectory, "checkpoint.pckp");

        // Load existing tokenizer if present
        if (File.Exists(tokenizerPath))
        {
            try { _tokenizer = BpeTokenizer.Load(tokenizerPath); }
            catch { /* corrupt tokenizer — treat as first session */ }
        }

        if (_tokenizer is null) return; // no model without a tokenizer

        EnsureModelInitialized();

        // Load checkpoint into the freshly constructed model + optimizer
        if (File.Exists(checkpointPath) && _model is not null && _optimizer is not null)
        {
            try { _globalStep = CheckpointManager.Load(checkpointPath, _model, _optimizer); }
            catch { _globalStep = 0; /* corrupt checkpoint — train from scratch */ }
        }
    }

    // ── Private: Model Initialization ───────────────────────────────────────

    private void EnsureModelInitialized()
    {
        if (_model is not null || _tokenizer is null) return;

        // Use the actual tokenizer VocabSize in case it differs from the target
        var modelConfig = _config.Model with { VocabSize = _tokenizer.VocabSize };
        _model = new PicoLLMModel(modelConfig);
        _optimizer = new AdamW(_config.Training.LearningRate);

        var schedule = new LearningRateSchedule(
            warmupSteps: _config.Training.WarmupSteps,
            totalSteps:  100_000,   // large; LR decays slowly across many sessions
            maxLr:       _config.Training.LearningRate);

        _trainingLoop = new TrainingLoop(
            _model,
            _optimizer,
            schedule,
            clipNorm:  _config.Training.MaxGradNorm,
            batchSize: _config.Training.BatchSize,
            seqLen:    _config.Training.SeqLen);
    }

    // ── Private: Event Emission ──────────────────────────────────────────────

    private void Emit(OrchestratorEvent ev) => OnProgress?.Invoke(ev);
}

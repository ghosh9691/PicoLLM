using System.Collections.Concurrent;
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
/// <see cref="BrowseAsync"/> fetches and parses a URL for immediate UI display, queuing its
/// text content for background training. Training runs on a thread-pool thread so the UI stays
/// responsive while pages are browsed.
/// </para>
/// <para>
/// <see cref="ProcessUrlsAsync"/> provides the original synchronous batch path (fetch → parse →
/// train) used by tests and non-interactive pipelines.
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
    private bool _stateLoaded;

    // Corpus accumulates all page text seen this orchestrator lifetime.
    private readonly StringBuilder _corpusBuffer = new();
    private readonly object _corpusLock = new();

    // ── Background training ───────────────────────────────────────────────────
    private record TrainingItem(string Url, string Text);

    private readonly ConcurrentQueue<TrainingItem> _trainingQueue = new();
    private readonly CancellationTokenSource _trainingCts = new();
    private Task? _trainingTask;

    /// <summary>Subscribe to receive pipeline progress events.</summary>
    public event Action<OrchestratorEvent>? OnProgress;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a new <see cref="PicoOrchestrator"/>.
    /// </summary>
    /// <param name="config">Pipeline configuration (model, training, data directory).</param>
    /// <param name="httpOverride">
    ///   Optional delegate that replaces the real HTTP fetcher. Used for testing.
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
    /// Fetches and parses a single URL for immediate display, then queues its content for
    /// background training. Returns the parsed page immediately — does NOT wait for training.
    /// Returns null if the fetch or parse fails.
    /// </summary>
    /// <param name="url">URL to fetch.</param>
    /// <param name="cancellationToken">Cancellation token (covers fetch + parse only).</param>
    public async Task<ParsedPage?> BrowseAsync(string url, CancellationToken cancellationToken = default)
    {
        EnsureStateLoaded();
        try
        {
            var result = await _fetchFunc(url, cancellationToken).ConfigureAwait(false);
            bool success = result.Status is BrowseStatus.Success or BrowseStatus.Redirect;
            Emit(new PageFetchedEvent(url, success, result.ErrorMessage));

            if (!success || string.IsNullOrWhiteSpace(result.HtmlContent)) return null;

            var page = await HtmlParser
                .ParseAsync(result.HtmlContent, url, cancellationToken)
                .ConfigureAwait(false);
            Emit(new PageParsedEvent(url, page.CleanText.Length, page.Images.Count, page.Links.Count, page));

            if (!string.IsNullOrWhiteSpace(page.CleanText))
            {
                lock (_corpusLock) _corpusBuffer.AppendLine(page.CleanText);
                _trainingQueue.Enqueue(new TrainingItem(url, page.CleanText));
                Emit(new TrainingQueueChangedEvent(_trainingQueue.Count));
                EnsureTrainingStarted();
            }

            return page;
        }
        catch (Exception ex)
        {
            Emit(new SessionErrorEvent(url, ex.Message));
            return null;
        }
    }

    /// <summary>
    /// Fetches, parses, and trains on a single URL synchronously (blocking until training
    /// completes). Equivalent to <see cref="ProcessUrlsAsync"/> with a single-element list.
    /// </summary>
    public async Task ProcessUrlAsync(string url, CancellationToken cancellationToken = default) =>
        await ProcessUrlsAsync([url], cancellationToken).ConfigureAwait(false);

    /// <summary>
    /// Processes a batch of URLs in order: fetch → parse → accumulate → tokenize → train.
    /// Failed URLs are reported via <see cref="SessionErrorEvent"/> and skipped; remaining
    /// URLs continue processing. Training runs synchronously on the caller's thread.
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
                Emit(new PageParsedEvent(url, page.CleanText.Length, page.Images.Count, page.Links.Count, page));

                if (string.IsNullOrWhiteSpace(page.CleanText)) continue;

                newPageTexts.Add(page.CleanText);
                lock (_corpusLock) _corpusBuffer.AppendLine(page.CleanText);
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
            string corpus;
            lock (_corpusLock) corpus = _corpusBuffer.ToString();
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
        string allText;
        lock (_corpusLock) allText = _corpusBuffer.ToString();
        var allTokens = _tokenizer.Encode(allText);
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
    /// Ends the current session: cancels background training, saves a training checkpoint,
    /// and exports the model to GGUF. Waits for the background training loop to finish.
    /// </summary>
    /// <param name="ggufOutputPath">Destination path for the exported .gguf file.</param>
    public void EndSession(string ggufOutputPath)
    {
        ArgumentNullException.ThrowIfNull(ggufOutputPath);

        // Stop background training and wait for graceful shutdown.
        _trainingCts.Cancel();
        try { _trainingTask?.Wait(); } catch (AggregateException) { /* training cancelled */ }

        if (_model is null || _tokenizer is null || _optimizer is null) return;

        Directory.CreateDirectory(_config.DataDirectory);
        var checkpointPath = Path.Combine(_config.DataDirectory, "checkpoint.pckp");
        CheckpointManager.Save(_model, _optimizer, _globalStep, checkpointPath);
        Emit(new CheckpointSavedEvent(checkpointPath));

        GgufExporter.Export(_model, _tokenizer, ggufOutputPath);
        var fileSize = new FileInfo(ggufOutputPath).Length;
        Emit(new GgufExportedEvent(ggufOutputPath, fileSize));
    }

    // ── Private: Background Training ─────────────────────────────────────────

    private void EnsureTrainingStarted()
    {
        if (_trainingTask is not null) return;
        _trainingTask = Task.Run(() => TrainingLoopAsync(_trainingCts.Token));
    }

    private async Task TrainingLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            if (!_trainingQueue.TryDequeue(out var item))
            {
                try { await Task.Delay(100, ct).ConfigureAwait(false); }
                catch (OperationCanceledException) { break; }
                continue;
            }

            Emit(new TrainingQueueChangedEvent(_trainingQueue.Count));

            // Train tokenizer on first session (no tokenizer.json existed at startup).
            if (_tokenizer is null)
            {
                string corpus;
                lock (_corpusLock) corpus = _corpusBuffer.ToString();
                var trainer = new BpeTrainer();
                var tokConfig = trainer.Train(corpus, _config.Model.VocabSize);
                _tokenizer = BpeTokenizer.FromConfig(tokConfig);

                Directory.CreateDirectory(_config.DataDirectory);
                _tokenizer.Save(Path.Combine(_config.DataDirectory, "tokenizer.json"));
                Emit(new TokenizerTrainedEvent(_tokenizer.VocabSize));
            }

            EnsureModelInitialized();

            string allText;
            lock (_corpusLock) allText = _corpusBuffer.ToString();
            var allTokens = _tokenizer!.Encode(allText);
            int minTokens = _config.Training.SeqLen + 1;
            if (allTokens.Length < minTokens) continue;

            for (int i = 0; i < _config.StepsPerPage; i++)
            {
                if (ct.IsCancellationRequested) break;
                var metrics = _trainingLoop!.Step(allTokens, _globalStep);
                _globalStep++;
                Emit(new TrainingStepEvent(
                    metrics.Step, metrics.Loss, metrics.LearningRate,
                    metrics.TokensPerSec, metrics.GradNorm));
            }
        }
    }

    // ── Private: State Loading ───────────────────────────────────────────────

    private void EnsureStateLoaded()
    {
        if (_stateLoaded) return;
        _stateLoaded = true;

        var tokenizerPath  = Path.Combine(_config.DataDirectory, "tokenizer.json");
        var checkpointPath = Path.Combine(_config.DataDirectory, "checkpoint.pckp");

        if (File.Exists(tokenizerPath))
        {
            try { _tokenizer = BpeTokenizer.Load(tokenizerPath); }
            catch { /* corrupt tokenizer — treat as first session */ }
        }

        if (_tokenizer is null) return; // no model without a tokenizer

        EnsureModelInitialized();

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

        var modelConfig = _config.Model with { VocabSize = _tokenizer.VocabSize };
        _model = new PicoLLMModel(modelConfig);
        _optimizer = new AdamW(_config.Training.LearningRate);

        var schedule = new LearningRateSchedule(
            warmupSteps: _config.Training.WarmupSteps,
            totalSteps:  100_000,
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

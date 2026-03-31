using FluentAssertions;
using PicoLLM.App.Orchestration;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Orchestrator;

public class PicoOrchestratorTests : IDisposable
{
    private readonly string _dataDir;

    public PicoOrchestratorTests()
    {
        _dataDir = Path.Combine(Path.GetTempPath(), "pico_orch_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dataDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_dataDir))
            Directory.Delete(_dataDir, recursive: true);
    }

    private static OrchestratorConfig MakeConfig(string dataDir) =>
        new(
            Model: new PicoLLM.Core.Model.ModelConfig(
                VocabSize:    300,   // 4 special + 256 bytes + 40 merges
                EmbedDim:     32,
                NumHeads:     2,
                NumLayers:    1,
                FfMultiplier: 2,
                MaxSeqLen:    64),
            Training: new TrainingConfig(BatchSize: 1, SeqLen: 16, LearningRate: 1e-4f, WarmupSteps: 5),
            DataDirectory: dataDir,
            StepsPerPage: 3);

    // ── Helper: fake HTTP that always returns the same HTML ──────────────────

    private static Func<string, CancellationToken, Task<BrowseResult>> FakeHttp(string html) =>
        (url, _) => Task.FromResult(new BrowseResult(
            Url:           url,
            Status:        BrowseStatus.Success,
            HtmlContent:   html,
            ErrorMessage:  null,
            HttpStatusCode: 200,
            ElapsedTime:   TimeSpan.Zero));

    private static Func<string, CancellationToken, Task<BrowseResult>> FailingHttp() =>
        (url, _) => Task.FromResult(new BrowseResult(
            Url:           url,
            Status:        BrowseStatus.Error,
            HtmlContent:   null,
            ErrorMessage:  "Connection refused",
            HttpStatusCode: 0,
            ElapsedTime:   TimeSpan.Zero));

    // ── State Management Tests ───────────────────────────────────────────────

    [Fact]
    public void Constructor_DoesNotThrow_WithFreshDataDirectory()
    {
        var act = () => new PicoOrchestrator(MakeConfig(_dataDir));
        act.Should().NotThrow();
    }

    [Fact]
    public async Task ProcessUrlsAsync_FreshStart_TrainsTokenizerAndModel()
    {
        // Large enough text for BPE to produce merges
        const string html = """
            <html><body>
            <p>The transformer architecture is the foundation of modern large language models.
            It uses self-attention mechanisms to process sequences of tokens in parallel.
            Each layer consists of multi-head attention followed by a feedforward network.
            The model learns to predict the next token given the preceding context.
            This training objective is called causal language modelling or next-token prediction.
            Neural networks with many layers can learn complex representations of text data.
            The attention mechanism allows each token to attend to all previous tokens.
            Position encodings provide information about token order in the sequence.
            Layer normalisation stabilises training by normalising activations within each layer.
            The feedforward network applies two linear transformations with a nonlinear activation.</p>
            </body></html>
            """;

        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));
        orch.OnProgress += events.Add;

        await orch.ProcessUrlsAsync(["https://example.com/page1"]);

        // Must have emitted a PageFetchedEvent
        events.OfType<PageFetchedEvent>().Should().ContainSingle(e => e.Success);

        // Must have emitted a PageParsedEvent with non-zero text
        events.OfType<PageParsedEvent>().Should().ContainSingle(e => e.TextLength > 0);

        // Must have trained a tokenizer (first session, no tokenizer.json existed)
        events.OfType<TokenizerTrainedEvent>().Should().ContainSingle();

        // Must have run some training steps
        events.OfType<TrainingStepEvent>().Should().HaveCount(config.StepsPerPage);

        // Tokenizer must be saved to disk
        File.Exists(Path.Combine(_dataDir, "tokenizer.json")).Should().BeTrue();
    }

    [Fact]
    public async Task ProcessUrlsAsync_SecondSession_ReusesExistingTokenizer()
    {
        const string html = """
            <html><body>
            <p>Machine learning models can be trained on large corpora of text data.
            The training process involves forward passes and backpropagation through the network.
            Gradient descent algorithms update the weights to minimise the loss function.
            Overfitting occurs when a model performs well on training data but poorly on new data.
            Regularisation techniques like weight decay help prevent overfitting in neural networks.</p>
            </body></html>
            """;

        var config = MakeConfig(_dataDir);

        // First session — trains tokenizer
        var orch1 = new PicoOrchestrator(config, FakeHttp(html));
        await orch1.ProcessUrlsAsync(["https://example.com/p1"]);

        // Second session — tokenizer.json exists; TokenizerTrainedEvent must NOT be emitted
        var events2 = new List<OrchestratorEvent>();
        var orch2 = new PicoOrchestrator(config, FakeHttp(html));
        orch2.OnProgress += events2.Add;
        await orch2.ProcessUrlsAsync(["https://example.com/p2"]);

        events2.OfType<TokenizerTrainedEvent>().Should().BeEmpty();
        events2.OfType<TrainingStepEvent>().Should().HaveCount(config.StepsPerPage);
    }

    [Fact]
    public async Task ProcessUrlsAsync_FailedUrl_EmitsSessionErrorAndContinues()
    {
        const string goodHtml = """
            <html><body>
            <p>The quick brown fox jumps over the lazy dog repeatedly for training data purposes.
            Token sequences are sampled from this corpus to create input output pairs for learning.
            The model processes these sequences and adjusts its internal weights accordingly.</p>
            </body></html>
            """;

        int callCount = 0;
        Func<string, CancellationToken, Task<BrowseResult>> mixedHttp = (url, ct) =>
        {
            callCount++;
            return callCount == 2
                ? Task.FromResult(new BrowseResult(url, BrowseStatus.Error, null, "404 Not Found", 404, TimeSpan.Zero))
                : Task.FromResult(new BrowseResult(url, BrowseStatus.Success, goodHtml, null, 200, TimeSpan.Zero));
        };

        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, mixedHttp);
        orch.OnProgress += events.Add;

        await orch.ProcessUrlsAsync([
            "https://example.com/good1",
            "https://example.com/bad",
            "https://example.com/good2"
        ]);

        // Error URL reported
        events.OfType<PageFetchedEvent>().Should().Contain(e => !e.Success);

        // Still got training steps from the two good pages
        events.OfType<TrainingStepEvent>().Should().HaveCountGreaterThan(0);
    }

    [Fact]
    public async Task EndSession_ProducesCheckpointAndGguf()
    {
        const string html = """
            <html><body>
            <p>Natural language processing is a subfield of artificial intelligence focused on text.
            Deep learning has transformed how computers understand and generate human language.
            Attention mechanisms allow models to focus on relevant parts of the input sequence.
            Pre-training on large corpora enables transfer learning to downstream tasks.
            Fine-tuning adapts a pre-trained model to a specific domain or task efficiently.</p>
            </body></html>
            """;

        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));
        await orch.ProcessUrlsAsync(["https://example.com/page1"]);

        var ggufPath = Path.Combine(_dataDir, "test_export.gguf");
        orch.EndSession(ggufPath);

        // Checkpoint must exist
        File.Exists(Path.Combine(_dataDir, "checkpoint.pckp")).Should().BeTrue();

        // GGUF must exist and be non-empty
        File.Exists(ggufPath).Should().BeTrue();
        new FileInfo(ggufPath).Length.Should().BeGreaterThan(0);
    }

    // ── BrowseAsync tests ────────────────────────────────────────────────────

    [Fact]
    public async Task BrowseAsync_ReturnsParsedPage_Immediately()
    {
        const string html = """
            <html><body>
            <h1>Hello</h1>
            <p>Quick brown fox content for testing the browse path.</p>
            </body></html>
            """;

        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));

        var page = await orch.BrowseAsync("https://example.com/page");

        page.Should().NotBeNull();
        page!.CleanText.Should().NotBeEmpty();
    }

    [Fact]
    public async Task BrowseAsync_EmitsPageFetchedAndPageParsedEvents()
    {
        const string html = """
            <html><body>
            <p>Content for browse event test.</p>
            </body></html>
            """;

        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));
        orch.OnProgress += events.Add;

        await orch.BrowseAsync("https://example.com/browse");

        events.OfType<PageFetchedEvent>().Should().ContainSingle(e => e.Success);
        events.OfType<PageParsedEvent>().Should().ContainSingle(e => e.TextLength > 0);
    }

    [Fact]
    public async Task BrowseAsync_FailedFetch_ReturnsNull_AndEmitsError()
    {
        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FailingHttp());
        orch.OnProgress += events.Add;

        var page = await orch.BrowseAsync("https://example.com/bad");

        page.Should().BeNull();
        events.OfType<PageFetchedEvent>().Should().ContainSingle(e => !e.Success);
    }

    [Fact]
    public async Task BrowseAsync_QueueDepthEventEmitted()
    {
        const string html = """
            <html><body>
            <p>Training queue depth event test content here.</p>
            </body></html>
            """;

        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));
        orch.OnProgress += events.Add;

        await orch.BrowseAsync("https://example.com/q");

        events.OfType<TrainingQueueChangedEvent>().Should().NotBeEmpty();
    }

    [Fact]
    public async Task BrowseAsync_MultiplePages_ProcessedInFifoOrder()
    {
        var fetchOrder = new List<string>();
        Func<string, CancellationToken, Task<BrowseResult>> orderedHttp = (url, _) =>
        {
            fetchOrder.Add(url);
            return Task.FromResult(new BrowseResult(
                Url: url, Status: BrowseStatus.Success,
                HtmlContent: "<html><body><p>Content for " + url + ".</p></body></html>",
                ErrorMessage: null, HttpStatusCode: 200, ElapsedTime: TimeSpan.Zero));
        };

        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, orderedHttp);

        await orch.BrowseAsync("https://example.com/p1");
        await orch.BrowseAsync("https://example.com/p2");
        await orch.BrowseAsync("https://example.com/p3");

        fetchOrder.Should().Equal(
            "https://example.com/p1",
            "https://example.com/p2",
            "https://example.com/p3");
    }

    [Fact]
    public async Task EndSession_EmitsCheckpointAndGgufEvents()
    {
        const string html = """
            <html><body>
            <p>Recurrent neural networks process sequences one token at a time in a loop.
            Long short-term memory networks address the vanishing gradient problem in RNNs.
            Transformers replace recurrence with self-attention for better parallelisation.
            The encoder processes the input while the decoder generates the output tokens.
            Modern large language models use only the decoder component for generation tasks.</p>
            </body></html>
            """;

        var events = new List<OrchestratorEvent>();
        var config = MakeConfig(_dataDir);
        var orch = new PicoOrchestrator(config, FakeHttp(html));
        orch.OnProgress += events.Add;

        await orch.ProcessUrlsAsync(["https://example.com/page1"]);

        var ggufPath = Path.Combine(_dataDir, "model.gguf");
        orch.EndSession(ggufPath);

        events.OfType<CheckpointSavedEvent>().Should().ContainSingle();
        events.OfType<GgufExportedEvent>().Should().ContainSingle(e => e.FileSizeBytes > 0);
    }
}

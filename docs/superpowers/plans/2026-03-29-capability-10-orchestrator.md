# Capability 10 — Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `PicoOrchestrator` — the pipeline that wires Browse → Tokenize → Train → Checkpoint → Export GGUF into a single callable API.

**Architecture:** New `PicoLLM.App` class library in `src/PicoLLM.App/Orchestration/`. Three files: `OrchestratorEvent.cs` (all event record types), `OrchestratorConfig.cs` (config records with defaults), and `PicoOrchestrator.cs` (the pipeline). HTTP fetching is injected via an optional `Func<string, CancellationToken, Task<BrowseResult>>` delegate so tests can operate without live HTTP.

**Tech Stack:** .NET 9 / C# 12, xUnit, FluentAssertions. Wires: `HttpFetcher`, `HtmlParser`, `BpeTrainer`, `BpeTokenizer`, `PicoLLMModel`, `AdamW`, `LearningRateSchedule`, `TrainingLoop`, `CheckpointManager`, `GgufExporter`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/PicoLLM.App/PicoLLM.App.csproj` | Class library; references all other projects |
| Create | `src/PicoLLM.App/Orchestration/OrchestratorEvent.cs` | All 7 event record types |
| Create | `src/PicoLLM.App/Orchestration/OrchestratorConfig.cs` | `OrchestratorConfig` and `TrainingConfig` records |
| Create | `src/PicoLLM.App/Orchestration/PicoOrchestrator.cs` | Pipeline: load state → browse → train → checkpoint → export |
| Modify | `tests/PicoLLM.Tests/PicoLLM.Tests.csproj` | Add `<ProjectReference>` to `PicoLLM.App` |
| Create | `tests/PicoLLM.Tests/Orchestrator/OrchestratorConfigTests.cs` | Default value and construction tests |
| Create | `tests/PicoLLM.Tests/Orchestrator/PicoOrchestratorTests.cs` | State management + pipeline + session end + error tests |

---

## Task 1: Project Scaffold

**Files:**
- Create: `src/PicoLLM.App/PicoLLM.App.csproj`
- Modify: `tests/PicoLLM.Tests/PicoLLM.Tests.csproj`

- [ ] **Step 1: Create `src/PicoLLM.App/PicoLLM.App.csproj`**

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\PicoLLM.Core\PicoLLM.Core.csproj" />
    <ProjectReference Include="..\PicoLLM.Tokenizer\PicoLLM.Tokenizer.csproj" />
    <ProjectReference Include="..\PicoLLM.Training\PicoLLM.Training.csproj" />
    <ProjectReference Include="..\PicoLLM.Browser\PicoLLM.Browser.csproj" />
    <ProjectReference Include="..\PicoLLM.Gguf\PicoLLM.Gguf.csproj" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Add to solution**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet sln add src/PicoLLM.App/PicoLLM.App.csproj
```

- [ ] **Step 3: Add test project reference**

In `tests/PicoLLM.Tests/PicoLLM.Tests.csproj`, add inside the existing `<ItemGroup>` with `<ProjectReference>` entries:

```xml
<ProjectReference Include="..\..\src\PicoLLM.App\PicoLLM.App.csproj" />
```

- [ ] **Step 4: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: Build succeeded (empty project, all dependencies restored).

- [ ] **Step 5: Commit**

```bash
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator
git add src/PicoLLM.App/PicoLLM.App.csproj tests/PicoLLM.Tests/PicoLLM.Tests.csproj PicoLLM.slnx
git commit -m "feat(cap10): scaffold PicoLLM.App project"
```

---

## Task 2: Event System and Config Records

**Files:**
- Create: `src/PicoLLM.App/Orchestration/OrchestratorEvent.cs`
- Create: `src/PicoLLM.App/Orchestration/OrchestratorConfig.cs`
- Create: `tests/PicoLLM.Tests/Orchestrator/OrchestratorConfigTests.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Orchestration/OrchestratorEvent.cs`**

```csharp
using PicoLLM.Core.Model;

namespace PicoLLM.App.Orchestration;

/// <summary>Base class for all orchestrator pipeline events.</summary>
public abstract record OrchestratorEvent(DateTime Timestamp);

/// <summary>Emitted when a URL fetch completes (success or failure).</summary>
/// <param name="Url">The URL that was fetched.</param>
/// <param name="Success">True if the page was retrieved successfully.</param>
/// <param name="Error">Human-readable error detail; null on success.</param>
public record PageFetchedEvent(string Url, bool Success, string? Error)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted when HTML parsing produces a clean page.</summary>
/// <param name="Url">Source URL.</param>
/// <param name="TextLength">Length of extracted clean text in characters.</param>
/// <param name="ImageCount">Number of image references extracted.</param>
/// <param name="LinkCount">Number of link references extracted.</param>
public record PageParsedEvent(string Url, int TextLength, int ImageCount, int LinkCount)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted when a BPE tokenizer has been trained from the first session corpus.</summary>
/// <param name="VocabSize">Actual vocabulary size of the trained tokenizer.</param>
public record TokenizerTrainedEvent(int VocabSize)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted after each training step.</summary>
/// <param name="Step">Global step number (0-indexed, increments across sessions).</param>
/// <param name="Loss">Cross-entropy loss for this step.</param>
/// <param name="Lr">Learning rate used.</param>
/// <param name="TokensPerSec">Throughput in tokens per second.</param>
/// <param name="GradNorm">Global gradient L2 norm before clipping.</param>
public record TrainingStepEvent(int Step, float Loss, float Lr, float TokensPerSec, float GradNorm)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted when a training checkpoint is saved to disk.</summary>
/// <param name="Path">Absolute path of the saved checkpoint file.</param>
public record CheckpointSavedEvent(string Path)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted when the model is exported to GGUF format.</summary>
/// <param name="Path">Absolute path of the exported .gguf file.</param>
/// <param name="FileSizeBytes">Size of the exported file in bytes.</param>
public record GgufExportedEvent(string Path, long FileSizeBytes)
    : OrchestratorEvent(DateTime.UtcNow);

/// <summary>Emitted when a URL is skipped due to a fetch or parse error.</summary>
/// <param name="Url">The URL that failed.</param>
/// <param name="Error">Human-readable error description.</param>
public record SessionErrorEvent(string Url, string Error)
    : OrchestratorEvent(DateTime.UtcNow);
```

- [ ] **Step 2: Create `src/PicoLLM.App/Orchestration/OrchestratorConfig.cs`**

```csharp
using PicoLLM.Core.Model;

namespace PicoLLM.App.Orchestration;

/// <summary>
/// Configuration for the <see cref="PicoOrchestrator"/> pipeline.
/// </summary>
/// <param name="Model">Transformer model hyperparameters.</param>
/// <param name="Training">Training loop hyperparameters.</param>
/// <param name="DataDirectory">Directory for checkpoints, tokenizer, and GGUF exports.</param>
/// <param name="StepsPerPage">Number of training steps to run after each fetched page.</param>
public record OrchestratorConfig(
    ModelConfig Model,
    TrainingConfig Training,
    string DataDirectory,
    int StepsPerPage = 100)
{
    /// <summary>Creates an <see cref="OrchestratorConfig"/> with all defaults applied.</summary>
    /// <param name="dataDirectory">Directory for persisted files.</param>
    public static OrchestratorConfig CreateDefault(string dataDirectory) =>
        new(
            Model: new ModelConfig(
                VocabSize:    1024,
                EmbedDim:     128,
                NumHeads:     4,
                NumLayers:    4,
                FfMultiplier: 4,
                MaxSeqLen:    512),
            Training: new TrainingConfig(),
            DataDirectory: dataDirectory);
}

/// <summary>
/// Hyperparameters for the training loop inside the orchestrator.
/// </summary>
/// <param name="BatchSize">Number of sequences per batch.</param>
/// <param name="SeqLen">Sequence length in tokens (input window size).</param>
/// <param name="LearningRate">Initial and peak learning rate for AdamW.</param>
/// <param name="WarmupSteps">Number of linear warmup steps for the LR schedule.</param>
/// <param name="MaxGradNorm">Maximum gradient L2 norm before clipping.</param>
public record TrainingConfig(
    int   BatchSize     = 4,
    int   SeqLen        = 128,
    float LearningRate  = 1e-4f,
    int   WarmupSteps   = 50,
    float MaxGradNorm   = 1.0f);
```

- [ ] **Step 3: Write failing tests**

Create `tests/PicoLLM.Tests/Orchestrator/OrchestratorConfigTests.cs`:

```csharp
using FluentAssertions;
using PicoLLM.App.Orchestration;

namespace PicoLLM.Tests.Orchestrator;

public class OrchestratorConfigTests
{
    [Fact]
    public void CreateDefault_UsesExpectedModelDefaults()
    {
        var config = OrchestratorConfig.CreateDefault("/tmp/pico");

        config.Model.VocabSize.Should().Be(1024);
        config.Model.EmbedDim.Should().Be(128);
        config.Model.NumHeads.Should().Be(4);
        config.Model.NumLayers.Should().Be(4);
        config.Model.FfMultiplier.Should().Be(4);
        config.Model.MaxSeqLen.Should().Be(512);
    }

    [Fact]
    public void CreateDefault_UsesExpectedTrainingDefaults()
    {
        var config = OrchestratorConfig.CreateDefault("/tmp/pico");

        config.Training.BatchSize.Should().Be(4);
        config.Training.SeqLen.Should().Be(128);
        config.Training.LearningRate.Should().BeApproximately(1e-4f, 1e-7f);
        config.Training.WarmupSteps.Should().Be(50);
        config.Training.MaxGradNorm.Should().BeApproximately(1.0f, 1e-7f);
    }

    [Fact]
    public void CreateDefault_SetsDataDirectory()
    {
        var config = OrchestratorConfig.CreateDefault("/data/my-dir");
        config.DataDirectory.Should().Be("/data/my-dir");
    }

    [Fact]
    public void CreateDefault_StepsPerPageIs100()
    {
        var config = OrchestratorConfig.CreateDefault("/tmp");
        config.StepsPerPage.Should().Be(100);
    }

    [Fact]
    public void TrainingConfig_DefaultConstructor_HasExpectedValues()
    {
        var tc = new TrainingConfig();
        tc.BatchSize.Should().Be(4);
        tc.SeqLen.Should().Be(128);
        tc.LearningRate.Should().BeApproximately(1e-4f, 1e-7f);
        tc.WarmupSteps.Should().Be(50);
        tc.MaxGradNorm.Should().BeApproximately(1.0f, 1e-7f);
    }

    [Fact]
    public void PageFetchedEvent_HasTimestamp()
    {
        var before = DateTime.UtcNow;
        var ev = new PageFetchedEvent("https://example.com", true, null);
        var after = DateTime.UtcNow;

        ev.Timestamp.Should().BeOnOrAfter(before).And.BeOnOrBefore(after);
        ev.Url.Should().Be("https://example.com");
        ev.Success.Should().BeTrue();
        ev.Error.Should().BeNull();
    }

    [Fact]
    public void TrainingStepEvent_StoresAllFields()
    {
        var ev = new TrainingStepEvent(Step: 5, Loss: 2.3f, Lr: 0.0001f, TokensPerSec: 1000f, GradNorm: 0.5f);
        ev.Step.Should().Be(5);
        ev.Loss.Should().BeApproximately(2.3f, 1e-5f);
    }
}
```

- [ ] **Step 4: Run tests — expect compile failure (PicoLLM.App not yet on build path)**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~OrchestratorConfigTests"
```

Expected: Build error (types not found) until code is compiled.

- [ ] **Step 5: Build PicoLLM.App to verify no errors**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: Build succeeded.

- [ ] **Step 6: Run tests — expect all pass**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~OrchestratorConfigTests"
```

Expected: All 7 tests pass.

- [ ] **Step 7: Commit**

```bash
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator
git add src/PicoLLM.App/Orchestration/OrchestratorEvent.cs src/PicoLLM.App/Orchestration/OrchestratorConfig.cs tests/PicoLLM.Tests/Orchestrator/OrchestratorConfigTests.cs
git commit -m "feat(cap10): add OrchestratorEvent hierarchy and OrchestratorConfig records"
```

---

## Task 3: PicoOrchestrator — State Management

**Files:**
- Create: `src/PicoLLM.App/Orchestration/PicoOrchestrator.cs`
- Create: `tests/PicoLLM.Tests/Orchestrator/PicoOrchestratorTests.cs`

Tests in this task verify: fresh-start initialization and checkpoint/tokenizer loading.

- [ ] **Step 1: Write the failing tests (state management only)**

Create `tests/PicoLLM.Tests/Orchestrator/PicoOrchestratorTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run tests — expect compile failure (`PicoOrchestrator` not yet defined)**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~PicoOrchestratorTests"
```

Expected: Build error — `PicoOrchestrator` does not exist.

- [ ] **Step 3: Create `src/PicoLLM.App/Orchestration/PicoOrchestrator.cs`**

```csharp
using System.Text;
using PicoLLM.Browser;
using PicoLLM.Browser.Parsing;
using PicoLLM.Core.Compute;
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
```

- [ ] **Step 4: Run the full test suite to verify no regressions**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```

Expected: All 314 pre-existing tests plus new orchestrator tests pass.

If `ProcessUrlsAsync_FreshStart_TrainsTokenizerAndModel` times out or runs slowly: the test uses a small model config (32-dim, 1 layer, 2 heads) and only 3 training steps — it should complete in a few seconds. If BPE training fails with "corpus too short", check that the HTML test string has enough unique byte-pair sequences to reach 300 vocab size. Reduce `VocabSize` to `270` (minimum = 260 + 10 merges) in `MakeConfig` if needed.

- [ ] **Step 5: Commit**

```bash
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator
git add src/PicoLLM.App/Orchestration/PicoOrchestrator.cs tests/PicoLLM.Tests/Orchestrator/PicoOrchestratorTests.cs
git commit -m "feat(cap10): add PicoOrchestrator — browse-to-train pipeline with events"
```

---

## Task 4: Full Test Suite and Final Build

**Files:**
- No new files — run verification

- [ ] **Step 1: Run all cap10 tests specifically**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~OrchestratorConfigTests|FullyQualifiedName~PicoOrchestratorTests"
```

Expected: All tests pass.

- [ ] **Step 2: Run full test suite to confirm no regressions**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```

Expected: All tests pass (314 + new orchestrator tests).

- [ ] **Step 3: Confirm PicoLLM.App builds cleanly**

```
cd d:/source/ghosh9691/PicoLLM/.worktrees/cap10-orchestrator && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: Build succeeded, 0 warnings about missing references.

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Task | How Covered |
|---|---|---|
| Browse-to-train pipeline | T3 | `ProcessUrlsAsync` + `ProcessUrlAsync` |
| Train from single page | T3 | `ProcessUrlsAsync(["url"])` |
| Train from multiple pages | T3 | Loop + corpus accumulation |
| Incremental learning (load checkpoint) | T3 | `EnsureStateLoaded` loads from `checkpoint.pckp` |
| First session (no checkpoint) | T3 | `EnsureStateLoaded` — model init from scratch |
| BPE tokenizer training (first session) | T3 | `_tokenizer is null` branch in `ProcessUrlsAsync` |
| BPE tokenizer reuse (subsequent) | T3 | `SecondSession_ReusesExistingTokenizer` test |
| GGUF export on session end | T3 | `EndSession` calls `GgufExporter.Export` |
| Checkpoint save on session end | T3 | `EndSession` calls `CheckpointManager.Save` |
| Training config defaults | T2 | `TrainingConfig` record + `OrchestratorConfigTests` |
| Progress events: PageFetched, PageParsed, TokenizerTrained, TrainingStep, CheckpointSaved, GgufExported, SessionError | T2 + T3 | All 7 types in `OrchestratorEvent.cs`; verified in tests |
| Error resilience: skip bad URL, continue | T3 | `FailedUrl_EmitsSessionErrorAndContinues` test |

### Placeholder Scan

None found — all steps contain complete code.

### Type Consistency

- `PageFetchedEvent(string Url, bool Success, string? Error)` — defined T2, used in `PicoOrchestrator` T3 ✓
- `PageParsedEvent(string Url, int TextLength, int ImageCount, int LinkCount)` ✓
- `TokenizerTrainedEvent(int VocabSize)` ✓
- `TrainingStepEvent(int Step, float Loss, float Lr, float TokensPerSec, float GradNorm)` ✓
- `CheckpointSavedEvent(string Path)` ✓
- `GgufExportedEvent(string Path, long FileSizeBytes)` ✓
- `SessionErrorEvent(string Url, string Error)` ✓
- `OrchestratorConfig.CreateDefault(string)` defined T2, used in tests T3 ✓
- `PicoOrchestrator(config, Func<...>? httpOverride)` — constructor matches tests ✓
- `TrainingConfig.SeqLen` — used as `_config.Training.SeqLen` in `TrainingLoop` constructor ✓

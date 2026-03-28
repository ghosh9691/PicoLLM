# Orchestrator — Technical Design

## Pipeline Flow

```
User provides URL(s) or browses interactively
         │
         ▼
┌─────────────────────┐
│  Load or Init State  │
│  - Checkpoint?       │──► Load model + optimizer + step
│  - Tokenizer?        │──► Load tokenizer.json
│  - Neither?          │──► Init fresh model
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  For each URL:       │
│  1. Check robots.txt │
│  2. Fetch HTML       │
│  3. Parse → text     │
│  4. Emit events      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Corpus Accumulation │
│  - Append page text  │
│  - If first session  │
│    and no tokenizer: │
│    train BPE now     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Training Loop       │
│  - Tokenize corpus   │
│  - Sample batches    │
│  - N steps per page  │
│  - Emit metrics      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Session End         │
│  - Save checkpoint   │
│  - Export GGUF       │
│  - Report to UI      │
└─────────────────────┘
```

## Core Class

```csharp
public class PicoOrchestrator
{
    private readonly OrchestratorConfig _config;
    private readonly HttpFetcher _fetcher;
    private readonly HtmlParser _parser;
    private BpeTokenizer? _tokenizer;
    private PicoLLMModel? _model;
    private AdamW? _optimizer;
    private int _globalStep;

    public event Action<OrchestratorEvent>? OnProgress;

    public async Task ProcessUrlAsync(string url) { ... }
    public async Task ProcessUrlsAsync(IEnumerable<string> urls) { ... }
    public void EndSession(string ggufOutputPath) { ... }
}

public record OrchestratorConfig(
    ModelConfig Model,
    TrainingConfig Training,
    string DataDirectory,       // where checkpoints/tokenizer/gguf go
    int StepsPerPage = 100);

public record TrainingConfig(
    int BatchSize = 4,
    int SeqLen = 128,
    float LearningRate = 1e-4f,
    int WarmupSteps = 50,
    float MaxGradNorm = 1.0f);
```

## Event System

```csharp
public abstract record OrchestratorEvent(DateTime Timestamp);
public record PageFetchedEvent(string Url, bool Success, string? Error) : OrchestratorEvent(DateTime.UtcNow);
public record PageParsedEvent(string Url, int TextLength, int ImageCount, int LinkCount) : OrchestratorEvent(DateTime.UtcNow);
public record TokenizerTrainedEvent(int VocabSize) : OrchestratorEvent(DateTime.UtcNow);
public record TrainingStepEvent(int Step, float Loss, float Lr, float TokensPerSec, float GradNorm) : OrchestratorEvent(DateTime.UtcNow);
public record CheckpointSavedEvent(string Path) : OrchestratorEvent(DateTime.UtcNow);
public record GgufExportedEvent(string Path, long FileSizeBytes) : OrchestratorEvent(DateTime.UtcNow);
public record SessionErrorEvent(string Url, string Error) : OrchestratorEvent(DateTime.UtcNow);
```

## File Layout on Disk

```
{DataDirectory}/
├── tokenizer.json          # BPE tokenizer (persists across sessions)
├── checkpoint.pckp         # Training checkpoint (model + optimizer + step)
├── picollm.gguf            # Latest GGUF export
└── corpus/
    └── session_{timestamp}.txt  # Raw text per session (optional, for debugging)
```

## Project Location

`src/PicoLLM.App/Orchestration/`:
- `PicoOrchestrator.cs`
- `OrchestratorConfig.cs`
- `OrchestratorEvent.cs` — all event types

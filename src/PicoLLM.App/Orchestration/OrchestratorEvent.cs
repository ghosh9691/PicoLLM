using PicoLLM.Browser.Parsing;

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
/// <param name="Page">The fully parsed page including clean text, links, and images.</param>
public record PageParsedEvent(string Url, int TextLength, int ImageCount, int LinkCount, ParsedPage Page)
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

/// <summary>Emitted when the background training queue depth changes (page added or training started).</summary>
/// <param name="QueueDepth">Number of pages currently waiting to be trained.</param>
public record TrainingQueueChangedEvent(int QueueDepth)
    : OrchestratorEvent(DateTime.UtcNow);

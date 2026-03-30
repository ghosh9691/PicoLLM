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

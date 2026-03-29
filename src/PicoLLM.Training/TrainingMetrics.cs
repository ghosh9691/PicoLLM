namespace PicoLLM.Training;

/// <summary>
/// Snapshot of training metrics captured after each step.
/// </summary>
/// <param name="Step">The zero-indexed training step number.</param>
/// <param name="Loss">Cross-entropy loss for this step.</param>
/// <param name="LearningRate">Learning rate used for the optimizer update.</param>
/// <param name="TokensPerSec">Throughput in tokens processed per second.</param>
/// <param name="GradNorm">Global gradient L2 norm before clipping.</param>
public record TrainingMetrics(
    int Step,
    float Loss,
    float LearningRate,
    float TokensPerSec,
    float GradNorm);

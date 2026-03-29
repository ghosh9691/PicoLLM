using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Training;

/// <summary>
/// Orchestrates one training step: sample batch → forward → loss → backward → clip → optimizer update → metrics.
/// </summary>
/// <remarks>
/// Input/target pair creation:
/// Each sequence is a window of (seq_len + 1) tokens from the corpus.
/// inputs  = tokens[start .. start+seq_len]
/// targets = tokens[start+1 .. start+seq_len+1]   (shifted right by 1)
/// </remarks>
public sealed class TrainingLoop
{
    private readonly PicoLLMModel _model;
    private readonly AdamW _optimizer;
    private readonly LearningRateSchedule _schedule;
    private readonly float _clipNorm;
    private readonly int _batchSize;
    private readonly int _seqLen;
    private readonly Random _rng;

    /// <summary>
    /// Initializes the training loop.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="optimizer">The AdamW optimizer.</param>
    /// <param name="schedule">The learning rate schedule.</param>
    /// <param name="clipNorm">Maximum gradient norm for clipping (e.g. 1.0).</param>
    /// <param name="batchSize">Number of sequences per batch.</param>
    /// <param name="seqLen">Sequence length (number of input tokens).</param>
    /// <param name="seed">Optional random seed for reproducible batch sampling.</param>
    public TrainingLoop(
        PicoLLMModel model,
        AdamW optimizer,
        LearningRateSchedule schedule,
        float clipNorm = 1.0f,
        int batchSize = 4,
        int seqLen = 64,
        int? seed = null)
    {
        _model     = model;
        _optimizer = optimizer;
        _schedule  = schedule;
        _clipNorm  = clipNorm;
        _batchSize = batchSize;
        _seqLen    = seqLen;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Runs one training step on the given flat token corpus.
    /// </summary>
    /// <param name="tokens">Flat array of token IDs. Must have length ≥ (seqLen + 1).</param>
    /// <param name="step">Current step number (0-indexed), used by the LR schedule.</param>
    /// <returns>Training metrics for this step.</returns>
    public TrainingMetrics Step(int[] tokens, int step)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Length < _seqLen + 1)
            throw new ArgumentException(
                $"Token corpus length {tokens.Length} must be >= seqLen+1 = {_seqLen + 1}.");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Update learning rate
        float lr = _schedule.GetLR(step);
        _optimizer.LearningRate = lr;

        // Sample batch
        var (inputs, targets) = SampleBatch(tokens);

        // Zero gradients
        _model.ZeroGrad();

        // Forward
        var logits = _model.Forward(inputs);

        // Loss
        float loss = CrossEntropyLoss.Forward(logits, targets);

        // Backward
        var gradLogits = CrossEntropyLoss.Backward(logits, targets);
        _model.Backward(gradLogits);

        // Clip gradients
        var allParams = _model.GetAllParameters();
        float gradNorm = GradientClipper.ClipGradNorm(allParams, _clipNorm);

        // Optimizer step
        _optimizer.Step(allParams);

        sw.Stop();
        float tokensPerSec = (_batchSize * _seqLen) / (float)sw.Elapsed.TotalSeconds;

        return new TrainingMetrics(step, loss, lr, tokensPerSec, gradNorm);
    }

    private (int[,] inputs, int[,] targets) SampleBatch(int[] tokens)
    {
        int maxStart = tokens.Length - _seqLen - 1;
        var inputs  = new int[_batchSize, _seqLen];
        var targets = new int[_batchSize, _seqLen];

        for (int b = 0; b < _batchSize; b++)
        {
            int start = _rng.Next(0, maxStart + 1);
            for (int s = 0; s < _seqLen; s++)
            {
                inputs[b, s]  = tokens[start + s];
                targets[b, s] = tokens[start + s + 1];
            }
        }

        return (inputs, targets);
    }
}

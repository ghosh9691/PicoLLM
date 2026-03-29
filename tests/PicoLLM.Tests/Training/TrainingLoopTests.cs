using FluentAssertions;
using PicoLLM.Core.Model;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class TrainingLoopTests
{
    private static ModelConfig TinyConfig() => new(
        VocabSize: 32, EmbedDim: 16, NumHeads: 2,
        NumLayers: 1, FfMultiplier: 2, MaxSeqLen: 32);

    [Fact]
    public void Step_ReturnsMetrics_WithCorrectStep()
    {
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);
        var schedule = new LearningRateSchedule(0, 100, 1e-3f);
        var loop = new TrainingLoop(model, optimizer, schedule,
            clipNorm: 1f, batchSize: 2, seqLen: 8, seed: 1);

        var tokens = Enumerable.Range(0, 200).Select(i => i % 32).ToArray();
        var metrics = loop.Step(tokens, step: 5);

        metrics.Step.Should().Be(5);
        metrics.Loss.Should().BeGreaterThan(0f);
        metrics.LearningRate.Should().BeGreaterThan(0f).And.BeLessThanOrEqualTo(1e-3f);
        metrics.TokensPerSec.Should().BeGreaterThan(0f);
        metrics.GradNorm.Should().BeGreaterThanOrEqualTo(0f);
    }

    [Fact]
    public void Step_LossIsFinite()
    {
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW(lr: 1e-3f, weightDecay: 0f);
        var schedule = new LearningRateSchedule(0, 50, 1e-3f);
        var loop = new TrainingLoop(model, optimizer, schedule,
            clipNorm: 1f, batchSize: 2, seqLen: 8, seed: 2);

        var tokens = Enumerable.Range(0, 200).Select(i => i % 32).ToArray();
        var metrics = loop.Step(tokens, 0);

        float.IsFinite(metrics.Loss).Should().BeTrue();
        float.IsNaN(metrics.Loss).Should().BeFalse();
    }

    [Fact]
    public void MultipleSteps_LossDecreases_OnSameData()
    {
        // Overfitting test: running 50 steps on identical data should drive loss down.
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW(lr: 5e-3f, weightDecay: 0f);
        var schedule = new LearningRateSchedule(0, 100, 5e-3f);
        var loop = new TrainingLoop(model, optimizer, schedule,
            clipNorm: 1f, batchSize: 2, seqLen: 8, seed: 42);

        // Small fixed corpus so the model can overfit
        var tokens = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        float firstLoss = loop.Step(tokens, 0).Loss;
        float lastLoss = firstLoss;
        for (int s = 1; s < 50; s++)
            lastLoss = loop.Step(tokens, s).Loss;

        lastLoss.Should().BeLessThan(firstLoss,
            $"loss should decrease after 50 steps (was {firstLoss:F4}, now {lastLoss:F4})");
    }
}

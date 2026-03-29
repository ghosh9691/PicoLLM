using FluentAssertions;
using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class CheckpointManagerTests : IDisposable
{
    private readonly string _tempPath = Path.Combine(Path.GetTempPath(), $"pico_test_{Guid.NewGuid()}.ckpt");

    public void Dispose()
    {
        if (File.Exists(_tempPath)) File.Delete(_tempPath);
    }

    private static ModelConfig TinyConfig() => new(
        VocabSize: 32, EmbedDim: 16, NumHeads: 2,
        NumLayers: 1, FfMultiplier: 2, MaxSeqLen: 32);

    [Fact]
    public void SaveLoad_RestoresStep()
    {
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW();

        CheckpointManager.Save(model, optimizer, step: 42, _tempPath);

        var model2 = new PicoLLMModel(TinyConfig(), seed: 99); // different seed
        var optimizer2 = new AdamW();
        int restoredStep = CheckpointManager.Load(_tempPath, model2, optimizer2);

        restoredStep.Should().Be(42);
    }

    [Fact]
    public void SaveLoad_RestoresModelWeights_IdenticalOutput()
    {
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW();

        var tokenIds = new[,] { { 1, 2, 3, 4 } };
        var logitsBefore = model.Forward(tokenIds).Data.ToArray();

        CheckpointManager.Save(model, optimizer, step: 1, _tempPath);

        var model2 = new PicoLLMModel(TinyConfig(), seed: 99);
        CheckpointManager.Load(_tempPath, model2, new AdamW());
        var logitsAfter = model2.Forward(tokenIds).Data.ToArray();

        logitsAfter.Should().BeEquivalentTo(logitsBefore,
            options => options.WithStrictOrdering().Using<float>(ctx =>
                ctx.Subject.Should().BeApproximately(ctx.Expectation, 1e-5f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public void SaveLoad_RestoresOptimizerStep()
    {
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer = new AdamW();

        // Run one optimizer step so moments are populated
        var tokens = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var schedule = new LearningRateSchedule(0, 100, 1e-3f);
        var loop = new TrainingLoop(model, optimizer, schedule,
            clipNorm: 1f, batchSize: 1, seqLen: 8, seed: 1);
        loop.Step(tokens, 0);

        CheckpointManager.Save(model, optimizer, step: 100, _tempPath);

        var model2 = new PicoLLMModel(TinyConfig(), seed: 1);
        var optimizer2 = new AdamW();
        CheckpointManager.Load(_tempPath, model2, optimizer2);

        optimizer2.StepCount.Should().Be(100);
    }

    [Fact]
    public void Load_BadMagic_ThrowsInvalidDataException()
    {
        File.WriteAllBytes(_tempPath, [0xFF, 0xFF, 0xFF, 0xFF]);
        var model = new PicoLLMModel(TinyConfig(), seed: 1);
        Action act = () => CheckpointManager.Load(_tempPath, model, new AdamW());
        act.Should().Throw<InvalidDataException>();
    }
}

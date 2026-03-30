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

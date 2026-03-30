using FluentAssertions;
using PicoLLM.App.ViewModels;

namespace PicoLLM.Tests.UiShell;

public class SettingsViewModelTests : IDisposable
{
    private readonly string _dir =
        Path.Combine(Path.GetTempPath(), "pico_settings_" + Guid.NewGuid().ToString("N"));

    public SettingsViewModelTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void DefaultValues_AreCorrect()
    {
        var vm = new SettingsViewModel(_dir);
        vm.DataDirectory.Should().NotBeNullOrEmpty();
        vm.VocabSize.Should().Be(1024);
        vm.EmbedDim.Should().Be(128);
        vm.NumHeads.Should().Be(4);
        vm.NumLayers.Should().Be(4);
        vm.MaxSeqLen.Should().Be(512);
        vm.StepsPerPage.Should().Be(100);
        vm.LearningRate.Should().BeApproximately(1e-4f, 1e-7f);
        vm.BatchSize.Should().Be(4);
        vm.SeqLen.Should().Be(128);
        vm.WarmupSteps.Should().Be(50);
    }

    [Fact]
    public void SaveAndLoad_RoundTripsAllValues()
    {
        var vm = new SettingsViewModel(_dir)
        {
            DataDirectory = "/custom/path",
            VocabSize     = 2048,
            EmbedDim      = 256,
            NumHeads      = 8,
            NumLayers     = 6,
            MaxSeqLen     = 1024,
            StepsPerPage  = 200,
            LearningRate  = 3e-4f,
            BatchSize     = 8,
            SeqLen        = 256,
            WarmupSteps   = 100
        };
        vm.Save();

        var vm2 = new SettingsViewModel(_dir);
        vm2.Load();

        vm2.DataDirectory.Should().Be("/custom/path");
        vm2.VocabSize.Should().Be(2048);
        vm2.EmbedDim.Should().Be(256);
        vm2.NumHeads.Should().Be(8);
        vm2.NumLayers.Should().Be(6);
        vm2.MaxSeqLen.Should().Be(1024);
        vm2.StepsPerPage.Should().Be(200);
        vm2.LearningRate.Should().BeApproximately(3e-4f, 1e-6f);
        vm2.BatchSize.Should().Be(8);
        vm2.SeqLen.Should().Be(256);
        vm2.WarmupSteps.Should().Be(100);
    }

    [Fact]
    public void Load_WhenNoFile_KeepsDefaults()
    {
        var vm = new SettingsViewModel(_dir);
        vm.Load(); // no settings.json exists yet
        vm.VocabSize.Should().Be(1024);
    }

    [Fact]
    public void ToOrchestratorConfig_ReflectsAllSettings()
    {
        var vm = new SettingsViewModel(_dir) { DataDirectory = _dir };
        var config = vm.ToOrchestratorConfig();
        config.Model.VocabSize.Should().Be(vm.VocabSize);
        config.Model.EmbedDim.Should().Be(vm.EmbedDim);
        config.Model.NumHeads.Should().Be(vm.NumHeads);
        config.Model.NumLayers.Should().Be(vm.NumLayers);
        config.Model.MaxSeqLen.Should().Be(vm.MaxSeqLen);
        config.StepsPerPage.Should().Be(vm.StepsPerPage);
        config.Training.LearningRate.Should().BeApproximately(vm.LearningRate, 1e-7f);
        config.DataDirectory.Should().Be(_dir);
    }
}

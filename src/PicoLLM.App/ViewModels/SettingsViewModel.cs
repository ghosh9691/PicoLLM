using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using PicoLLM.App.Orchestration;
using PicoLLM.Core.Model;

namespace PicoLLM.App.ViewModels;

/// <summary>
/// Persisted application settings including data directory and model/training hyperparameters.
/// Saved as JSON to the platform AppData directory, not to <see cref="DataDirectory"/>
/// (which avoids a chicken-and-egg problem when DataDirectory itself changes).
/// </summary>
public partial class SettingsViewModel : ObservableObject
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly string _settingsDir;

    private string SettingsFilePath => Path.Combine(_settingsDir, "settings.json");

    // ── Data directory ────────────────────────────────────────────────────────

    /// <summary>Directory for tokenizer, checkpoints, and GGUF exports.</summary>
    [ObservableProperty] private string _dataDirectory = GetPlatformDefaultDataDir();

    // ── Model hyperparameters ─────────────────────────────────────────────────

    /// <summary>Target BPE vocabulary size.</summary>
    [ObservableProperty] private int _vocabSize = 1024;

    /// <summary>Token embedding dimension.</summary>
    [ObservableProperty] private int _embedDim = 128;

    /// <summary>Number of attention heads per layer.</summary>
    [ObservableProperty] private int _numHeads = 4;

    /// <summary>Number of transformer decoder layers.</summary>
    [ObservableProperty] private int _numLayers = 4;

    /// <summary>Maximum context length in tokens.</summary>
    [ObservableProperty] private int _maxSeqLen = 512;

    /// <summary>Training steps to run per fetched page.</summary>
    [ObservableProperty] private int _stepsPerPage = 100;

    // ── Training hyperparameters ──────────────────────────────────────────────

    /// <summary>Peak learning rate for AdamW.</summary>
    [ObservableProperty] private float _learningRate = 1e-4f;

    /// <summary>Number of sequences per training batch.</summary>
    [ObservableProperty] private int _batchSize = 4;

    /// <summary>Input sequence length for training (tokens).</summary>
    [ObservableProperty] private int _seqLen = 128;

    /// <summary>Linear LR warmup steps.</summary>
    [ObservableProperty] private int _warmupSteps = 50;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a <see cref="SettingsViewModel"/> that persists to <paramref name="settingsDir"/>.
    /// Defaults to the platform AppData directory when <paramref name="settingsDir"/> is null.
    /// </summary>
    public SettingsViewModel(string? settingsDir = null)
    {
        _settingsDir = settingsDir ?? GetPlatformAppDataDir();
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// <summary>Loads settings from <c>settings.json</c>. Silently keeps defaults if file absent or corrupt.</summary>
    public void Load()
    {
        var path = SettingsFilePath;
        if (!File.Exists(path)) return;
        try
        {
            var dto = JsonSerializer.Deserialize<SettingsDto>(File.ReadAllText(path), JsonOptions);
            if (dto is null) return;
            DataDirectory = dto.DataDirectory ?? DataDirectory;
            VocabSize     = dto.VocabSize;
            EmbedDim      = dto.EmbedDim;
            NumHeads      = dto.NumHeads;
            NumLayers     = dto.NumLayers;
            MaxSeqLen     = dto.MaxSeqLen;
            StepsPerPage  = dto.StepsPerPage;
            LearningRate  = dto.LearningRate;
            BatchSize     = dto.BatchSize;
            SeqLen        = dto.SeqLen;
            WarmupSteps   = dto.WarmupSteps;
        }
        catch { /* corrupt file — use defaults */ }
    }

    /// <summary>Saves current settings to <c>settings.json</c>.</summary>
    public void Save()
    {
        Directory.CreateDirectory(_settingsDir);
        var dto = new SettingsDto(DataDirectory, VocabSize, EmbedDim, NumHeads, NumLayers,
            MaxSeqLen, StepsPerPage, LearningRate, BatchSize, SeqLen, WarmupSteps);
        File.WriteAllText(SettingsFilePath, JsonSerializer.Serialize(dto, JsonOptions));
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// <summary>Builds an <see cref="OrchestratorConfig"/> from the current settings.</summary>
    public OrchestratorConfig ToOrchestratorConfig() =>
        new(
            Model: new ModelConfig(
                VocabSize:    VocabSize,
                EmbedDim:     EmbedDim,
                NumHeads:     NumHeads,
                NumLayers:    NumLayers,
                FfMultiplier: 4,
                MaxSeqLen:    MaxSeqLen),
            Training: new TrainingConfig(
                BatchSize:    BatchSize,
                SeqLen:       SeqLen,
                LearningRate: LearningRate,
                WarmupSteps:  WarmupSteps,
                MaxGradNorm:  1.0f),
            DataDirectory: DataDirectory,
            StepsPerPage:  StepsPerPage);

    // ── Platform helpers ──────────────────────────────────────────────────────

    private static string GetPlatformAppDataDir() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "PicoLLM");

    private static string GetPlatformDefaultDataDir() =>
        Path.Combine(GetPlatformAppDataDir(), "sessions");

    // ── DTO ───────────────────────────────────────────────────────────────────

    private record SettingsDto(
        string? DataDirectory,
        int     VocabSize,
        int     EmbedDim,
        int     NumHeads,
        int     NumLayers,
        int     MaxSeqLen,
        int     StepsPerPage,
        float   LearningRate,
        int     BatchSize,
        int     SeqLen,
        int     WarmupSteps);
}

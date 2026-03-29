namespace PicoLLM.Core.Model;

/// <summary>
/// Immutable configuration for a PicoLLM transformer model.
/// All hyperparameters are specified here and passed to the model constructor.
/// </summary>
/// <param name="VocabSize">Number of tokens in the vocabulary.</param>
/// <param name="EmbedDim">Hidden dimension / embedding size. Must be divisible by NumHeads.</param>
/// <param name="NumHeads">Number of attention heads. EmbedDim must be divisible by this.</param>
/// <param name="NumLayers">Number of stacked decoder blocks.</param>
/// <param name="FfMultiplier">Feedforward expansion factor (ff_dim = EmbedDim × FfMultiplier).</param>
/// <param name="MaxSeqLen">Maximum supported sequence length (for positional encoding).</param>
/// <param name="Dropout">Dropout probability (0.0 = disabled; used for training config only).</param>
public record ModelConfig(
    int VocabSize,
    int EmbedDim,
    int NumHeads,
    int NumLayers,
    int FfMultiplier,
    int MaxSeqLen,
    float Dropout = 0f);

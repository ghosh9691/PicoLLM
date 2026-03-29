using PicoLLM.Core.Activations;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Position-wise feedforward network with GELU activation.
/// Architecture: Linear(embed_dim → ff_dim) → GELU → Linear(ff_dim → embed_dim).
/// The hidden (ff) dimension is embed_dim × ff_multiplier (typically 4).
/// Input/output shape: [batch, seq, embed_dim].
/// </summary>
public sealed class FeedForward
{
    private readonly LinearLayer _up;    // [embed_dim, ff_dim]
    private readonly LinearLayer _down;  // [ff_dim, embed_dim]

    /// <summary>The hidden (expanded) dimension: embed_dim × ff_multiplier.</summary>
    public int FfDim { get; }

    /// <summary>
    /// Initializes the feedforward sublayer.
    /// </summary>
    /// <param name="embedDim">Input and output dimension.</param>
    /// <param name="ffMultiplier">Expansion factor for the hidden layer (default 4).</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public FeedForward(int embedDim, int ffMultiplier = 4, int? seed = null)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (ffMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(ffMultiplier));

        FfDim = embedDim * ffMultiplier;
        _up   = new LinearLayer(embedDim, FfDim,   useBias: true, seed: seed);
        _down = new LinearLayer(FfDim,   embedDim, useBias: true, seed: seed.HasValue ? seed + 1 : null);
    }

    /// <summary>
    /// Forward pass: up-project → GELU → down-project.
    /// </summary>
    /// <param name="x">Input [batch, seq, embed_dim].</param>
    /// <returns>Output [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        var hidden = GELU.Forward(_up.Forward(x));
        return _down.Forward(hidden);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _up.ZeroGrad();
        _down.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters from both linear layers.</summary>
    public IEnumerable<Tensor> Parameters() =>
        _up.Parameters().Concat(_down.Parameters());
}

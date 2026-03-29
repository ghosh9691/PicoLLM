using PicoLLM.Core.Activations;
using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Position-wise feedforward network with GELU activation.
/// Architecture: Linear(embed_dim → ff_dim) → GELU → Linear(ff_dim → embed_dim).
/// The hidden (ff) dimension is embed_dim × ff_multiplier (typically 4).
/// Input/output shape: [batch, seq, embed_dim].
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
public sealed class FeedForward : ILayer
{
    private readonly LinearLayer _up;    // [embed_dim, ff_dim]
    private readonly LinearLayer _down;  // [ff_dim, embed_dim]
    private Tensor? _upOutput;           // cached pre-GELU values for backward

    /// <summary>The hidden (expanded) dimension: embed_dim × ff_multiplier.</summary>
    public int FfDim { get; }

    /// <summary>The up-projection linear layer [embed_dim → ff_dim].</summary>
    public LinearLayer Up => _up;

    /// <summary>The down-projection linear layer [ff_dim → embed_dim].</summary>
    public LinearLayer Down => _down;

    /// <summary>
    /// Initializes the feedforward sublayer.
    /// </summary>
    /// <param name="embedDim">Input and output dimension.</param>
    /// <param name="ffMultiplier">Expansion factor for the hidden layer (default 4).</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public FeedForward(int embedDim, int ffMultiplier = 4, int? seed = null,
        IComputeProvider? computeProvider = null)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (ffMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(ffMultiplier));

        FfDim = embedDim * ffMultiplier;
        _up   = new LinearLayer(embedDim, FfDim,   useBias: true, seed: seed,
                    computeProvider: computeProvider);
        _down = new LinearLayer(FfDim,   embedDim, useBias: true,
                    seed: seed.HasValue ? seed + 1 : null,
                    computeProvider: computeProvider);
    }

    /// <summary>
    /// Forward pass: up-project → GELU → down-project.
    /// Caches pre-GELU values for <see cref="Backward"/>.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        _upOutput = _up.Forward(x);
        var hidden = GELU.Forward(_upOutput);
        return _down.Forward(hidden);
    }

    /// <summary>
    /// Backward pass.
    /// d_down_input = down.Backward(gradOutput)
    /// d_up_output  = d_down_input * GELU'(_upOutput)
    /// dx           = up.Backward(d_up_output)
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        if (_upOutput is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        var dHidden    = _down.Backward(gradOutput);
        var dUpOutput  = TensorMath.Multiply(dHidden, GELU.Backward(_upOutput));
        return _up.Backward(dUpOutput);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _up.ZeroGrad();
        _down.ZeroGrad();
        _upOutput = null;
    }

    /// <summary>Returns all learnable parameters from both linear layers.</summary>
    public IEnumerable<Parameter> Parameters() =>
        _up.Parameters().Concat(_down.Parameters());
}

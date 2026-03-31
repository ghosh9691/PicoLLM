using PicoLLM.Core.Activations;
using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Position-wise feedforward network using the SwiGLU gated architecture (LLaMA-style).
/// Architecture: (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down.
/// No bias terms on any projection. The hidden (ff) dimension is embed_dim × ff_multiplier.
/// Input/output shape: [batch, seq, embed_dim].
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
public sealed class FeedForward : ILayer
{
    private readonly LinearLayer _gate;  // [embed_dim → ff_dim], no bias
    private readonly LinearLayer _up;    // [embed_dim → ff_dim], no bias
    private readonly LinearLayer _down;  // [ff_dim → embed_dim], no bias

    // Cached intermediate values for backward pass
    private Tensor? _lastGatePreAct;   // x · W_gate  (pre-SiLU)
    private Tensor? _lastGateOutput;   // SiLU(x · W_gate)
    private Tensor? _lastUpOutput;     // x · W_up

    /// <summary>The hidden (expanded) dimension: embed_dim × ff_multiplier.</summary>
    public int FfDim { get; }

    /// <summary>The gate-projection linear layer [embed_dim → ff_dim].</summary>
    public LinearLayer Gate => _gate;

    /// <summary>The up-projection linear layer [embed_dim → ff_dim].</summary>
    public LinearLayer Up => _up;

    /// <summary>The down-projection linear layer [ff_dim → embed_dim].</summary>
    public LinearLayer Down => _down;

    /// <summary>
    /// Initializes the SwiGLU feedforward sublayer.
    /// </summary>
    /// <param name="embedDim">Input and output dimension.</param>
    /// <param name="ffMultiplier">Expansion factor for the hidden layer (default 4).</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    /// <param name="computeProvider">Optional compute provider for matrix multiplication.</param>
    public FeedForward(int embedDim, int ffMultiplier = 4, int? seed = null,
        IComputeProvider? computeProvider = null)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (ffMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(ffMultiplier));

        FfDim = embedDim * ffMultiplier;
        _gate = new LinearLayer(embedDim, FfDim,   useBias: false, seed: seed,
                    computeProvider: computeProvider);
        _up   = new LinearLayer(embedDim, FfDim,   useBias: false,
                    seed: seed.HasValue ? seed + 1 : null,
                    computeProvider: computeProvider);
        _down = new LinearLayer(FfDim,   embedDim, useBias: false,
                    seed: seed.HasValue ? seed + 2 : null,
                    computeProvider: computeProvider);
    }

    /// <summary>
    /// Forward pass: SwiGLU — (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down.
    /// Caches intermediate values for <see cref="Backward"/>.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        _lastGatePreAct = _gate.Forward(x);
        _lastGateOutput = SiLU.Forward(_lastGatePreAct);
        _lastUpOutput   = _up.Forward(x);
        var hidden = TensorMath.Multiply(_lastGateOutput, _lastUpOutput);
        return _down.Forward(hidden);
    }

    /// <summary>
    /// Backward pass through SwiGLU.
    /// d_hidden     = down.Backward(gradOutput)
    /// d_gate_act   = d_hidden ⊙ _lastUpOutput
    /// d_up_pre     = d_hidden ⊙ _lastGateOutput
    /// d_gate_pre   = d_gate_act ⊙ SiLU'(_lastGatePreAct)
    /// dx           = gate.Backward(d_gate_pre) + up.Backward(d_up_pre)
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        if (_lastGatePreAct is null || _lastGateOutput is null || _lastUpOutput is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        var dHidden    = _down.Backward(gradOutput);
        var dGateAct   = TensorMath.Multiply(dHidden, _lastUpOutput);
        var dUpPre     = TensorMath.Multiply(dHidden, _lastGateOutput);
        var dGatePre   = TensorMath.Multiply(dGateAct, SiLU.Derivative(_lastGatePreAct));
        var dxFromGate = _gate.Backward(dGatePre);
        var dxFromUp   = _up.Backward(dUpPre);
        return TensorMath.Add(dxFromGate, dxFromUp);
    }

    /// <summary>Zeros the accumulated gradients and clears cached activations.</summary>
    public void ZeroGrad()
    {
        _gate.ZeroGrad();
        _up.ZeroGrad();
        _down.ZeroGrad();
        _lastGatePreAct = null;
        _lastGateOutput = null;
        _lastUpOutput   = null;
    }

    /// <summary>Returns all learnable parameters: gate, then up, then down.</summary>
    public IEnumerable<Parameter> Parameters() =>
        _gate.Parameters().Concat(_up.Parameters()).Concat(_down.Parameters());
}

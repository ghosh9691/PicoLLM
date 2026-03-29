using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Training;

/// <summary>
/// Contract for a neural network layer that supports both forward and backward passes.
/// All inner transformer layers (Linear, LayerNorm, MHA, FFN, DecoderBlock) implement this.
/// </summary>
/// <remarks>
/// Typical usage pattern:
/// <code>
/// // Forward pass — layer caches inputs internally
/// var output = layer.Forward(input);
///
/// // ... compute loss ...
///
/// // Backward pass — layer uses its cached inputs to compute gradients
/// var gradInput = layer.Backward(gradOutput);
/// </code>
/// The layer accumulates gradients into its <see cref="Parameters"/> on each <see cref="Backward"/> call.
/// Call <see cref="ZeroGrad"/> before each new forward/backward cycle.
/// </remarks>
public interface ILayer
{
    /// <summary>
    /// Forward pass. The layer MUST cache the input (and any intermediate activations)
    /// required for the subsequent <see cref="Backward"/> call.
    /// </summary>
    Tensor Forward(Tensor input);

    /// <summary>
    /// Backward pass. Accumulates parameter gradients and returns the gradient
    /// with respect to the layer's input (to be passed to the previous layer).
    /// </summary>
    Tensor Backward(Tensor gradOutput);

    /// <summary>Returns all trainable parameters in this layer.</summary>
    IEnumerable<Parameter> Parameters();

    /// <summary>Zeros all accumulated gradients. Call before each training step.</summary>
    void ZeroGrad();
}

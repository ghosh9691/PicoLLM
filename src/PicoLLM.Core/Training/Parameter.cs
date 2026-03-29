using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Training;

/// <summary>
/// Wraps a trainable weight tensor and its accumulated gradient tensor.
/// Every learnable parameter in the model is stored in one of these.
/// The optimizer reads <see cref="Data"/> and <see cref="Grad"/> each step.
/// </summary>
public sealed class Parameter
{
    /// <summary>The weight values. Updated by the optimizer each step.</summary>
    public Tensor Data { get; }

    /// <summary>
    /// Accumulated gradient. Populated by backward passes, zeroed by <see cref="ZeroGrad"/>.
    /// The optimizer reads this to compute the parameter update.
    /// </summary>
    public Tensor Grad { get; }

    /// <summary>
    /// Initializes a parameter with the given data tensor.
    /// A zero gradient tensor of the same shape is created automatically.
    /// </summary>
    public Parameter(Tensor data)
    {
        ArgumentNullException.ThrowIfNull(data);
        Data = data;
        Grad = TensorFactory.Zeros(data.GetShape());
    }

    /// <summary>Zeros the gradient tensor. Call before each forward/backward pass.</summary>
    public void ZeroGrad() => Grad.MutableData.Clear();
}

using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Rectified Linear Unit activation function: f(x) = max(0, x).
/// Used in feedforward sublayers when GELU is not desired.
/// </summary>
public static class ReLU
{
    /// <summary>Applies ReLU element-wise: output[i] = max(0, input[i]).</summary>
    public static Tensor Forward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = MathF.Max(0f, src[i]);
        return new Tensor(input.GetShape(), result);
    }

    /// <summary>
    /// ReLU derivative with respect to the pre-activation input.
    /// d/dx ReLU(x) = 1 if x > 0, else 0.
    /// Multiply this by the upstream gradient to get the backward pass output.
    /// </summary>
    public static Tensor Backward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = src[i] > 0f ? 1f : 0f;
        return new Tensor(input.GetShape(), result);
    }
}

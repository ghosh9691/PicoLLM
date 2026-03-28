using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x)).
/// Squashes values to the range (0, 1). Used in gating mechanisms.
/// </summary>
public static class Sigmoid
{
    /// <summary>Applies sigmoid element-wise.</summary>
    public static Tensor Forward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = SigmoidScalar(src[i]);
        return new Tensor(input.GetShape(), result);
    }

    /// <summary>
    /// Sigmoid derivative with respect to the pre-activation input.
    /// d/dx σ(x) = σ(x) · (1 − σ(x)).
    /// </summary>
    public static Tensor Backward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            float s = SigmoidScalar(src[i]);
            result[i] = s * (1f - s);
        }
        return new Tensor(input.GetShape(), result);
    }

    internal static float SigmoidScalar(float x) => 1f / (1f + MathF.Exp(-x));
}

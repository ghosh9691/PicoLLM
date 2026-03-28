using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Hyperbolic tangent activation function: tanh(x) = (e^x − e^(-x)) / (e^x + e^(-x)).
/// Squashes values to the range (−1, 1).
/// </summary>
public static class Tanh
{
    /// <summary>Applies tanh element-wise.</summary>
    public static Tensor Forward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = MathF.Tanh(src[i]);
        return new Tensor(input.GetShape(), result);
    }

    /// <summary>
    /// Tanh derivative with respect to the pre-activation input.
    /// d/dx tanh(x) = 1 − tanh²(x) = sech²(x).
    /// </summary>
    public static Tensor Backward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            float t = MathF.Tanh(src[i]);
            result[i] = 1f - t * t;
        }
        return new Tensor(input.GetShape(), result);
    }
}

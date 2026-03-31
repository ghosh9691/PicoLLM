using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Sigmoid Linear Unit activation function.
/// SiLU(x) = x · σ(x), where σ is the sigmoid function.
/// Used as the gate activation in the SwiGLU feedforward sublayer.
/// </summary>
public static class SiLU
{
    /// <summary>Applies SiLU element-wise: SiLU(x) = x · σ(x).</summary>
    public static Tensor Forward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = src[i] * Sigmoid.SigmoidScalar(src[i]);
        return new Tensor(input.GetShape(), result);
    }

    /// <summary>
    /// SiLU derivative with respect to the pre-activation input.
    /// d/dx SiLU(x) = σ(x) · (1 + x · (1 − σ(x))).
    /// </summary>
    public static Tensor Derivative(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            float s = Sigmoid.SigmoidScalar(src[i]);
            result[i] = s * (1f + src[i] * (1f - s));
        }
        return new Tensor(input.GetShape(), result);
    }
}

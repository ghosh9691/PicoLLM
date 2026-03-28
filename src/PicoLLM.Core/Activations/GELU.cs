using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Gaussian Error Linear Unit activation function.
/// Used in the feedforward sublayer of transformer decoder blocks (as in GPT-2).
/// Approximation: GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))).
/// </summary>
public static class GELU
{
    /// <summary>√(2/π) constant used in the GELU approximation.</summary>
    private const float Sqrt2OverPi = 0.7978845608f;

    /// <summary>Cubic coefficient in the GELU approximation.</summary>
    private const float CubicCoeff = 0.044715f;

    /// <summary>Applies GELU element-wise using the tanh approximation.</summary>
    public static Tensor Forward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = GeluScalar(src[i]);
        return new Tensor(input.GetShape(), result);
    }

    /// <summary>
    /// GELU derivative with respect to the pre-activation input.
    /// d/dx GELU(x) = 0.5·tanh(inner) + (0.5·x·sech²(inner)·d_inner) + 0.5
    /// where inner = √(2/π)·(x + 0.044715·x³) and d_inner = √(2/π)·(1 + 3·0.044715·x²).
    /// </summary>
    public static Tensor Backward(Tensor input)
    {
        var src = input.Data;
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = GeluDerivScalar(src[i]);
        return new Tensor(input.GetShape(), result);
    }

    private static float GeluScalar(float x)
    {
        float inner = Sqrt2OverPi * (x + CubicCoeff * x * x * x);
        return 0.5f * x * (1f + MathF.Tanh(inner));
    }

    private static float GeluDerivScalar(float x)
    {
        float inner = Sqrt2OverPi * (x + CubicCoeff * x * x * x);
        float tanhInner = MathF.Tanh(inner);
        float sech2 = 1f - tanhInner * tanhInner; // sech²(inner)
        float dInner = Sqrt2OverPi * (1f + 3f * CubicCoeff * x * x);
        return 0.5f * (1f + tanhInner) + 0.5f * x * sech2 * dInner;
    }
}

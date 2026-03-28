using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Layer normalization with learnable scale (gamma) and shift (beta) parameters.
/// Normalizes each vector of length <see cref="EmbedDim"/> along the last dimension.
/// Formula: output = gamma * (x - mean) / sqrt(var + eps) + beta
/// </summary>
public sealed class LayerNorm
{
    private const float Eps = 1e-5f;

    /// <summary>Scale parameter, shape [embed_dim], initialized to 1.</summary>
    public Tensor Gamma { get; }

    /// <summary>Shift parameter, shape [embed_dim], initialized to 0.</summary>
    public Tensor Beta { get; }

    /// <summary>Accumulated gradient for gamma.</summary>
    public Tensor GammaGrad { get; }

    /// <summary>Accumulated gradient for beta.</summary>
    public Tensor BetaGrad { get; }

    /// <summary>Number of features in the last dimension.</summary>
    public int EmbedDim { get; }

    /// <summary>Initializes LayerNorm for vectors of length <paramref name="embedDim"/>.</summary>
    public LayerNorm(int embedDim)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        EmbedDim = embedDim;
        Gamma = TensorFactory.Ones(embedDim);
        Beta = TensorFactory.Zeros(embedDim);
        GammaGrad = TensorFactory.Zeros(embedDim);
        BetaGrad = TensorFactory.Zeros(embedDim);
    }

    /// <summary>
    /// Forward pass. Normalizes each last-dimension vector independently.
    /// Input shape: [..., embed_dim]. Output shape equals input shape.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        int D = EmbedDim;
        if (x.Shape[x.Rank - 1] != D)
            throw new ArgumentException(
                $"Last dimension {x.Shape[x.Rank - 1]} does not match EmbedDim {D}.");

        int outerSize = x.Length / D;
        var src = x.Data;
        var gamma = Gamma.Data;
        var beta = Beta.Data;
        var result = new float[x.Length];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * D;

            // Compute mean along last dimension
            float mean = 0f;
            for (int d = 0; d < D; d++) mean += src[offset + d];
            mean /= D;

            // Compute variance along last dimension
            float var_ = 0f;
            for (int d = 0; d < D; d++)
            {
                float diff = src[offset + d] - mean;
                var_ += diff * diff;
            }
            var_ /= D;

            // Normalize, scale, shift
            float invStd = 1f / MathF.Sqrt(var_ + Eps);
            for (int d = 0; d < D; d++)
                result[offset + d] = gamma[d] * ((src[offset + d] - mean) * invStd) + beta[d];
        }

        return new Tensor(x.GetShape(), result);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        GammaGrad.MutableData.Clear();
        BetaGrad.MutableData.Clear();
    }

    /// <summary>Returns all learnable parameters: [Gamma, Beta].</summary>
    public IEnumerable<Tensor> Parameters() { yield return Gamma; yield return Beta; }
}

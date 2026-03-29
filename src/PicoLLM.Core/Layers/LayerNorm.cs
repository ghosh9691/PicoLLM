using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Layer normalization with learnable scale (gamma) and shift (beta) parameters.
/// Normalizes each vector of length <see cref="EmbedDim"/> along the last dimension.
/// Formula: output = gamma * (x - mean) / sqrt(var + eps) + beta
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
public sealed class LayerNorm : ILayer
{
    private const float Eps = 1e-5f;

    private readonly Parameter _gammaParam;
    private readonly Parameter _betaParam;
    private Tensor? _lastInput;

    /// <summary>Scale parameter, shape [embed_dim], initialized to 1.</summary>
    public Tensor Gamma => _gammaParam.Data;

    /// <summary>Shift parameter, shape [embed_dim], initialized to 0.</summary>
    public Tensor Beta => _betaParam.Data;

    /// <summary>Accumulated gradient for gamma.</summary>
    public Tensor GammaGrad => _gammaParam.Grad;

    /// <summary>Accumulated gradient for beta.</summary>
    public Tensor BetaGrad => _betaParam.Grad;

    /// <summary>Number of features in the last dimension.</summary>
    public int EmbedDim { get; }

    /// <summary>Initializes LayerNorm for vectors of length <paramref name="embedDim"/>.</summary>
    public LayerNorm(int embedDim)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        EmbedDim = embedDim;
        _gammaParam = new Parameter(TensorFactory.Ones(embedDim));
        _betaParam  = new Parameter(TensorFactory.Zeros(embedDim));
    }

    /// <summary>
    /// Forward pass. Normalizes each last-dimension vector independently.
    /// Caches input for <see cref="Backward"/>.
    /// Input shape: [..., embed_dim]. Output shape equals input shape.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        int D = EmbedDim;
        if (x.Shape[x.Rank - 1] != D)
            throw new ArgumentException(
                $"Last dimension {x.Shape[x.Rank - 1]} does not match EmbedDim {D}.");

        _lastInput = x;
        int outerSize = x.Length / D;
        var src = x.Data;
        var gamma = Gamma.Data;
        var beta = Beta.Data;
        var result = new float[x.Length];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * D;
            float mean = 0f;
            for (int d = 0; d < D; d++) mean += src[offset + d];
            mean /= D;

            float var_ = 0f;
            for (int d = 0; d < D; d++)
            {
                float diff = src[offset + d] - mean;
                var_ += diff * diff;
            }
            var_ /= D;

            float invStd = 1f / MathF.Sqrt(var_ + Eps);
            for (int d = 0; d < D; d++)
                result[offset + d] = gamma[d] * ((src[offset + d] - mean) * invStd) + beta[d];
        }

        return new Tensor(x.GetShape(), result);
    }

    /// <summary>
    /// Backward pass. Computes gradients for gamma and beta, returns dx.
    /// dx = (1/D) * invStd * [D*dxHat - Σ(dxHat) - x̂*Σ(dxHat*x̂)]
    /// where dxHat = dL/dy * gamma.
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        if (_lastInput is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        int D = EmbedDim;
        int outerSize = _lastInput.Length / D;
        var src = _lastInput.Data;
        var dOut = gradOutput.Data;
        var gamma = Gamma.Data;
        var gammaGrad = GammaGrad.MutableData;
        var betaGrad  = BetaGrad.MutableData;
        var dx = new float[_lastInput.Length];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * D;

            // Recompute statistics (same as forward)
            float mean = 0f;
            for (int d = 0; d < D; d++) mean += src[offset + d];
            mean /= D;

            float var_ = 0f;
            for (int d = 0; d < D; d++)
            {
                float diff = src[offset + d] - mean;
                var_ += diff * diff;
            }
            var_ /= D;
            float invStd = 1f / MathF.Sqrt(var_ + Eps);

            // x̂ = (x - mean) * invStd
            var xHat = new float[D];
            for (int d = 0; d < D; d++) xHat[d] = (src[offset + d] - mean) * invStd;

            // Accumulate gamma and beta gradients
            for (int d = 0; d < D; d++)
            {
                gammaGrad[d] += dOut[offset + d] * xHat[d];
                betaGrad[d]  += dOut[offset + d];
            }

            // dL/dx̂ = dL/dy * gamma
            var dxHat = new float[D];
            for (int d = 0; d < D; d++) dxHat[d] = dOut[offset + d] * gamma[d];

            float sum1 = 0f, sum2 = 0f;
            for (int d = 0; d < D; d++) { sum1 += dxHat[d] * xHat[d]; sum2 += dxHat[d]; }

            // dx = (1/D) * invStd * (D*dxHat - sum2 - x̂*sum1)
            for (int d = 0; d < D; d++)
                dx[offset + d] = (1f / D) * invStd * (D * dxHat[d] - sum2 - xHat[d] * sum1);
        }

        return new Tensor(_lastInput.GetShape(), dx);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _gammaParam.ZeroGrad();
        _betaParam.ZeroGrad();
        _lastInput = null;
    }

    /// <summary>Returns all learnable parameters: [Gamma, Beta].</summary>
    public IEnumerable<Parameter> Parameters()
    {
        yield return _gammaParam;
        yield return _betaParam;
    }
}

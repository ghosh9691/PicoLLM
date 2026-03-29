using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Fully-connected (dense) linear layer: output = input @ Weights + Bias.
/// Supports 2D input [rows, in_features] and 3D input [batch, seq, in_features].
/// All leading dimensions are preserved; only the last dimension is transformed.
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
public sealed class LinearLayer : ILayer
{
    private readonly Parameter _weightsParam;
    private readonly Parameter? _biasParam;
    private Tensor? _lastInput;

    /// <summary>Weight matrix, shape [in_features, out_features].</summary>
    public Tensor Weights => _weightsParam.Data;

    /// <summary>Bias vector, shape [out_features], or null if bias is disabled.</summary>
    public Tensor? Bias => _biasParam?.Data;

    /// <summary>Accumulated gradient for weights (same shape as Weights).</summary>
    public Tensor WeightGrad => _weightsParam.Grad;

    /// <summary>Accumulated gradient for bias, or null if bias is disabled.</summary>
    public Tensor? BiasGrad => _biasParam?.Grad;

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>
    /// Initializes a linear layer with Xavier uniform weight initialization.
    /// </summary>
    /// <param name="inFeatures">Number of input features.</param>
    /// <param name="outFeatures">Number of output features.</param>
    /// <param name="useBias">Whether to include a bias term.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public LinearLayer(int inFeatures, int outFeatures, bool useBias = true, int? seed = null)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weightsParam = new Parameter(
            TensorFactory.XavierUniform([inFeatures, outFeatures], inFeatures, outFeatures, seed));

        if (useBias)
            _biasParam = new Parameter(TensorFactory.Zeros(outFeatures));
    }

    /// <summary>
    /// Forward pass. output = input @ Weights + Bias.
    /// Caches input for use in <see cref="Backward"/>.
    /// Supports any rank ≥ 2: all leading dimensions pass through unchanged.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        int rank = input.Rank;
        int lastDim = input.Shape[rank - 1];
        if (lastDim != InFeatures)
            throw new ArgumentException(
                $"Last dimension {lastDim} does not match InFeatures {InFeatures}.");

        _lastInput = input;

        int rows = input.Length / InFeatures;
        var flat = TensorMath.Reshape(input, rows, InFeatures);
        var result = TensorMath.MatMul(flat, Weights);

        if (Bias is not null)
        {
            var bias = Bias.Data;
            var r = result.MutableData;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < OutFeatures; j++)
                    r[i * OutFeatures + j] += bias[j];
        }

        var outShape = new int[rank];
        for (int i = 0; i < rank - 1; i++) outShape[i] = input.Shape[i];
        outShape[rank - 1] = OutFeatures;
        return TensorMath.Reshape(result, outShape);
    }

    /// <summary>
    /// Backward pass.
    /// Accumulates dW and db into parameter gradients.
    /// Returns gradient w.r.t. input (dx = gradOutput @ W^T).
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        if (_lastInput is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        int rank = _lastInput.Rank;
        int rows = _lastInput.Length / InFeatures;

        var gradFlat  = TensorMath.Reshape(gradOutput, rows, OutFeatures); // [rows, out]
        var inputFlat = TensorMath.Reshape(_lastInput, rows, InFeatures);  // [rows, in]

        // dW += input^T @ grad
        var dW = TensorMath.MatMul(TensorMath.Transpose(inputFlat, 0, 1), gradFlat); // [in, out]
        var wg = WeightGrad.MutableData;
        var dwData = dW.Data;
        for (int i = 0; i < wg.Length; i++) wg[i] += dwData[i];

        // db += Σ(grad, axis=0)
        if (_biasParam is not null)
        {
            var bg = _biasParam.Grad.MutableData;
            var gd = gradFlat.Data;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < OutFeatures; j++)
                    bg[j] += gd[i * OutFeatures + j];
        }

        // dx = grad @ W^T → [rows, in], reshape to original shape
        var dxFlat = TensorMath.MatMul(gradFlat, TensorMath.Transpose(Weights, 0, 1));
        return TensorMath.Reshape(dxFlat, _lastInput.GetShape());
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _weightsParam.ZeroGrad();
        _biasParam?.ZeroGrad();
        _lastInput = null;
    }

    /// <summary>Returns all learnable parameters (Weights, then Bias if present).</summary>
    public IEnumerable<Parameter> Parameters()
    {
        yield return _weightsParam;
        if (_biasParam is not null) yield return _biasParam;
    }
}

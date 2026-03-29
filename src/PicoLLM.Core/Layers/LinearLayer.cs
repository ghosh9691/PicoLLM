using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Fully-connected (dense) linear layer: output = input @ Weights + Bias.
/// Supports 2D input [rows, in_features] and 3D input [batch, seq, in_features].
/// All leading dimensions are preserved; only the last dimension is transformed.
/// </summary>
public sealed class LinearLayer
{
    /// <summary>Weight matrix, shape [in_features, out_features].</summary>
    public Tensor Weights { get; }

    /// <summary>Bias vector, shape [out_features], or null if bias is disabled.</summary>
    public Tensor? Bias { get; }

    /// <summary>Accumulated gradient for weights.</summary>
    public Tensor WeightGrad { get; }

    /// <summary>Accumulated gradient for bias, or null if bias is disabled.</summary>
    public Tensor? BiasGrad { get; }

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
        Weights = TensorFactory.XavierUniform([inFeatures, outFeatures], inFeatures, outFeatures, seed);
        WeightGrad = TensorFactory.Zeros(inFeatures, outFeatures);

        if (useBias)
        {
            Bias = TensorFactory.Zeros(outFeatures);
            BiasGrad = TensorFactory.Zeros(outFeatures);
        }
    }

    /// <summary>
    /// Forward pass. output = input @ Weights + Bias.
    /// Supports any rank ≥ 2: all leading dimensions pass through unchanged.
    /// </summary>
    /// <param name="input">Input tensor with last dimension = InFeatures.</param>
    /// <returns>Output tensor with last dimension = OutFeatures.</returns>
    public Tensor Forward(Tensor input)
    {
        int rank = input.Rank;
        int lastDim = input.Shape[rank - 1];
        if (lastDim != InFeatures)
            throw new ArgumentException(
                $"Last dimension {lastDim} does not match InFeatures {InFeatures}.");

        // Flatten all leading dims into one row dimension, matmul, then restore shape.
        int rows = input.Length / InFeatures;
        var flat = TensorMath.Reshape(input, rows, InFeatures);  // [rows, in]
        var result = TensorMath.MatMul(flat, Weights);            // [rows, out]

        if (Bias is not null)
        {
            var bias = Bias.Data;
            var r = result.MutableData;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < OutFeatures; j++)
                    r[i * OutFeatures + j] += bias[j];
        }

        // Restore original leading dims + OutFeatures
        var outShape = new int[rank];
        for (int i = 0; i < rank - 1; i++) outShape[i] = input.Shape[i];
        outShape[rank - 1] = OutFeatures;
        return TensorMath.Reshape(result, outShape);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        WeightGrad.MutableData.Clear();
        BiasGrad?.MutableData.Clear();
    }

    /// <summary>Returns all learnable parameters (Weights, then Bias if present).</summary>
    public IEnumerable<Tensor> Parameters()
    {
        yield return Weights;
        if (Bias is not null) yield return Bias;
    }
}

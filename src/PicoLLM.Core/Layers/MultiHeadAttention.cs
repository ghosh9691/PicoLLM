using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Scaled dot-product multi-head self-attention with causal masking.
/// Input shape: [batch, seq, embed_dim]. Output shape: same.
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
/// <remarks>
/// Forward pipeline (GPT-2 style):
/// <list type="number">
/// <item>Project x → Q, K, V via three separate linear layers: each [B,S,E].</item>
/// <item>Reshape [B,S,E] → [B,H,S,head_dim] for Q, K, V.</item>
/// <item>scores = (Q @ K^T) / sqrt(head_dim) → [B,H,S,S].</item>
/// <item>Apply causal mask (upper triangle → −∞).</item>
/// <item>weights = softmax(scores, axis=−1) → [B,H,S,S].</item>
/// <item>context = weights @ V → [B,H,S,head_dim].</item>
/// <item>Reshape context back to [B,S,E].</item>
/// <item>Apply output projection.</item>
/// </list>
/// </remarks>
public sealed class MultiHeadAttention : ILayer
{
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _embedDim;

    // Cached for backward
    private Tensor? _lastInput;
    private Tensor? _qHeads;   // [B,H,S,D] after permute
    private Tensor? _kHeads;   // [B,H,S,D] after permute
    private Tensor? _vHeads;   // [B,H,S,D] after permute
    private Tensor? _attnWeights; // [B,H,S,S] after softmax
    private readonly IComputeProvider? _computeProvider;

    /// <summary>Query projection [embed_dim, embed_dim].</summary>
    public LinearLayer QueryProj { get; }

    /// <summary>Key projection [embed_dim, embed_dim].</summary>
    public LinearLayer KeyProj { get; }

    /// <summary>Value projection [embed_dim, embed_dim].</summary>
    public LinearLayer ValueProj { get; }

    /// <summary>Output projection [embed_dim, embed_dim].</summary>
    public LinearLayer OutputProj { get; }

    /// <summary>
    /// Initializes multi-head self-attention.
    /// </summary>
    /// <param name="embedDim">Model hidden dimension. Must be divisible by numHeads.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public MultiHeadAttention(int embedDim, int numHeads, int? seed = null,
        IComputeProvider? computeProvider = null)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException(
                $"embedDim {embedDim} must be divisible by numHeads {numHeads}.");
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim  = embedDim / numHeads;
        _computeProvider = computeProvider;

        QueryProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed, computeProvider: computeProvider);
        KeyProj    = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 1 : null, computeProvider: computeProvider);
        ValueProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 2 : null, computeProvider: computeProvider);
        OutputProj = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 3 : null, computeProvider: computeProvider);
    }

    /// <summary>
    /// Forward pass with causal masking. Caches activations for <see cref="Backward"/>.
    /// </summary>
    /// <param name="x">Input tensor [batch, seq, embed_dim].</param>
    /// <returns>Output tensor [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        if (x.Rank != 3)
            throw new ArgumentException("Input must be rank 3: [batch, seq, embed_dim].");

        int B = x.Shape[0], S = x.Shape[1], E = x.Shape[2];
        if (E != _embedDim)
            throw new ArgumentException($"Expected embed_dim {_embedDim}, got {E}.");

        int H = _numHeads, D = _headDim;
        _lastInput = x;

        // 1. Project: [B,S,E]
        var q = QueryProj.Forward(x);
        var k = KeyProj.Forward(x);
        var v = ValueProj.Forward(x);

        // 2. Reshape → [B,S,H,D] → permute(0,2,1,3) → [B,H,S,D]
        q = TensorMath.Permute(TensorMath.Reshape(q, B, S, H, D), [0, 2, 1, 3]);
        k = TensorMath.Permute(TensorMath.Reshape(k, B, S, H, D), [0, 2, 1, 3]);
        v = TensorMath.Permute(TensorMath.Reshape(v, B, S, H, D), [0, 2, 1, 3]);

        _qHeads = q; _kHeads = k; _vHeads = v;

        // 3. Scaled dot-product: Q @ K^T / sqrt(D)
        var kT = TensorMath.Transpose(k, 2, 3);
        var rawScores = _computeProvider is not null
            ? _computeProvider.BatchedMatMul(q, kT)
            : TensorMath.BatchedMatMul(q, kT);
        var scores = TensorMath.Multiply(rawScores, 1f / MathF.Sqrt(D));

        // 4. Causal mask
        scores = ApplyCausalMask(scores, B, H, S);

        // 5. Softmax + cache
        var weights = TensorMath.Softmax(scores, axis: 3);
        _attnWeights = weights;

        // 6. Context: weights @ V → [B,H,S,D]
        var context = _computeProvider is not null
            ? _computeProvider.BatchedMatMul(weights, v)
            : TensorMath.BatchedMatMul(weights, v);

        // 7. Permute(0,2,1,3) → [B,S,H,D] → reshape → [B,S,E]
        context = TensorMath.Reshape(TensorMath.Permute(context, [0, 2, 1, 3]), B, S, E);

        // 8. Output projection
        return OutputProj.Forward(context);
    }

    /// <summary>
    /// Backward pass. Propagates gradients through the full attention mechanism.
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        if (_lastInput is null || _qHeads is null || _kHeads is null || _vHeads is null || _attnWeights is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        int B = _lastInput.Shape[0], S = _lastInput.Shape[1], E = _lastInput.Shape[2];
        int H = _numHeads, D = _headDim;
        float scale = 1f / MathF.Sqrt(D);

        // 1. Output projection backward → dContext [B,S,E]
        var dContext = OutputProj.Backward(gradOutput);

        // 2. Reshape dContext [B,S,E] → [B,S,H,D] → permute(0,2,1,3) → [B,H,S,D]
        var dContextHeads = TensorMath.Permute(
            TensorMath.Reshape(dContext, B, S, H, D), [0, 2, 1, 3]);

        // 3. dV = attnWeights^T @ dContextHeads
        //    dWeights = dContextHeads @ vHeads^T
        var dV = TensorMath.BatchedMatMul(TensorMath.Transpose(_attnWeights, 2, 3), dContextHeads); // [B,H,S,D]
        var dWeights = TensorMath.BatchedMatMul(dContextHeads, TensorMath.Transpose(_vHeads, 2, 3)); // [B,H,S,S]

        // 4. Softmax backward
        var dScores = SoftmaxBackward(dWeights, _attnWeights); // [B,H,S,S]

        // 5. dQ = dScores @ kHeads * scale  [B,H,S,D]
        //    dK = dScores^T @ qHeads * scale [B,H,S,D]
        var dQ = TensorMath.Multiply(TensorMath.BatchedMatMul(dScores, _kHeads), scale);
        var dK = TensorMath.Multiply(TensorMath.BatchedMatMul(TensorMath.Transpose(dScores, 2, 3), _qHeads), scale);

        // 6. Permute(0,2,1,3) → [B,S,H,D] → reshape → [B,S,E]
        dQ = TensorMath.Reshape(TensorMath.Permute(dQ, [0, 2, 1, 3]), B, S, E);
        dK = TensorMath.Reshape(TensorMath.Permute(dK, [0, 2, 1, 3]), B, S, E);
        dV = TensorMath.Reshape(TensorMath.Permute(dV, [0, 2, 1, 3]), B, S, E);

        // 7. Projection backward and sum three input gradient paths
        var dxQ = QueryProj.Backward(dQ);
        var dxK = KeyProj.Backward(dK);
        var dxV = ValueProj.Backward(dV);

        return TensorMath.Add(TensorMath.Add(dxQ, dxK), dxV);
    }

    /// <summary>
    /// Backward through softmax applied along axis 3 (key dimension).
    /// d_logit[k] = softmax[k] * (dOut[k] - Σ_j(dOut[j]*softmax[j]))
    /// </summary>
    private static Tensor SoftmaxBackward(Tensor dOut, Tensor softmaxOut)
    {
        int B = softmaxOut.Shape[0], H = softmaxOut.Shape[1];
        int Sq = softmaxOut.Shape[2], Sk = softmaxOut.Shape[3];
        var so = softmaxOut.Data;
        var dO = dOut.Data;
        var result = new float[B * H * Sq * Sk];

        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int q = 0; q < Sq; q++)
                {
                    int offset = ((b * H + h) * Sq + q) * Sk;
                    float dot = 0f;
                    for (int k = 0; k < Sk; k++) dot += dO[offset + k] * so[offset + k];
                    for (int k = 0; k < Sk; k++)
                        result[offset + k] = so[offset + k] * (dO[offset + k] - dot);
                }

        return new Tensor([B, H, Sq, Sk], result);
    }

    private static Tensor ApplyCausalMask(Tensor scores, int B, int H, int S)
    {
        var data = scores.Data.ToArray();
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < S; i++)
                    for (int j = i + 1; j < S; j++)
                        data[((b * H + h) * S + i) * S + j] = float.NegativeInfinity;
        return new Tensor([B, H, S, S], data);
    }

    /// <summary>Zeros the accumulated gradients in all projection layers.</summary>
    public void ZeroGrad()
    {
        QueryProj.ZeroGrad();
        KeyProj.ZeroGrad();
        ValueProj.ZeroGrad();
        OutputProj.ZeroGrad();
        _lastInput = null; _qHeads = null; _kHeads = null; _vHeads = null; _attnWeights = null;
    }

    /// <summary>Returns all learnable parameters from all four projections.</summary>
    public IEnumerable<Parameter> Parameters() =>
        QueryProj.Parameters()
            .Concat(KeyProj.Parameters())
            .Concat(ValueProj.Parameters())
            .Concat(OutputProj.Parameters());
}

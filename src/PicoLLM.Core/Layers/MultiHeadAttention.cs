using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Scaled dot-product multi-head self-attention with causal masking.
/// Input shape: [batch, seq, embed_dim]. Output shape: same.
/// </summary>
/// <remarks>
/// Pipeline (GPT-2 style):
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
public sealed class MultiHeadAttention
{
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _embedDim;

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
    public MultiHeadAttention(int embedDim, int numHeads, int? seed = null)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException(
                $"embedDim {embedDim} must be divisible by numHeads {numHeads}.");
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;

        QueryProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed);
        KeyProj    = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 1 : null);
        ValueProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 2 : null);
        OutputProj = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 3 : null);
    }

    /// <summary>
    /// Forward pass with causal masking.
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

        // 1. Linear projections: each [B, S, E]
        var q = QueryProj.Forward(x);
        var k = KeyProj.Forward(x);
        var v = ValueProj.Forward(x);

        // 2. Reshape [B,S,E] → [B,S,H,D] then permute(0,2,1,3) → [B,H,S,D]
        q = TensorMath.Permute(TensorMath.Reshape(q, B, S, H, D), [0, 2, 1, 3]);
        k = TensorMath.Permute(TensorMath.Reshape(k, B, S, H, D), [0, 2, 1, 3]);
        v = TensorMath.Permute(TensorMath.Reshape(v, B, S, H, D), [0, 2, 1, 3]);

        // 3. Scaled dot-product: Q @ K^T / sqrt(D)
        //    Transpose K: [B,H,S,D] → [B,H,D,S], then batched matmul → [B,H,S,S]
        var kT = TensorMath.Transpose(k, 2, 3);
        var scores = TensorMath.BatchedMatMul(q, kT);
        scores = TensorMath.Multiply(scores, 1f / MathF.Sqrt(D));

        // 4. Causal mask: positions j > i → −∞
        scores = ApplyCausalMask(scores, B, H, S);

        // 5. Softmax over key dimension (axis 3)
        var weights = TensorMath.Softmax(scores, axis: 3);

        // 6. Weighted sum of values: [B,H,S,S] @ [B,H,S,D] → [B,H,S,D]
        var context = TensorMath.BatchedMatMul(weights, v);

        // 7. Permute(0,2,1,3) → [B,S,H,D], reshape → [B,S,E]
        context = TensorMath.Reshape(
            TensorMath.Permute(context, [0, 2, 1, 3]),
            B, S, E);

        // 8. Output projection
        return OutputProj.Forward(context);
    }

    private static Tensor ApplyCausalMask(Tensor scores, int B, int H, int S)
    {
        var data = scores.Data.ToArray();
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < S; i++)
                    for (int j = i + 1; j < S; j++)
                    {
                        int idx = ((b * H + h) * S + i) * S + j;
                        data[idx] = float.NegativeInfinity;
                    }
        return new Tensor([B, H, S, S], data);
    }

    /// <summary>Zeros the accumulated gradients in all projection layers.</summary>
    public void ZeroGrad()
    {
        QueryProj.ZeroGrad();
        KeyProj.ZeroGrad();
        ValueProj.ZeroGrad();
        OutputProj.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters from all four projections.</summary>
    public IEnumerable<Tensor> Parameters() =>
        QueryProj.Parameters()
            .Concat(KeyProj.Parameters())
            .Concat(ValueProj.Parameters())
            .Concat(OutputProj.Parameters());
}

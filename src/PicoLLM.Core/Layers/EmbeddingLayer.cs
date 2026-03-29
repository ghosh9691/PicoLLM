using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Combined embedding layer: token embedding + sinusoidal positional encoding.
/// This is the first layer of the transformer: it converts token ID sequences
/// into continuous vector representations.
/// </summary>
/// <remarks>
/// Forward for a single sequence:
/// <code>output[pos] = TokenEmbedding[tokenId[pos]] + PE[pos]</code>
///
/// Forward for a batch: the positional encoding is identical for every sample
/// and is added independently to each sequence in the batch.
/// </remarks>
public sealed class EmbeddingLayer
{
    private readonly TokenEmbedding _tokenEmbed;
    private readonly PositionalEncoding _positionalEncoding;

    /// <summary>The underlying token embedding (access for weight updates).</summary>
    public TokenEmbedding TokenEmbedding => _tokenEmbed;

    /// <summary>
    /// Initializes the combined embedding layer.
    /// </summary>
    /// <param name="vocabSize">Number of tokens in the vocabulary.</param>
    /// <param name="embedDim">Embedding dimension (must match transformer hidden size).</param>
    /// <param name="maxSeqLen">Maximum supported sequence length.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public EmbeddingLayer(int vocabSize, int embedDim, int maxSeqLen = 2048, int? seed = null)
    {
        _tokenEmbed = new TokenEmbedding(vocabSize, embedDim, seed);
        _positionalEncoding = new PositionalEncoding(maxSeqLen, embedDim);
    }

    /// <summary>
    /// Forward pass for a single token sequence.
    /// </summary>
    /// <param name="tokenIds">Token IDs, length = seq_len.</param>
    /// <returns>Tensor of shape [seq_len, embed_dim].</returns>
    public Tensor Forward(int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var embedded = _tokenEmbed.Forward(tokenIds);
        var posEnc = _positionalEncoding.GetEncoding(tokenIds.Length);
        return TensorMath.Add(embedded, posEnc);
    }

    /// <summary>
    /// Batched forward pass.
    /// </summary>
    /// <param name="batchTokenIds">Array of token ID sequences, one per batch item.</param>
    /// <returns>Tensor of shape [batch, seq_len, embed_dim].</returns>
    public Tensor ForwardBatch(int[][] batchTokenIds)
    {
        ArgumentNullException.ThrowIfNull(batchTokenIds);
        if (batchTokenIds.Length == 0)
            throw new ArgumentException("Batch must not be empty.", nameof(batchTokenIds));

        int batch = batchTokenIds.Length;
        int seqLen = batchTokenIds[0].Length;
        int embedDim = _tokenEmbed.EmbedDim;

        // Verify all sequences have the same length
        foreach (var seq in batchTokenIds)
            if (seq.Length != seqLen)
                throw new ArgumentException(
                    "All sequences in the batch must have the same length.", nameof(batchTokenIds));

        var posEnc = _positionalEncoding.GetEncoding(seqLen); // [seq_len, embed_dim]
        var result = new Tensor(batch, seqLen, embedDim);
        var r = result.MutableData;
        var pe = posEnc.Data;

        for (int b = 0; b < batch; b++)
        {
            var embedded = _tokenEmbed.Forward(batchTokenIds[b]); // [seq_len, embed_dim]
            var emb = embedded.Data;
            int batchOffset = b * seqLen * embedDim;
            for (int s = 0; s < seqLen; s++)
            {
                int seqOffset = s * embedDim;
                for (int d = 0; d < embedDim; d++)
                    r[batchOffset + seqOffset + d] = emb[seqOffset + d] + pe[seqOffset + d];
            }
        }

        return result;
    }

    /// <summary>
    /// Backward pass for a single sequence.
    /// Propagates gradients back to the token embedding weights.
    /// The positional encoding has no learnable parameters so its gradient is ignored.
    /// </summary>
    /// <param name="gradOutput">Upstream gradient, shape [seq_len, embed_dim].</param>
    /// <param name="tokenIds">Token IDs used in the forward pass.</param>
    public void Backward(Tensor gradOutput, int[] tokenIds)
    {
        _tokenEmbed.Backward(gradOutput, tokenIds);
    }

    /// <summary>
    /// Backward pass for a batched sequence.
    /// Splits the [batch, seq, embed_dim] gradient and accumulates into token embedding rows.
    /// </summary>
    /// <param name="gradOutput">Upstream gradient, shape [batch, seq, embed_dim].</param>
    /// <param name="batchTokenIds">Token IDs used in the forward pass, one array per batch item.</param>
    public void BackwardBatch(Tensor gradOutput, int[][] batchTokenIds)
    {
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(batchTokenIds);

        int batch   = batchTokenIds.Length;
        int seqLen  = batchTokenIds[0].Length;
        int embedDim = _tokenEmbed.EmbedDim;

        for (int b = 0; b < batch; b++)
        {
            // Slice [seq, embed] from [batch, seq, embed]
            var seqGrad = TensorMath.Slice(
                TensorMath.Reshape(gradOutput, batch * seqLen, embedDim),
                b * seqLen, seqLen);
            _tokenEmbed.Backward(seqGrad, batchTokenIds[b]);
        }
    }

    /// <summary>Zeros accumulated gradients in the token embedding.</summary>
    public void ZeroGrad()
    {
        _tokenEmbed.ZeroGrad();
    }
}

using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Learnable token embedding lookup table.
/// Maps each token ID to a dense vector of size <see cref="EmbedDim"/>.
/// Equivalent to a one-hot multiply against the weight matrix, but implemented
/// as a direct row-lookup for efficiency.
/// </summary>
/// <remarks>
/// Forward: For each token ID t, return row t from <see cref="Weights"/>.
/// Output shape: [seq_len, embed_dim].
///
/// Backward: Accumulate the upstream gradient into the rows of <see cref="WeightGrad"/>
/// that were accessed during the forward pass (sparse update).
/// </remarks>
public sealed class TokenEmbedding
{
    /// <summary>
    /// Embedding weight matrix, shape [vocab_size, embed_dim].
    /// Each row is the embedding vector for one token.
    /// </summary>
    public Tensor Weights { get; }

    /// <summary>
    /// Accumulated gradient for <see cref="Weights"/>, same shape [vocab_size, embed_dim].
    /// Reset to zero by the optimizer after each parameter update.
    /// </summary>
    public Tensor WeightGrad { get; }

    /// <summary>Number of tokens in the vocabulary.</summary>
    public int VocabSize { get; }

    /// <summary>Dimension of each embedding vector.</summary>
    public int EmbedDim { get; }

    /// <summary>
    /// Initializes a new token embedding layer.
    /// Weights are drawn from N(0, 0.02) — the GPT-2 convention.
    /// </summary>
    /// <param name="vocabSize">Number of tokens in the vocabulary.</param>
    /// <param name="embedDim">Size of each embedding vector.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public TokenEmbedding(int vocabSize, int embedDim, int? seed = null)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));

        VocabSize = vocabSize;
        EmbedDim = embedDim;
        Weights = TensorFactory.RandomNormal([vocabSize, embedDim], mean: 0f, std: 0.02f, seed: seed);
        WeightGrad = TensorFactory.Zeros(vocabSize, embedDim);
    }

    /// <summary>
    /// Internal constructor for loading saved weights.
    /// </summary>
    internal TokenEmbedding(Tensor weights)
    {
        VocabSize = weights.Shape[0];
        EmbedDim = weights.Shape[1];
        Weights = weights;
        WeightGrad = TensorFactory.Zeros(VocabSize, EmbedDim);
    }

    /// <summary>
    /// Forward pass: look up the embedding vector for each token ID.
    /// </summary>
    /// <param name="tokenIds">Sequence of token IDs (length = seq_len).</param>
    /// <returns>Tensor of shape [seq_len, embed_dim].</returns>
    public Tensor Forward(int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        int seqLen = tokenIds.Length;
        var result = new Tensor(seqLen, EmbedDim);
        var w = Weights.Data;
        var r = result.MutableData;

        for (int i = 0; i < seqLen; i++)
        {
            int id = tokenIds[i];
            if ((uint)id >= (uint)VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {id} is out of range [0, {VocabSize}).");
            int srcOffset = id * EmbedDim;
            int dstOffset = i * EmbedDim;
            w.Slice(srcOffset, EmbedDim).CopyTo(r.Slice(dstOffset, EmbedDim));
        }

        return result;
    }

    /// <summary>
    /// Backward pass: accumulate gradient into the rows that were used in the forward pass.
    /// Only the rows corresponding to <paramref name="tokenIds"/> are updated (sparse gradient).
    /// </summary>
    /// <param name="gradOutput">Upstream gradient, shape [seq_len, embed_dim].</param>
    /// <param name="tokenIds">The same token IDs used in the corresponding forward call.</param>
    public void Backward(Tensor gradOutput, int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(gradOutput);
        ArgumentNullException.ThrowIfNull(tokenIds);

        var grad = gradOutput.Data;
        var wg = WeightGrad.MutableData;

        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            int srcOffset = i * EmbedDim;
            int dstOffset = id * EmbedDim;
            for (int d = 0; d < EmbedDim; d++)
                wg[dstOffset + d] += grad[srcOffset + d];
        }
    }

    /// <summary>Zeros out the accumulated gradient (called by the optimizer after each step).</summary>
    public void ZeroGrad()
    {
        WeightGrad.MutableData.Clear();
    }
}

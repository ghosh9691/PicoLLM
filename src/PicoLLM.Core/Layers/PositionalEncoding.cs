using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Sinusoidal positional encoding as described in "Attention Is All You Need" (Vaswani et al., 2017).
/// Not learned — computed once at construction and cached.
/// </summary>
/// <remarks>
/// Formulas:
/// <code>
/// PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))
/// </code>
/// where <c>pos</c> is the token position and <c>i</c> is the dimension index.
///
/// The encoding is deterministic: given the same (seq_len, embed_dim) pair,
/// the result is always identical.
/// </remarks>
public sealed class PositionalEncoding
{
    private readonly Tensor _table; // shape: [max_seq_len, embed_dim]

    /// <summary>Maximum sequence length this encoding supports.</summary>
    public int MaxSeqLen { get; }

    /// <summary>Embedding dimensionality.</summary>
    public int EmbedDim { get; }

    /// <summary>
    /// Precomputes sinusoidal encoding up to <paramref name="maxSeqLen"/> positions.
    /// </summary>
    public PositionalEncoding(int maxSeqLen, int embedDim)
    {
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));

        MaxSeqLen = maxSeqLen;
        EmbedDim = embedDim;
        _table = Compute(maxSeqLen, embedDim);
    }

    /// <summary>
    /// Returns the positional encoding for the first <paramref name="seqLen"/> positions.
    /// </summary>
    /// <param name="seqLen">Number of positions required (must be ≤ MaxSeqLen).</param>
    /// <returns>Tensor of shape [seqLen, embed_dim].</returns>
    public Tensor GetEncoding(int seqLen)
    {
        if (seqLen <= 0 || seqLen > MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen),
                $"seqLen {seqLen} must be in [1, {MaxSeqLen}].");
        return TensorMath.Slice(_table, 0, seqLen);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private static Tensor Compute(int maxSeqLen, int embedDim)
    {
        var data = new float[maxSeqLen * embedDim];

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < embedDim / 2; i++)
            {
                double angle = pos / Math.Pow(10000.0, 2.0 * i / embedDim);
                data[pos * embedDim + 2 * i] = (float)Math.Sin(angle);
                data[pos * embedDim + 2 * i + 1] = (float)Math.Cos(angle);
            }
            // Handle odd embedDim: last dimension uses sin only
            if (embedDim % 2 == 1)
            {
                double angle = pos / Math.Pow(10000.0, (embedDim - 1.0) / embedDim);
                data[pos * embedDim + embedDim - 1] = (float)Math.Sin(angle);
            }
        }

        return new Tensor(new[] { maxSeqLen, embedDim }, data);
    }
}

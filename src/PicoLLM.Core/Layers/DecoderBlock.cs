using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// A single GPT-style decoder block with pre-norm residual connections.
/// </summary>
/// <remarks>
/// Forward pass:
/// <code>
/// x = x + Attention(LayerNorm(x))   // self-attention sublayer
/// x = x + FFN(LayerNorm(x))         // feedforward sublayer
/// </code>
/// Using pre-norm (norm before sublayer) rather than post-norm improves
/// training stability for deep transformers.
/// </remarks>
public sealed class DecoderBlock
{
    private readonly LayerNorm _attnNorm;
    private readonly LayerNorm _ffnNorm;

    /// <summary>The multi-head self-attention sublayer.</summary>
    public MultiHeadAttention Attention { get; }

    /// <summary>The feedforward sublayer.</summary>
    public FeedForward FFN { get; }

    /// <summary>
    /// Initializes a decoder block.
    /// </summary>
    /// <param name="embedDim">Hidden dimension (must be divisible by numHeads).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="ffMultiplier">FFN expansion factor (default 4).</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public DecoderBlock(int embedDim, int numHeads, int ffMultiplier = 4, int? seed = null)
    {
        _attnNorm = new LayerNorm(embedDim);
        Attention = new MultiHeadAttention(embedDim, numHeads, seed);
        _ffnNorm  = new LayerNorm(embedDim);
        FFN       = new FeedForward(embedDim, ffMultiplier, seed);
    }

    /// <summary>
    /// Forward pass with pre-norm residual connections.
    /// </summary>
    /// <param name="x">Input [batch, seq, embed_dim].</param>
    /// <returns>Output [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        x = TensorMath.Add(x, Attention.Forward(_attnNorm.Forward(x)));
        x = TensorMath.Add(x, FFN.Forward(_ffnNorm.Forward(x)));
        return x;
    }

    /// <summary>Zeros all accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _attnNorm.ZeroGrad();
        Attention.ZeroGrad();
        _ffnNorm.ZeroGrad();
        FFN.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters in this block.</summary>
    public IEnumerable<Tensor> Parameters() =>
        _attnNorm.Parameters()
            .Concat(Attention.Parameters())
            .Concat(_ffnNorm.Parameters())
            .Concat(FFN.Parameters());
}

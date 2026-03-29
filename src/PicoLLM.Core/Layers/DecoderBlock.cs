using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Core.Layers;

/// <summary>
/// A single GPT-style decoder block with pre-norm residual connections.
/// Implements <see cref="ILayer"/> for use in the training pipeline.
/// </summary>
/// <remarks>
/// Forward:
/// <code>
/// x = x + Attention(LayerNorm(x))   // self-attention sublayer
/// x = x + FFN(LayerNorm(x))         // feedforward sublayer
/// </code>
/// Backward (reverse order):
/// <code>
/// dY1  = gradOut + _ffnNorm.Backward(FFN.Backward(gradOut))
/// dX   = dY1    + _attnNorm.Backward(Attention.Backward(dY1))
/// </code>
/// </remarks>
public sealed class DecoderBlock : ILayer
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
    public DecoderBlock(int embedDim, int numHeads, int ffMultiplier = 4, int? seed = null)
    {
        _attnNorm = new LayerNorm(embedDim);
        Attention = new MultiHeadAttention(embedDim, numHeads, seed);
        _ffnNorm  = new LayerNorm(embedDim);
        FFN       = new FeedForward(embedDim, ffMultiplier, seed);
    }

    /// <summary>Forward pass with pre-norm residual connections.</summary>
    public Tensor Forward(Tensor x)
    {
        x = TensorMath.Add(x, Attention.Forward(_attnNorm.Forward(x)));
        x = TensorMath.Add(x, FFN.Forward(_ffnNorm.Forward(x)));
        return x;
    }

    /// <summary>
    /// Backward pass. Reverses the two residual sublayers.
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        // FFN sublayer backward (reverse of: y1 = x + FFN(LN2(x)))
        var dFfnOut = FFN.Backward(gradOutput);          // grad w.r.t. LN2 output
        var dLn2In  = _ffnNorm.Backward(dFfnOut);        // grad w.r.t. y1 through FFN path
        var gradY1  = TensorMath.Add(gradOutput, dLn2In); // total grad w.r.t. y1

        // Attention sublayer backward (reverse of: y1 = x + Attn(LN1(x)))
        var dAttnOut = Attention.Backward(gradY1);
        var dLn1In   = _attnNorm.Backward(dAttnOut);
        return TensorMath.Add(gradY1, dLn1In);
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
    public IEnumerable<Parameter> Parameters() =>
        _attnNorm.Parameters()
            .Concat(Attention.Parameters())
            .Concat(_ffnNorm.Parameters())
            .Concat(FFN.Parameters());
}

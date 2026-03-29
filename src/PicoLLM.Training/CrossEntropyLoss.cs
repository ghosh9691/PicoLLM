using PicoLLM.Core.Tensors;

namespace PicoLLM.Training;

/// <summary>
/// Cross-entropy loss for next-token prediction.
/// Expects logits [batch, seq, vocab_size] and integer target IDs [batch, seq].
/// </summary>
/// <remarks>
/// Forward:  loss = −mean( log( softmax(logits)[b, s, target[b,s]] ) )
///
/// Backward: dLogits = softmax(logits)
///           dLogits[b, s, target[b,s]] −= 1
///           dLogits /= (batch × seq)
/// </remarks>
public static class CrossEntropyLoss
{
    /// <summary>
    /// Computes the mean cross-entropy loss.
    /// </summary>
    /// <param name="logits">Model output [batch, seq, vocab_size].</param>
    /// <param name="targets">Target token IDs [batch, seq]. Each value in [0, vocab_size).</param>
    /// <returns>Scalar mean loss.</returns>
    public static float Forward(Tensor logits, int[,] targets)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(targets);
        if (logits.Rank != 3)
            throw new ArgumentException("logits must be rank 3: [batch, seq, vocab].");

        int B = logits.Shape[0], S = logits.Shape[1], V = logits.Shape[2];
        var src = logits.Data;
        double totalLoss = 0.0;

        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
            {
                int offset = (b * S + s) * V;
                int target = targets[b, s];

                // Numerically stable: subtract max before softmax
                float maxVal = float.NegativeInfinity;
                for (int v = 0; v < V; v++) maxVal = MathF.Max(maxVal, src[offset + v]);

                float sumExp = 0f;
                for (int v = 0; v < V; v++) sumExp += MathF.Exp(src[offset + v] - maxVal);

                float logProb = (src[offset + target] - maxVal) - MathF.Log(sumExp);
                totalLoss -= logProb;
            }

        return (float)(totalLoss / (B * S));
    }

    /// <summary>
    /// Computes the gradient of the loss w.r.t. logits.
    /// </summary>
    /// <param name="logits">Model output [batch, seq, vocab_size].</param>
    /// <param name="targets">Target token IDs [batch, seq].</param>
    /// <returns>Gradient tensor [batch, seq, vocab_size].</returns>
    public static Tensor Backward(Tensor logits, int[,] targets)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(targets);

        int B = logits.Shape[0], S = logits.Shape[1], V = logits.Shape[2];
        var src = logits.Data;
        var grad = new float[B * S * V];
        float scale = 1f / (B * S);

        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
            {
                int offset = (b * S + s) * V;

                // Compute softmax
                float maxVal = float.NegativeInfinity;
                for (int v = 0; v < V; v++) maxVal = MathF.Max(maxVal, src[offset + v]);
                float sumExp = 0f;
                for (int v = 0; v < V; v++) sumExp += MathF.Exp(src[offset + v] - maxVal);

                for (int v = 0; v < V; v++)
                    grad[offset + v] = MathF.Exp(src[offset + v] - maxVal) / sumExp * scale;

                // Subtract 1 at the target position
                grad[offset + targets[b, s]] -= scale;
            }

        return TensorFactory.FromArray([B, S, V], grad);
    }
}

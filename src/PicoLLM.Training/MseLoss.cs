using PicoLLM.Core.Tensors;

namespace PicoLLM.Training;

/// <summary>
/// Mean-squared-error loss: loss = mean( (pred − target)² ).
/// Used for regression tasks or debugging the training pipeline.
/// </summary>
public static class MseLoss
{
    /// <summary>
    /// Computes the MSE loss.
    /// </summary>
    /// <param name="pred">Predictions tensor (any shape).</param>
    /// <param name="target">Targets tensor (same shape as pred).</param>
    /// <returns>Scalar mean loss.</returns>
    public static float Forward(Tensor pred, Tensor target)
    {
        ArgumentNullException.ThrowIfNull(pred);
        ArgumentNullException.ThrowIfNull(target);
        if (pred.Length != target.Length)
            throw new ArgumentException("pred and target must have the same number of elements.");

        var p = pred.Data; var t = target.Data;
        double sum = 0.0;
        for (int i = 0; i < p.Length; i++) { float d = p[i] - t[i]; sum += d * d; }
        return (float)(sum / p.Length);
    }

    /// <summary>
    /// Computes the gradient of the MSE loss w.r.t. predictions.
    /// dL/dpred = 2 * (pred − target) / N
    /// </summary>
    public static Tensor Backward(Tensor pred, Tensor target)
    {
        ArgumentNullException.ThrowIfNull(pred);
        ArgumentNullException.ThrowIfNull(target);

        var p = pred.Data; var t = target.Data;
        var grad = new float[p.Length];
        float scale = 2f / p.Length;
        for (int i = 0; i < p.Length; i++) grad[i] = scale * (p[i] - t[i]);
        return TensorFactory.FromArray(pred.GetShape(), grad);
    }
}

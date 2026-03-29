using PicoLLM.Core.Training;

namespace PicoLLM.Training;

/// <summary>
/// Gradient clipping by global L2 norm to prevent exploding gradients.
/// </summary>
public static class GradientClipper
{
    /// <summary>
    /// Computes the global gradient norm across all parameters and clips if it exceeds maxNorm.
    /// </summary>
    /// <param name="parameters">All parameters whose gradients should be clipped.</param>
    /// <param name="maxNorm">Maximum allowed global gradient norm.</param>
    /// <returns>The global gradient norm BEFORE clipping (useful for metrics).</returns>
    public static float ClipGradNorm(IEnumerable<Parameter> parameters, float maxNorm)
    {
        if (maxNorm <= 0f) throw new ArgumentOutOfRangeException(nameof(maxNorm), "maxNorm must be positive.");

        var paramList = parameters.ToList();

        // Compute global L2 norm
        double sumSq = 0.0;
        foreach (var p in paramList)
        {
            var grad = p.Grad.Data;
            for (int i = 0; i < grad.Length; i++) sumSq += grad[i] * grad[i];
        }

        float globalNorm = (float)Math.Sqrt(sumSq);

        // Scale gradients if norm exceeds maxNorm
        if (globalNorm > maxNorm)
        {
            float scale = maxNorm / globalNorm;
            foreach (var p in paramList)
            {
                var grad = p.Grad.MutableData;
                for (int i = 0; i < grad.Length; i++) grad[i] *= scale;
            }
        }

        return globalNorm;
    }
}

using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Training;

/// <summary>
/// AdamW optimizer with decoupled weight decay.
/// Tracks first and second moment estimates per parameter.
/// </summary>
/// <remarks>
/// Update rule (step t):
/// <code>
/// m = β1·m + (1−β1)·g
/// v = β2·v + (1−β2)·g²
/// m̂ = m / (1−β1^t)
/// v̂ = v / (1−β2^t)
/// p -= lr · (m̂ / (√v̂ + ε) + weightDecay · p)
/// </code>
/// </remarks>
public sealed class AdamW
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private float _lr;
    private int _step;

    // Per-parameter moment tensors
    private readonly Dictionary<Parameter, (float[] m, float[] v)> _state = new();

    /// <summary>Current step count (incremented each <see cref="Step"/> call).</summary>
    public int StepCount => _step;

    /// <summary>Current learning rate (can be updated by the scheduler each step).</summary>
    public float LearningRate
    {
        get => _lr;
        set => _lr = value;
    }

    /// <summary>
    /// Initializes the AdamW optimizer.
    /// </summary>
    /// <param name="lr">Learning rate.</param>
    /// <param name="beta1">First moment decay (default 0.9).</param>
    /// <param name="beta2">Second moment decay (default 0.999).</param>
    /// <param name="eps">Numerical stability constant (default 1e-8).</param>
    /// <param name="weightDecay">Decoupled weight decay coefficient (default 0.01).</param>
    public AdamW(float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
                 float eps = 1e-8f, float weightDecay = 0.01f)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
    }

    /// <summary>
    /// Performs one optimization step.
    /// Increments the internal step counter, updates moments, and applies weight decay.
    /// </summary>
    /// <param name="parameters">All trainable parameters to update.</param>
    public void Step(IEnumerable<Parameter> parameters)
    {
        _step++;
        float b1t = 1f - MathF.Pow(_beta1, _step);  // bias correction denominator for m
        float b2t = 1f - MathF.Pow(_beta2, _step);  // bias correction denominator for v

        foreach (var p in parameters)
        {
            if (!_state.TryGetValue(p, out var state))
            {
                state = (new float[p.Data.Length], new float[p.Data.Length]);
                _state[p] = state;
            }

            var (m, v) = state;
            var data = p.Data.MutableData;
            var grad = p.Grad.Data;

            for (int i = 0; i < data.Length; i++)
            {
                float g = grad[i];
                m[i] = _beta1 * m[i] + (1f - _beta1) * g;
                v[i] = _beta2 * v[i] + (1f - _beta2) * g * g;
                float mHat = m[i] / b1t;
                float vHat = v[i] / b2t;
                data[i] -= _lr * (mHat / (MathF.Sqrt(vHat) + _eps) + _weightDecay * data[i]);
            }
        }
    }

    /// <summary>
    /// Returns a snapshot of the optimizer state for checkpointing.
    /// Each entry corresponds to one parameter in the order they were registered.
    /// </summary>
    public IReadOnlyDictionary<Parameter, (float[] m, float[] v)> State => _state;

    /// <summary>
    /// Restores optimizer state from a checkpoint.
    /// The parameters and their moment arrays must match in count and shape.
    /// </summary>
    public void LoadState(IEnumerable<Parameter> parameters, IEnumerable<(float[] m, float[] v)> moments)
    {
        _state.Clear();
        using var pe = parameters.GetEnumerator();
        using var me = moments.GetEnumerator();
        while (pe.MoveNext() && me.MoveNext())
            _state[pe.Current] = me.Current;
    }

    /// <summary>Resets the step counter. Used when loading a checkpoint.</summary>
    public void SetStep(int step) => _step = step;
}

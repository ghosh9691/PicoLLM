namespace PicoLLM.Training;

/// <summary>
/// Linear warmup followed by cosine decay learning rate schedule.
/// </summary>
/// <remarks>
/// Phase 1 — warmup (step &lt; warmupSteps):
///   lr = maxLr × step / warmupSteps
///
/// Phase 2 — cosine decay (step ≥ warmupSteps):
///   progress = (step − warmupSteps) / (totalSteps − warmupSteps)
///   lr = minLr + 0.5 × (maxLr − minLr) × (1 + cos(π × progress))
/// </remarks>
public sealed class LearningRateSchedule
{
    private readonly int _warmupSteps;
    private readonly int _totalSteps;
    private readonly float _maxLr;
    private readonly float _minLr;

    /// <summary>
    /// Initializes the learning rate schedule.
    /// </summary>
    /// <param name="warmupSteps">Steps during which lr rises linearly from 0 to maxLr.</param>
    /// <param name="totalSteps">Total number of training steps.</param>
    /// <param name="maxLr">Peak learning rate (reached at end of warmup).</param>
    /// <param name="minLr">Minimum learning rate (reached at totalSteps).</param>
    public LearningRateSchedule(int warmupSteps, int totalSteps, float maxLr, float minLr = 0f)
    {
        if (warmupSteps < 0) throw new ArgumentOutOfRangeException(nameof(warmupSteps));
        if (totalSteps <= 0) throw new ArgumentOutOfRangeException(nameof(totalSteps));
        if (warmupSteps > totalSteps)
            throw new ArgumentException("warmupSteps must not exceed totalSteps.");

        _warmupSteps = warmupSteps;
        _totalSteps  = totalSteps;
        _maxLr  = maxLr;
        _minLr  = minLr;
    }

    /// <summary>
    /// Returns the learning rate for the given step (0-indexed).
    /// </summary>
    public float GetLR(int step)
    {
        if (step < 0) return _minLr;
        if (step >= _totalSteps) return _minLr;

        if (_warmupSteps > 0 && step < _warmupSteps)
            return _maxLr * ((float)step / _warmupSteps);

        int decaySteps = _totalSteps - _warmupSteps;
        if (decaySteps <= 0) return _maxLr;

        float progress = (float)(step - _warmupSteps) / decaySteps;
        return _minLr + 0.5f * (_maxLr - _minLr) * (1f + MathF.Cos(MathF.PI * progress));
    }
}

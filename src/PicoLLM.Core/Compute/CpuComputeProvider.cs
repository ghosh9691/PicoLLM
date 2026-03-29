using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Compute;

/// <summary>
/// CPU compute provider. Delegates directly to <see cref="TensorMath"/>.
/// This is the default provider used when no GPU is available or configured.
/// </summary>
public sealed class CpuComputeProvider : IComputeProvider
{
    /// <inheritdoc/>
    public string Name => "CPU";

    /// <inheritdoc/>
    public bool IsGpu => false;

    /// <inheritdoc/>
    public Tensor MatMul(Tensor a, Tensor b) => TensorMath.MatMul(a, b);

    /// <inheritdoc/>
    public Tensor BatchedMatMul(Tensor a, Tensor b) => TensorMath.BatchedMatMul(a, b);

    /// <inheritdoc/>
    public void Dispose() { /* no resources to release */ }
}

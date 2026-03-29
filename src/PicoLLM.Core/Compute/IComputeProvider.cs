using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Compute;

/// <summary>
/// Abstraction over matrix multiplication compute backends (CPU or GPU).
/// Implement this interface to plug in a different compute device.
/// </summary>
/// <remarks>
/// All implementors must be thread-safe for concurrent <see cref="MatMul"/>
/// and <see cref="BatchedMatMul"/> calls.
/// </remarks>
public interface IComputeProvider : IDisposable
{
    /// <summary>Human-readable name of the backend (e.g. "CPU", "CUDA:RTX 4090").</summary>
    string Name { get; }

    /// <summary>True if this provider executes on a GPU.</summary>
    bool IsGpu { get; }

    /// <summary>
    /// 2D matrix multiplication: result = a @ b.
    /// </summary>
    /// <param name="a">Rank-2 tensor [M, K].</param>
    /// <param name="b">Rank-2 tensor [K, N].</param>
    /// <returns>Rank-2 tensor [M, N].</returns>
    Tensor MatMul(Tensor a, Tensor b);

    /// <summary>
    /// Batched 4D matrix multiplication over the batch and head dimensions.
    /// result[b,h] = a[b,h] @ b[b,h] for all b,h.
    /// </summary>
    /// <param name="a">Rank-4 tensor [B, H, S1, D].</param>
    /// <param name="b">Rank-4 tensor [B, H, D, S2].</param>
    /// <returns>Rank-4 tensor [B, H, S1, S2].</returns>
    Tensor BatchedMatMul(Tensor a, Tensor b);
}

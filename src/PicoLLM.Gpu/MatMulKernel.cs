using ILGPU;
using ILGPU.Runtime;

namespace PicoLLM.Gpu;

/// <summary>
/// ILGPU kernel for 2D matrix multiplication on the GPU.
/// Each thread computes one element of the result matrix.
/// This naive implementation is correct but not cache-optimized;
/// tiled/shared-memory variants would be needed for production use.
/// </summary>
public static class MatMulKernel
{
    /// <summary>
    /// Computes result[row, col] = Σ_k a[row, k] * b[k, col].
    /// Called once per output element via Index2D (row=X, col=Y).
    /// </summary>
    public static void Execute(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> result,
        int k)
    {
        float sum = 0f;
        for (int i = 0; i < k; i++)
            sum += a[index.X, i] * b[i, index.Y];
        result[index.X, index.Y] = sum;
    }
}

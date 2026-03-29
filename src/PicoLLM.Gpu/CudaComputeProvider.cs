using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Gpu;

/// <summary>
/// GPU compute provider using ILGPU + CUDA.
/// Falls back to CPU automatically for small matrices where transfer overhead
/// exceeds the compute gain.
/// </summary>
/// <remarks>
/// Dispose this provider when training is complete to release GPU memory.
/// </remarks>
public sealed class CudaComputeProvider : IComputeProvider
{
    /// <summary>M*N*K must exceed this value to use GPU (otherwise transfer overhead dominates).</summary>
    public const int GpuThreshold = 65536;

    private readonly Context _context;
    private readonly CudaAccelerator _accelerator;
    private readonly CpuComputeProvider _cpuFallback;
    private readonly Action<
        Index2D,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>,
        int> _kernel;
    private bool _disposed;

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public bool IsGpu => true;

    /// <summary>
    /// Initialises the CUDA accelerator and compiles the MatMul kernel.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if no CUDA device is found.</exception>
    public CudaComputeProvider()
    {
        _context = Context.CreateDefault();
        var devices = _context.GetCudaDevices();
        if (devices.Count == 0)
        {
            _context.Dispose();
            throw new InvalidOperationException("No CUDA-capable GPU detected.");
        }

        _accelerator = devices[0].CreateCudaAccelerator(_context);
        Name = $"CUDA:{_accelerator.Name}";
        _cpuFallback = new CpuComputeProvider();

        _kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            int>(MatMulKernel.Execute);
    }

    /// <inheritdoc/>
    public Tensor MatMul(Tensor a, Tensor b)
    {
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];

        if ((long)M * N * K <= GpuThreshold)
            return _cpuFallback.MatMul(a, b);

        using var bufA = _accelerator.Allocate2DDenseX<float>(new LongIndex2D(M, K));
        using var bufB = _accelerator.Allocate2DDenseX<float>(new LongIndex2D(K, N));
        using var bufR = _accelerator.Allocate2DDenseX<float>(new LongIndex2D(M, N));

        bufA.CopyFromCPU(To2D(a.Data.ToArray(), M, K));
        bufB.CopyFromCPU(To2D(b.Data.ToArray(), K, N));

        _kernel(new Index2D(M, N), bufA.View, bufB.View, bufR.View, K);
        _accelerator.Synchronize();

        return TensorFactory.FromData([M, N], Flatten(bufR.GetAsArray2D(), M, N));
    }

    /// <inheritdoc/>
    public Tensor BatchedMatMul(Tensor a, Tensor b)
    {
        int B = a.Shape[0], H = a.Shape[1], S1 = a.Shape[2], D = a.Shape[3];
        int S2 = b.Shape[3];
        var result = new float[B * H * S1 * S2];

        for (int bIdx = 0; bIdx < B; bIdx++)
        {
            for (int hIdx = 0; hIdx < H; hIdx++)
            {
                var sliceA = ExtractSlice(a, bIdx, hIdx, S1, D);
                var sliceB = ExtractSlice(b, bIdx, hIdx, D, S2);
                var sliceR = MatMul(sliceA, sliceB);
                InsertSlice(result, sliceR.Data.ToArray(), bIdx, hIdx, B, H, S1, S2);
            }
        }

        return TensorFactory.FromData([B, H, S1, S2], result);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _cpuFallback.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
        _disposed = true;
    }

    private static Tensor ExtractSlice(Tensor t, int b, int h, int rows, int cols)
    {
        int offset = (b * t.Shape[1] + h) * rows * cols;
        var src = t.Data;
        var slice = new float[rows * cols];
        for (int i = 0; i < slice.Length; i++) slice[i] = src[offset + i];
        return TensorFactory.FromData([rows, cols], slice);
    }

    private static void InsertSlice(float[] dest, float[] slice, int b, int h, int B, int H, int S1, int S2)
    {
        int offset = (b * H + h) * S1 * S2;
        Array.Copy(slice, 0, dest, offset, slice.Length);
    }

    private static float[,] To2D(float[] flat, int rows, int cols)
    {
        var arr = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                arr[i, j] = flat[i * cols + j];
        return arr;
    }

    private static float[] Flatten(float[,] arr, int rows, int cols)
    {
        var flat = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = arr[i, j];
        return flat;
    }
}

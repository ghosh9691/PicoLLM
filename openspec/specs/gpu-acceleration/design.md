# GPU Acceleration — Technical Design

## Library Choice: ILGPU

ILGPU is a .NET GPU computing library that compiles C# to CUDA PTX. It requires no native CUDA SDK installation — the NuGet package includes everything needed.

```xml
<PackageReference Include="ILGPU" Version="1.5.1" />
<PackageReference Include="ILGPU.Algorithms" Version="1.5.1" />
```

## Compute Provider Interface

```csharp
public interface IComputeProvider : IDisposable
{
    string Name { get; }
    bool IsGpu { get; }
    Tensor MatMul(Tensor a, Tensor b);
    Tensor BatchedMatMul(Tensor a, Tensor b);
}

public class CpuComputeProvider : IComputeProvider
{
    // Delegates to TensorMath.MatMul
}

public class CudaComputeProvider : IComputeProvider
{
    private Context _context;
    private Accelerator _accelerator;
    // Uses ILGPU kernels
}
```

## GPU Detection

```csharp
public static class GpuDetector
{
    public static GpuInfo? DetectNvidia()
    {
        using var context = Context.CreateDefault();
        var cudaDevices = context.GetCudaDevices();
        if (cudaDevices.Length == 0) return null;

        var device = cudaDevices[0];
        return new GpuInfo(device.Name, device.MemorySize, ...);
    }
}
```

## ILGPU Kernel for MatMul

```csharp
static void MatMulKernel(
    Index2D index,
    ArrayView2D<float, Stride2D.DenseX> a,
    ArrayView2D<float, Stride2D.DenseX> b,
    ArrayView2D<float, Stride2D.DenseX> result,
    int K)
{
    float sum = 0;
    for (int k = 0; k < K; k++)
        sum += a[index.X, k] * b[k, index.Y];
    result[index.X, index.Y] = sum;
}
```

## Size Threshold

Only use GPU when M*N*K > 65536 (configurable). Below this threshold, CPU is faster due to transfer overhead.

## Project Location

`src/PicoLLM.Gpu/`:
- `GpuDetector.cs`
- `GpuInfo.cs`
- `IComputeProvider.cs`
- `CpuComputeProvider.cs`
- `CudaComputeProvider.cs`
- `MatMulKernel.cs`

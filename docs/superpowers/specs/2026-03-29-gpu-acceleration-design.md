# GPU Acceleration — Design

**Date:** 2026-03-29
**Capability:** 9 — gpu-acceleration
**Project:** PicoLLM.Gpu (new) + additions to PicoLLM.Core

## Scope

- `IComputeProvider` interface and `CpuComputeProvider` added to `PicoLLM.Core` (zero GPU dependency)
- New `PicoLLM.Gpu` project (references Core) with ILGPU + ILGPU.Algorithms NuGet
- `MultiHeadAttention` and `FeedForward` in Core updated to accept optional `IComputeProvider`
- Test project gains reference to Gpu

## Architecture

| Location | File | Responsibility |
|----------|------|----------------|
| PicoLLM.Core | `IComputeProvider.cs` | Interface: `Name`, `IsGpu`, `MatMul`, `BatchedMatMul` |
| PicoLLM.Core | `CpuComputeProvider.cs` | Delegates to `TensorMath.MatMul` |
| PicoLLM.Gpu | `GpuInfo.cs` | Record: GPU name, VRAM, compute capability |
| PicoLLM.Gpu | `GpuDetector.cs` | `DetectNvidia()` — ILGPU Context scan, returns `GpuInfo?` |
| PicoLLM.Gpu | `CudaComputeProvider.cs` | ILGPU accelerator, MatMul kernel, CPU↔GPU transfers |
| PicoLLM.Gpu | `MatMulKernel.cs` | Static ILGPU kernel method (Index2D, ArrayView2D) |

## Compute Provider Interface

```csharp
public interface IComputeProvider : IDisposable
{
    string Name { get; }
    bool IsGpu { get; }
    Tensor MatMul(Tensor a, Tensor b);
    Tensor BatchedMatMul(Tensor a, Tensor b);
}
```

## Size Threshold

`CudaComputeProvider` uses CPU for operations where `M * N * K <= 65536` (transfer overhead exceeds compute gain). Threshold is a configurable constant.

## Memory Lifecycle

For each GPU MatMul:
1. Copy input tensors A, B to GPU `MemoryBuffer2D<float>`
2. Allocate output buffer on GPU
3. Launch kernel
4. Copy result back to CPU `Tensor`
5. Dispose all GPU buffers

## Layer Integration

`MultiHeadAttention` and `FeedForward` accept an optional `IComputeProvider? computeProvider = null`, defaulting to `new CpuComputeProvider()`. No breaking changes to existing call sites.

## Tests

- `CpuComputeProvider`: correct matmul results (unit)
- CPU vs GPU result comparison: tolerance 1e-4 for [512,256] × [256,512]
- No-GPU fallback: `CudaComputeProvider` or `GpuDetector` works gracefully when no GPU available
- Size threshold: small tensors routed to CPU even when GPU is available
- Performance benchmark: GPU vs CPU for large matrices (informational, not a pass/fail test)

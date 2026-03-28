# GPU Acceleration — Implementation Tasks

## Setup
- [ ] 1.1 Create `PicoLLM.Gpu` project, reference PicoLLM.Core
- [ ] 1.2 Add ILGPU and ILGPU.Algorithms NuGet packages

## Interface
- [ ] 2.1 Define `IComputeProvider` interface in PicoLLM.Core (no GPU dependency)
- [ ] 2.2 Implement `CpuComputeProvider` in PicoLLM.Core (delegates to TensorMath)
- [ ] 2.3 Write tests: CpuComputeProvider produces correct matmul results

## GPU Detection
- [ ] 3.1 Implement `GpuDetector.DetectNvidia()` using ILGPU Context
- [ ] 3.2 Implement `GpuInfo` record (name, VRAM, compute capability)
- [ ] 3.3 Handle no-GPU case gracefully (return null, no exception)

## CUDA Provider
- [ ] 4.1 Implement `CudaComputeProvider` with ILGPU accelerator init
- [ ] 4.2 Implement ILGPU MatMul kernel
- [ ] 4.3 Implement CPU↔GPU memory transfer helpers
- [ ] 4.4 Implement size threshold check (use CPU for small tensors)
- [ ] 4.5 Implement proper Dispose pattern for GPU resources

## Validation
- [ ] 5.1 Write CPU vs GPU result comparison tests (tolerance 1e-4)
- [ ] 5.2 Write fallback test: provider works when no GPU available
- [ ] 5.3 Write performance benchmark: GPU vs CPU for large matrices

## Integration
- [ ] 6.1 Wire `IComputeProvider` into transformer attention matmul
- [ ] 6.2 Wire into feedforward layer matmul
- [ ] 6.3 Add GPU status to training metrics output

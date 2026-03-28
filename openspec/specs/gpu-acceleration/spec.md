# GPU Acceleration Specification

## Purpose

Optionally offload compute-intensive tensor operations (primarily matrix multiplication) to NVIDIA GPUs using ILGPU, a .NET GPU computing library. This is a nice-to-have feature that demonstrates GPU acceleration concepts.

### Requirement: GPU Detection

The system SHALL detect the presence of NVIDIA GPUs at runtime.

#### Scenario: GPU available
- **GIVEN** a machine with an NVIDIA GPU with CUDA support
- **WHEN** GPU detection runs
- **THEN** the system reports GPU name, compute capability, and VRAM

#### Scenario: No GPU available
- **GIVEN** a machine with no NVIDIA GPU
- **WHEN** GPU detection runs
- **THEN** the system reports "No GPU detected, using CPU" and continues normally

### Requirement: GPU Matrix Multiplication

The system SHALL provide a GPU-accelerated matrix multiplication that can substitute for the CPU implementation.

#### Scenario: GPU matmul produces same results
- **GIVEN** two matrices A [512, 256] and B [256, 512]
- **WHEN** multiplied on GPU and CPU separately
- **THEN** results match within floating-point tolerance (1e-4)

### Requirement: Transparent Fallback

The system SHALL automatically fall back to CPU if GPU initialization fails or if tensors are too small to benefit from GPU transfer overhead.

#### Scenario: Small tensor stays on CPU
- **GIVEN** a matrix multiply of [4, 4] × [4, 4]
- **WHEN** the compute provider is queried
- **THEN** CPU is used (transfer overhead exceeds compute benefit)

### Requirement: Compute Provider Abstraction

The system SHALL define an `IComputeProvider` interface so the training loop can use either CPU or GPU without code changes.

#### Scenario: Swap providers
- **GIVEN** an `IComputeProvider` interface with `MatMul`, `BatchedMatMul`, etc.
- **WHEN** the GPU provider is unavailable
- **THEN** the CPU provider is used with identical API

### Requirement: Memory Management

The system SHALL explicitly manage GPU memory transfers, copying tensors to GPU before compute and results back to CPU after.

#### Scenario: Transfer lifecycle
- **GIVEN** a matrix multiply on GPU
- **WHEN** executed
- **THEN** input tensors are copied to GPU memory
- **AND** computation runs on GPU
- **AND** result tensor is copied back to CPU memory
- **AND** GPU buffers are released

# Core Tensor — Implementation Tasks

## Setup
- [ ] 1.1 Create solution `PicoLLM.sln` with projects: `PicoLLM.Core`, `PicoLLM.Tests`
- [ ] 1.2 Configure `PicoLLM.Core` as `net9.0` class library with nullable enabled, zero NuGet packages
- [ ] 1.3 Configure `PicoLLM.Tests` with xUnit + FluentAssertions

## Tensor Type
- [ ] 2.1 Implement `Tensor` class with `float[] _data`, `int[] _shape`, `int[] _strides`
- [ ] 2.2 Implement stride computation (row-major / C-order)
- [ ] 2.3 Implement multi-index to flat-index conversion with bounds checking
- [ ] 2.4 Implement `this[params int[] indices]` indexer (get/set)
- [ ] 2.5 Implement `Rank`, `Shape`, `Length` properties
- [ ] 2.6 Write tests: construction, indexing, bounds checking, properties

## Factory Methods
- [ ] 3.1 Implement `TensorFactory.Zeros(shape)`, `Ones(shape)`, `Fill(shape, value)`
- [ ] 3.2 Implement `TensorFactory.Random(shape, seed)` — uniform [0, 1)
- [ ] 3.3 Implement `TensorFactory.RandomNormal(shape, mean, std, seed)`
- [ ] 3.4 Implement `TensorFactory.XavierUniform(fanIn, fanOut, seed)`
- [ ] 3.5 Implement `TensorFactory.XavierNormal(fanIn, fanOut, seed)`
- [ ] 3.6 Write tests for each factory method

## Element-wise Operations
- [ ] 4.1 Implement `Add(a, b)`, `Subtract(a, b)`, `Multiply(a, b)`, `Divide(a, b)` — shape must match
- [ ] 4.2 Implement scalar versions: `Add(tensor, scalar)`, `Multiply(tensor, scalar)`, etc.
- [ ] 4.3 Implement operator overloads: `+`, `-`, `*`, `/` for tensor-tensor and tensor-scalar
- [ ] 4.4 Write tests for all element-wise ops

## Matrix Multiplication
- [ ] 5.1 Implement `MatMul(a, b)` for 2D tensors [M,K] × [K,N] → [M,N]
- [ ] 5.2 Implement `BatchedMatMul(a, b)` for 4D tensors [B,H,S1,D] × [B,H,D,S2] → [B,H,S1,S2]
- [ ] 5.3 Write tests: known small matrices, identity matrix, batched attention shapes

## Reshape, Transpose, View
- [ ] 6.1 Implement `Reshape(tensor, newShape)` — validates total elements match
- [ ] 6.2 Implement `Transpose(tensor, dim0, dim1)` — swaps two dimensions
- [ ] 6.3 Implement `Permute(tensor, axes)` — general axis reordering
- [ ] 6.4 Write tests: reshape round-trip, transpose correctness, permute for attention

## Reduction & Softmax
- [ ] 7.1 Implement `Sum(tensor, axis)`, `Mean(tensor, axis)`, `Max(tensor, axis)`, `ArgMax(tensor, axis)`
- [ ] 7.2 Implement `Softmax(tensor, axis)` with numerical stability (subtract max)
- [ ] 7.3 Implement `ApplyMask(tensor, mask, fillValue)` for causal masking
- [ ] 7.4 Write tests: reductions on known data, softmax sums to 1, causal mask blocks future

## Activation Functions
- [ ] 8.1 Implement `ReLU(tensor)`, `GELU(tensor)`, `Sigmoid(tensor)`, `Tanh(tensor)`
- [ ] 8.2 Write tests for each activation with known input/output pairs

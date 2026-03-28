# Core Tensor — Technical Design

## Data Structure

```csharp
public class Tensor
{
    private readonly float[] _data;
    private readonly int[] _shape;
    private readonly int[] _strides;  // precomputed for fast index math

    public ReadOnlySpan<float> Data => _data;
    public ReadOnlySpan<int> Shape => _shape;
    public int Rank => _shape.Length;
    public int Length => _data.Length;
}
```

## Stride Computation

Strides are row-major (C-order): `strides[i] = product(shape[i+1..])`. For shape [2, 3, 4]: strides = [12, 4, 1].

Flat index from multi-index: `flatIndex = sum(index[i] * strides[i])`.

## Matrix Multiplication

Use the naive triple-loop for clarity (educational purpose). Optimize with loop tiling only if GPU offload is not available and performance is a concern.

```
for i in 0..M:
    for j in 0..N:
        sum = 0
        for k in 0..K:
            sum += A[i,k] * B[k,j]
        C[i,j] = sum
```

For batched matmul, add outer loops over batch and head dimensions.

## Numerically Stable Softmax

```
max_val = max(x)
shifted = x - max_val
exp_vals = exp(shifted)
sum_exp = sum(exp_vals)
result = exp_vals / sum_exp
```

## Weight Initialization

- **Xavier Uniform**: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
- **Xavier Normal**: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
- **Kaiming (for ReLU)**: N(0, std) where std = sqrt(2 / fan_in)

Use `System.Random` with a seed for reproducibility.

## Project Location

All code in `src/PicoLLM.Core/Tensors/`. Key files:
- `Tensor.cs` — Core type
- `TensorMath.cs` — Static math operations (matmul, softmax, etc.)
- `TensorFactory.cs` — Creation helpers (zeros, ones, random, Xavier)

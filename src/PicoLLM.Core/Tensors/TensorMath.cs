namespace PicoLLM.Core.Tensors;

/// <summary>
/// Static class providing all mathematical operations on <see cref="Tensor"/> instances.
/// Operations return new tensors; inputs are never mutated.
/// </summary>
public static class TensorMath
{
    // ── Element-wise: tensor × tensor ──────────────────────────────────────

    /// <summary>Element-wise addition. Shapes must match exactly.</summary>
    public static Tensor Add(Tensor a, Tensor b)
    {
        AssertSameShape(a, b, nameof(Add));
        var result = new float[a.Length];
        var ad = a.Data; var bd = b.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] + bd[i];
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Element-wise subtraction. Shapes must match exactly.</summary>
    public static Tensor Subtract(Tensor a, Tensor b)
    {
        AssertSameShape(a, b, nameof(Subtract));
        var result = new float[a.Length];
        var ad = a.Data; var bd = b.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] - bd[i];
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Element-wise multiplication (Hadamard product). Shapes must match exactly.</summary>
    public static Tensor Multiply(Tensor a, Tensor b)
    {
        AssertSameShape(a, b, nameof(Multiply));
        var result = new float[a.Length];
        var ad = a.Data; var bd = b.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] * bd[i];
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Element-wise division. Shapes must match exactly.</summary>
    public static Tensor Divide(Tensor a, Tensor b)
    {
        AssertSameShape(a, b, nameof(Divide));
        var result = new float[a.Length];
        var ad = a.Data; var bd = b.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] / bd[i];
        return new Tensor(a.GetShape(), result);
    }

    // ── Element-wise: tensor × scalar ──────────────────────────────────────

    /// <summary>Adds a scalar to every element.</summary>
    public static Tensor Add(Tensor a, float scalar)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] + scalar;
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Subtracts a scalar from every element.</summary>
    public static Tensor Subtract(Tensor a, float scalar)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] - scalar;
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Multiplies every element by a scalar.</summary>
    public static Tensor Multiply(Tensor a, float scalar)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] * scalar;
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Divides every element by a scalar.</summary>
    public static Tensor Divide(Tensor a, float scalar)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = ad[i] / scalar;
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Applies MathF.Exp to every element.</summary>
    public static Tensor Exp(Tensor a)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = MathF.Exp(ad[i]);
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Applies MathF.Log to every element.</summary>
    public static Tensor Log(Tensor a)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = MathF.Log(ad[i]);
        return new Tensor(a.GetShape(), result);
    }

    /// <summary>Applies MathF.Sqrt to every element.</summary>
    public static Tensor Sqrt(Tensor a)
    {
        var result = new float[a.Length];
        var ad = a.Data;
        for (int i = 0; i < result.Length; i++) result[i] = MathF.Sqrt(ad[i]);
        return new Tensor(a.GetShape(), result);
    }

    // ── Matrix multiplication ───────────────────────────────────────────────

    /// <summary>
    /// 2D matrix multiplication: [M, K] × [K, N] → [M, N].
    /// Uses the naive triple-loop so learners can trace the arithmetic.
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires 2D tensors.");
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];
        if (b.Shape[0] != K)
            throw new ArgumentException(
                $"Inner dimensions do not match: [{M},{K}] × [{b.Shape[0]},{N}].");

        var result = new float[M * N];
        var ad = a.Data; var bd = b.Data;
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
            {
                float aik = ad[i * K + k];
                for (int j = 0; j < N; j++)
                    result[i * N + j] += aik * bd[k * N + j];
            }
        return new Tensor([M, N], result);
    }

    /// <summary>
    /// Batched matrix multiplication for attention: [B, H, S1, D] × [B, H, D, S2] → [B, H, S1, S2].
    /// Each (batch, head) pair is an independent 2D matmul.
    /// </summary>
    public static Tensor BatchedMatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 4 || b.Rank != 4)
            throw new ArgumentException("BatchedMatMul requires 4D tensors.");

        int B = a.Shape[0], H = a.Shape[1], S1 = a.Shape[2], D = a.Shape[3];
        int S2 = b.Shape[3];
        if (b.Shape[0] != B || b.Shape[1] != H || b.Shape[2] != D)
            throw new ArgumentException(
                $"BatchedMatMul dimension mismatch: [{B},{H},{S1},{D}] × [{b.Shape[0]},{b.Shape[1]},{b.Shape[2]},{S2}].");

        var result = new float[B * H * S1 * S2];
        var ad = a.Data; var bd = b.Data;

        for (int batch = 0; batch < B; batch++)
            for (int head = 0; head < H; head++)
                for (int i = 0; i < S1; i++)
                    for (int k = 0; k < D; k++)
                    {
                        float aval = ad[((batch * H + head) * S1 + i) * D + k];
                        for (int j = 0; j < S2; j++)
                            result[((batch * H + head) * S1 + i) * S2 + j] +=
                                aval * bd[((batch * H + head) * D + k) * S2 + j];
                    }

        return new Tensor([B, H, S1, S2], result);
    }

    // ── Reshape, Transpose, Permute ────────────────────────────────────────

    /// <summary>
    /// Returns a tensor with a new shape but the same data (no copy).
    /// The product of all dimensions must equal the original tensor's Length.
    /// Use -1 for at most one dimension to infer its size automatically.
    /// </summary>
    public static Tensor Reshape(Tensor tensor, params int[] newShape)
    {
        // Resolve any -1 dimension
        int inferredIdx = -1;
        int known = 1;
        for (int i = 0; i < newShape.Length; i++)
        {
            if (newShape[i] == -1)
            {
                if (inferredIdx >= 0)
                    throw new ArgumentException("Only one dimension can be inferred (-1).");
                inferredIdx = i;
            }
            else known *= newShape[i];
        }
        if (inferredIdx >= 0)
        {
            if (tensor.Length % known != 0)
                throw new ArgumentException(
                    $"Cannot infer dimension: {tensor.Length} is not divisible by {known}.");
            newShape = (int[])newShape.Clone();
            newShape[inferredIdx] = tensor.Length / known;
        }

        int total = 1;
        foreach (var d in newShape) total *= d;
        if (total != tensor.Length)
            throw new ArgumentException(
                $"Reshape: total elements {total} ≠ original {tensor.Length}.");

        // Share the underlying array — no copy
        return new Tensor(newShape, tensor.RawData);
    }

    /// <summary>
    /// Swaps two dimensions and returns a new tensor with data reordered.
    /// Example: Transpose([3,4], 0, 1) → [4,3].
    /// </summary>
    public static Tensor Transpose(Tensor tensor, int dim0, int dim1)
    {
        int rank = tensor.Rank;
        if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank)
            throw new ArgumentOutOfRangeException($"Dimensions {dim0},{dim1} out of range for rank-{rank} tensor.");

        var axes = new int[rank];
        for (int i = 0; i < rank; i++) axes[i] = i;
        axes[dim0] = dim1;
        axes[dim1] = dim0;
        return Permute(tensor, axes);
    }

    /// <summary>
    /// General axis reordering. Returns a new tensor with data physically reordered.
    /// axes must be a permutation of [0, rank).
    /// </summary>
    public static Tensor Permute(Tensor tensor, int[] axes)
    {
        int rank = tensor.Rank;
        if (axes.Length != rank)
            throw new ArgumentException($"axes length {axes.Length} must equal rank {rank}.");

        var oldShape = tensor.GetShape();
        var newShape = new int[rank];
        for (int i = 0; i < rank; i++) newShape[i] = oldShape[axes[i]];

        var oldStrides = tensor.Strides;
        var newStrides = new int[rank]; // strides in the NEW tensor corresponding to OLD axes
        // We need: for each position in the new tensor, compute flat index in old tensor.
        // New strides (in new index space) = old strides of the permuted axes.
        var newToOldStrides = new int[rank];
        for (int i = 0; i < rank; i++) newToOldStrides[i] = oldStrides[axes[i]];

        var result = new float[tensor.Length];
        var src = tensor.Data;
        var newTotalStrides = ComputeStrides(newShape);

        // Iterate over every element in the new shape
        for (int flatNew = 0; flatNew < tensor.Length; flatNew++)
        {
            // Decompose flatNew into newShape indices
            int rem = flatNew;
            int flatOld = 0;
            for (int d = 0; d < rank; d++)
            {
                int idx = rem / newTotalStrides[d];
                rem %= newTotalStrides[d];
                flatOld += idx * newToOldStrides[d];
            }
            result[flatNew] = src[flatOld];
        }

        return new Tensor(newShape, result);
    }

    /// <summary>
    /// Slices rows [startRow, startRow+rowCount) along axis 0.
    /// Only supports slicing along the first axis for positional encoding lookup.
    /// </summary>
    public static Tensor Slice(Tensor tensor, int startRow, int rowCount)
    {
        int[] shape = tensor.GetShape();
        if (startRow < 0 || startRow + rowCount > shape[0])
            throw new ArgumentOutOfRangeException(
                $"Slice [{startRow},{startRow + rowCount}) out of bounds for axis-0 size {shape[0]}.");

        int rowSize = tensor.Length / shape[0];
        var result = new float[rowCount * rowSize];
        tensor.Data.Slice(startRow * rowSize, rowCount * rowSize).CopyTo(result);

        var newShape = (int[])shape.Clone();
        newShape[0] = rowCount;
        return new Tensor(newShape, result);
    }

    // ── Reductions ─────────────────────────────────────────────────────────

    /// <summary>
    /// Sums elements along the specified axis.
    /// Result has the same rank with size 1 along the reduced axis, then squeezed.
    /// </summary>
    public static Tensor Sum(Tensor tensor, int axis)
    {
        return ReduceAxis(tensor, axis, 0f, (acc, x) => acc + x);
    }

    /// <summary>Computes the mean along the specified axis.</summary>
    public static Tensor Mean(Tensor tensor, int axis)
    {
        int axisSize = tensor.Shape[axis];
        return Multiply(Sum(tensor, axis), 1f / axisSize);
    }

    /// <summary>Computes the maximum value along the specified axis.</summary>
    public static Tensor Max(Tensor tensor, int axis)
    {
        return ReduceAxis(tensor, axis, float.NegativeInfinity, MathF.Max);
    }

    /// <summary>
    /// Returns the index of the maximum value along the specified axis.
    /// Result dtype is float (cast to int at call site for index use).
    /// </summary>
    public static Tensor ArgMax(Tensor tensor, int axis)
    {
        int rank = tensor.Rank;
        int[] shape = tensor.GetShape();
        int axisSize = shape[axis];

        // Build output shape (axis dimension removed)
        var outShape = new int[rank - 1];
        for (int i = 0, j = 0; i < rank; i++)
            if (i != axis) outShape[j++] = shape[i];

        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        var result = new float[outerSize * innerSize];
        var src = tensor.Data;

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;
                for (int a = 0; a < axisSize; a++)
                {
                    float v = src[(outer * axisSize + a) * innerSize + inner];
                    if (v > maxVal) { maxVal = v; maxIdx = a; }
                }
                result[outer * innerSize + inner] = maxIdx;
            }

        return new Tensor(outShape.Length == 0 ? [1] : outShape, result);
    }

    // ── Softmax ────────────────────────────────────────────────────────────

    /// <summary>
    /// Numerically stable softmax along the specified axis.
    /// Subtracts the per-slice max before exponentiation to avoid overflow.
    /// </summary>
    public static Tensor Softmax(Tensor tensor, int axis)
    {
        int rank = tensor.Rank;
        int[] shape = tensor.GetShape();
        int axisSize = shape[axis];

        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        var src = tensor.Data;
        var result = new float[tensor.Length];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max
                float maxVal = float.NegativeInfinity;
                for (int a = 0; a < axisSize; a++)
                    maxVal = MathF.Max(maxVal, src[(outer * axisSize + a) * innerSize + inner]);

                // Sum of exp(x - max)
                float sumExp = 0f;
                for (int a = 0; a < axisSize; a++)
                    sumExp += MathF.Exp(src[(outer * axisSize + a) * innerSize + inner] - maxVal);

                // Normalize
                for (int a = 0; a < axisSize; a++)
                {
                    int idx = (outer * axisSize + a) * innerSize + inner;
                    result[idx] = MathF.Exp(src[idx] - maxVal) / sumExp;
                }
            }

        return new Tensor(shape, result);
    }

    // ── Masking ────────────────────────────────────────────────────────────

    /// <summary>
    /// Applies a boolean mask to a tensor, replacing masked positions with <paramref name="fillValue"/>.
    /// <paramref name="mask"/> must have the same shape as <paramref name="tensor"/>.
    /// A mask value of <c>true</c> means "keep"; <c>false</c> means "fill".
    /// </summary>
    public static Tensor ApplyMask(Tensor tensor, bool[] mask, float fillValue = float.NegativeInfinity)
    {
        if (mask.Length != tensor.Length)
            throw new ArgumentException(
                $"Mask length {mask.Length} does not match tensor length {tensor.Length}.");

        var result = new float[tensor.Length];
        var src = tensor.Data;
        for (int i = 0; i < result.Length; i++)
            result[i] = mask[i] ? src[i] : fillValue;

        return new Tensor(tensor.GetShape(), result);
    }

    /// <summary>
    /// Creates a causal (lower-triangular) mask for a square attention score tensor of shape [seq, seq].
    /// Returns a bool[] where true = keep (positions where j ≤ i) and false = mask out (j > i).
    /// </summary>
    public static bool[] CausalMask(int seqLen)
    {
        var mask = new bool[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                mask[i * seqLen + j] = j <= i;
        return mask;
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static void AssertSameShape(Tensor a, Tensor b, string opName)
    {
        if (a.Rank != b.Rank)
            throw new ArgumentException(
                $"{opName}: rank mismatch {a.Rank} vs {b.Rank}.");
        for (int i = 0; i < a.Rank; i++)
            if (a.Shape[i] != b.Shape[i])
                throw new ArgumentException(
                    $"{opName}: shape mismatch at dim {i}: {a.Shape[i]} vs {b.Shape[i]}.");
    }

    private static Tensor ReduceAxis(Tensor tensor, int axis, float identity, Func<float, float, float> combine)
    {
        int rank = tensor.Rank;
        int[] shape = tensor.GetShape();
        int axisSize = shape[axis];

        var outShape = new int[rank - 1];
        for (int i = 0, j = 0; i < rank; i++)
            if (i != axis) outShape[j++] = shape[i];
        if (outShape.Length == 0) outShape = [1];

        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        var result = new float[outerSize * innerSize];
        var src = tensor.Data;

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                float acc = identity;
                for (int a = 0; a < axisSize; a++)
                    acc = combine(acc, src[(outer * axisSize + a) * innerSize + inner]);
                result[outer * innerSize + inner] = acc;
            }

        return new Tensor(outShape, result);
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }
}

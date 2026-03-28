namespace PicoLLM.Core.Tensors;

/// <summary>
/// An N-dimensional array of single-precision floats stored in row-major (C-order) layout.
/// This is the fundamental data structure for all tensor operations in PicoLLM.
/// Strides are pre-computed so multi-index to flat-index conversion is O(rank).
/// </summary>
public sealed class Tensor
{
    private readonly float[] _data;
    private readonly int[] _shape;
    private readonly int[] _strides;

    /// <summary>Creates a zero-initialized tensor with the given shape.</summary>
    /// <param name="shape">One or more positive dimension sizes.</param>
    public Tensor(params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));
        foreach (var dim in shape)
            if (dim <= 0)
                throw new ArgumentException($"Each dimension must be positive. Got {dim}.", nameof(shape));

        _shape = (int[])shape.Clone();
        _strides = ComputeStrides(shape);
        int total = 1;
        foreach (var d in shape) total *= d;
        _data = new float[total];
    }

    /// <summary>
    /// Internal constructor used by factory methods and math operations.
    /// Takes ownership of the provided data array (no copy).
    /// </summary>
    internal Tensor(int[] shape, float[] data)
    {
        _shape = (int[])shape.Clone();
        _strides = ComputeStrides(shape);
        int total = 1;
        foreach (var d in shape) total *= d;
        if (data.Length != total)
            throw new ArgumentException(
                $"Data length {data.Length} does not match shape product {total}.");
        _data = data;
    }

    /// <summary>Read-only view of the underlying float array.</summary>
    public ReadOnlySpan<float> Data => _data;

    /// <summary>
    /// Mutable span over the underlying float array.
    /// Used by the optimizer and backward passes to update weights in-place.
    /// </summary>
    public Span<float> MutableData => _data;

    /// <summary>Read-only view of the dimension sizes.</summary>
    public ReadOnlySpan<int> Shape => _shape;

    /// <summary>Read-only view of the strides (elements to skip per dimension).</summary>
    public ReadOnlySpan<int> Strides => _strides;

    /// <summary>Total number of elements across all dimensions.</summary>
    public int Length => _data.Length;

    /// <summary>Number of dimensions (rank of the tensor).</summary>
    public int Rank => _shape.Length;

    /// <summary>Returns a copy of the shape array (safe for external modification).</summary>
    public int[] GetShape() => (int[])_shape.Clone();

    /// <summary>
    /// Exposes the raw float array for serialization and GPU transfer.
    /// Callers must not resize or replace the array.
    /// </summary>
    internal float[] RawData => _data;

    /// <summary>Gets or sets an element using a multi-dimensional index.</summary>
    /// <param name="indices">One index per dimension. Must be within bounds.</param>
    /// <exception cref="IndexOutOfRangeException">If any index is out of bounds.</exception>
    public float this[params int[] indices]
    {
        get => _data[ComputeFlatIndex(indices)];
        set => _data[ComputeFlatIndex(indices)] = value;
    }

    private int ComputeFlatIndex(int[] indices)
    {
        if (indices.Length != _shape.Length)
            throw new IndexOutOfRangeException(
                $"Expected {_shape.Length} index dimensions but got {indices.Length}.");
        int flat = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            if ((uint)indices[i] >= (uint)_shape[i])
                throw new IndexOutOfRangeException(
                    $"Index {indices[i]} is out of range for dimension {i} with size {_shape[i]}.");
            flat += indices[i] * _strides[i];
        }
        return flat;
    }

    /// <summary>
    /// Computes row-major strides: strides[i] = product of shape[i+1..end].
    /// Example: shape [2, 3, 4] → strides [12, 4, 1].
    /// </summary>
    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    // ── Operator overloads ──────────────────────────────────────────────────

    /// <summary>Element-wise addition of two same-shape tensors.</summary>
    public static Tensor operator +(Tensor a, Tensor b) => TensorMath.Add(a, b);

    /// <summary>Element-wise subtraction of two same-shape tensors.</summary>
    public static Tensor operator -(Tensor a, Tensor b) => TensorMath.Subtract(a, b);

    /// <summary>Element-wise multiplication of two same-shape tensors.</summary>
    public static Tensor operator *(Tensor a, Tensor b) => TensorMath.Multiply(a, b);

    /// <summary>Element-wise division of two same-shape tensors.</summary>
    public static Tensor operator /(Tensor a, Tensor b) => TensorMath.Divide(a, b);

    /// <summary>Adds a scalar to every element.</summary>
    public static Tensor operator +(Tensor a, float scalar) => TensorMath.Add(a, scalar);

    /// <summary>Subtracts a scalar from every element.</summary>
    public static Tensor operator -(Tensor a, float scalar) => TensorMath.Subtract(a, scalar);

    /// <summary>Multiplies every element by a scalar.</summary>
    public static Tensor operator *(Tensor a, float scalar) => TensorMath.Multiply(a, scalar);

    /// <summary>Multiplies every element by a scalar (scalar on left).</summary>
    public static Tensor operator *(float scalar, Tensor a) => TensorMath.Multiply(a, scalar);

    /// <summary>Divides every element by a scalar.</summary>
    public static Tensor operator /(Tensor a, float scalar) => TensorMath.Divide(a, scalar);

    /// <summary>Negates every element.</summary>
    public static Tensor operator -(Tensor a) => TensorMath.Multiply(a, -1f);

    /// <inheritdoc/>
    public override string ToString()
    {
        var shape = string.Join(", ", _shape);
        return $"Tensor[{shape}] ({_data.Length} elements)";
    }
}

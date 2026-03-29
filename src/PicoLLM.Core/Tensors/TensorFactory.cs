namespace PicoLLM.Core.Tensors;

/// <summary>
/// Static factory methods for creating <see cref="Tensor"/> instances with common initializations.
/// All random methods accept an optional seed for reproducibility in experiments.
/// </summary>
public static class TensorFactory
{
    // ── Constant fills ─────────────────────────────────────────────────────

    /// <summary>Creates a tensor of the given shape filled with zeros.</summary>
    public static Tensor Zeros(params int[] shape) => new Tensor(shape);

    /// <summary>Creates a tensor of the given shape filled with ones.</summary>
    public static Tensor Ones(params int[] shape)
    {
        int total = Product(shape);
        var data = new float[total];
        Array.Fill(data, 1f);
        return new Tensor(shape, data);
    }

    /// <summary>Creates a tensor of the given shape filled with a constant value.</summary>
    public static Tensor Fill(float value, params int[] shape)
    {
        int total = Product(shape);
        var data = new float[total];
        Array.Fill(data, value);
        return new Tensor(shape, data);
    }

    // ── Random fills ───────────────────────────────────────────────────────

    /// <summary>Creates a tensor filled with uniform random values in [0, 1).</summary>
    /// <param name="shape">Dimension sizes.</param>
    /// <param name="seed">Random seed. Use null for a non-deterministic seed.</param>
    public static Tensor Random(int[] shape, int? seed = null)
    {
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        int total = Product(shape);
        var data = new float[total];
        for (int i = 0; i < total; i++) data[i] = (float)rng.NextDouble();
        return new Tensor(shape, data);
    }

    /// <summary>Creates a tensor filled with values drawn from N(mean, std²).</summary>
    /// <param name="shape">Dimension sizes.</param>
    /// <param name="mean">Distribution mean.</param>
    /// <param name="std">Distribution standard deviation.</param>
    /// <param name="seed">Random seed. Use null for a non-deterministic seed.</param>
    public static Tensor RandomNormal(int[] shape, float mean = 0f, float std = 1f, int? seed = null)
    {
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        int total = Product(shape);
        var data = new float[total];
        for (int i = 0; i < total; i++)
            data[i] = mean + std * SampleNormal(rng);
        return new Tensor(shape, data);
    }

    // ── Weight initialization ──────────────────────────────────────────────

    /// <summary>
    /// Xavier (Glorot) Uniform initialization.
    /// Samples from U(-limit, limit) where limit = sqrt(6 / (fanIn + fanOut)).
    /// Used for linear layers without ReLU activations.
    /// </summary>
    public static Tensor XavierUniform(int[] shape, int fanIn, int fanOut, int? seed = null)
    {
        float limit = MathF.Sqrt(6f / (fanIn + fanOut));
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        int total = Product(shape);
        var data = new float[total];
        for (int i = 0; i < total; i++)
            data[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        return new Tensor(shape, data);
    }

    /// <summary>
    /// Xavier (Glorot) Normal initialization.
    /// Samples from N(0, std) where std = sqrt(2 / (fanIn + fanOut)).
    /// </summary>
    public static Tensor XavierNormal(int[] shape, int fanIn, int fanOut, int? seed = null)
    {
        float std = MathF.Sqrt(2f / (fanIn + fanOut));
        return RandomNormal(shape, 0f, std, seed);
    }

    /// <summary>
    /// Kaiming (He) Normal initialization for ReLU activations.
    /// Samples from N(0, std) where std = sqrt(2 / fanIn).
    /// </summary>
    public static Tensor KaimingNormal(int[] shape, int fanIn, int? seed = null)
    {
        float std = MathF.Sqrt(2f / fanIn);
        return RandomNormal(shape, 0f, std, seed);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a tensor from an existing float array (defensive copy).
    /// Used by the training pipeline to wrap computed gradient arrays.
    /// </summary>
    public static Tensor FromArray(int[] shape, float[] data)
    {
        var copy = (float[])data.Clone();
        return new Tensor(shape, copy);
    }

    /// <summary>
    /// Creates a tensor with the given shape backed by the provided data array.
    /// The data is copied — the caller retains ownership of the original array.
    /// </summary>
    /// <param name="shape">Dimension sizes. Their product must equal <c>data.Length</c>.</param>
    /// <param name="data">Float values in row-major order.</param>
    public static Tensor FromData(int[] shape, float[] data)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(data);
        return new Tensor(shape, (float[])data.Clone());
    }

    /// <summary>Creates a 1-D identity tensor with the given size (diagonal of 1s).</summary>
    public static Tensor Eye(int size)
    {
        var data = new float[size * size];
        for (int i = 0; i < size; i++) data[i * size + i] = 1f;
        return new Tensor([size, size], data);
    }

    private static int Product(int[] shape)
    {
        int total = 1;
        foreach (var d in shape) total *= d;
        return total;
    }

    /// <summary>
    /// Box-Muller transform: generates a standard normal sample from two uniform [0,1) values.
    /// </summary>
    private static float SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}

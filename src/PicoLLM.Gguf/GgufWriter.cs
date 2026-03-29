using System.Text;

namespace PicoLLM.Gguf;

/// <summary>
/// Low-level little-endian binary writer for the GGUF v3 format.
/// All multi-byte values are written in little-endian byte order.
/// Wraps a <see cref="Stream"/> and provides type-safe GGUF primitives.
/// </summary>
public sealed class GgufWriter : IDisposable
{
    private readonly BinaryWriter _writer;
    private bool _disposed;

    /// <summary>Creates a GgufWriter that writes to <paramref name="stream"/>.</summary>
    public GgufWriter(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        // BinaryWriter defaults to little-endian on all platforms
        _writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
    }

    /// <summary>Current position in the underlying stream.</summary>
    public long Position => _writer.BaseStream.Position;

    // ── GGUF Header ─────────────────────────────────────────────────────────

    /// <summary>
    /// Writes the 24-byte GGUF v3 file header.
    /// </summary>
    /// <param name="tensorCount">Total number of tensors in the file.</param>
    /// <param name="kvCount">Total number of metadata key-value pairs.</param>
    public void WriteHeader(int tensorCount, int kvCount)
    {
        _writer.Write(GgufConstants.Magic);          // 4 bytes: "GGUF"
        _writer.Write(GgufConstants.Version);        // uint32: 3
        _writer.Write((ulong)tensorCount);           // uint64: tensor count
        _writer.Write((ulong)kvCount);               // uint64: KV count
    }

    // ── GGUF String ─────────────────────────────────────────────────────────

    /// <summary>
    /// Writes a GGUF string: uint64 byte-length followed by raw UTF-8 bytes (no null terminator).
    /// </summary>
    public void WriteGgufString(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        _writer.Write((ulong)bytes.Length);
        _writer.Write(bytes);
    }

    // ── Metadata KV Writers ──────────────────────────────────────────────────

    /// <summary>Writes a metadata KV pair with a string value.</summary>
    public void WriteMetadataString(string key, string value)
    {
        WriteGgufString(key);
        _writer.Write((uint)GgufValueType.String);
        WriteGgufString(value);
    }

    /// <summary>Writes a metadata KV pair with a uint32 value.</summary>
    public void WriteMetadataUint32(string key, uint value)
    {
        WriteGgufString(key);
        _writer.Write((uint)GgufValueType.Uint32);
        _writer.Write(value);
    }

    /// <summary>Writes a metadata KV pair with a float32 value.</summary>
    public void WriteMetadataFloat32(string key, float value)
    {
        WriteGgufString(key);
        _writer.Write((uint)GgufValueType.Float32);
        _writer.Write(value);
    }

    /// <summary>Writes a metadata KV pair containing an array of strings.</summary>
    public void WriteMetadataStringArray(string key, string[] values)
    {
        WriteGgufString(key);
        _writer.Write((uint)GgufValueType.Array);
        _writer.Write((uint)GgufValueType.String);  // element type
        _writer.Write((ulong)values.Length);
        foreach (var v in values) WriteGgufString(v);
    }

    /// <summary>Writes a metadata KV pair containing an array of int32 values.</summary>
    public void WriteMetadataInt32Array(string key, int[] values)
    {
        WriteGgufString(key);
        _writer.Write((uint)GgufValueType.Array);
        _writer.Write((uint)GgufValueType.Int32);   // element type
        _writer.Write((ulong)values.Length);
        foreach (var v in values) _writer.Write(v);
    }

    // ── Tensor Info ──────────────────────────────────────────────────────────

    /// <summary>
    /// Writes one tensor info entry: name, dimension count, shape dimensions, data type, data offset.
    /// The <paramref name="dataOffset"/> is relative to the start of the tensor data section
    /// (i.e., the byte immediately after the alignment padding).
    /// </summary>
    public void WriteTensorInfo(string name, int[] shape, GgufDataType type, ulong dataOffset)
    {
        WriteGgufString(name);
        _writer.Write((uint)shape.Length);           // n_dims
        foreach (var dim in shape)
            _writer.Write((ulong)dim);               // dims as uint64
        _writer.Write((uint)type);                   // data type
        _writer.Write(dataOffset);                   // offset into data section
    }

    // ── Tensor Data ──────────────────────────────────────────────────────────

    /// <summary>
    /// Writes raw float32 values in little-endian order.
    /// </summary>
    public void WriteTensorData(float[] data)
    {
        foreach (var f in data) _writer.Write(f);
    }

    // ── Padding ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Writes zero bytes until the stream position is a multiple of <paramref name="alignment"/>.
    /// If the position is already aligned, writes nothing.
    /// </summary>
    public void Pad(int alignment)
    {
        long pos = _writer.BaseStream.Position;
        long remainder = pos % alignment;
        if (remainder == 0) return;
        long padding = alignment - remainder;
        _writer.Write(new byte[padding]);
    }

    // ── Raw Bytes (for testing) ───────────────────────────────────────────────

    /// <summary>Writes raw bytes directly. Used in tests to set up alignment scenarios.</summary>
    public void WriteRawBytes(byte[] bytes) => _writer.Write(bytes);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _writer.Dispose();
        _disposed = true;
    }
}

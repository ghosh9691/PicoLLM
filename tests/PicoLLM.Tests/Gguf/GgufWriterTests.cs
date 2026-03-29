using FluentAssertions;
using PicoLLM.Gguf;

namespace PicoLLM.Tests.Gguf;

public class GgufWriterTests
{
    private static (GgufWriter Writer, MemoryStream Stream) CreateWriter()
    {
        var ms = new MemoryStream();
        var writer = new GgufWriter(ms);
        return (writer, ms);
    }

    [Fact]
    public void WriteHeader_WritesCorrectBytes()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteHeader(tensorCount: 5, kvCount: 3);
        writer.Dispose();

        var bytes = ms.ToArray();

        // Magic: "GGUF"
        bytes[0..4].Should().Equal(0x47, 0x47, 0x55, 0x46);
        // Version: 3 (uint32 LE)
        bytes[4..8].Should().Equal(0x03, 0x00, 0x00, 0x00);
        // Tensor count: 5 (uint64 LE)
        bytes[8..16].Should().Equal(0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        // KV count: 3 (uint64 LE)
        bytes[16..24].Should().Equal(0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        // Total header size: 24 bytes
        bytes.Length.Should().Be(24);
    }

    [Fact]
    public void WriteGgufString_CorrectLengthPrefixAndUtf8()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteGgufString("llama");
        writer.Dispose();

        var bytes = ms.ToArray();
        // uint64 length = 5
        bytes[0..8].Should().Equal(0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        // UTF-8 bytes: l=0x6C, l=0x6C, a=0x61, m=0x6D, a=0x61
        bytes[8..13].Should().Equal(0x6C, 0x6C, 0x61, 0x6D, 0x61);
        bytes.Length.Should().Be(13);
    }

    [Fact]
    public void WriteMetadataUint32_WritesKeyValueTypeAndValue()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteMetadataUint32("test.key", 42u);
        writer.Dispose();

        var bytes = ms.ToArray();
        using var reader = new BinaryReader(new MemoryStream(bytes));

        // Key: uint64 len + utf8
        ulong keyLen = reader.ReadUInt64();
        keyLen.Should().Be(8); // "test.key" = 8 chars
        var keyBytes = reader.ReadBytes((int)keyLen);
        System.Text.Encoding.UTF8.GetString(keyBytes).Should().Be("test.key");

        // Value type: UINT32 = 4
        uint valueType = reader.ReadUInt32();
        valueType.Should().Be((uint)GgufValueType.Uint32);

        // Value: 42
        uint value = reader.ReadUInt32();
        value.Should().Be(42u);
    }

    [Fact]
    public void WriteMetadataFloat32_WritesKeyValueTypeAndValue()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteMetadataFloat32("eps", 1e-5f);
        writer.Dispose();

        var bytes = ms.ToArray();
        using var reader = new BinaryReader(new MemoryStream(bytes));

        ulong keyLen = reader.ReadUInt64();
        reader.ReadBytes((int)keyLen); // skip key
        uint valueType = reader.ReadUInt32();
        valueType.Should().Be((uint)GgufValueType.Float32);
        float value = reader.ReadSingle();
        value.Should().BeApproximately(1e-5f, 1e-10f);
    }

    [Fact]
    public void WriteMetadataString_WritesKeyValueTypeAndValue()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteMetadataString("general.architecture", "llama");
        writer.Dispose();

        var bytes = ms.ToArray();
        using var reader = new BinaryReader(new MemoryStream(bytes));

        ulong keyLen = reader.ReadUInt64();
        reader.ReadBytes((int)keyLen); // skip key
        uint valueType = reader.ReadUInt32();
        valueType.Should().Be((uint)GgufValueType.String);
        ulong strLen = reader.ReadUInt64();
        var strBytes = reader.ReadBytes((int)strLen);
        System.Text.Encoding.UTF8.GetString(strBytes).Should().Be("llama");
    }

    [Fact]
    public void Pad_AlignsToBoundary()
    {
        var (writer, ms) = CreateWriter();
        // Write 3 bytes, then pad to 32-byte boundary → 29 zero bytes
        writer.WriteRawBytes(new byte[] { 0x01, 0x02, 0x03 });
        writer.Pad(32);
        writer.Dispose();

        var bytes = ms.ToArray();
        bytes.Length.Should().Be(32);
        bytes[3..32].Should().AllBeEquivalentTo((byte)0);
    }

    [Fact]
    public void Pad_AlreadyAligned_WritesNothing()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteRawBytes(new byte[32]);
        writer.Pad(32);
        writer.Dispose();

        ms.Length.Should().Be(32);
    }

    [Fact]
    public void WriteTensorData_WritesFloatsLittleEndian()
    {
        var (writer, ms) = CreateWriter();
        writer.WriteTensorData(new float[] { 1.0f, 2.0f });
        writer.Dispose();

        var bytes = ms.ToArray();
        bytes.Length.Should().Be(8); // 2 × 4 bytes
        BitConverter.ToSingle(bytes, 0).Should().BeApproximately(1.0f, 1e-6f);
        BitConverter.ToSingle(bytes, 4).Should().BeApproximately(2.0f, 1e-6f);
    }
}

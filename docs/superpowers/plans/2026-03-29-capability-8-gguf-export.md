# Capability 8 — GGUF Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a trained PicoLLM model to GGUF v3 binary format so it can be loaded by Ollama (`ollama create`) or LM Studio.

**Architecture:** New `PicoLLM.Gguf` project with no NuGet dependencies. `GgufWriter` handles all low-level binary serialisation. `TensorNameMapper` translates PicoLLM internal tensor references to llama.cpp conventions. `GgufExporter` orchestrates the two-pass write (first compute offsets, then write). `GgufValidator` reads back and verifies the header. Before the Gguf project can access model internals, several private fields in Core must be promoted to public properties.

**Tech Stack:** .NET 9 / C# 12, `System.IO.BinaryWriter`, xUnit, FluentAssertions.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/PicoLLM.Core/Layers/DecoderBlock.cs` | Add `AttnNorm` and `FfnNorm` public properties |
| Modify | `src/PicoLLM.Core/Layers/FeedForward.cs` | Add `Up` and `Down` public properties |
| Modify | `src/PicoLLM.Core/Model/PicoLLMModel.cs` | Add `Embedding`, `Blocks`, `FinalNorm`, `LmHead` public properties |
| Modify | `src/PicoLLM.Tokenizer/BpeTokenizer.cs` | Add `GetVocabRepresentations()` public method |
| Create | `src/PicoLLM.Gguf/PicoLLM.Gguf.csproj` | New project file |
| Create | `src/PicoLLM.Gguf/GgufConstants.cs` | Magic bytes, version, alignment, enums, metadata key constants |
| Create | `src/PicoLLM.Gguf/GgufWriter.cs` | Low-level `BinaryWriter` wrapper for all GGUF primitives |
| Create | `src/PicoLLM.Gguf/TensorNameMapper.cs` | Internal tensor path → llama.cpp name translation |
| Create | `src/PicoLLM.Gguf/GgufExporter.cs` | High-level export orchestrator |
| Create | `src/PicoLLM.Gguf/GgufValidator.cs` | Read-back header/tensor-count verifier |
| Modify | `tests/PicoLLM.Tests/PicoLLM.Tests.csproj` | Add `<ProjectReference>` to PicoLLM.Gguf |
| Create | `tests/PicoLLM.Tests/Gguf/GgufWriterTests.cs` | Binary output unit tests |
| Create | `tests/PicoLLM.Tests/Gguf/TensorNameMapperTests.cs` | Name mapping unit tests |
| Create | `tests/PicoLLM.Tests/Gguf/GgufExporterTests.cs` | Integration tests: export → validate |

---

## Task 1: Project Scaffold

**Files:**
- Create: `src/PicoLLM.Gguf/PicoLLM.Gguf.csproj`
- Modify: `tests/PicoLLM.Tests/PicoLLM.Tests.csproj`

- [ ] **Step 1: Create the Gguf project file**

```xml
<!-- src/PicoLLM.Gguf/PicoLLM.Gguf.csproj -->
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\PicoLLM.Core\PicoLLM.Core.csproj" />
    <ProjectReference Include="..\PicoLLM.Tokenizer\PicoLLM.Tokenizer.csproj" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Add the Gguf project reference to the test project**

Open `tests/PicoLLM.Tests/PicoLLM.Tests.csproj` and add to the existing `<ItemGroup>` that contains `<ProjectReference>` entries:

```xml
<ProjectReference Include="..\..\src\PicoLLM.Gguf\PicoLLM.Gguf.csproj" />
```

The full updated `<ItemGroup>` block should be:

```xml
<ItemGroup>
  <ProjectReference Include="..\..\src\PicoLLM.Core\PicoLLM.Core.csproj" />
  <ProjectReference Include="..\..\src\PicoLLM.Tokenizer\PicoLLM.Tokenizer.csproj" />
  <ProjectReference Include="..\..\src\PicoLLM.Training\PicoLLM.Training.csproj" />
  <ProjectReference Include="..\..\src\PicoLLM.Browser\PicoLLM.Browser.csproj" />
  <ProjectReference Include="..\..\src\PicoLLM.Gguf\PicoLLM.Gguf.csproj" />
</ItemGroup>
```

- [ ] **Step 3: Verify the project builds**

```
dotnet build src/PicoLLM.Gguf/PicoLLM.Gguf.csproj
```
Expected: Build succeeded (empty project).

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.Gguf/PicoLLM.Gguf.csproj tests/PicoLLM.Tests/PicoLLM.Tests.csproj
git commit -m "feat(cap8): scaffold PicoLLM.Gguf project"
```

---

## Task 2: Expose Model Internals for Tensor Extraction

The GGUF exporter needs to walk every tensor in the model. Several internal fields must become public properties. These are additive changes — no existing behaviour changes.

**Files:**
- Modify: `src/PicoLLM.Core/Layers/DecoderBlock.cs`
- Modify: `src/PicoLLM.Core/Layers/FeedForward.cs`
- Modify: `src/PicoLLM.Core/Model/PicoLLMModel.cs`
- Modify: `src/PicoLLM.Tokenizer/BpeTokenizer.cs`

- [ ] **Step 1: Add `AttnNorm` and `FfnNorm` to `DecoderBlock`**

In `src/PicoLLM.Core/Layers/DecoderBlock.cs`, add two public properties after the existing `FFN` property declaration (around line 31):

```csharp
/// <summary>The pre-attention layer normalisation sublayer.</summary>
public LayerNorm AttnNorm => _attnNorm;

/// <summary>The pre-feedforward layer normalisation sublayer.</summary>
public LayerNorm FfnNorm => _ffnNorm;
```

- [ ] **Step 2: Add `Up` and `Down` to `FeedForward`**

In `src/PicoLLM.Core/Layers/FeedForward.cs`, add two public properties after the existing `FfDim` property (around line 21):

```csharp
/// <summary>The up-projection linear layer [embed_dim → ff_dim].</summary>
public LinearLayer Up => _up;

/// <summary>The down-projection linear layer [ff_dim → embed_dim].</summary>
public LinearLayer Down => _down;
```

- [ ] **Step 3: Add tensor-access properties to `PicoLLMModel`**

In `src/PicoLLM.Core/Model/PicoLLMModel.cs`, add four public properties after the existing `Config` property (around line 32):

```csharp
/// <summary>The combined token + positional embedding layer.</summary>
public EmbeddingLayer Embedding => _embedding;

/// <summary>The decoder blocks, one per transformer layer.</summary>
public IReadOnlyList<DecoderBlock> Blocks => _blocks;

/// <summary>The final layer normalisation applied before the LM head.</summary>
public LayerNorm FinalNorm => _finalNorm;

/// <summary>The language model head linear projection [embed_dim → vocab_size].</summary>
public LinearLayer LmHead => _lmHead;
```

- [ ] **Step 4: Add `GetVocabRepresentations()` to `BpeTokenizer`**

In `src/PicoLLM.Tokenizer/BpeTokenizer.cs`, add this public method after the `Decode` method (around line 148):

```csharp
/// <summary>
/// Returns string representations and token types for all tokens in the vocabulary,
/// ordered by token ID from 0 to VocabSize-1. Used by the GGUF exporter.
/// </summary>
/// <returns>
/// A tuple of:
/// - <c>Tokens</c>: string representation of each token (special name, hex byte, or decoded text).
/// - <c>TokenTypes</c>: GGUF token type for each token (3=special, 6=byte, 1=normal).
/// </returns>
public (string[] Tokens, int[] TokenTypes) GetVocabRepresentations()
{
    var tokens = new string[_vocabSize];
    var types  = new int[_vocabSize];

    // Special tokens: IDs 0-3
    tokens[PadId] = "<|pad|>"; types[PadId] = 3;
    tokens[UnkId] = "<|unk|>"; types[UnkId] = 3;
    tokens[BosId] = "<|bos|>"; types[BosId] = 3;
    tokens[EosId] = "<|eos|>"; types[EosId] = 3;

    // Byte tokens: IDs 4-259 (ByteOffset + 0..255)
    for (int b = 0; b < 256; b++)
    {
        int id = b + ByteOffset;
        tokens[id] = $"<0x{b:X2}>";
        types[id]  = 6;
    }

    // Merged tokens: IDs 260+ — decode byte sequence as UTF-8 (best effort)
    for (int id = ByteOffset + 256; id < _vocabSize; id++)
    {
        if (_vocab.TryGetValue(id, out var bytes) && bytes.Length > 0)
        {
            try   { tokens[id] = System.Text.Encoding.UTF8.GetString(bytes); }
            catch { tokens[id] = $"<token_{id}>"; }
        }
        else
        {
            tokens[id] = $"<token_{id}>";
        }
        types[id] = 1;
    }

    return (tokens, types);
}
```

- [ ] **Step 5: Build everything to verify no errors**

```
dotnet build src/PicoLLM.Core/PicoLLM.Core.csproj && dotnet build src/PicoLLM.Tokenizer/PicoLLM.Tokenizer.csproj && dotnet build src/PicoLLM.Gguf/PicoLLM.Gguf.csproj
```
Expected: All three projects build successfully, 0 errors.

- [ ] **Step 6: Run existing tests to verify no regressions**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```
Expected: All existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/PicoLLM.Core/Layers/DecoderBlock.cs src/PicoLLM.Core/Layers/FeedForward.cs src/PicoLLM.Core/Model/PicoLLMModel.cs src/PicoLLM.Tokenizer/BpeTokenizer.cs
git commit -m "feat(cap8): expose model internals for GGUF tensor extraction"
```

---

## Task 3: Constants and Enums

**Files:**
- Create: `src/PicoLLM.Gguf/GgufConstants.cs`

- [ ] **Step 1: Create the file**

```csharp
// src/PicoLLM.Gguf/GgufConstants.cs
namespace PicoLLM.Gguf;

/// <summary>GGUF metadata value types (encodes what follows the key in a KV pair).</summary>
public enum GgufValueType : uint
{
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

/// <summary>GGUF tensor data types (stored in tensor info entries).</summary>
public enum GgufDataType : uint
{
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
}

/// <summary>
/// Fixed constants and metadata key strings for the GGUF v3 format.
/// PicoLLM exports using the "llama" architecture identifier for Ollama compatibility.
/// </summary>
/// <remarks>
/// <b>Known tradeoff:</b> PicoLLM uses standard LayerNorm (mean + variance normalization),
/// while the llama architecture in llama.cpp assumes RMSNorm. The exported file will be
/// structurally valid GGUF but may produce slightly different inference results than a
/// natively-trained LLaMA model. To achieve full inference compatibility, migrate the
/// model to use RMSNorm.
/// </remarks>
public static class GgufConstants
{
    /// <summary>4-byte file magic: ASCII "GGUF".</summary>
    public static readonly byte[] Magic = "GGUF"u8.ToArray();

    /// <summary>GGUF format version we produce.</summary>
    public const uint Version = 3;

    /// <summary>Tensor data section alignment boundary in bytes.</summary>
    public const int Alignment = 32;

    // ── General metadata keys ────────────────────────────────────────────────
    public const string KeyArchitecture      = "general.architecture";
    public const string KeyName              = "general.name";
    public const string KeyFileType          = "general.file_type";

    // ── Llama architecture metadata keys ────────────────────────────────────
    public const string KeyContextLength     = "llama.context_length";
    public const string KeyEmbeddingLength   = "llama.embedding_length";
    public const string KeyBlockCount        = "llama.block_count";
    public const string KeyFeedForwardLength = "llama.feed_forward_length";
    public const string KeyHeadCount         = "llama.attention.head_count";
    public const string KeyHeadCountKv       = "llama.attention.head_count_kv";
    public const string KeyRopeDimCount      = "llama.rope.dimension_count";
    public const string KeyLayerNormEps      = "llama.attention.layer_norm_rms_epsilon";

    // ── Tokenizer metadata keys ──────────────────────────────────────────────
    public const string KeyTokenizerModel    = "tokenizer.ggml.model";
    public const string KeyTokenizerTokens   = "tokenizer.ggml.tokens";
    public const string KeyTokenizerTypes    = "tokenizer.ggml.token_type";
    public const string KeyBosTokenId        = "tokenizer.ggml.bos_token_id";
    public const string KeyEosTokenId        = "tokenizer.ggml.eos_token_id";
    public const string KeyPadTokenId        = "tokenizer.ggml.padding_token_id";

    /// <summary>Total number of metadata KV pairs written by GgufExporter.</summary>
    public const int MetadataKvCount = 17;
}
```

- [ ] **Step 2: Build to verify no errors**

```
dotnet build src/PicoLLM.Gguf/PicoLLM.Gguf.csproj
```
Expected: Build succeeded.

- [ ] **Step 3: Commit**

```bash
git add src/PicoLLM.Gguf/GgufConstants.cs
git commit -m "feat(cap8): add GgufConstants — enums, magic, metadata key strings"
```

---

## Task 4: Low-Level Binary Writer

**Files:**
- Create: `src/PicoLLM.Gguf/GgufWriter.cs`
- Create: `tests/PicoLLM.Tests/Gguf/GgufWriterTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Gguf/GgufWriterTests.cs
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
```

- [ ] **Step 2: Run tests — expect compile failure**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~GgufWriterTests"
```
Expected: Build error — `GgufWriter` does not exist.

- [ ] **Step 3: Implement GgufWriter**

```csharp
// src/PicoLLM.Gguf/GgufWriter.cs
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
    internal void WriteRawBytes(byte[] bytes) => _writer.Write(bytes);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _writer.Dispose();
        _disposed = true;
    }
}
```

- [ ] **Step 4: Run tests — expect all pass**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~GgufWriterTests"
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Gguf/GgufWriter.cs tests/PicoLLM.Tests/Gguf/GgufWriterTests.cs
git commit -m "feat(cap8): add GgufWriter with header/metadata/tensor binary primitives"
```

---

## Task 5: Tensor Name Mapper

**Files:**
- Create: `src/PicoLLM.Gguf/TensorNameMapper.cs`
- Create: `tests/PicoLLM.Tests/Gguf/TensorNameMapperTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Gguf/TensorNameMapperTests.cs
using FluentAssertions;
using PicoLLM.Gguf;

namespace PicoLLM.Tests.Gguf;

public class TensorNameMapperTests
{
    [Fact]
    public void TokenEmbeddingWeights_MapsCorrectly()
    {
        TensorNameMapper.ToGgufName("Embedding.TokenEmbedding.Weights")
            .Should().Be("token_embd.weight");
    }

    [Theory]
    [InlineData(0, "Attention.QueryProj.Weights",  "blk.0.attn_q.weight")]
    [InlineData(0, "Attention.KeyProj.Weights",    "blk.0.attn_k.weight")]
    [InlineData(0, "Attention.ValueProj.Weights",  "blk.0.attn_v.weight")]
    [InlineData(0, "Attention.OutputProj.Weights", "blk.0.attn_output.weight")]
    [InlineData(0, "AttnNorm.Gamma",               "blk.0.attn_norm.weight")]
    [InlineData(0, "FFN.Up.Weights",               "blk.0.ffn_up.weight")]
    [InlineData(0, "FFN.Down.Weights",             "blk.0.ffn_down.weight")]
    [InlineData(0, "FfnNorm.Gamma",                "blk.0.ffn_norm.weight")]
    [InlineData(3, "Attention.QueryProj.Weights",  "blk.3.attn_q.weight")]
    [InlineData(3, "FFN.Up.Weights",               "blk.3.ffn_up.weight")]
    public void BlockTensors_MapsWithLayerIndex(int layerIndex, string suffix, string expected)
    {
        TensorNameMapper.ToGgufName($"Block.{layerIndex}.{suffix}")
            .Should().Be(expected);
    }

    [Fact]
    public void FinalNormGamma_MapsCorrectly()
    {
        TensorNameMapper.ToGgufName("FinalNorm.Gamma")
            .Should().Be("output_norm.weight");
    }

    [Fact]
    public void LmHeadWeights_MapsCorrectly()
    {
        TensorNameMapper.ToGgufName("LmHead.Weights")
            .Should().Be("output.weight");
    }

    [Fact]
    public void UnknownName_ThrowsArgumentException()
    {
        var act = () => TensorNameMapper.ToGgufName("Unknown.Tensor");
        act.Should().Throw<ArgumentException>();
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~TensorNameMapperTests"
```
Expected: Build error — `TensorNameMapper` does not exist.

- [ ] **Step 3: Implement TensorNameMapper**

```csharp
// src/PicoLLM.Gguf/TensorNameMapper.cs
using System.Text.RegularExpressions;

namespace PicoLLM.Gguf;

/// <summary>
/// Translates PicoLLM internal tensor path identifiers to llama.cpp naming conventions
/// used by GGUF. Layer-indexed paths (Block.{i}.*) substitute the integer index at runtime.
/// </summary>
public static class TensorNameMapper
{
    // Maps suffix after "Block.{i}." to llama.cpp suffix after "blk.{i}."
    private static readonly Dictionary<string, string> BlockSuffixMap = new()
    {
        ["AttnNorm.Gamma"]               = "attn_norm.weight",
        ["Attention.QueryProj.Weights"]  = "attn_q.weight",
        ["Attention.KeyProj.Weights"]    = "attn_k.weight",
        ["Attention.ValueProj.Weights"]  = "attn_v.weight",
        ["Attention.OutputProj.Weights"] = "attn_output.weight",
        ["FfnNorm.Gamma"]                = "ffn_norm.weight",
        ["FFN.Up.Weights"]               = "ffn_up.weight",
        ["FFN.Down.Weights"]             = "ffn_down.weight",
    };

    private static readonly Dictionary<string, string> TopLevelMap = new()
    {
        ["Embedding.TokenEmbedding.Weights"] = "token_embd.weight",
        ["FinalNorm.Gamma"]                  = "output_norm.weight",
        ["LmHead.Weights"]                   = "output.weight",
    };

    private static readonly Regex BlockPattern =
        new(@"^Block\.(\d+)\.(.+)$", RegexOptions.Compiled);

    /// <summary>
    /// Converts a PicoLLM internal tensor path to its llama.cpp GGUF name.
    /// </summary>
    /// <param name="internalPath">
    ///   Either a top-level path (e.g. "Embedding.TokenEmbedding.Weights")
    ///   or a block-indexed path (e.g. "Block.2.Attention.QueryProj.Weights").
    /// </param>
    /// <exception cref="ArgumentException">Thrown if the path has no known mapping.</exception>
    public static string ToGgufName(string internalPath)
    {
        ArgumentNullException.ThrowIfNull(internalPath);

        if (TopLevelMap.TryGetValue(internalPath, out var topLevel))
            return topLevel;

        var match = BlockPattern.Match(internalPath);
        if (match.Success)
        {
            int layerIndex = int.Parse(match.Groups[1].Value);
            string suffix  = match.Groups[2].Value;
            if (BlockSuffixMap.TryGetValue(suffix, out var ggufSuffix))
                return $"blk.{layerIndex}.{ggufSuffix}";
        }

        throw new ArgumentException(
            $"No GGUF name mapping found for internal path: '{internalPath}'",
            nameof(internalPath));
    }
}
```

- [ ] **Step 4: Run tests — expect all pass**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~TensorNameMapperTests"
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Gguf/TensorNameMapper.cs tests/PicoLLM.Tests/Gguf/TensorNameMapperTests.cs
git commit -m "feat(cap8): add TensorNameMapper — internal paths to llama.cpp names"
```

---

## Task 6: GGUF Exporter and Validator

**Files:**
- Create: `src/PicoLLM.Gguf/GgufExporter.cs`
- Create: `src/PicoLLM.Gguf/GgufValidator.cs`
- Create: `tests/PicoLLM.Tests/Gguf/GgufExporterTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Gguf/GgufExporterTests.cs
using FluentAssertions;
using PicoLLM.Core.Model;
using PicoLLM.Gguf;
using PicoLLM.Tokenizer;

namespace PicoLLM.Tests.Gguf;

public class GgufExporterTests : IDisposable
{
    private readonly string _tempFile;

    public GgufExporterTests()
    {
        _tempFile = Path.GetTempFileName();
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile)) File.Delete(_tempFile);
    }

    private static (PicoLLMModel Model, BpeTokenizer Tokenizer) CreateSmallModel()
    {
        // Minimal config for fast tests: 2 layers, 32-dim embed, 2 heads
        var config = new ModelConfig(
            VocabSize: 260 + 10,  // 4 special + 256 byte + 10 merged = 270
            EmbedDim: 32,
            NumHeads: 2,
            NumLayers: 2,
            FfMultiplier: 2,
            MaxSeqLen: 16);
        var model = new PicoLLMModel(config, seed: 42);

        // Build a tiny tokenizer with the matching vocab size
        var tokConfig = new TokenizerConfig
        {
            Version = 1,
            VocabSize = config.VocabSize,
            SpecialTokens = new Dictionary<string, int>
            {
                ["<|pad|>"] = 0, ["<|unk|>"] = 1, ["<|bos|>"] = 2, ["<|eos|>"] = 3
            },
            ByteTokens = Enumerable.Range(0, 256)
                .ToDictionary(b => (b + 4).ToString(), b => new List<int> { b }),
            Merges = Enumerable.Range(0, 10)
                .Select(i => new List<int> { 4 + i, 5 + i, 260 + i })
                .ToList()
        };
        var tokenizer = BpeTokenizer.FromConfig(tokConfig);
        return (model, tokenizer);
    }

    [Fact]
    public void Export_WritesValidGgufMagic()
    {
        var (model, tokenizer) = CreateSmallModel();
        GgufExporter.Export(model, tokenizer, _tempFile);

        var bytes = File.ReadAllBytes(_tempFile);
        bytes[0..4].Should().Equal(0x47, 0x47, 0x55, 0x46); // "GGUF"
    }

    [Fact]
    public void Export_WritesVersion3()
    {
        var (model, tokenizer) = CreateSmallModel();
        GgufExporter.Export(model, tokenizer, _tempFile);

        var bytes = File.ReadAllBytes(_tempFile);
        uint version = BitConverter.ToUInt32(bytes, 4);
        version.Should().Be(3u);
    }

    [Fact]
    public void Export_TensorCountMatchesExpected()
    {
        var (model, tokenizer) = CreateSmallModel();
        GgufExporter.Export(model, tokenizer, _tempFile);

        var bytes = File.ReadAllBytes(_tempFile);
        ulong tensorCount = BitConverter.ToUInt64(bytes, 8);
        // 2 layers × 8 tensors + 3 (token_embd + output_norm + output) = 19
        tensorCount.Should().Be(19UL);
    }

    [Fact]
    public void Export_FileIsNotEmpty()
    {
        var (model, tokenizer) = CreateSmallModel();
        GgufExporter.Export(model, tokenizer, _tempFile);
        new FileInfo(_tempFile).Length.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Validate_ExportedFile_Passes()
    {
        var (model, tokenizer) = CreateSmallModel();
        GgufExporter.Export(model, tokenizer, _tempFile);

        var result = GgufValidator.Validate(_tempFile);
        result.IsValid.Should().BeTrue(result.Error ?? "");
    }

    [Fact]
    public void Validate_CorruptedMagic_Fails()
    {
        File.WriteAllBytes(_tempFile, new byte[] { 0x00, 0x00, 0x00, 0x00 });
        var result = GgufValidator.Validate(_tempFile);
        result.IsValid.Should().BeFalse();
        result.Error.Should().Contain("magic");
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~GgufExporterTests"
```
Expected: Build error — `GgufExporter`, `GgufValidator` do not exist.

- [ ] **Step 3: Implement GgufValidator first (needed by tests)**

```csharp
// src/PicoLLM.Gguf/GgufValidator.cs
namespace PicoLLM.Gguf;

/// <summary>
/// Reads a GGUF file and verifies its header is structurally valid.
/// Useful for smoke-testing exports without loading the full model.
/// </summary>
public static class GgufValidator
{
    /// <summary>Result of a GGUF file validation check.</summary>
    /// <param name="IsValid">True if the file passed all checks.</param>
    /// <param name="TensorCount">Number of tensors declared in the header; 0 on failure.</param>
    /// <param name="KvCount">Number of KV pairs declared in the header; 0 on failure.</param>
    /// <param name="Error">Human-readable error description; null on success.</param>
    public record ValidationResult(bool IsValid, ulong TensorCount, ulong KvCount, string? Error);

    /// <summary>
    /// Validates the GGUF file at <paramref name="path"/>.
    /// Checks: magic bytes, version number, header completeness.
    /// </summary>
    public static ValidationResult Validate(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            using var reader = new BinaryReader(fs);

            if (fs.Length < 24)
                return new ValidationResult(false, 0, 0, "File too small to contain a GGUF header");

            // Magic
            var magic = reader.ReadBytes(4);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
                return new ValidationResult(false, 0, 0,
                    $"Invalid magic bytes: expected GGUF, got {System.Text.Encoding.ASCII.GetString(magic)}");

            // Version
            uint version = reader.ReadUInt32();
            if (version != 3)
                return new ValidationResult(false, 0, 0,
                    $"Unsupported GGUF version: {version} (expected 3)");

            // Counts
            ulong tensorCount = reader.ReadUInt64();
            ulong kvCount     = reader.ReadUInt64();

            return new ValidationResult(true, tensorCount, kvCount, null);
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, 0, 0, ex.Message);
        }
    }
}
```

- [ ] **Step 4: Implement GgufExporter**

```csharp
// src/PicoLLM.Gguf/GgufExporter.cs
using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;
using PicoLLM.Tokenizer;

namespace PicoLLM.Gguf;

/// <summary>
/// Exports a trained <see cref="PicoLLMModel"/> plus its <see cref="BpeTokenizer"/>
/// to a GGUF v3 binary file.
/// </summary>
/// <remarks>
/// Uses a two-pass strategy:
/// <list type="number">
///   <item>Collect all named tensors and pre-compute data offsets.</item>
///   <item>Write header → metadata KVs → tensor info array → padding → tensor data.</item>
/// </list>
/// </remarks>
public static class GgufExporter
{
    /// <summary>
    /// Writes the model and tokenizer to <paramref name="outputPath"/> in GGUF v3 format.
    /// </summary>
    /// <param name="model">The trained PicoLLM model.</param>
    /// <param name="tokenizer">The matching BPE tokenizer.</param>
    /// <param name="outputPath">Destination file path (created or overwritten).</param>
    public static void Export(PicoLLMModel model, BpeTokenizer tokenizer, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(outputPath);

        var namedTensors = CollectNamedTensors(model);
        var offsets = ComputeOffsets(namedTensors);

        using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(namedTensors.Count, GgufConstants.MetadataKvCount);
        WriteMetadata(writer, model.Config, tokenizer);
        WriteTensorInfoArray(writer, namedTensors, offsets);
        writer.Pad(GgufConstants.Alignment);
        WriteTensorDataSection(writer, namedTensors);
    }

    // ── Tensor Collection ────────────────────────────────────────────────────

    private static List<(string GgufName, Tensor Data)> CollectNamedTensors(PicoLLMModel model)
    {
        var result = new List<(string, Tensor)>();

        // Token embedding
        result.Add(("token_embd.weight", model.Embedding.TokenEmbedding.Weights));

        // Decoder blocks
        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i];
            result.Add(($"blk.{i}.attn_norm.weight",   block.AttnNorm.Gamma));
            result.Add(($"blk.{i}.attn_q.weight",      block.Attention.QueryProj.Weights));
            result.Add(($"blk.{i}.attn_k.weight",      block.Attention.KeyProj.Weights));
            result.Add(($"blk.{i}.attn_v.weight",      block.Attention.ValueProj.Weights));
            result.Add(($"blk.{i}.attn_output.weight", block.Attention.OutputProj.Weights));
            result.Add(($"blk.{i}.ffn_norm.weight",    block.FfnNorm.Gamma));
            result.Add(($"blk.{i}.ffn_up.weight",      block.FFN.Up.Weights));
            result.Add(($"blk.{i}.ffn_down.weight",    block.FFN.Down.Weights));
        }

        // Final norm + LM head
        result.Add(("output_norm.weight", model.FinalNorm.Gamma));
        result.Add(("output.weight",      model.LmHead.Weights));

        return result;
    }

    // ── Offset Computation ────────────────────────────────────────────────────

    private static ulong[] ComputeOffsets(List<(string Name, Tensor Data)> tensors)
    {
        var offsets = new ulong[tensors.Count];
        ulong cursor = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = cursor;
            ulong byteSize = (ulong)(tensors[i].Data.Length * sizeof(float));
            cursor += byteSize;
            // Align to 32 bytes
            ulong remainder = cursor % GgufConstants.Alignment;
            if (remainder != 0) cursor += (ulong)GgufConstants.Alignment - remainder;
        }
        return offsets;
    }

    // ── Metadata Writing ─────────────────────────────────────────────────────

    private static void WriteMetadata(GgufWriter writer, ModelConfig config, BpeTokenizer tokenizer)
    {
        int ffDim = config.EmbedDim * config.FfMultiplier;
        int ropeDim = config.EmbedDim / config.NumHeads;

        writer.WriteMetadataString(GgufConstants.KeyArchitecture, "llama");
        writer.WriteMetadataString(GgufConstants.KeyName,         "PicoLLM");
        writer.WriteMetadataUint32(GgufConstants.KeyFileType,     0u);  // 0 = F32

        writer.WriteMetadataUint32(GgufConstants.KeyContextLength,     (uint)config.MaxSeqLen);
        writer.WriteMetadataUint32(GgufConstants.KeyEmbeddingLength,   (uint)config.EmbedDim);
        writer.WriteMetadataUint32(GgufConstants.KeyBlockCount,        (uint)config.NumLayers);
        writer.WriteMetadataUint32(GgufConstants.KeyFeedForwardLength, (uint)ffDim);
        writer.WriteMetadataUint32(GgufConstants.KeyHeadCount,         (uint)config.NumHeads);
        writer.WriteMetadataUint32(GgufConstants.KeyHeadCountKv,       (uint)config.NumHeads);
        writer.WriteMetadataUint32(GgufConstants.KeyRopeDimCount,      (uint)ropeDim);
        writer.WriteMetadataFloat32(GgufConstants.KeyLayerNormEps,     1e-5f);

        var (tokens, tokenTypes) = tokenizer.GetVocabRepresentations();
        writer.WriteMetadataString(GgufConstants.KeyTokenizerModel, "gpt2");
        writer.WriteMetadataStringArray(GgufConstants.KeyTokenizerTokens, tokens);
        writer.WriteMetadataInt32Array(GgufConstants.KeyTokenizerTypes,  tokenTypes);
        writer.WriteMetadataUint32(GgufConstants.KeyBosTokenId, (uint)BpeTokenizer.BosId);
        writer.WriteMetadataUint32(GgufConstants.KeyEosTokenId, (uint)BpeTokenizer.EosId);
        writer.WriteMetadataUint32(GgufConstants.KeyPadTokenId, (uint)BpeTokenizer.PadId);
    }

    // ── Tensor Info Array ────────────────────────────────────────────────────

    private static void WriteTensorInfoArray(
        GgufWriter writer,
        List<(string Name, Tensor Data)> tensors,
        ulong[] offsets)
    {
        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, tensor) = tensors[i];
            writer.WriteTensorInfo(name, tensor.GetShape(), GgufDataType.F32, offsets[i]);
        }
    }

    // ── Tensor Data Section ───────────────────────────────────────────────────

    private static void WriteTensorDataSection(
        GgufWriter writer,
        List<(string Name, Tensor Data)> tensors)
    {
        foreach (var (_, tensor) in tensors)
        {
            writer.WriteTensorData(tensor.Data.ToArray());
            writer.Pad(GgufConstants.Alignment);
        }
    }
}
```

- [ ] **Step 5: Run all Capability 8 tests**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~GgufWriterTests|FullyQualifiedName~TensorNameMapperTests|FullyQualifiedName~GgufExporterTests"
```
Expected: All tests pass.

- [ ] **Step 6: Run full test suite to check for regressions**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/PicoLLM.Gguf/GgufExporter.cs src/PicoLLM.Gguf/GgufValidator.cs tests/PicoLLM.Tests/Gguf/GgufExporterTests.cs
git commit -m "feat(cap8): add GgufExporter and GgufValidator — completes capability 8"
```

---

## Self-Review

**Spec coverage check:**
- [x] GGUF v3 header (magic, version, tensor count, kv count) → Task 4 (`WriteHeader_WritesCorrectBytes`), Task 6 (`Export_WritesValidGgufMagic`, `Export_WritesVersion3`)
- [x] Required metadata KV pairs → Task 6 (`WriteMetadata` covers all 17 keys including architecture, dimensions, tokenizer)
- [x] Tokenizer metadata (tokens array, token types, BOS/EOS/PAD IDs) → Task 2 (`GetVocabRepresentations`), Task 6 (`WriteMetadata`)
- [x] Tensor data in F32 format with llama.cpp names → Task 5 (TensorNameMapper), Task 6 (`CollectNamedTensors`)
- [x] Tensor alignment to 32-byte boundaries → Task 4 (`Pad_AlignsToBoundary`, `Pad_AlreadyAligned_WritesNothing`), Task 6 (`ComputeOffsets` + `WriteTensorDataSection`)
- [x] Little-endian byte order → Task 4 (`WriteTensorData_WritesFloatsLittleEndian`), confirmed: `BinaryWriter` is LE by default on all platforms
- [x] GGUF string encoding (uint64 length + UTF-8, no null terminator) → Task 4 (`WriteGgufString_CorrectLengthPrefixAndUtf8`)
- [x] Tensor count matches actual model tensor count → Task 6 (`Export_TensorCountMatchesExpected` — 2 layers × 8 + 3 = 19)
- [x] GgufValidator reads back and checks → Task 6 (`Validate_ExportedFile_Passes`, `Validate_CorruptedMagic_Fails`)

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `GgufWriter` methods match signatures used in `GgufExporter`. `GgufDataType.F32` used in both `WriteTensorInfo` and `GgufExporter`. `GgufConstants.MetadataKvCount = 17` matches the 17 KV pairs written in `WriteMetadata`. `GgufValidator.ValidationResult` record fields (`IsValid`, `Error`) match test assertions.

**One note for manual follow-up:** After full training, test Ollama import manually with `ollama create picollm -f Modelfile` where the Modelfile contains `FROM ./picollm.gguf`. This cannot be automated in the unit test suite.

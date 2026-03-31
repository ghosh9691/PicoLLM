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
        // 2 layers × 9 tensors (+ ffn_gate.weight) + 3 (token_embd + output_norm + output) = 21
        tensorCount.Should().Be(21UL);
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

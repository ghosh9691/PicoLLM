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
    [InlineData(0, "FFN.Gate.Weights",             "blk.0.ffn_gate.weight")]
    [InlineData(0, "FFN.Up.Weights",               "blk.0.ffn_up.weight")]
    [InlineData(0, "FFN.Down.Weights",             "blk.0.ffn_down.weight")]
    [InlineData(0, "FfnNorm.Gamma",                "blk.0.ffn_norm.weight")]
    [InlineData(3, "Attention.QueryProj.Weights",  "blk.3.attn_q.weight")]
    [InlineData(3, "FFN.Gate.Weights",             "blk.3.ffn_gate.weight")]
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

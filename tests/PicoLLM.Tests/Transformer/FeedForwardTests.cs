using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class FeedForwardTests
{
    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var ffn = new FeedForward(embedDim: 64, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 8, 64], seed: 1);
        var out_ = ffn.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 8, 64);
    }

    [Fact]
    public void FfDim_IsEmbedDimTimesMultiplier()
    {
        var ffn = new FeedForward(embedDim: 32, ffMultiplier: 4, seed: 1);
        ffn.FfDim.Should().Be(128);
    }

    [Fact]
    public void Forward_NoNaN_WithRandomInput()
    {
        var ffn = new FeedForward(embedDim: 128, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 128], seed: 42);
        var out_ = ffn.Forward(x);
        foreach (float v in out_.Data) float.IsNaN(v).Should().BeFalse();
    }

    [Fact]
    public void Forward_SingleToken_CorrectOutputShape()
    {
        var ffn = new FeedForward(embedDim: 16, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 1, 16], seed: 1);
        var out_ = ffn.Forward(x);
        out_.Shape.ToArray().Should().Equal(1, 1, 16);
    }

    [Fact]
    public void Parameters_ReturnsThreeWeightTensors()
    {
        // SwiGLU: _gate + _up + _down, each weights-only (no bias) = 3 total
        var ffn = new FeedForward(embedDim: 8, ffMultiplier: 4, seed: 1);
        ffn.Parameters().Should().HaveCount(3);
    }
}

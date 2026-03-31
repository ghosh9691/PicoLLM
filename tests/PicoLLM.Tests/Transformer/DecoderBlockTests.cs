using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class DecoderBlockTests
{
    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var block = new DecoderBlock(embedDim: 64, numHeads: 4, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 8, 64], seed: 1);
        var out_ = block.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 8, 64);
    }

    [Fact]
    public void Forward_NoNaN_WithRandomInput()
    {
        var block = new DecoderBlock(embedDim: 32, numHeads: 4, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 32], seed: 42);
        var out_ = block.Forward(x);
        foreach (float v in out_.Data) float.IsNaN(v).Should().BeFalse();
    }

    [Fact]
    public void Forward_4Layers_Stacked_CorrectOutputShape()
    {
        int B = 1, S = 6, E = 32;
        var x = TensorFactory.RandomNormal([B, S, E], seed: 1);
        for (int i = 0; i < 4; i++)
        {
            var block = new DecoderBlock(embedDim: E, numHeads: 4, ffMultiplier: 4, seed: i);
            x = block.Forward(x);
        }
        x.Shape.ToArray().Should().Equal(B, S, E);
    }

    [Fact]
    public void Parameters_CountIsCorrect()
    {
        // Per block:
        //   attn LayerNorm: 2 (gamma, beta)
        //   MHA: 4 projections × 2 (weights+bias) = 8
        //   ffn LayerNorm: 2
        //   FFN (SwiGLU): 3 unbiased weight-only linears = 3
        //   total = 15
        var block = new DecoderBlock(embedDim: 16, numHeads: 4, ffMultiplier: 4, seed: 1);
        block.Parameters().Should().HaveCount(15);
    }
}

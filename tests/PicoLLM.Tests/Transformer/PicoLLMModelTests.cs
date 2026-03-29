using FluentAssertions;
using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class PicoLLMModelTests
{
    private static ModelConfig SmallConfig() => new(
        VocabSize: 512, EmbedDim: 64, NumHeads: 4,
        NumLayers: 2, FfMultiplier: 4, MaxSeqLen: 32);

    [Fact]
    public void Forward_LogitsShape_IsCorrect()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } }; // [2, 5]
        var logits = model.Forward(tokenIds);
        logits.Shape.ToArray().Should().Equal(2, 5, 512);
    }

    [Fact]
    public void Forward_NoNaN_NoInf_InLogits()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 0, 1, 2 } };
        var logits = model.Forward(tokenIds);
        foreach (float v in logits.Data)
        {
            float.IsNaN(v).Should().BeFalse();
            float.IsInfinity(v).Should().BeFalse();
        }
    }

    [Fact]
    public void TotalParameters_IsPositive()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        model.TotalParameters().Should().BeGreaterThan(0);
    }

    [Fact]
    public void TotalParameters_KnownConfig_CorrectCount()
    {
        // vocab=512, embed=64, heads=4, layers=2, ff_mult=4
        //
        // Token embedding:       512 × 64              = 32 768
        //
        // Per decoder block:
        //   attn LayerNorm:       64 + 64              =    128  (gamma + beta)
        //   MHA 4 projections:    4 × (64×64 + 64)     = 16 640  (weights + bias each)
        //   ffn LayerNorm:        64 + 64              =    128
        //   FFN up:               64×256 + 256         = 16 640
        //   FFN down:             256×64 + 64          = 16 448
        //   Block total:          128 + 16 640 + 128 + 16 640 + 16 448 = 49 984
        //
        // 2 blocks:              2 × 49 984            = 99 968
        // Final LayerNorm:        64 + 64              =    128
        // LM head:                64×512 + 512         = 33 280
        //
        // Grand total: 32 768 + 99 968 + 128 + 33 280 = 166 144
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        model.TotalParameters().Should().Be(166_144);
    }

    [Fact]
    public void GetAllParameters_SumMatchesTotalParameters()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        int total = model.GetAllParameters().Sum(p => p.Data.Length);
        total.Should().Be(model.TotalParameters());
    }

    [Fact]
    public void Forward_Softmax_ProducesValidProbabilityDistribution()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 1, 2, 3 } };
        var logits = model.Forward(tokenIds); // [1, 3, 512]

        // Extract logits at the last position
        var lastLogits = new float[512];
        for (int v = 0; v < 512; v++) lastLogits[v] = logits[0, 2, v];

        float maxL = lastLogits.Max();
        float sumExp = lastLogits.Sum(l => MathF.Exp(l - maxL));
        float[] probs = lastLogits.Select(l => MathF.Exp(l - maxL) / sumExp).ToArray();

        probs.Sum().Should().BeApproximately(1f, 1e-4f);
        probs.All(p => p >= 0f).Should().BeTrue();
    }

    [Fact]
    public void Forward_SingleToken_Works()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 42 } };
        var logits = model.Forward(tokenIds);
        logits.Shape.ToArray().Should().Equal(1, 1, 512);
    }
}

using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class LayerNormTests
{
    private const float Tol = 1e-4f;

    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var ln = new LayerNorm(8);
        var x = TensorFactory.RandomNormal([2, 4, 8], seed: 1);
        var out_ = ln.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 4, 8);
    }

    [Fact]
    public void Forward_Normalized_MeanNearZero_VarNearOne()
    {
        var ln = new LayerNorm(64);
        var x = TensorFactory.RandomNormal([1, 1, 64], seed: 42);
        var out_ = ln.Forward(x);
        var data = out_.Data.ToArray();
        float mean = data.Average();
        float var_ = data.Select(v => (v - mean) * (v - mean)).Average();
        mean.Should().BeApproximately(0f, 1e-3f);
        var_.Should().BeApproximately(1f, 1e-3f);
    }

    [Fact]
    public void Forward_GammaEffect_ScalesOutput()
    {
        var x = TensorFactory.RandomNormal([1, 1, 4], seed: 7);

        var ln2 = new LayerNorm(4);
        ln2.Gamma.MutableData.Fill(2f);
        var out2 = ln2.Forward(x);

        var ln1 = new LayerNorm(4);
        var out1 = ln1.Forward(x);

        for (int i = 0; i < 4; i++)
            out2[0, 0, i].Should().BeApproximately(2f * out1[0, 0, i], Tol);
    }

    [Fact]
    public void Forward_BetaEffect_ShiftsOutput()
    {
        var x = TensorFactory.RandomNormal([1, 1, 4], seed: 3);

        var ln5 = new LayerNorm(4);
        ln5.Beta.MutableData.Fill(5f);
        var out5 = ln5.Forward(x);

        var ln0 = new LayerNorm(4);
        var out0 = ln0.Forward(x);

        for (int i = 0; i < 4; i++)
            out5[0, 0, i].Should().BeApproximately(out0[0, 0, i] + 5f, Tol);
    }

    [Fact]
    public void Forward_2DInput_Works()
    {
        var ln = new LayerNorm(8);
        var x = TensorFactory.RandomNormal([4, 8], seed: 1);
        var out_ = ln.Forward(x);
        out_.Shape.ToArray().Should().Equal(4, 8);
    }

    [Fact]
    public void Parameters_ReturnsTwoTensors_GammaAndBeta()
    {
        var ln = new LayerNorm(16);
        var parameters = ln.Parameters().ToList();
        parameters.Should().HaveCount(2);
    }
}

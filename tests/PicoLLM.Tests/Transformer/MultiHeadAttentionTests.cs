using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class MultiHeadAttentionTests
{
    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var mha = new MultiHeadAttention(embedDim: 128, numHeads: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 6, 128], seed: 1);
        var out_ = mha.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 6, 128);
    }

    [Fact]
    public void Forward_SingleHead_OutputShape()
    {
        var mha = new MultiHeadAttention(embedDim: 32, numHeads: 1, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 32], seed: 1);
        var out_ = mha.Forward(x);
        out_.Shape.ToArray().Should().Equal(1, 5, 32);
    }

    [Fact]
    public void Forward_NoNaN_NoInf_WithRandomInput()
    {
        var mha = new MultiHeadAttention(embedDim: 8, numHeads: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 4, 8], seed: 5);
        var out_ = mha.Forward(x);
        foreach (float v in out_.Data)
        {
            float.IsNaN(v).Should().BeFalse();
            float.IsInfinity(v).Should().BeFalse();
        }
    }

    [Fact]
    public void Forward_CausalMask_BlocksFuturePositions()
    {
        // With identity projections, Q=K=V=x. Position 0 should only attend to itself.
        var mha = new MultiHeadAttention(embedDim: 4, numHeads: 1, seed: 1);
        SetIdentity(mha.QueryProj.Weights);
        SetIdentity(mha.KeyProj.Weights);
        SetIdentity(mha.ValueProj.Weights);
        SetIdentity(mha.OutputProj.Weights);
        mha.QueryProj.Bias!.MutableData.Clear();
        mha.KeyProj.Bias!.MutableData.Clear();
        mha.ValueProj.Bias!.MutableData.Clear();
        mha.OutputProj.Bias!.MutableData.Clear();

        // 3 orthogonal tokens: [1,0,0,0], [0,1,0,0], [0,0,1,0]
        var xData = new float[] { 1f, 0f, 0f, 0f,  0f, 1f, 0f, 0f,  0f, 0f, 1f, 0f };
        var x = new Tensor([1, 3, 4], xData);
        var out_ = mha.Forward(x);

        // Position 0 attended only to itself (token [1,0,0,0]) → output ≈ [1,0,0,0]
        out_[0, 0, 0].Should().BeApproximately(1f, 0.01f);
        out_[0, 0, 1].Should().BeApproximately(0f, 0.01f);
        out_[0, 0, 2].Should().BeApproximately(0f, 0.01f);
        out_[0, 0, 3].Should().BeApproximately(0f, 0.01f);
    }

    [Fact]
    public void EmbedDim_NotDivisibleByNumHeads_Throws()
    {
        Action act = () => new MultiHeadAttention(embedDim: 10, numHeads: 3);
        act.Should().Throw<ArgumentException>();
    }

    private static void SetIdentity(Tensor t)
    {
        t.MutableData.Clear();
        int n = t.Shape[0];
        for (int i = 0; i < n; i++) t.MutableData[i * n + i] = 1f;
    }
}

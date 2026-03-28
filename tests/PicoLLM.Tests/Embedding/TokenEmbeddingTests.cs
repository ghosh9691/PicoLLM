using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Embedding;

public class TokenEmbeddingTests
{
    private const float Tol = 1e-5f;

    // ── Forward pass ─────────────────────────────────────────────────────────

    [Fact]
    public void Forward_OutputShape_IsSeqLenByEmbedDim()
    {
        var layer = new TokenEmbedding(vocabSize: 512, embedDim: 64, seed: 1);
        var ids = new[] { 2, 45, 67, 3 };
        var output = layer.Forward(ids);
        output.Shape.ToArray().Should().Equal(4, 64);
    }

    [Fact]
    public void Forward_SameId_SameRow()
    {
        var layer = new TokenEmbedding(vocabSize: 100, embedDim: 8, seed: 42);
        var out1 = layer.Forward([10]);
        var out2 = layer.Forward([10]);
        out1.Data.ToArray().Should().Equal(out2.Data.ToArray());
    }

    [Fact]
    public void Forward_DifferentIds_DifferentRows()
    {
        var layer = new TokenEmbedding(vocabSize: 100, embedDim: 8, seed: 42);
        var out1 = layer.Forward([10]);
        var out2 = layer.Forward([20]);
        out1.Data.ToArray().Should().NotEqual(out2.Data.ToArray());
    }

    [Fact]
    public void Forward_RowMatchesWeightRow()
    {
        var layer = new TokenEmbedding(vocabSize: 50, embedDim: 4, seed: 7);
        int id = 5;
        var output = layer.Forward([id]);
        var w = layer.Weights.Data;
        for (int d = 0; d < 4; d++)
            output[0, d].Should().BeApproximately(w[id * 4 + d], Tol);
    }

    [Fact]
    public void Forward_OutOfRangeId_Throws()
    {
        var layer = new TokenEmbedding(vocabSize: 10, embedDim: 4);
        Action act = () => layer.Forward([10]); // id 10 is out of range for vocab 10
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ── Backward pass ────────────────────────────────────────────────────────

    [Fact]
    public void Backward_AccumulatesGradInUsedRows()
    {
        var layer = new TokenEmbedding(vocabSize: 20, embedDim: 4, seed: 1);
        int[] ids = [3, 7];
        _ = layer.Forward(ids);

        var grad = TensorFactory.Ones(2, 4); // upstream gradient: all ones
        layer.Backward(grad, ids);

        // Row 3 and row 7 of WeightGrad should have accumulated values
        float row3sum = 0f, row7sum = 0f;
        for (int d = 0; d < 4; d++)
        {
            row3sum += layer.WeightGrad[3, d];
            row7sum += layer.WeightGrad[7, d];
        }
        row3sum.Should().BeApproximately(4f, Tol); // 4 dimensions × 1.0 gradient
        row7sum.Should().BeApproximately(4f, Tol);
    }

    [Fact]
    public void Backward_UnusedRowsGradRemainsZero()
    {
        var layer = new TokenEmbedding(vocabSize: 20, embedDim: 4, seed: 1);
        int[] ids = [3];
        _ = layer.Forward(ids);
        var grad = TensorFactory.Ones(1, 4);
        layer.Backward(grad, ids);

        // Row 5 was never accessed — gradient should stay zero
        for (int d = 0; d < 4; d++)
            layer.WeightGrad[5, d].Should().Be(0f);
    }

    [Fact]
    public void ZeroGrad_ClearsAccumulatedGradient()
    {
        var layer = new TokenEmbedding(vocabSize: 20, embedDim: 4, seed: 1);
        int[] ids = [3];
        _ = layer.Forward(ids);
        layer.Backward(TensorFactory.Ones(1, 4), ids);
        layer.ZeroGrad();
        foreach (var v in layer.WeightGrad.Data) v.Should().Be(0f);
    }
}

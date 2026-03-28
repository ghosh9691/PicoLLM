using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Embedding;

public class EmbeddingLayerTests
{
    private const float Tol = 1e-5f;

    // ── Single-sequence forward ───────────────────────────────────────────────

    [Fact]
    public void Forward_OutputShape_IsSeqLenByEmbedDim()
    {
        var layer = new EmbeddingLayer(vocabSize: 512, embedDim: 128, seed: 1);
        var ids = new[] { 2, 45, 67, 3 };
        var output = layer.Forward(ids);
        output.Shape.ToArray().Should().Equal(4, 128);
    }

    [Fact]
    public void Forward_IncludesPositionalEncoding()
    {
        var layer = new EmbeddingLayer(vocabSize: 100, embedDim: 8, seed: 1);
        var ids = new[] { 5 };
        var output = layer.Forward(ids); // [1, 8]

        // Manually compute: token emb[5] + PE(0)
        var pe = new PositionalEncoding(2048, 8);
        var peRow0 = pe.GetEncoding(1);
        var wRow5 = layer.TokenEmbedding.Weights;

        for (int d = 0; d < 8; d++)
            output[0, d].Should().BeApproximately(wRow5[5, d] + peRow0[0, d], Tol);
    }

    [Fact]
    public void Forward_SameInput_SameOutput()
    {
        var layer = new EmbeddingLayer(vocabSize: 100, embedDim: 16, seed: 7);
        var ids = new[] { 1, 2, 3 };
        var o1 = layer.Forward(ids);
        var o2 = layer.Forward(ids);
        o1.Data.ToArray().Should().Equal(o2.Data.ToArray());
    }

    // ── Batched forward ───────────────────────────────────────────────────────

    [Fact]
    public void ForwardBatch_OutputShape_IsBatchBySeqByEmbed()
    {
        var layer = new EmbeddingLayer(vocabSize: 512, embedDim: 64, seed: 1);
        var batch = new[] { new[] { 2, 10, 20 }, new[] { 3, 11, 21 } };
        var output = layer.ForwardBatch(batch);
        output.Shape.ToArray().Should().Equal(2, 3, 64);
    }

    [Fact]
    public void ForwardBatch_PositionalEncodingIdenticalAcrossBatch()
    {
        // Both batch items at the same position should have the same PE contribution.
        // Since the weight lookup differs per token, we can't compare full outputs —
        // but if we set all weights to zero, the output equals PE alone.
        var layer = new EmbeddingLayer(vocabSize: 512, embedDim: 4, seed: 1);
        // Zero out all weights
        layer.TokenEmbedding.Weights.MutableData.Clear();

        var batch = new[] { new[] { 5, 6 }, new[] { 7, 8 } };
        var output = layer.ForwardBatch(batch);

        // With zero token weights, output = PE only
        var pe = new PositionalEncoding(2048, 4);
        var peSlice = pe.GetEncoding(2);

        // Both batch items should match PE at each position
        for (int b = 0; b < 2; b++)
            for (int s = 0; s < 2; s++)
                for (int d = 0; d < 4; d++)
                    output[b, s, d].Should().BeApproximately(peSlice[s, d], Tol);
    }

    [Fact]
    public void ForwardBatch_UnequalSeqLen_Throws()
    {
        var layer = new EmbeddingLayer(vocabSize: 100, embedDim: 8);
        var batch = new[] { new[] { 1, 2, 3 }, new[] { 4, 5 } }; // different lengths
        Action act = () => layer.ForwardBatch(batch);
        act.Should().Throw<ArgumentException>();
    }

    // ── Backward ─────────────────────────────────────────────────────────────

    [Fact]
    public void Backward_PropagatesGradToTokenEmbedding()
    {
        var layer = new EmbeddingLayer(vocabSize: 50, embedDim: 4, seed: 1);
        int[] ids = [10, 20];
        _ = layer.Forward(ids);
        var grad = TensorFactory.Ones(2, 4);
        layer.Backward(grad, ids);

        // Rows 10 and 20 should have accumulated gradient
        float row10sum = 0f;
        for (int d = 0; d < 4; d++) row10sum += layer.TokenEmbedding.WeightGrad[10, d];
        row10sum.Should().BeApproximately(4f, Tol);
    }

    [Fact]
    public void ZeroGrad_ClearsEmbeddingGrad()
    {
        var layer = new EmbeddingLayer(vocabSize: 50, embedDim: 4, seed: 1);
        int[] ids = [5];
        _ = layer.Forward(ids);
        layer.Backward(TensorFactory.Ones(1, 4), ids);
        layer.ZeroGrad();
        foreach (var v in layer.TokenEmbedding.WeightGrad.Data) v.Should().Be(0f);
    }
}

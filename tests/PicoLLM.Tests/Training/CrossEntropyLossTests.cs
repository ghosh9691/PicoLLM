using FluentAssertions;
using PicoLLM.Core.Tensors;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class CrossEntropyLossTests
{
    private const float Tol = 1e-4f;

    [Fact]
    public void Forward_KnownValues_CorrectLoss()
    {
        // 1 batch, 1 seq, vocab=2
        // logits = [0, 1] → softmax ≈ [0.269, 0.731]
        // target = 0 → loss = -log(0.269) ≈ 1.3133
        var logits = new Tensor([1, 1, 2], [0f, 1f]);
        var targets = new int[,] { { 0 } };
        float loss = CrossEntropyLoss.Forward(logits, targets);

        float expectedProb = MathF.Exp(0f) / (MathF.Exp(0f) + MathF.Exp(1f));
        float expectedLoss = -MathF.Log(expectedProb);
        loss.Should().BeApproximately(expectedLoss, Tol);
    }

    [Fact]
    public void Forward_PerfectPrediction_LossNearZero()
    {
        // Very high logit at target position → loss ≈ 0
        var logits = new Tensor([1, 1, 3], [0f, 100f, 0f]);
        var targets = new int[,] { { 1 } };
        float loss = CrossEntropyLoss.Forward(logits, targets);
        loss.Should().BeLessThan(0.01f);
    }

    [Fact]
    public void Forward_Batch_LossIsAveraged()
    {
        // 2 batches of the same logits/targets → same loss as 1 batch
        var logits1 = new Tensor([1, 1, 2], [1f, 2f]);
        var targets1 = new int[,] { { 0 } };
        float loss1 = CrossEntropyLoss.Forward(logits1, targets1);

        var logits2 = new Tensor([2, 1, 2], [1f, 2f, 1f, 2f]);
        var targets2 = new int[,] { { 0 }, { 0 } };
        float loss2 = CrossEntropyLoss.Forward(logits2, targets2);

        loss1.Should().BeApproximately(loss2, Tol);
    }

    [Fact]
    public void Backward_OutputShape_MatchesLogits()
    {
        var logits = new Tensor([2, 3, 8], Enumerable.Range(0, 2 * 3 * 8).Select(i => (float)i).ToArray());
        var targets = new int[2, 3];
        var grad = CrossEntropyLoss.Backward(logits, targets);
        grad.Shape.ToArray().Should().Equal(2, 3, 8);
    }

    [Fact]
    public void Backward_SumsToZero_PerPosition()
    {
        // The gradient (softmax - one_hot) sums to 0 over vocab at each position
        // because sum(softmax) = 1 and sum(one_hot) = 1
        var logits = TensorFactory.RandomNormal([1, 1, 5], seed: 1);
        var targets = new int[,] { { 2 } };
        var grad = CrossEntropyLoss.Backward(logits, targets);

        float sum = grad.Data.ToArray().Sum();
        sum.Should().BeApproximately(0f, 1e-5f);
    }

    [Fact]
    public void Backward_TargetPosition_HasNegativeGrad()
    {
        // grad[target] = (softmax[target] - 1) / N < 0
        var logits = new Tensor([1, 1, 3], [1f, 2f, 3f]);
        var targets = new int[,] { { 2 } }; // highest logit is target
        var grad = CrossEntropyLoss.Backward(logits, targets);
        grad[0, 0, 2].Should().BeLessThan(0f);
    }
}

using FluentAssertions;
using PicoLLM.Core.Activations;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Activations;

public class ActivationsTests
{
    private const float Tol = 1e-4f;

    private static Tensor FromValues(params float[] values)
    {
        var t = new Tensor(values.Length);
        for (int i = 0; i < values.Length; i++) t[i] = values[i];
        return t;
    }

    // ── ReLU ────────────────────────────────────────────────────────────────

    [Fact]
    public void ReLU_Forward_ClampsNegativeToZero()
    {
        var t = FromValues(-3f, 0f, 2f, -1f, 5f);
        var out_ = ReLU.Forward(t);
        out_.Data.ToArray().Should().Equal(0f, 0f, 2f, 0f, 5f);
    }

    [Fact]
    public void ReLU_Backward_IsStepFunction()
    {
        var t = FromValues(-3f, 0f, 2f);
        var grad = ReLU.Backward(t);
        grad.Data.ToArray().Should().Equal(0f, 0f, 1f);
    }

    // ── GELU ────────────────────────────────────────────────────────────────

    [Fact]
    public void GELU_Forward_ZeroInputIsZero()
    {
        var t = FromValues(0f);
        var out_ = GELU.Forward(t);
        out_.Data[0].Should().BeApproximately(0f, Tol);
    }

    [Fact]
    public void GELU_Forward_LargePositiveApproachesX()
    {
        // For large x, GELU(x) ≈ x
        var t = FromValues(10f);
        var out_ = GELU.Forward(t);
        out_.Data[0].Should().BeApproximately(10f, 0.01f);
    }

    [Fact]
    public void GELU_Forward_LargeNegativeApproachesZero()
    {
        // For large negative x, GELU(x) ≈ 0
        var t = FromValues(-10f);
        var out_ = GELU.Forward(t);
        out_.Data[0].Should().BeApproximately(0f, 0.01f);
    }

    [Fact]
    public void GELU_Backward_PositiveNearOne()
    {
        // At x=2, GELU derivative should be close to 1
        var t = FromValues(2f);
        var deriv = GELU.Backward(t);
        deriv.Data[0].Should().BeInRange(0.9f, 1.1f);
    }

    // ── Sigmoid ─────────────────────────────────────────────────────────────

    [Fact]
    public void Sigmoid_Forward_ZeroInputIsHalf()
    {
        var t = FromValues(0f);
        var out_ = Sigmoid.Forward(t);
        out_.Data[0].Should().BeApproximately(0.5f, Tol);
    }

    [Fact]
    public void Sigmoid_Forward_OutputInRange()
    {
        var t = FromValues(-100f, 0f, 100f);
        var out_ = Sigmoid.Forward(t);
        foreach (var v in out_.Data)
        {
            v.Should().BeGreaterThanOrEqualTo(0f);
            v.Should().BeLessThanOrEqualTo(1f);
        }
    }

    [Fact]
    public void Sigmoid_Backward_AtZeroIsQuarter()
    {
        var t = FromValues(0f);
        var grad = Sigmoid.Backward(t);
        grad.Data[0].Should().BeApproximately(0.25f, Tol); // σ(0)·(1−σ(0)) = 0.5·0.5
    }

    // ── Tanh ────────────────────────────────────────────────────────────────

    [Fact]
    public void Tanh_Forward_ZeroInputIsZero()
    {
        var t = FromValues(0f);
        var out_ = Tanh.Forward(t);
        out_.Data[0].Should().BeApproximately(0f, Tol);
    }

    [Fact]
    public void Tanh_Forward_OutputInRange()
    {
        var t = FromValues(-10f, 0f, 10f);
        var out_ = Tanh.Forward(t);
        foreach (var v in out_.Data)
        {
            v.Should().BeGreaterThanOrEqualTo(-1f);
            v.Should().BeLessThanOrEqualTo(1f);
        }
    }

    [Fact]
    public void Tanh_Backward_AtZeroIsOne()
    {
        var t = FromValues(0f);
        var grad = Tanh.Backward(t);
        grad.Data[0].Should().BeApproximately(1f, Tol); // 1 - tanh²(0) = 1
    }

    // ── Softmax ─────────────────────────────────────────────────────────────

    [Fact]
    public void Softmax_Forward_DefaultLastAxis_SumsToOne()
    {
        var t = new Tensor(2, 3);
        t[0, 0] = 1f; t[0, 1] = 2f; t[0, 2] = 3f;
        t[1, 0] = 0.5f; t[1, 1] = 1.5f; t[1, 2] = 2.5f;
        var out_ = PicoLLM.Core.Activations.Softmax.Forward(t);
        (out_[0, 0] + out_[0, 1] + out_[0, 2]).Should().BeApproximately(1f, 1e-5f);
        (out_[1, 0] + out_[1, 1] + out_[1, 2]).Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public void Softmax_Forward_NegativeAxisResolvesCorrectly()
    {
        var t = FromValues(1f, 2f, 3f);
        var reshapedT = TensorMath.Reshape(t, 1, 3);
        var out_ = PicoLLM.Core.Activations.Softmax.Forward(reshapedT, axis: -1);
        (out_[0, 0] + out_[0, 1] + out_[0, 2]).Should().BeApproximately(1f, 1e-5f);
    }
}

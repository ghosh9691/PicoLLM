using FluentAssertions;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class GradientClipperTests
{
    [Fact]
    public void ClipGradNorm_ReturnsGlobalNorm()
    {
        // grads = [3, 4] → norm = 5
        var p = new Parameter(TensorFactory.Zeros(2));
        p.Grad.MutableData[0] = 3f;
        p.Grad.MutableData[1] = 4f;

        float norm = GradientClipper.ClipGradNorm([p], maxNorm: 10f);
        norm.Should().BeApproximately(5f, 1e-4f);
    }

    [Fact]
    public void ClipGradNorm_ScalesGrads_WhenAboveMax()
    {
        // grads = [3, 4], norm = 5, maxNorm = 1 → grads = [0.6, 0.8]
        var p = new Parameter(TensorFactory.Zeros(2));
        p.Grad.MutableData[0] = 3f;
        p.Grad.MutableData[1] = 4f;

        GradientClipper.ClipGradNorm([p], maxNorm: 1f);

        p.Grad.Data[0].Should().BeApproximately(0.6f, 1e-4f);
        p.Grad.Data[1].Should().BeApproximately(0.8f, 1e-4f);
    }

    [Fact]
    public void ClipGradNorm_NoOp_WhenBelowMax()
    {
        var p = new Parameter(TensorFactory.Zeros(2));
        p.Grad.MutableData[0] = 0.1f;
        p.Grad.MutableData[1] = 0.1f;

        GradientClipper.ClipGradNorm([p], maxNorm: 10f);

        p.Grad.Data[0].Should().BeApproximately(0.1f, 1e-5f);
        p.Grad.Data[1].Should().BeApproximately(0.1f, 1e-5f);
    }

    [Fact]
    public void ClipGradNorm_AcrossMultipleParameters()
    {
        // p1 grads = [3], p2 grads = [4] → global norm = 5
        var p1 = new Parameter(TensorFactory.Zeros(1));
        var p2 = new Parameter(TensorFactory.Zeros(1));
        p1.Grad.MutableData[0] = 3f;
        p2.Grad.MutableData[0] = 4f;

        float norm = GradientClipper.ClipGradNorm([p1, p2], maxNorm: 1f);
        norm.Should().BeApproximately(5f, 1e-4f);
        p1.Grad.Data[0].Should().BeApproximately(0.6f, 1e-4f);
        p2.Grad.Data[0].Should().BeApproximately(0.8f, 1e-4f);
    }
}

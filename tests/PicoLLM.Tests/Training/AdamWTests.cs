using FluentAssertions;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;
using PicoLLM.Training;

namespace PicoLLM.Tests.Training;

public class AdamWTests
{
    [Fact]
    public void Step_MovesParameterTowardLowerLoss()
    {
        // Minimize f(x) = x² by gradient descent.
        // At x=3, gradient = 2x = 6.
        var data = TensorFactory.Fill(3f, 1);
        var param = new Parameter(data);
        param.Grad.MutableData[0] = 6f; // manual gradient

        var optimizer = new AdamW(lr: 0.1f, weightDecay: 0f);
        optimizer.Step([param]);

        param.Data.Data[0].Should().BeLessThan(3f, "parameter should move toward lower loss");
    }

    [Fact]
    public void Step_Converges_On_QuadraticFunction()
    {
        // Minimize x² + y²: start at (3, 4), run 500 steps with analytical gradient
        var data = TensorFactory.FromArray([2], [3f, 4f]);
        var param = new Parameter(data);
        var optimizer = new AdamW(lr: 0.01f, weightDecay: 0f);

        for (int i = 0; i < 2000; i++)
        {
            param.ZeroGrad();
            // grad of (x²+y²) = [2x, 2y]
            param.Grad.MutableData[0] = 2f * param.Data.Data[0];
            param.Grad.MutableData[1] = 2f * param.Data.Data[1];
            optimizer.Step([param]);
        }

        param.Data.Data[0].Should().BeApproximately(0f, 0.05f);
        param.Data.Data[1].Should().BeApproximately(0f, 0.05f);
    }

    [Fact]
    public void Step_IncreasesStepCount()
    {
        var param = new Parameter(TensorFactory.Zeros(1));
        var optimizer = new AdamW();
        optimizer.Step([param]);
        optimizer.Step([param]);
        optimizer.StepCount.Should().Be(2);
    }

    [Fact]
    public void WeightDecay_ShrinksMagnitude()
    {
        // With zero gradient and non-zero weight decay, weights should shrink
        var data = TensorFactory.Fill(1f, 4);
        var param = new Parameter(data);
        var optimizer = new AdamW(lr: 0.01f, weightDecay: 0.1f);
        // Leave gradient at 0 (ZeroGrad not needed, it's already 0)
        optimizer.Step([param]);
        param.Data.Data[0].Should().BeLessThan(1f);
    }

    [Fact]
    public void LearningRate_IsUpdatable()
    {
        var optimizer = new AdamW(lr: 1e-4f);
        optimizer.LearningRate = 1e-3f;
        optimizer.LearningRate.Should().BeApproximately(1e-3f, 1e-8f);
    }
}

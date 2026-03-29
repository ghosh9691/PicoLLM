using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class LinearLayerTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void Forward_2D_CorrectOutput()
    {
        // [2, 3] input, weights [3,2]: first two rows of identity
        var layer = new LinearLayer(3, 2, useBias: true);
        // Weights row 0 = [1,0], row 1 = [0,1], row 2 = [0,0]
        layer.Weights.MutableData[0] = 1f; layer.Weights.MutableData[1] = 0f;
        layer.Weights.MutableData[2] = 0f; layer.Weights.MutableData[3] = 1f;
        layer.Weights.MutableData[4] = 0f; layer.Weights.MutableData[5] = 0f;
        layer.Bias!.MutableData.Clear();

        var input = new Tensor([2, 3], [1f, 2f, 3f, 4f, 5f, 6f]);
        var output = layer.Forward(input);

        output.Shape.ToArray().Should().Equal(2, 2);
        output[0, 0].Should().BeApproximately(1f, Tol);
        output[0, 1].Should().BeApproximately(2f, Tol);
        output[1, 0].Should().BeApproximately(4f, Tol);
        output[1, 1].Should().BeApproximately(5f, Tol);
    }

    [Fact]
    public void Forward_3D_OutputShape()
    {
        var layer = new LinearLayer(128, 64, useBias: true, seed: 1);
        var x = TensorFactory.RandomNormal([2, 10, 128], seed: 1);
        var out_ = layer.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 10, 64);
    }

    [Fact]
    public void Forward_NoBias_Works()
    {
        var layer = new LinearLayer(4, 4, useBias: false);
        layer.Bias.Should().BeNull();
        var x = TensorFactory.RandomNormal([1, 4], seed: 1);
        var out_ = layer.Forward(x);
        out_.Shape.ToArray().Should().Equal(1, 4);
    }

    [Fact]
    public void Forward_BiasAdded_Correctly()
    {
        var layer = new LinearLayer(2, 2, useBias: true);
        // Identity weights
        layer.Weights.MutableData[0] = 1f; layer.Weights.MutableData[1] = 0f;
        layer.Weights.MutableData[2] = 0f; layer.Weights.MutableData[3] = 1f;
        layer.Bias!.MutableData[0] = 10f;
        layer.Bias!.MutableData[1] = 20f;

        var x = new Tensor([1, 2], [3f, 4f]);
        var out_ = layer.Forward(x);
        out_[0, 0].Should().BeApproximately(13f, Tol);
        out_[0, 1].Should().BeApproximately(24f, Tol);
    }

    [Fact]
    public void Parameters_WithBias_ReturnsTwoTensors()
    {
        var layer = new LinearLayer(4, 8, useBias: true);
        layer.Parameters().Should().HaveCount(2);
    }

    [Fact]
    public void Parameters_NoBias_ReturnsOneTensor()
    {
        var layer = new LinearLayer(4, 8, useBias: false);
        layer.Parameters().Should().HaveCount(1);
    }
}

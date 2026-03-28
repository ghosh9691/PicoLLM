using FluentAssertions;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Tensors;

public class TensorFactoryTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void Zeros_AllElementsAreZero()
    {
        var t = TensorFactory.Zeros(3, 4);
        foreach (var v in t.Data) v.Should().Be(0f);
        t.Shape.ToArray().Should().Equal(3, 4);
    }

    [Fact]
    public void Ones_AllElementsAreOne()
    {
        var t = TensorFactory.Ones(2, 5);
        foreach (var v in t.Data) v.Should().Be(1f);
    }

    [Fact]
    public void Fill_AllElementsEqualValue()
    {
        var t = TensorFactory.Fill(7.5f, 3, 3);
        foreach (var v in t.Data) v.Should().BeApproximately(7.5f, Tol);
    }

    [Fact]
    public void Random_ValuesInRange()
    {
        var t = TensorFactory.Random([100], seed: 42);
        foreach (var v in t.Data)
        {
            v.Should().BeGreaterThanOrEqualTo(0f);
            v.Should().BeLessThan(1f);
        }
    }

    [Fact]
    public void Random_SameSeed_ProducesSameValues()
    {
        var a = TensorFactory.Random([50], seed: 123);
        var b = TensorFactory.Random([50], seed: 123);
        a.Data.ToArray().Should().Equal(b.Data.ToArray());
    }

    [Fact]
    public void RandomNormal_ApproximatelyZeroMean()
    {
        var t = TensorFactory.RandomNormal([10000], mean: 0f, std: 1f, seed: 1);
        float mean = t.Data.ToArray().Average();
        mean.Should().BeInRange(-0.1f, 0.1f); // within 3-sigma for large N
    }

    [Fact]
    public void XavierUniform_ValuesWithinLimit()
    {
        int fanIn = 64, fanOut = 128;
        float limit = MathF.Sqrt(6f / (fanIn + fanOut));
        var t = TensorFactory.XavierUniform([64, 128], fanIn, fanOut, seed: 7);
        foreach (var v in t.Data)
        {
            v.Should().BeGreaterThanOrEqualTo(-limit);
            v.Should().BeLessThanOrEqualTo(limit);
        }
    }

    [Fact]
    public void XavierNormal_ApproximatelyZeroMean()
    {
        var t = TensorFactory.XavierNormal([5000], 128, 128, seed: 9);
        float mean = t.Data.ToArray().Average();
        mean.Should().BeInRange(-0.05f, 0.05f);
    }

    [Fact]
    public void Eye_DiagonalOnes()
    {
        var e = TensorFactory.Eye(4);
        e.Shape.ToArray().Should().Equal(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                e[i, j].Should().Be(i == j ? 1f : 0f);
    }
}

using FluentAssertions;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Tensors;

public class TensorTests
{
    // ── Construction ────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_2D_AllocatesCorrectElements()
    {
        var t = new Tensor(3, 4);
        t.Length.Should().Be(12);
        t.Shape.ToArray().Should().Equal(3, 4);
        t.Rank.Should().Be(2);
    }

    [Fact]
    public void Constructor_4D_Allocates4096Elements()
    {
        var t = new Tensor(2, 4, 16, 32);
        t.Length.Should().Be(2 * 4 * 16 * 32);
    }

    [Fact]
    public void Constructor_ZeroDimension_Throws()
    {
        Action act = () => _ = new Tensor(3, 0, 4);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_EmptyShape_Throws()
    {
        Action act = () => _ = new Tensor(Array.Empty<int>());
        act.Should().Throw<ArgumentException>();
    }

    // ── Strides ─────────────────────────────────────────────────────────────

    [Fact]
    public void Strides_2D_AreRowMajor()
    {
        var t = new Tensor(3, 4);
        t.Strides.ToArray().Should().Equal(4, 1);
    }

    [Fact]
    public void Strides_3D_AreRowMajor()
    {
        var t = new Tensor(2, 3, 4);
        t.Strides.ToArray().Should().Equal(12, 4, 1);
    }

    // ── Indexer ─────────────────────────────────────────────────────────────

    [Fact]
    public void Indexer_SetAndGet_RoundTrips()
    {
        var t = new Tensor(3, 4);
        t[1, 2] = 5.0f;
        t[1, 2].Should().Be(5.0f);
    }

    [Fact]
    public void Indexer_OutOfBounds_Throws()
    {
        var t = new Tensor(3, 4);
        Action act = () => _ = t[3, 0]; // row 3 is out of bounds for size 3
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void Indexer_WrongRank_Throws()
    {
        var t = new Tensor(3, 4);
        Action act = () => _ = t[1, 2, 0];
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void Indexer_3D_LinearStorageCorrect()
    {
        var t = new Tensor(2, 3, 4);
        t[1, 2, 3] = 99f;
        // flat index = 1*12 + 2*4 + 3 = 23
        t.Data[23].Should().Be(99f);
    }

    // ── Properties ──────────────────────────────────────────────────────────

    [Fact]
    public void GetShape_ReturnsCopy()
    {
        var t = new Tensor(2, 3);
        var shape = t.GetShape();
        shape[0] = 99; // mutate copy
        t.Shape[0].Should().Be(2); // original unchanged
    }

    [Fact]
    public void MutableData_AllowsInPlaceWrite()
    {
        var t = new Tensor(4);
        t.MutableData[2] = 7f;
        t[2].Should().Be(7f);
    }

    [Fact]
    public void ToString_ContainsShape()
    {
        var t = new Tensor(2, 3);
        t.ToString().Should().Contain("2").And.Contain("3");
    }
}

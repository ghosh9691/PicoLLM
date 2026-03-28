using FluentAssertions;
using PicoLLM.Core.Layers;

namespace PicoLLM.Tests.Embedding;

public class PositionalEncodingTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void GetEncoding_ReturnsCorrectShape()
    {
        var pe = new PositionalEncoding(maxSeqLen: 512, embedDim: 64);
        var enc = pe.GetEncoding(4);
        enc.Shape.ToArray().Should().Equal(4, 64);
    }

    [Fact]
    public void GetEncoding_Position0_Dim0IsSin0()
    {
        var pe = new PositionalEncoding(maxSeqLen: 512, embedDim: 64);
        var enc = pe.GetEncoding(1);
        // PE(0, 0) = sin(0 / 10000^0) = sin(0) = 0
        enc[0, 0].Should().BeApproximately(0f, Tol);
    }

    [Fact]
    public void GetEncoding_Position0_Dim1IsCos0()
    {
        var pe = new PositionalEncoding(maxSeqLen: 512, embedDim: 64);
        var enc = pe.GetEncoding(1);
        // PE(0, 1) = cos(0 / ...) = cos(0) = 1
        enc[0, 1].Should().BeApproximately(1f, Tol);
    }

    [Fact]
    public void GetEncoding_EvenDimIsSin_OddDimIsCos()
    {
        var pe = new PositionalEncoding(maxSeqLen: 512, embedDim: 64);
        int seqLen = 10;
        var enc = pe.GetEncoding(seqLen);

        // Verify sin/cos pattern for position pos=1, i=2 (dim 4 and 5)
        int pos = 1, i = 2;
        double angle = pos / Math.Pow(10000.0, 2.0 * i / 64.0);
        enc[pos, 2 * i].Should().BeApproximately((float)Math.Sin(angle), Tol);
        enc[pos, 2 * i + 1].Should().BeApproximately((float)Math.Cos(angle), Tol);
    }

    [Fact]
    public void GetEncoding_Deterministic_SameCallSameResult()
    {
        var pe = new PositionalEncoding(maxSeqLen: 100, embedDim: 16);
        var e1 = pe.GetEncoding(5);
        var e2 = pe.GetEncoding(5);
        e1.Data.ToArray().Should().Equal(e2.Data.ToArray());
    }

    [Fact]
    public void GetEncoding_ExceedsMaxSeqLen_Throws()
    {
        var pe = new PositionalEncoding(maxSeqLen: 10, embedDim: 8);
        Action act = () => pe.GetEncoding(11);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GetEncoding_EmbedDim4_HandComputedValues()
    {
        // With embedDim=4, for position pos and i in {0,1}:
        // PE(pos,0) = sin(pos / 10000^(0/4)) = sin(pos)
        // PE(pos,1) = cos(pos / 10000^(0/4)) = cos(pos)
        // PE(pos,2) = sin(pos / 10000^(2/4)) = sin(pos/100)
        // PE(pos,3) = cos(pos / 10000^(2/4)) = cos(pos/100)
        var pe = new PositionalEncoding(maxSeqLen: 10, embedDim: 4);
        var enc = pe.GetEncoding(3);

        // pos=1
        enc[1, 0].Should().BeApproximately((float)Math.Sin(1.0), Tol);
        enc[1, 1].Should().BeApproximately((float)Math.Cos(1.0), Tol);
        enc[1, 2].Should().BeApproximately((float)Math.Sin(1.0 / 100.0), Tol);
        enc[1, 3].Should().BeApproximately((float)Math.Cos(1.0 / 100.0), Tol);
    }
}

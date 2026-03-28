using FluentAssertions;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Tensors;

public class TensorMathTests
{
    private const float Tolerance = 1e-5f;

    // ── Element-wise ────────────────────────────────────────────────────────

    [Fact]
    public void Add_TwoTensors_ElementWise()
    {
        var a = new Tensor(2, 3);
        var b = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 2f; a[1, 2] = 3f;
        b[0, 0] = 4f; b[0, 1] = 5f; b[1, 2] = 6f;
        var c = TensorMath.Add(a, b);
        c[0, 0].Should().BeApproximately(5f, Tolerance);
        c[0, 1].Should().BeApproximately(7f, Tolerance);
        c[1, 2].Should().BeApproximately(9f, Tolerance);
    }

    [Fact]
    public void Add_ShapeMismatch_Throws()
    {
        var a = new Tensor(2, 3);
        var b = new Tensor(3, 2);
        Action act = () => TensorMath.Add(a, b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Multiply_Scalar_ScalesAllElements()
    {
        var a = TensorFactory.Ones(2, 3);
        var c = TensorMath.Multiply(a, 3f);
        foreach (var v in c.Data) v.Should().BeApproximately(3f, Tolerance);
    }

    [Fact]
    public void OperatorPlus_DelegatesCorrectly()
    {
        var a = TensorFactory.Fill(2f, 3);
        var b = TensorFactory.Fill(3f, 3);
        var c = a + b;
        foreach (var v in c.Data) v.Should().BeApproximately(5f, Tolerance);
    }

    [Fact]
    public void OperatorStar_Scalar_Works()
    {
        var a = TensorFactory.Fill(2f, 4);
        var c = a * 5f;
        foreach (var v in c.Data) v.Should().BeApproximately(10f, Tolerance);
    }

    [Fact]
    public void Negate_Operator_NegatesAllElements()
    {
        var a = TensorFactory.Fill(3f, 4);
        var c = -a;
        foreach (var v in c.Data) v.Should().BeApproximately(-3f, Tolerance);
    }

    // ── MatMul ──────────────────────────────────────────────────────────────

    [Fact]
    public void MatMul_KnownValues_Correct()
    {
        // [2,3] × [3,2] = [2,2]
        var a = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 5f; a[1, 2] = 6f;

        var b = new Tensor(3, 2);
        b[0, 0] = 7f;  b[0, 1] = 8f;
        b[1, 0] = 9f;  b[1, 1] = 10f;
        b[2, 0] = 11f; b[2, 1] = 12f;

        var c = TensorMath.MatMul(a, b);
        c.Shape.ToArray().Should().Equal(2, 2);
        c[0, 0].Should().BeApproximately(58f, Tolerance);  // 1*7+2*9+3*11
        c[0, 1].Should().BeApproximately(64f, Tolerance);  // 1*8+2*10+3*12
        c[1, 0].Should().BeApproximately(139f, Tolerance); // 4*7+5*9+6*11
        c[1, 1].Should().BeApproximately(154f, Tolerance); // 4*8+5*10+6*12
    }

    [Fact]
    public void MatMul_IdentityMatrix_ReturnsSameTensor()
    {
        var a = new Tensor(3, 3);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) a[i, j] = i * 3 + j + 1f;
        var identity = TensorFactory.Eye(3);
        var result = TensorMath.MatMul(a, identity);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result[i, j].Should().BeApproximately(a[i, j], Tolerance);
    }

    [Fact]
    public void BatchedMatMul_Shape_Correct()
    {
        var q = TensorFactory.Random([2, 4, 8, 16], seed: 1);
        var kt = TensorFactory.Random([2, 4, 16, 8], seed: 2);
        var scores = TensorMath.BatchedMatMul(q, kt);
        scores.Shape.ToArray().Should().Equal(2, 4, 8, 8);
    }

    // ── Reshape ─────────────────────────────────────────────────────────────

    [Fact]
    public void Reshape_PreservesData()
    {
        var a = TensorFactory.Fill(3f, 2, 3);
        var b = TensorMath.Reshape(a, 6);
        b.Shape.ToArray().Should().Equal(6);
        foreach (var v in b.Data) v.Should().BeApproximately(3f, Tolerance);
    }

    [Fact]
    public void Reshape_InferredDimension_Works()
    {
        var a = TensorFactory.Ones(2, 3, 4);
        var b = TensorMath.Reshape(a, 2, -1);
        b.Shape.ToArray().Should().Equal(2, 12);
    }

    [Fact]
    public void Reshape_WrongTotal_Throws()
    {
        var a = TensorFactory.Ones(2, 3);
        Action act = () => TensorMath.Reshape(a, 7);
        act.Should().Throw<ArgumentException>();
    }

    // ── Transpose ───────────────────────────────────────────────────────────

    [Fact]
    public void Transpose_2D_CorrectShape()
    {
        var a = new Tensor(3, 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                a[i, j] = i * 4 + j;
        var b = TensorMath.Transpose(a, 0, 1);
        b.Shape.ToArray().Should().Equal(4, 3);
        b[2, 1].Should().BeApproximately(a[1, 2], Tolerance);
        b[3, 0].Should().BeApproximately(a[0, 3], Tolerance);
    }

    [Fact]
    public void Transpose_LastTwoDims_ForAttention()
    {
        var k = TensorFactory.Random([2, 4, 8, 16], seed: 42);
        var kt = TensorMath.Transpose(k, 2, 3);
        kt.Shape.ToArray().Should().Equal(2, 4, 16, 8);
    }

    // ── Slice ────────────────────────────────────────────────────────────────

    [Fact]
    public void Slice_FirstNRows_CorrectShape()
    {
        var a = TensorFactory.Ones(10, 8);
        var s = TensorMath.Slice(a, 0, 4);
        s.Shape.ToArray().Should().Equal(4, 8);
    }

    // ── Reductions ───────────────────────────────────────────────────────────

    [Fact]
    public void Sum_Axis1_CorrectValues()
    {
        var a = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 5f; a[1, 2] = 6f;
        var s = TensorMath.Sum(a, axis: 1);
        s.Shape.ToArray().Should().Equal(2);
        s[0].Should().BeApproximately(6f, Tolerance);
        s[1].Should().BeApproximately(15f, Tolerance);
    }

    [Fact]
    public void Mean_Axis0_CorrectValues()
    {
        var a = new Tensor(2, 2);
        a[0, 0] = 1f; a[0, 1] = 3f;
        a[1, 0] = 5f; a[1, 1] = 7f;
        var m = TensorMath.Mean(a, axis: 0);
        m[0].Should().BeApproximately(3f, Tolerance);
        m[1].Should().BeApproximately(5f, Tolerance);
    }

    [Fact]
    public void Max_Axis1_CorrectValues()
    {
        var a = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 9f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 2f; a[1, 2] = 6f;
        var m = TensorMath.Max(a, axis: 1);
        m[0].Should().BeApproximately(9f, Tolerance);
        m[1].Should().BeApproximately(6f, Tolerance);
    }

    [Fact]
    public void ArgMax_Axis1_CorrectIndices()
    {
        var a = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 9f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 2f; a[1, 2] = 6f;
        var idx = TensorMath.ArgMax(a, axis: 1);
        idx[0].Should().BeApproximately(1f, Tolerance); // index 1 is max in row 0
        idx[1].Should().BeApproximately(2f, Tolerance); // index 2 is max in row 1
    }

    // ── Softmax ──────────────────────────────────────────────────────────────

    [Fact]
    public void Softmax_RowSumsToOne()
    {
        var a = new Tensor(2, 3);
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 0.1f; a[1, 1] = 0.2f; a[1, 2] = 0.7f;
        var s = TensorMath.Softmax(a, axis: 1);
        float row0sum = s[0, 0] + s[0, 1] + s[0, 2];
        float row1sum = s[1, 0] + s[1, 1] + s[1, 2];
        row0sum.Should().BeApproximately(1f, Tolerance);
        row1sum.Should().BeApproximately(1f, Tolerance);
    }

    [Fact]
    public void Softmax_NumericallyStable_NoNaN()
    {
        var a = new Tensor(1, 3);
        a[0, 0] = 1000f; a[0, 1] = 1001f; a[0, 2] = 1002f;
        var s = TensorMath.Softmax(a, axis: 1);
        foreach (var v in s.Data) float.IsNaN(v).Should().BeFalse();
        (s[0, 0] + s[0, 1] + s[0, 2]).Should().BeApproximately(1f, Tolerance);
    }

    // ── Masking ──────────────────────────────────────────────────────────────

    [Fact]
    public void ApplyMask_FillsNegInfAndKeepsOthers()
    {
        var scores = new Tensor(2, 2);
        scores[0, 0] = 1f; scores[0, 1] = 2f;
        scores[1, 0] = 3f; scores[1, 1] = 4f;

        var mask = TensorMath.CausalMask(2);
        var masked = TensorMath.ApplyMask(scores, mask);

        masked[0, 0].Should().BeApproximately(1f, Tolerance);
        masked[0, 1].Should().Be(float.NegativeInfinity);
        masked[1, 0].Should().BeApproximately(3f, Tolerance);
        masked[1, 1].Should().BeApproximately(4f, Tolerance);
    }

    [Fact]
    public void CausalMask_AfterSoftmax_FuturePositionsAreZero()
    {
        var scores = new Tensor(1, 1, 3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                scores[0, 0, i, j] = 1f;

        var flatMask2D = TensorMath.CausalMask(3);
        // Expand mask to [1,1,3,3]
        var mask = new bool[9];
        Array.Copy(flatMask2D, mask, 9);
        var maskedTensor = new Tensor(1, 1, 3, 3);
        var scoresFlat = scores.Data;
        for (int i = 0; i < 9; i++)
        {
            int b = i / 3, s = i % 3;
            maskedTensor[0, 0, b, s] = mask[i] ? scoresFlat[i] : float.NegativeInfinity;
        }
        var softOut = TensorMath.Softmax(maskedTensor, axis: 3);
        // Position [0,0,0,1] and [0,0,0,2] should be ~0 (masked future)
        softOut[0, 0, 0, 1].Should().BeApproximately(0f, 1e-4f);
        softOut[0, 0, 0, 2].Should().BeApproximately(0f, 1e-4f);
    }
}

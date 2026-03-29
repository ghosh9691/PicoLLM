using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Training;

/// <summary>
/// Numerical gradient checks for backward passes.
/// For each layer: compute analytic grad via Backward(), then approximate
/// with finite differences (f(x+h) - f(x-h)) / 2h and compare.
/// </summary>
public class BackwardPassTests
{
    private const float H = 1e-3f;     // finite-difference step
    private const float Tol = 1e-2f;   // relative tolerance (analytic vs numerical)

    // ── Helper: compute a scalar loss from a tensor (sum of all elements) ──

    private static float ScalarLoss(Tensor t) => t.Data.ToArray().Sum();

    // ── LinearLayer ──────────────────────────────────────────────────────────

    [Fact]
    public void LinearLayer_Backward_GradInput_NumericalCheck()
    {
        var layer = new LinearLayer(4, 3, useBias: true, seed: 1);
        var x = TensorFactory.RandomNormal([2, 4], seed: 42);

        // Analytic grad w.r.t. x
        var y = layer.Forward(x);
        var gradOut = TensorFactory.Ones(2, 3); // dL/dy = 1 for L = sum(y)
        var dxAnalytic = layer.Backward(gradOut).Data.ToArray();

        // Numerical grad w.r.t. x
        var xData = x.Data.ToArray();
        var dxNumerical = new float[xData.Length];
        for (int i = 0; i < xData.Length; i++)
        {
            var xp = (float[])xData.Clone(); xp[i] += H;
            var xm = (float[])xData.Clone(); xm[i] -= H;
            float fp = ScalarLoss(layer.Forward(TensorFactory.FromArray(x.GetShape(), xp)));
            layer.ZeroGrad();
            float fm = ScalarLoss(layer.Forward(TensorFactory.FromArray(x.GetShape(), xm)));
            layer.ZeroGrad();
            dxNumerical[i] = (fp - fm) / (2 * H);
        }

        for (int i = 0; i < dxAnalytic.Length; i++)
            dxAnalytic[i].Should().BeApproximately(dxNumerical[i], 0.05f);
    }

    [Fact]
    public void LinearLayer_Backward_GradWeight_NonZeroAndFinite()
    {
        var layer = new LinearLayer(3, 2, useBias: false, seed: 1);
        var x = TensorFactory.RandomNormal([2, 3], seed: 7);

        layer.Forward(x);
        layer.Backward(TensorFactory.Ones(2, 2));

        var wGradAnalytic = layer.WeightGrad.Data.ToArray();
        wGradAnalytic.Any(g => g != 0f).Should().BeTrue("weight gradients must be non-zero");
        wGradAnalytic.All(float.IsFinite).Should().BeTrue("all weight gradients must be finite");
    }

    // ── LayerNorm ─────────────────────────────────────────────────────────────

    [Fact]
    public void LayerNorm_Backward_GradInput_NumericalCheck()
    {
        var ln = new LayerNorm(6);
        var x = TensorFactory.RandomNormal([2, 6], seed: 3);

        ln.Forward(x);
        var gradOut = TensorFactory.Ones(2, 6);
        var dxAnalytic = ln.Backward(gradOut).Data.ToArray();

        var xData = x.Data.ToArray();
        var dxNumerical = new float[xData.Length];
        for (int i = 0; i < xData.Length; i++)
        {
            var xp = (float[])xData.Clone(); xp[i] += H;
            var xm = (float[])xData.Clone(); xm[i] -= H;
            var ln2 = new LayerNorm(6);
            ln2.Gamma.MutableData.Clear();
            ln.Gamma.Data.CopyTo(ln2.Gamma.MutableData);
            ln2.Beta.MutableData.Clear();
            ln.Beta.Data.CopyTo(ln2.Beta.MutableData);
            float fp = ScalarLoss(ln2.Forward(TensorFactory.FromArray(x.GetShape(), xp)));
            float fm = ScalarLoss(ln2.Forward(TensorFactory.FromArray(x.GetShape(), xm)));
            dxNumerical[i] = (fp - fm) / (2 * H);
        }

        for (int i = 0; i < dxAnalytic.Length; i++)
            dxAnalytic[i].Should().BeApproximately(dxNumerical[i], 0.05f);
    }

    // ── FeedForward ───────────────────────────────────────────────────────────

    [Fact]
    public void FeedForward_Backward_OutputShape()
    {
        var ffn = new FeedForward(8, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 3, 8], seed: 1);
        ffn.Forward(x);
        var dx = ffn.Backward(TensorFactory.Ones(1, 3, 8));
        dx.Shape.ToArray().Should().Equal(1, 3, 8);
    }

    [Fact]
    public void FeedForward_Backward_GradientsAreFinite()
    {
        var ffn = new FeedForward(16, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([2, 4, 16], seed: 2);
        ffn.Forward(x);
        var dx = ffn.Backward(TensorFactory.RandomNormal([2, 4, 16], seed: 3));
        foreach (float v in dx.Data) float.IsFinite(v).Should().BeTrue();
    }

    // ── DecoderBlock ─────────────────────────────────────────────────────────

    [Fact]
    public void DecoderBlock_Backward_OutputShape()
    {
        var block = new DecoderBlock(embedDim: 16, numHeads: 2, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 4, 16], seed: 1);
        block.Forward(x);
        var dx = block.Backward(TensorFactory.Ones(1, 4, 16));
        dx.Shape.ToArray().Should().Equal(1, 4, 16);
    }

    [Fact]
    public void DecoderBlock_Backward_GradientsAreFinite()
    {
        var block = new DecoderBlock(embedDim: 16, numHeads: 4, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 3, 16], seed: 1);
        block.Forward(x);
        var gradOut = TensorFactory.RandomNormal([1, 3, 16], seed: 2);
        var dx = block.Backward(gradOut);
        foreach (float v in dx.Data) float.IsFinite(v).Should().BeTrue();
    }

    [Fact]
    public void DecoderBlock_Backward_AccumulatesParamGrads()
    {
        var block = new DecoderBlock(embedDim: 8, numHeads: 2, ffMultiplier: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 3, 8], seed: 1);
        block.Forward(x);
        block.Backward(TensorFactory.Ones(1, 3, 8));

        bool anyNonZero = block.Parameters().Any(p => p.Grad.Data.ToArray().Any(g => g != 0f));
        anyNonZero.Should().BeTrue("at least some param gradients should be non-zero after backward");
    }
}

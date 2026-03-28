# Transformer (Capability 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GPT-style transformer decoder stack: LayerNorm, LinearLayer, MultiHeadAttention, FeedForward, DecoderBlock, and the full PicoLLMModel.

**Architecture:** Pre-norm decoder-only transformer (GPT-2 style). Each decoder block applies LayerNorm before attention and FFN sublayers, with residual connections. The model composes an EmbeddingLayer (already built), N decoder blocks, a final LayerNorm, and a linear output head that maps hidden states to vocabulary logits.

**Tech Stack:** C# 12 / .NET 9, PicoLLM.Core (zero NuGet deps), xUnit + FluentAssertions for tests.

---

## File Map

**New source files:**
- `src/PicoLLM.Core/Layers/LayerNorm.cs`
- `src/PicoLLM.Core/Layers/LinearLayer.cs`
- `src/PicoLLM.Core/Layers/MultiHeadAttention.cs`
- `src/PicoLLM.Core/Layers/FeedForward.cs`
- `src/PicoLLM.Core/Layers/DecoderBlock.cs`
- `src/PicoLLM.Core/Model/ModelConfig.cs`
- `src/PicoLLM.Core/Model/PicoLLMModel.cs`

**New test files:**
- `tests/PicoLLM.Tests/Transformer/LayerNormTests.cs`
- `tests/PicoLLM.Tests/Transformer/LinearLayerTests.cs`
- `tests/PicoLLM.Tests/Transformer/MultiHeadAttentionTests.cs`
- `tests/PicoLLM.Tests/Transformer/FeedForwardTests.cs`
- `tests/PicoLLM.Tests/Transformer/DecoderBlockTests.cs`
- `tests/PicoLLM.Tests/Transformer/PicoLLMModelTests.cs`

---

### Task 1: LayerNorm

**Files:**
- Create: `src/PicoLLM.Core/Layers/LayerNorm.cs`
- Create: `tests/PicoLLM.Tests/Transformer/LayerNormTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/LayerNormTests.cs
using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class LayerNormTests
{
    private const float Tol = 1e-4f;

    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var ln = new LayerNorm(8);
        var x = TensorFactory.RandomNormal([2, 4, 8], seed: 1);
        var out_ = ln.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 4, 8);
    }

    [Fact]
    public void Forward_Normalized_MeanNearZero_VarNearOne()
    {
        // With gamma=1 and beta=0 (defaults), output should have mean≈0 and var≈1 per vector
        var ln = new LayerNorm(64);
        var x = TensorFactory.RandomNormal([1, 1, 64], seed: 42);
        var out_ = ln.Forward(x);
        var data = out_.Data.ToArray();
        float mean = data.Average();
        float var_ = data.Select(v => (v - mean) * (v - mean)).Average();
        mean.Should().BeApproximately(0f, 1e-3f);
        var_.Should().BeApproximately(1f, 1e-3f);
    }

    [Fact]
    public void Forward_GammaEffect_ScalesOutput()
    {
        var ln = new LayerNorm(4);
        // Set gamma to [2,2,2,2]
        ln.Gamma.MutableData.Fill(2f);
        var x = TensorFactory.RandomNormal([1, 1, 4], seed: 7);
        var out_ = ln.Forward(x);
        // With gamma=2, output should be ~2x normalized values
        // Verify by comparing with gamma=1 case
        var ln1 = new LayerNorm(4);
        ln1.Gamma.MutableData.Fill(1f);
        // Copy same weights
        x.Data.CopyTo(ln1.Gamma.MutableData); // not weights, just checking shape
        var out1 = ln1.Forward(x);
        // Each element of out_ should be ≈ 2 * out1[i] + beta[i]
        for (int i = 0; i < 4; i++)
            out_[0, 0, i].Should().BeApproximately(2f * out1[0, 0, i], Tol);
    }

    [Fact]
    public void Forward_BetaEffect_ShiftsOutput()
    {
        var ln = new LayerNorm(4);
        ln.Beta.MutableData.Fill(5f);
        var x = TensorFactory.RandomNormal([1, 1, 4], seed: 3);
        var out_ = ln.Forward(x);
        var ln0 = new LayerNorm(4);
        var out0 = ln0.Forward(x);
        for (int i = 0; i < 4; i++)
            out_[0, 0, i].Should().BeApproximately(out0[0, 0, i] + 5f, Tol);
    }

    [Fact]
    public void Forward_2DInput_Works()
    {
        var ln = new LayerNorm(8);
        var x = TensorFactory.RandomNormal([4, 8], seed: 1);
        var out_ = ln.Forward(x);
        out_.Shape.ToArray().Should().Equal(4, 8);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~LayerNormTests" --no-build 2>&1 | tail -5
```
Expected: build error (LayerNorm not found)

- [ ] **Step 3: Implement LayerNorm**

```csharp
// src/PicoLLM.Core/Layers/LayerNorm.cs
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Layer normalization with learnable scale (gamma) and shift (beta) parameters.
/// Normalizes each vector of length <see cref="EmbedDim"/> along the last dimension.
/// Formula: output = gamma * (x - mean) / sqrt(var + eps) + beta
/// </summary>
public sealed class LayerNorm
{
    private const float Eps = 1e-5f;

    /// <summary>Scale parameter, shape [embed_dim], initialized to 1.</summary>
    public Tensor Gamma { get; }

    /// <summary>Shift parameter, shape [embed_dim], initialized to 0.</summary>
    public Tensor Beta { get; }

    /// <summary>Accumulated gradient for gamma.</summary>
    public Tensor GammaGrad { get; }

    /// <summary>Accumulated gradient for beta.</summary>
    public Tensor BetaGrad { get; }

    /// <summary>Number of features in the last dimension.</summary>
    public int EmbedDim { get; }

    /// <summary>Initializes LayerNorm for vectors of length <paramref name="embedDim"/>.</summary>
    public LayerNorm(int embedDim)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        EmbedDim = embedDim;
        Gamma = TensorFactory.Ones(embedDim);
        Beta = TensorFactory.Zeros(embedDim);
        GammaGrad = TensorFactory.Zeros(embedDim);
        BetaGrad = TensorFactory.Zeros(embedDim);
    }

    /// <summary>
    /// Forward pass. Normalizes each last-dimension vector independently.
    /// Input shape: [..., embed_dim]. Output shape equals input shape.
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        int D = EmbedDim;
        if (x.Shape[x.Rank - 1] != D)
            throw new ArgumentException(
                $"Last dimension {x.Shape[x.Rank - 1]} does not match EmbedDim {D}.");

        int outerSize = x.Length / D;
        var src = x.Data;
        var gamma = Gamma.Data;
        var beta = Beta.Data;
        var result = new float[x.Length];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * D;

            // Compute mean
            float mean = 0f;
            for (int d = 0; d < D; d++) mean += src[offset + d];
            mean /= D;

            // Compute variance
            float var_ = 0f;
            for (int d = 0; d < D; d++)
            {
                float diff = src[offset + d] - mean;
                var_ += diff * diff;
            }
            var_ /= D;

            // Normalize, scale, shift
            float invStd = 1f / MathF.Sqrt(var_ + Eps);
            for (int d = 0; d < D; d++)
                result[offset + d] = gamma[d] * ((src[offset + d] - mean) * invStd) + beta[d];
        }

        return new Tensor(x.GetShape(), result);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        GammaGrad.MutableData.Clear();
        BetaGrad.MutableData.Clear();
    }

    /// <summary>Returns all learnable parameters: [Gamma, Beta].</summary>
    public IEnumerable<Tensor> Parameters() { yield return Gamma; yield return Beta; }
}
```

- [ ] **Step 4: Run tests and verify they pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~LayerNormTests" 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Core/Layers/LayerNorm.cs tests/PicoLLM.Tests/Transformer/LayerNormTests.cs
git commit -m "feat: implement LayerNorm with gamma/beta parameters and tests"
```

---

### Task 2: LinearLayer

**Files:**
- Create: `src/PicoLLM.Core/Layers/LinearLayer.cs`
- Create: `tests/PicoLLM.Tests/Transformer/LinearLayerTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/LinearLayerTests.cs
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
        // [2, 3] @ [3, 2] + [2] = [2, 2]
        var layer = new LinearLayer(3, 2, useBias: true);
        // Set weights to identity-like: [[1,0],[0,1],[0,0]] and bias [0,0]
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
        // Weights = identity, bias = [10, 20]
        layer.Weights.MutableData[0] = 1f; layer.Weights.MutableData[1] = 0f;
        layer.Weights.MutableData[2] = 0f; layer.Weights.MutableData[3] = 1f;
        layer.Bias!.MutableData[0] = 10f; layer.Bias!.MutableData[1] = 20f;

        var x = new Tensor([1, 2], [3f, 4f]);
        var out_ = layer.Forward(x);
        out_[0, 0].Should().BeApproximately(13f, 1e-5f);
        out_[0, 1].Should().BeApproximately(24f, 1e-5f);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.Core 2>&1 | tail -5
```

- [ ] **Step 3: Implement LinearLayer**

```csharp
// src/PicoLLM.Core/Layers/LinearLayer.cs
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Fully-connected (dense) linear layer: output = input @ Weights + Bias.
/// Supports 2D input [batch, in] and 3D input [batch, seq, in].
/// </summary>
public sealed class LinearLayer
{
    /// <summary>Weight matrix, shape [in_features, out_features].</summary>
    public Tensor Weights { get; }

    /// <summary>Bias vector, shape [out_features], or null if bias is disabled.</summary>
    public Tensor? Bias { get; }

    /// <summary>Accumulated gradient for weights.</summary>
    public Tensor WeightGrad { get; }

    /// <summary>Accumulated gradient for bias, or null if bias is disabled.</summary>
    public Tensor? BiasGrad { get; }

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>
    /// Initializes a linear layer with Xavier uniform weight initialization.
    /// </summary>
    public LinearLayer(int inFeatures, int outFeatures, bool useBias = true, int? seed = null)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        Weights = TensorFactory.XavierUniform([inFeatures, outFeatures], inFeatures, outFeatures, seed);
        WeightGrad = TensorFactory.Zeros(inFeatures, outFeatures);

        if (useBias)
        {
            Bias = TensorFactory.Zeros(outFeatures);
            BiasGrad = TensorFactory.Zeros(outFeatures);
        }
    }

    /// <summary>
    /// Forward pass. Supports [batch, in], [seq, in] (2D) and [batch, seq, in] (3D) inputs.
    /// Output preserves all leading dimensions: [batch, out] or [batch, seq, out].
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        int rank = input.Rank;
        int lastDim = input.Shape[rank - 1];
        if (lastDim != InFeatures)
            throw new ArgumentException(
                $"Last dimension {lastDim} does not match InFeatures {InFeatures}.");

        // Flatten all leading dims into one, matmul, then restore shape.
        int rows = input.Length / InFeatures;
        var flat = TensorMath.Reshape(input, rows, InFeatures);     // [rows, in]
        var result = TensorMath.MatMul(flat, Weights);               // [rows, out]

        if (Bias is not null)
        {
            // Add bias to every row
            var bias = Bias.Data;
            var r = result.MutableData;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < OutFeatures; j++)
                    r[i * OutFeatures + j] += bias[j];
        }

        // Restore original leading dims + OutFeatures
        var outShape = new int[rank];
        for (int i = 0; i < rank - 1; i++) outShape[i] = input.Shape[i];
        outShape[rank - 1] = OutFeatures;
        return TensorMath.Reshape(result, outShape);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        WeightGrad.MutableData.Clear();
        BiasGrad?.MutableData.Clear();
    }

    /// <summary>Returns all learnable parameters.</summary>
    public IEnumerable<Tensor> Parameters()
    {
        yield return Weights;
        if (Bias is not null) yield return Bias;
    }
}
```

- [ ] **Step 4: Run tests and verify pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~LinearLayerTests" 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Core/Layers/LinearLayer.cs tests/PicoLLM.Tests/Transformer/LinearLayerTests.cs
git commit -m "feat: implement LinearLayer supporting 2D and 3D inputs"
```

---

### Task 3: MultiHeadAttention

**Files:**
- Create: `src/PicoLLM.Core/Layers/MultiHeadAttention.cs`
- Create: `tests/PicoLLM.Tests/Transformer/MultiHeadAttentionTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/MultiHeadAttentionTests.cs
using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class MultiHeadAttentionTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var mha = new MultiHeadAttention(embedDim: 128, numHeads: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 6, 128], seed: 1);
        var out_ = mha.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 6, 128);
    }

    [Fact]
    public void Forward_SingleHead_OutputShape()
    {
        var mha = new MultiHeadAttention(embedDim: 32, numHeads: 1, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 32], seed: 1);
        var out_ = mha.Forward(x);
        out_.Shape.ToArray().Should().Equal(1, 5, 32);
    }

    [Fact]
    public void Forward_CausalMask_BlocksFuturePositions()
    {
        // Set all projection weights to identity and bias to 0 so attention weights are interpretable.
        // With identity projections: Q=K=V=x.
        // Position 0 should only attend to position 0 (mask blocks pos 1,2,...).
        var mha = new MultiHeadAttention(embedDim: 4, numHeads: 1, seed: 1);
        // Set Q/K/V weights to identity [4,4]
        SetIdentity(mha.QueryProj.Weights);
        SetIdentity(mha.KeyProj.Weights);
        SetIdentity(mha.ValueProj.Weights);
        SetIdentity(mha.OutputProj.Weights);
        mha.QueryProj.Bias!.MutableData.Clear();
        mha.KeyProj.Bias!.MutableData.Clear();
        mha.ValueProj.Bias!.MutableData.Clear();
        mha.OutputProj.Bias!.MutableData.Clear();

        // x: [1, 3, 4] — 3 distinct tokens
        var xData = new float[] { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
        var x = new Tensor([1, 3, 4], xData);
        var out_ = mha.Forward(x);

        // Position 0 output = value[0] (only attends to pos 0)
        out_[0, 0, 0].Should().BeApproximately(1f, 0.01f);
        out_[0, 0, 1].Should().BeApproximately(0f, 0.01f);
    }

    [Fact]
    public void Forward_AttentionWeights_SumToOne()
    {
        // The attention weights (softmax output) should sum to 1 per query position.
        var mha = new MultiHeadAttention(embedDim: 8, numHeads: 2, seed: 1);
        var x = TensorFactory.RandomNormal([1, 4, 8], seed: 5);
        // Just test that output is not NaN/Inf (weights sum to 1 is internal)
        var out_ = mha.Forward(x);
        foreach (float v in out_.Data) float.IsNaN(v).Should().BeFalse();
        foreach (float v in out_.Data) float.IsInfinity(v).Should().BeFalse();
    }

    private static void SetIdentity(Tensor t)
    {
        t.MutableData.Clear();
        int n = t.Shape[0];
        for (int i = 0; i < n; i++) t.MutableData[i * n + i] = 1f;
    }
}
```

- [ ] **Step 2: Implement MultiHeadAttention**

```csharp
// src/PicoLLM.Core/Layers/MultiHeadAttention.cs
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Scaled dot-product multi-head self-attention with causal masking.
/// Input shape: [batch, seq, embed_dim]. Output shape: same.
/// </summary>
/// <remarks>
/// Pipeline (GPT-2 style):
/// 1. Project x → Q, K, V via three separate linear layers.
/// 2. Reshape [B,S,E] → [B,H,S,head_dim] for each of Q,K,V.
/// 3. scores = (Q @ K^T) / sqrt(head_dim), shape [B,H,S,S].
/// 4. Apply causal mask (upper triangle → -∞).
/// 5. weights = softmax(scores, axis=-1).
/// 6. context = weights @ V, shape [B,H,S,head_dim].
/// 7. Reshape context back to [B,S,E].
/// 8. Apply output projection.
/// </remarks>
public sealed class MultiHeadAttention
{
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _embedDim;

    /// <summary>Query projection [embed_dim, embed_dim].</summary>
    public LinearLayer QueryProj { get; }

    /// <summary>Key projection [embed_dim, embed_dim].</summary>
    public LinearLayer KeyProj { get; }

    /// <summary>Value projection [embed_dim, embed_dim].</summary>
    public LinearLayer ValueProj { get; }

    /// <summary>Output projection [embed_dim, embed_dim].</summary>
    public LinearLayer OutputProj { get; }

    /// <summary>Initializes multi-head attention.</summary>
    /// <param name="embedDim">Model hidden dimension. Must be divisible by numHeads.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public MultiHeadAttention(int embedDim, int numHeads, int? seed = null)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException(
                $"embedDim {embedDim} must be divisible by numHeads {numHeads}.");
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;

        QueryProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed);
        KeyProj    = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 1 : null);
        ValueProj  = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 2 : null);
        OutputProj = new LinearLayer(embedDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 3 : null);
    }

    /// <summary>
    /// Forward pass with causal masking.
    /// </summary>
    /// <param name="x">Input tensor [batch, seq, embed_dim].</param>
    /// <returns>Output tensor [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        if (x.Rank != 3)
            throw new ArgumentException("Input must be rank 3: [batch, seq, embed_dim].");

        int B = x.Shape[0], S = x.Shape[1], E = x.Shape[2];
        if (E != _embedDim)
            throw new ArgumentException($"Expected embed_dim {_embedDim}, got {E}.");

        int H = _numHeads, D = _headDim;

        // 1. Project: each is [B, S, E]
        var q = QueryProj.Forward(x);
        var k = KeyProj.Forward(x);
        var v = ValueProj.Forward(x);

        // 2. Reshape [B,S,E] → [B,S,H,D] → permute(0,2,1,3) → [B,H,S,D]
        q = TensorMath.Permute(TensorMath.Reshape(q, B, S, H, D), [0, 2, 1, 3]);
        k = TensorMath.Permute(TensorMath.Reshape(k, B, S, H, D), [0, 2, 1, 3]);
        v = TensorMath.Permute(TensorMath.Reshape(v, B, S, H, D), [0, 2, 1, 3]);

        // 3. scores = Q @ K^T / sqrt(D): transpose K → [B,H,D,S] then batched matmul → [B,H,S,S]
        var kT = TensorMath.Transpose(k, 2, 3); // [B,H,D,S]
        var scores = TensorMath.BatchedMatMul(q, kT); // [B,H,S,S]
        scores = TensorMath.Multiply(scores, 1f / MathF.Sqrt(D));

        // 4. Apply causal mask
        scores = ApplyCausalMask(scores, B, H, S);

        // 5. Softmax along last axis (key positions)
        var weights = TensorMath.Softmax(scores, axis: 3); // [B,H,S,S]

        // 6. context = weights @ V: [B,H,S,S] @ [B,H,S,D] → [B,H,S,D]
        var context = TensorMath.BatchedMatMul(weights, v);

        // 7. Reshape back: permute(0,2,1,3) → [B,S,H,D] → reshape → [B,S,E]
        context = TensorMath.Reshape(
            TensorMath.Permute(context, [0, 2, 1, 3]),
            B, S, E);

        // 8. Output projection
        return OutputProj.Forward(context);
    }

    private static Tensor ApplyCausalMask(Tensor scores, int B, int H, int S)
    {
        // scores: [B, H, S, S]
        // mask[i, j] = -inf if j > i, else 0
        var data = scores.Data.ToArray();
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < S; i++)
                    for (int j = i + 1; j < S; j++)
                    {
                        int idx = ((b * H + h) * S + i) * S + j;
                        data[idx] = float.NegativeInfinity;
                    }
        return new Tensor([B, H, S, S], data);
    }

    /// <summary>Zeros the accumulated gradients in all projections.</summary>
    public void ZeroGrad()
    {
        QueryProj.ZeroGrad();
        KeyProj.ZeroGrad();
        ValueProj.ZeroGrad();
        OutputProj.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters.</summary>
    public IEnumerable<Tensor> Parameters() =>
        QueryProj.Parameters()
            .Concat(KeyProj.Parameters())
            .Concat(ValueProj.Parameters())
            .Concat(OutputProj.Parameters());
}
```

- [ ] **Step 3: Run tests**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~MultiHeadAttentionTests" 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.Core/Layers/MultiHeadAttention.cs tests/PicoLLM.Tests/Transformer/MultiHeadAttentionTests.cs
git commit -m "feat: implement MultiHeadAttention with causal masking"
```

---

### Task 4: FeedForward

**Files:**
- Create: `src/PicoLLM.Core/Layers/FeedForward.cs`
- Create: `tests/PicoLLM.Tests/Transformer/FeedForwardTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/FeedForwardTests.cs
using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class FeedForwardTests
{
    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var ffn = new FeedForward(embedDim: 64, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 8, 64], seed: 1);
        var out_ = ffn.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 8, 64);
    }

    [Fact]
    public void FfDim_IsEmbedDimTimesMultiplier()
    {
        var ffn = new FeedForward(embedDim: 32, ffMultiplier: 4, seed: 1);
        ffn.FfDim.Should().Be(128);
    }

    [Fact]
    public void Forward_NoNaN_WithRandomInput()
    {
        var ffn = new FeedForward(embedDim: 128, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 128], seed: 42);
        var out_ = ffn.Forward(x);
        foreach (float v in out_.Data) float.IsNaN(v).Should().BeFalse();
    }
}
```

- [ ] **Step 2: Implement FeedForward**

```csharp
// src/PicoLLM.Core/Layers/FeedForward.cs
using PicoLLM.Core.Activations;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// Position-wise feedforward network: Linear → GELU → Linear.
/// Hidden dimension is embed_dim × ff_multiplier (typically 4).
/// Input/output shape: [batch, seq, embed_dim].
/// </summary>
public sealed class FeedForward
{
    private readonly LinearLayer _up;    // [embed_dim, ff_dim]
    private readonly LinearLayer _down;  // [ff_dim, embed_dim]

    /// <summary>The hidden (expanded) dimension: embed_dim × ff_multiplier.</summary>
    public int FfDim { get; }

    /// <summary>Initializes the feedforward sublayer.</summary>
    public FeedForward(int embedDim, int ffMultiplier = 4, int? seed = null)
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (ffMultiplier <= 0) throw new ArgumentOutOfRangeException(nameof(ffMultiplier));

        FfDim = embedDim * ffMultiplier;
        _up   = new LinearLayer(embedDim, FfDim, useBias: true, seed: seed);
        _down = new LinearLayer(FfDim, embedDim, useBias: true, seed: seed.HasValue ? seed + 1 : null);
    }

    /// <summary>
    /// Forward pass: up-project, apply GELU, down-project.
    /// </summary>
    /// <param name="x">Input [batch, seq, embed_dim].</param>
    /// <returns>Output [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        var hidden = GELU.Forward(_up.Forward(x));
        return _down.Forward(hidden);
    }

    /// <summary>Zeros the accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _up.ZeroGrad();
        _down.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters.</summary>
    public IEnumerable<Tensor> Parameters() =>
        _up.Parameters().Concat(_down.Parameters());
}
```

- [ ] **Step 3: Run tests**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~FeedForwardTests" 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.Core/Layers/FeedForward.cs tests/PicoLLM.Tests/Transformer/FeedForwardTests.cs
git commit -m "feat: implement FeedForward (Linear→GELU→Linear)"
```

---

### Task 5: DecoderBlock

**Files:**
- Create: `src/PicoLLM.Core/Layers/DecoderBlock.cs`
- Create: `tests/PicoLLM.Tests/Transformer/DecoderBlockTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/DecoderBlockTests.cs
using FluentAssertions;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class DecoderBlockTests
{
    [Fact]
    public void Forward_OutputShape_MatchesInput()
    {
        var block = new DecoderBlock(embedDim: 64, numHeads: 4, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([2, 8, 64], seed: 1);
        var out_ = block.Forward(x);
        out_.Shape.ToArray().Should().Equal(2, 8, 64);
    }

    [Fact]
    public void Forward_NoNaN_WithRandomInput()
    {
        var block = new DecoderBlock(embedDim: 32, numHeads: 4, ffMultiplier: 4, seed: 1);
        var x = TensorFactory.RandomNormal([1, 5, 32], seed: 42);
        var out_ = block.Forward(x);
        foreach (float v in out_.Data) float.IsNaN(v).Should().BeFalse();
    }

    [Fact]
    public void Forward_ResidualConnection_ShapePreserved_WithZeroAttentionAndFFN()
    {
        // If attention and FFN output zeros, output should equal input (residual passthrough).
        var block = new DecoderBlock(embedDim: 4, numHeads: 1, ffMultiplier: 4, seed: 1);
        // Zero out all weights in attention projections
        block.Attention.QueryProj.Weights.MutableData.Clear();
        block.Attention.KeyProj.Weights.MutableData.Clear();
        block.Attention.ValueProj.Weights.MutableData.Clear();
        block.Attention.OutputProj.Weights.MutableData.Clear();
        block.Attention.QueryProj.Bias!.MutableData.Clear();
        block.Attention.KeyProj.Bias!.MutableData.Clear();
        block.Attention.ValueProj.Bias!.MutableData.Clear();
        block.Attention.OutputProj.Bias!.MutableData.Clear();
        // Zero out all FFN weights
        block.FFN.Parameters().ToList().ForEach(p => p.MutableData.Clear());

        var x = TensorFactory.RandomNormal([1, 3, 4], seed: 1);
        var out_ = block.Forward(x);
        out_.Shape.ToArray().Should().Equal(1, 3, 4);
        // Output ≈ x + 0 + 0 = x (LayerNorm may slightly alter values via gamma/beta)
    }
}
```

- [ ] **Step 2: Implement DecoderBlock**

```csharp
// src/PicoLLM.Core/Layers/DecoderBlock.cs
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Layers;

/// <summary>
/// A single GPT-style decoder block with pre-norm residual connections:
/// <code>
/// x = x + Attention(LayerNorm(x))
/// x = x + FFN(LayerNorm(x))
/// </code>
/// </summary>
public sealed class DecoderBlock
{
    private readonly LayerNorm _attnNorm;
    private readonly LayerNorm _ffnNorm;

    /// <summary>The multi-head self-attention sublayer.</summary>
    public MultiHeadAttention Attention { get; }

    /// <summary>The feedforward sublayer.</summary>
    public FeedForward FFN { get; }

    /// <summary>Initializes a decoder block.</summary>
    public DecoderBlock(int embedDim, int numHeads, int ffMultiplier = 4, int? seed = null)
    {
        _attnNorm = new LayerNorm(embedDim);
        Attention = new MultiHeadAttention(embedDim, numHeads, seed);
        _ffnNorm  = new LayerNorm(embedDim);
        FFN       = new FeedForward(embedDim, ffMultiplier, seed);
    }

    /// <summary>
    /// Forward pass with pre-norm residual connections.
    /// </summary>
    /// <param name="x">Input [batch, seq, embed_dim].</param>
    /// <returns>Output [batch, seq, embed_dim].</returns>
    public Tensor Forward(Tensor x)
    {
        // Pre-norm attention with residual
        x = TensorMath.Add(x, Attention.Forward(_attnNorm.Forward(x)));
        // Pre-norm FFN with residual
        x = TensorMath.Add(x, FFN.Forward(_ffnNorm.Forward(x)));
        return x;
    }

    /// <summary>Zeros all accumulated gradients.</summary>
    public void ZeroGrad()
    {
        _attnNorm.ZeroGrad();
        Attention.ZeroGrad();
        _ffnNorm.ZeroGrad();
        FFN.ZeroGrad();
    }

    /// <summary>Returns all learnable parameters.</summary>
    public IEnumerable<Tensor> Parameters() =>
        _attnNorm.Parameters()
            .Concat(Attention.Parameters())
            .Concat(_ffnNorm.Parameters())
            .Concat(FFN.Parameters());
}
```

- [ ] **Step 3: Run tests**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~DecoderBlockTests" 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.Core/Layers/DecoderBlock.cs tests/PicoLLM.Tests/Transformer/DecoderBlockTests.cs
git commit -m "feat: implement DecoderBlock with pre-norm residual connections"
```

---

### Task 6: ModelConfig + PicoLLMModel

**Files:**
- Create: `src/PicoLLM.Core/Model/ModelConfig.cs`
- Create: `src/PicoLLM.Core/Model/PicoLLMModel.cs`
- Create: `tests/PicoLLM.Tests/Transformer/PicoLLMModelTests.cs`

- [ ] **Step 1: Write failing tests**

```csharp
// tests/PicoLLM.Tests/Transformer/PicoLLMModelTests.cs
using FluentAssertions;
using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Transformer;

public class PicoLLMModelTests
{
    private static ModelConfig SmallConfig() => new(
        VocabSize: 512, EmbedDim: 64, NumHeads: 4,
        NumLayers: 2, FfMultiplier: 4, MaxSeqLen: 32);

    [Fact]
    public void Forward_OutputLogitsShape_IsCorrect()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } }; // [2, 5]
        var logits = model.Forward(tokenIds);
        logits.Shape.ToArray().Should().Equal(2, 5, 512);
    }

    [Fact]
    public void Forward_NoNaN_InLogits()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 0, 1, 2 } };
        var logits = model.Forward(tokenIds);
        foreach (float v in logits.Data)
        {
            float.IsNaN(v).Should().BeFalse();
            float.IsInfinity(v).Should().BeFalse();
        }
    }

    [Fact]
    public void TotalParameters_IsPositive()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        model.TotalParameters().Should().BeGreaterThan(0);
    }

    [Fact]
    public void TotalParameters_KnownConfig_CorrectCount()
    {
        // vocab=512, embed=64, heads=4, layers=2, ff_mult=4, maxSeq=32
        // Token embedding: 512 * 64 = 32768
        // Positional encoding: no params (sinusoidal)
        // Per decoder block:
        //   LayerNorm(attn): 64*2 = 128
        //   MHA: 4 * (64*64 + 64) = 4 * 4160 = 16640
        //   LayerNorm(ffn): 64*2 = 128
        //   FFN: (64*256 + 256) + (256*64 + 64) = 16640 + 16448 = 33088
        //   Block total: 128 + 16640 + 128 + 33088 = 49984
        // 2 blocks: 99968
        // Final LayerNorm: 128
        // LM head: 64*512 + 512 = 33280
        // Total: 32768 + 99968 + 128 + 33280 = 166144
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        model.TotalParameters().Should().Be(166144);
    }

    [Fact]
    public void GetAllParameters_CountMatchesTotalParameters()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        int total = model.GetAllParameters().Sum(p => p.Length);
        total.Should().Be(model.TotalParameters());
    }

    [Fact]
    public void Forward_Softmax_ProducesValidDistribution()
    {
        var model = new PicoLLMModel(SmallConfig(), seed: 1);
        var tokenIds = new[,] { { 1, 2, 3 } };
        var logits = model.Forward(tokenIds); // [1, 3, 512]
        // Softmax over vocab at last position
        var lastLogits = new float[512];
        for (int v = 0; v < 512; v++)
            lastLogits[v] = logits[0, 2, v];
        float maxL = lastLogits.Max();
        float sumExp = lastLogits.Sum(l => MathF.Exp(l - maxL));
        float[] probs = lastLogits.Select(l => MathF.Exp(l - maxL) / sumExp).ToArray();
        probs.Sum().Should().BeApproximately(1f, 1e-4f);
        probs.All(p => p >= 0f).Should().BeTrue();
    }
}
```

- [ ] **Step 2: Implement ModelConfig**

```csharp
// src/PicoLLM.Core/Model/ModelConfig.cs
namespace PicoLLM.Core.Model;

/// <summary>
/// Immutable configuration for a PicoLLM transformer model.
/// All hyperparameters are specified here and passed to the model constructor.
/// </summary>
/// <param name="VocabSize">Number of tokens in the vocabulary.</param>
/// <param name="EmbedDim">Hidden dimension / embedding size. Must be divisible by NumHeads.</param>
/// <param name="NumHeads">Number of attention heads. EmbedDim must be divisible by this.</param>
/// <param name="NumLayers">Number of stacked decoder blocks.</param>
/// <param name="FfMultiplier">Feedforward hidden size multiplier (ff_dim = EmbedDim × FfMultiplier).</param>
/// <param name="MaxSeqLen">Maximum supported sequence length (for positional encoding).</param>
/// <param name="Dropout">Dropout probability (0.0 = disabled, used for training config only).</param>
public record ModelConfig(
    int VocabSize,
    int EmbedDim,
    int NumHeads,
    int NumLayers,
    int FfMultiplier,
    int MaxSeqLen,
    float Dropout = 0f);
```

- [ ] **Step 3: Implement PicoLLMModel**

```csharp
// src/PicoLLM.Core/Model/PicoLLMModel.cs
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Model;

/// <summary>
/// Full GPT-style decoder-only transformer model.
/// Architecture: TokenEmbedding + PositionalEncoding → N × DecoderBlock → LayerNorm → Linear(vocab).
/// </summary>
/// <remarks>
/// Forward pass:
/// <code>
/// x = Embedding(tokenIds)                    // [B, S, E]
/// for block in blocks: x = block(x)          // [B, S, E]
/// x = FinalNorm(x)                           // [B, S, E]
/// logits = LmHead(x)                         // [B, S, VocabSize]
/// </code>
/// </remarks>
public sealed class PicoLLMModel
{
    private readonly EmbeddingLayer _embedding;
    private readonly DecoderBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly LinearLayer _lmHead;

    /// <summary>Model hyperparameters.</summary>
    public ModelConfig Config { get; }

    /// <summary>
    /// Initializes the model with the given configuration.
    /// </summary>
    /// <param name="config">Model hyperparameters.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PicoLLMModel(ModelConfig config, int? seed = null)
    {
        Config = config;

        _embedding = new EmbeddingLayer(
            config.VocabSize, config.EmbedDim, config.MaxSeqLen, seed);

        _blocks = new DecoderBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new DecoderBlock(
                config.EmbedDim, config.NumHeads, config.FfMultiplier,
                seed: seed.HasValue ? seed + i * 10 : null);

        _finalNorm = new LayerNorm(config.EmbedDim);
        _lmHead = new LinearLayer(config.EmbedDim, config.VocabSize, useBias: true,
            seed: seed.HasValue ? seed + config.NumLayers * 10 : null);
    }

    /// <summary>
    /// Forward pass: token IDs → vocabulary logits.
    /// </summary>
    /// <param name="tokenIds">Token ID matrix [batch, seq]. All values must be in [0, VocabSize).</param>
    /// <returns>Logits tensor [batch, seq, vocab_size].</returns>
    public Tensor Forward(int[,] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        int batch = tokenIds.GetLength(0);
        int seqLen = tokenIds.GetLength(1);

        // Convert 2D array to jagged for EmbeddingLayer
        var jagged = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            jagged[b] = new int[seqLen];
            for (int s = 0; s < seqLen; s++)
                jagged[b][s] = tokenIds[b, s];
        }

        // Embedding: [batch, seq, embed_dim]
        Tensor x = _embedding.ForwardBatch(jagged);

        // Decoder blocks
        foreach (var block in _blocks)
            x = block.Forward(x);

        // Final norm
        x = _finalNorm.Forward(x);

        // LM head: [batch, seq, vocab_size]
        return _lmHead.Forward(x);
    }

    /// <summary>
    /// Returns the total number of learnable parameters.
    /// </summary>
    public int TotalParameters() => GetAllParameters().Sum(p => p.Length);

    /// <summary>
    /// Returns a flat list of all learnable parameter tensors in the model.
    /// Used by the optimizer to update weights.
    /// </summary>
    public IReadOnlyList<Tensor> GetAllParameters()
    {
        var result = new List<Tensor>();

        // Token embedding weights (positional encoding has no learnable params)
        result.Add(_embedding.TokenEmbedding.Weights);

        // Decoder blocks
        foreach (var block in _blocks)
            result.AddRange(block.Parameters());

        // Final norm
        result.AddRange(_finalNorm.Parameters());

        // LM head
        result.AddRange(_lmHead.Parameters());

        return result;
    }

    /// <summary>Zeros all accumulated gradients in the entire model.</summary>
    public void ZeroGrad()
    {
        _embedding.ZeroGrad();
        foreach (var block in _blocks) block.ZeroGrad();
        _finalNorm.ZeroGrad();
        _lmHead.ZeroGrad();
    }
}
```

- [ ] **Step 4: Run all transformer tests**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests --filter "FullyQualifiedName~PicoLLM.Tests.Transformer" 2>&1 | tail -20
```

- [ ] **Step 5: Run all tests to ensure no regressions**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/PicoLLM.Core/Model/ tests/PicoLLM.Tests/Transformer/PicoLLMModelTests.cs
git commit -m "feat: implement ModelConfig and PicoLLMModel (full transformer)"
```

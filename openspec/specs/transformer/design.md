# Transformer — Technical Design

## Architecture Overview (GPT-2 style decoder-only)

```
Input Token IDs
    │
    ▼
[Token Embedding + Positional Encoding]
    │
    ▼
┌─────────────────────────────┐
│   Decoder Block × N         │
│  ┌────────────────────────┐ │
│  │ LayerNorm              │ │
│  │ Multi-Head Attention   │ │
│  │ + Residual Connection  │ │
│  ├────────────────────────┤ │
│  │ LayerNorm              │ │
│  │ FFN (Linear→GELU→Linear│ │
│  │ + Residual Connection  │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
[Final LayerNorm]
    │
    ▼
[Linear → vocab_size] (logits)
```

## Layer Normalization

```csharp
public class LayerNorm
{
    public Tensor Gamma; // [embed_dim], init to 1.0
    public Tensor Beta;  // [embed_dim], init to 0.0
    private const float Eps = 1e-5f;

    public Tensor Forward(Tensor x)
    {
        // mean and var along last dim
        // normalized = (x - mean) / sqrt(var + eps)
        // return gamma * normalized + beta
    }
}
```

## Multi-Head Self-Attention

Key classes:
```csharp
public class LinearLayer  // no bias option for Q/K/V projections
{
    public Tensor Weights; // [in_features, out_features]
    public Tensor? Bias;   // [out_features] or null
}

public class MultiHeadAttention
{
    private LinearLayer _queryProj;  // [embed_dim, embed_dim]
    private LinearLayer _keyProj;    // [embed_dim, embed_dim]
    private LinearLayer _valueProj;  // [embed_dim, embed_dim]
    private LinearLayer _outputProj; // [embed_dim, embed_dim]
    private int _numHeads;
    private int _headDim;            // embed_dim / num_heads
}
```

### Attention Computation Steps

1. Project: Q = x @ Wq, K = x @ Wk, V = x @ Wv — each [batch, seq, embed_dim]
2. Reshape: [batch, seq, embed_dim] → [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
3. Scores: attn = (Q @ K^T) / sqrt(head_dim) — [batch, heads, seq, seq]
4. Mask: upper triangle → -∞ (causal)
5. Weights: softmax(attn) — [batch, heads, seq, seq]
6. Context: weights @ V — [batch, heads, seq, head_dim]
7. Concat: [batch, heads, seq, head_dim] → [batch, seq, embed_dim]
8. Project: output @ Wo — [batch, seq, embed_dim]

## Feedforward Network

```csharp
public class FeedForward
{
    private LinearLayer _up;   // [embed_dim, ff_dim]
    private LinearLayer _down; // [ff_dim, embed_dim]

    public Tensor Forward(Tensor x)
    {
        var hidden = TensorMath.GELU(_up.Forward(x));
        return _down.Forward(hidden);
    }
}
```

Where ff_dim = embed_dim × ff_multiplier (typically 4).

## Decoder Block

```csharp
public class DecoderBlock
{
    private LayerNorm _attnNorm;
    private MultiHeadAttention _attn;
    private LayerNorm _ffnNorm;
    private FeedForward _ffn;

    public Tensor Forward(Tensor x)
    {
        // Pre-norm residual
        x = TensorMath.Add(x, _attn.Forward(_attnNorm.Forward(x)));
        x = TensorMath.Add(x, _ffn.Forward(_ffnNorm.Forward(x)));
        return x;
    }
}
```

## Full Model

```csharp
public class PicoLLMModel
{
    private EmbeddingLayer _embedding;
    private DecoderBlock[] _blocks;
    private LayerNorm _finalNorm;
    private LinearLayer _lmHead;  // [embed_dim, vocab_size]
    public ModelConfig Config { get; }

    public Tensor Forward(int[,] tokenIds) { ... }
    public int TotalParameters() { ... }
}

public record ModelConfig(
    int VocabSize, int EmbedDim, int NumHeads,
    int NumLayers, int FfMultiplier, int MaxSeqLen);
```

## Weight Sharing (Optional)

The output head (lm_head) can share weights with the token embedding matrix (weight tying). This is common in small models and reduces parameter count.

## Project Location

`src/PicoLLM.Core/Layers/`:
- `LayerNorm.cs`
- `LinearLayer.cs`
- `MultiHeadAttention.cs`
- `FeedForward.cs`
- `DecoderBlock.cs`

`src/PicoLLM.Core/Model/`:
- `PicoLLMModel.cs`
- `ModelConfig.cs`

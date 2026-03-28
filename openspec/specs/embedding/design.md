# Embedding — Technical Design

## Token Embedding

A simple lookup table: `float[vocab_size, embed_dim]`. Forward pass indexes rows by token ID.

```csharp
public class TokenEmbedding
{
    public Tensor Weights; // shape: [vocab_size, embed_dim]

    public Tensor Forward(int[] tokenIds)
    {
        // For each token ID, copy the corresponding row from Weights
        // Output shape: [seq_len, embed_dim]
    }

    public void Backward(Tensor gradOutput, int[] tokenIds)
    {
        // Accumulate gradients into the rows that were used
    }
}
```

Initialize weights with Xavier Normal or N(0, 0.02).

## Sinusoidal Positional Encoding

Not learned — computed once and cached up to `max_seq_len`.

```
PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))
```

Precompute a tensor of shape `[max_seq_len, embed_dim]` at initialization. During forward pass, slice `[0..seq_len, :]` and add to token embeddings.

## Combined Forward

```csharp
public class EmbeddingLayer
{
    private TokenEmbedding _tokenEmbed;
    private Tensor _positionalEncoding; // precomputed, not trainable

    public Tensor Forward(int[] tokenIds)
    {
        var embedded = _tokenEmbed.Forward(tokenIds);
        var posSlice = _positionalEncoding.Slice(0, tokenIds.Length);
        return TensorMath.Add(embedded, posSlice);
    }
}
```

## Batched Operation

For batch input of shape `[batch, seq_len]`, produce output `[batch, seq_len, embed_dim]`. The same positional encoding is broadcast across the batch dimension.

## Project Location

`src/PicoLLM.Core/Layers/`:
- `TokenEmbedding.cs`
- `PositionalEncoding.cs`
- `EmbeddingLayer.cs` — combines both

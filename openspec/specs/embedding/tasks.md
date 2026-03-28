# Embedding — Implementation Tasks

## Token Embedding
- [ ] 1.1 Implement `TokenEmbedding` with weights tensor [vocab_size, embed_dim]
- [ ] 1.2 Implement `Forward(int[] tokenIds)` → Tensor [seq_len, embed_dim]
- [ ] 1.3 Implement `Backward(gradOutput, tokenIds)` — sparse row gradient accumulation
- [ ] 1.4 Write tests: embed known IDs, verify output shape, verify gradient rows

## Positional Encoding
- [ ] 2.1 Implement `PositionalEncoding` with precomputed sinusoidal table
- [ ] 2.2 Verify sin/cos alternation on even/odd indices
- [ ] 2.3 Write tests: compare against hand-computed values for small dimensions

## Combined Embedding
- [ ] 3.1 Implement `EmbeddingLayer` combining token + positional
- [ ] 3.2 Implement batched forward: [batch, seq_len] → [batch, seq_len, embed_dim]
- [ ] 3.3 Write tests: output shape, positional encoding broadcast across batch

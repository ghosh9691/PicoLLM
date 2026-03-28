# Transformer — Implementation Tasks

## Layer Normalization
- [ ] 1.1 Implement `LayerNorm` with gamma (scale) and beta (shift) parameters
- [ ] 1.2 Implement forward: compute mean, variance along last dim, normalize, scale+shift
- [ ] 1.3 Store intermediate values needed for backward pass
- [ ] 1.4 Write tests: normalized output has mean≈0 var≈1, gamma/beta effect

## Linear Layer
- [ ] 2.1 Implement `LinearLayer` with Weights tensor and optional Bias
- [ ] 2.2 Implement forward: output = input @ Weights + Bias
- [ ] 2.3 Support batched input: [batch, seq, in] → [batch, seq, out]
- [ ] 2.4 Write tests: known weight matrix, verify output values

## Multi-Head Self-Attention
- [ ] 3.1 Implement Q, K, V projection via three LinearLayer instances
- [ ] 3.2 Implement reshape [batch, seq, embed] → [batch, heads, seq, head_dim]
- [ ] 3.3 Implement scaled dot-product: (Q @ K^T) / sqrt(head_dim)
- [ ] 3.4 Implement causal mask generation (upper triangle → -inf)
- [ ] 3.5 Apply mask then softmax to attention scores
- [ ] 3.6 Compute context: attn_weights @ V
- [ ] 3.7 Reshape back [batch, heads, seq, head_dim] → [batch, seq, embed]
- [ ] 3.8 Apply output projection
- [ ] 3.9 Write tests: output shape, causal mask blocks future, attention weights sum to 1

## Feedforward Network
- [ ] 4.1 Implement `FeedForward` with up-projection and down-projection
- [ ] 4.2 Apply GELU activation between layers
- [ ] 4.3 Write tests: output shape, ff_dim = embed_dim * multiplier

## Decoder Block
- [ ] 5.1 Implement `DecoderBlock` with pre-norm residual pattern
- [ ] 5.2 Wire: x + Attention(LayerNorm(x)), then x + FFN(LayerNorm(x))
- [ ] 5.3 Write tests: output shape matches input shape, residual passthrough

## Full Model
- [ ] 6.1 Implement `ModelConfig` record
- [ ] 6.2 Implement `PicoLLMModel` composing embedding + N blocks + final norm + lm_head
- [ ] 6.3 Implement `Forward(tokenIds)` → logits [batch, seq, vocab]
- [ ] 6.4 Implement `TotalParameters()` count
- [ ] 6.5 Implement `GetAllParameters()` — flat list of all trainable tensors
- [ ] 6.6 Write tests: forward pass shapes, parameter count for known config
- [ ] 6.7 Integration test: random tokens → logits → softmax → valid probability distribution

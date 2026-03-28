# Transformer Specification

## Purpose

Implement the GPT-style transformer decoder block: multi-head self-attention, feedforward network, layer normalization, and residual connections. This is the core architecture that makes a language model.

### Requirement: Layer Normalization

The system SHALL implement layer normalization with learnable scale (gamma) and shift (beta) parameters.

#### Scenario: Normalize a vector
- **GIVEN** an input tensor of shape [batch, seq, embed_dim]
- **WHEN** LayerNorm is applied along the last dimension
- **THEN** each [embed_dim] vector has mean ≈ 0 and variance ≈ 1 (before scale/shift)
- **AND** the output shape equals the input shape

### Requirement: Multi-Head Self-Attention

The system SHALL implement scaled dot-product attention with multiple heads, including causal masking for autoregressive generation.

#### Scenario: Attention with 4 heads
- **GIVEN** input of shape [batch, seq, embed_dim] with embed_dim=128 and num_heads=4
- **WHEN** attention is computed
- **THEN** head_dim = 128/4 = 32
- **AND** Q, K, V are projected via linear layers [embed_dim, embed_dim]
- **AND** reshaped to [batch, heads, seq, head_dim]
- **AND** scores = (Q @ K^T) / sqrt(head_dim)
- **AND** causal mask applied (upper triangle → -inf)
- **AND** softmax applied to scores
- **AND** output = scores @ V, reshaped back to [batch, seq, embed_dim]
- **AND** final linear projection applied

#### Scenario: Causal masking prevents future attention
- **GIVEN** a sequence of length 5
- **WHEN** attention weights are computed with causal masking
- **THEN** position 0 attends only to position 0
- **AND** position 2 attends to positions 0, 1, 2 only
- **AND** position 4 attends to all 5 positions

### Requirement: Feedforward Network

The system SHALL implement a two-layer feedforward network with GELU activation: Linear(embed_dim, ff_dim) → GELU → Linear(ff_dim, embed_dim).

#### Scenario: FFN dimensions
- **GIVEN** embed_dim=128 and ff_multiplier=4
- **WHEN** the FFN processes input of shape [batch, seq, 128]
- **THEN** the hidden layer has dimension 512 (128 × 4)
- **AND** the output returns to shape [batch, seq, 128]

### Requirement: Transformer Decoder Block

The system SHALL combine the above into a single decoder block with pre-norm residual connections:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

#### Scenario: Single block forward pass
- **GIVEN** input of shape [batch, seq, embed_dim]
- **WHEN** a decoder block processes it
- **THEN** output has the same shape [batch, seq, embed_dim]
- **AND** the residual connection preserves gradient flow

### Requirement: Stacked Decoder

The system SHALL stack N decoder blocks sequentially to form the full transformer.

#### Scenario: 4-layer transformer
- **GIVEN** num_layers=4, embed_dim=128, num_heads=4, ff_multiplier=4
- **WHEN** input passes through all 4 blocks
- **THEN** each block's output feeds into the next block's input
- **AND** the final output shape is [batch, seq, embed_dim]

### Requirement: Output Head

The system SHALL include a final LayerNorm followed by a linear projection to vocabulary size, producing logits for next-token prediction.

#### Scenario: Logits output
- **GIVEN** transformer output of shape [batch, seq, embed_dim] and vocab_size=512
- **WHEN** the output head is applied
- **THEN** logits have shape [batch, seq, vocab_size]
- **AND** each position's logits represent unnormalized scores over the vocabulary

### Requirement: Full Model Configuration

The system SHALL accept a configuration object specifying all hyperparameters.

#### Scenario: Model config
- **GIVEN** a config with vocab_size=512, embed_dim=128, num_heads=4, num_layers=4, ff_multiplier=4, max_seq_len=512, dropout=0.0
- **WHEN** the model is instantiated
- **THEN** all layers are initialized with the specified dimensions
- **AND** total parameter count is computable and reportable

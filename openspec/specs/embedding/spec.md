# Embedding Specification

## Purpose

Convert token IDs into dense vector representations and add positional information so the transformer knows token ordering.

### Requirement: Token Embedding

The system SHALL map each token ID to a learnable dense vector of dimension `embed_dim`.

#### Scenario: Embed a token sequence
- **GIVEN** a vocabulary of size 512 and embed_dim of 128
- **WHEN** token IDs [2, 45, 67, 3] are embedded
- **THEN** the output is a tensor of shape [4, 128]
- **AND** each row is the learned embedding vector for that token ID

### Requirement: Positional Encoding

The system SHALL add positional information to token embeddings using sinusoidal positional encoding (as described in "Attention Is All You Need").

#### Scenario: Positional encoding values
- **GIVEN** a sequence of length 4 and embed_dim 128
- **WHEN** positional encoding is computed
- **THEN** even indices use sin(pos / 10000^(2i/d)) and odd indices use cos(pos / 10000^(2i/d))
- **AND** the encoding is deterministic (not learned)

### Requirement: Combined Embedding

The system SHALL produce the final embedding as: `output = TokenEmbedding(tokens) + PositionalEncoding(seq_len)`

#### Scenario: Combined output shape
- **GIVEN** input token IDs of shape [batch, seq_len]
- **WHEN** the combined embedding is computed
- **THEN** the output shape is [batch, seq_len, embed_dim]

### Requirement: Embedding Weights Are Trainable

The token embedding lookup table weights SHALL be updated during backpropagation.

#### Scenario: Gradient update
- **GIVEN** a loss gradient flowing back to the embedding layer
- **WHEN** backpropagation is performed
- **THEN** only the rows corresponding to token IDs used in the forward pass are updated

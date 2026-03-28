# Training Specification

## Purpose

Implement the training pipeline: loss computation, backpropagation through the entire transformer, AdamW optimizer, and the training loop that accepts tokenized text and updates model weights.

### Requirement: Cross-Entropy Loss

The system SHALL compute cross-entropy loss between predicted logits and target token IDs.

#### Scenario: Loss computation
- **GIVEN** logits of shape [batch, seq, vocab_size] and target IDs of shape [batch, seq]
- **WHEN** cross-entropy loss is computed
- **THEN** loss = -mean(log(softmax(logits)[target]))
- **AND** the loss is a single scalar float

### Requirement: Backpropagation Through Transformer

The system SHALL compute gradients for every trainable parameter in the model via reverse-mode automatic differentiation (manual chain rule).

#### Scenario: Gradient shapes match parameter shapes
- **GIVEN** a forward pass producing a scalar loss
- **WHEN** backward is called
- **THEN** every trainable parameter has a gradient tensor of identical shape
- **AND** gradients are accumulated (not replaced) within a batch

### Requirement: Backward for Each Layer Type

The system SHALL implement backward passes for: LinearLayer, LayerNorm, MultiHeadAttention, FeedForward, DecoderBlock, TokenEmbedding, and the output head.

#### Scenario: LinearLayer backward
- **GIVEN** forward: y = x @ W + b, and gradient dL/dy
- **WHEN** backward is computed
- **THEN** dL/dW = x^T @ dL/dy, dL/db = sum(dL/dy, axis=0), dL/dx = dL/dy @ W^T

### Requirement: AdamW Optimizer

The system SHALL implement the AdamW optimizer with configurable learning rate, beta1, beta2, epsilon, and weight decay.

#### Scenario: Single optimization step
- **GIVEN** parameters with computed gradients, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
- **WHEN** an optimizer step is taken
- **THEN** first and second moment estimates are updated
- **AND** bias-corrected moments are computed
- **AND** parameters are updated: param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

### Requirement: Gradient Clipping

The system SHALL support gradient clipping by global norm to prevent exploding gradients.

#### Scenario: Clip gradients
- **GIVEN** a max norm of 1.0 and current global gradient norm of 5.0
- **WHEN** clipping is applied
- **THEN** all gradients are scaled by (1.0 / 5.0)

### Requirement: Training Loop

The system SHALL implement a training loop that: splits text into sequences of `max_seq_len`, creates input/target pairs (shifted by 1), runs forward pass, computes loss, runs backward pass, clips gradients, and takes an optimizer step.

#### Scenario: Training on a batch
- **GIVEN** tokenized text as a flat array of token IDs
- **WHEN** a training step is executed with batch_size=4 and seq_len=64
- **THEN** 4 sequences of length 64 are sampled
- **AND** targets are the same sequences shifted right by 1
- **AND** loss decreases over multiple steps on the same data (overfitting test)

### Requirement: Learning Rate Schedule

The system SHALL support a linear warmup followed by cosine decay learning rate schedule.

#### Scenario: Warmup then decay
- **GIVEN** warmup_steps=100, total_steps=1000, max_lr=1e-4, min_lr=1e-6
- **WHEN** step 50 is reached
- **THEN** lr = max_lr * (50/100)
- **WHEN** step 550 is reached (halfway through decay)
- **THEN** lr is between max_lr and min_lr following cosine curve

### Requirement: Training Metrics

The system SHALL report loss, learning rate, tokens/second, and gradient norm at each step.

#### Scenario: Metrics reporting
- **GIVEN** a training step completes
- **WHEN** metrics are queried
- **THEN** loss (float), lr (float), tokens_per_sec (float), grad_norm (float) are available

### Requirement: Checkpoint Save/Load

The system SHALL save and load full training state (model weights, optimizer state, step count) to/from disk in a binary format for resuming training.

#### Scenario: Resume training
- **GIVEN** a checkpoint saved at step 500
- **WHEN** training is resumed from that checkpoint
- **THEN** the model, optimizer moments, and step counter are restored
- **AND** training continues from step 501 with identical behavior

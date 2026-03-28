# Training — Implementation Tasks

## Setup
- [ ] 1.1 Create `PicoLLM.Training` project, reference PicoLLM.Core and PicoLLM.Tokenizer

## Parameter & Layer Infrastructure
- [ ] 2.1 Implement `Parameter` class wrapping Tensor + Grad
- [ ] 2.2 Define `ILayer` interface (Forward, Backward, Parameters)
- [ ] 2.3 Refactor all Core layers (LinearLayer, LayerNorm, etc.) to implement ILayer
- [ ] 2.4 Implement `PicoLLMModel.Parameters()` — collect all parameters across all layers
- [ ] 2.5 Implement `PicoLLMModel.ZeroGrad()` — zero all parameter gradients

## Backward Passes
- [ ] 3.1 Implement `LinearLayer.Backward(gradOutput)` — compute dW, db, dx
- [ ] 3.2 Implement `LayerNorm.Backward(gradOutput)` — compute dGamma, dBeta, dx
- [ ] 3.3 Implement `MultiHeadAttention.Backward(gradOutput)` — chain through Q/K/V projections, softmax, masking
- [ ] 3.4 Implement `FeedForward.Backward(gradOutput)` — chain through GELU and linear layers
- [ ] 3.5 Implement `DecoderBlock.Backward(gradOutput)` — residual connections
- [ ] 3.6 Implement `TokenEmbedding.Backward(gradOutput, tokenIds)` — sparse row updates
- [ ] 3.7 Implement output head (lm_head) backward
- [ ] 3.8 Implement `PicoLLMModel.Backward(gradLogits)` — full chain in reverse
- [ ] 3.9 Write gradient check tests: numerical gradient vs analytical gradient for each layer

## Loss
- [ ] 4.1 Implement `CrossEntropyLoss.Forward(logits, targets)` → scalar loss
- [ ] 4.2 Implement `CrossEntropyLoss.Backward(logits, targets)` → grad_logits
- [ ] 4.3 Write tests: known logits/targets, verify loss value and gradient

## Optimizer
- [ ] 5.1 Implement `AdamW` with configurable lr, beta1, beta2, eps, weight_decay
- [ ] 5.2 Implement first/second moment tracking per parameter
- [ ] 5.3 Implement bias correction
- [ ] 5.4 Implement weight decay (decoupled)
- [ ] 5.5 Write tests: parameter moves toward lower loss on simple function

## Gradient Clipping
- [ ] 6.1 Implement global gradient norm computation
- [ ] 6.2 Implement `ClipGradNorm(parameters, maxNorm)`
- [ ] 6.3 Write tests: verify clipping scales gradients correctly

## Learning Rate Schedule
- [ ] 7.1 Implement `LearningRateSchedule` with linear warmup + cosine decay
- [ ] 7.2 Write tests: warmup ramp, peak, decay curve

## Training Loop
- [ ] 8.1 Implement sequence sampling from flat token array
- [ ] 8.2 Implement input/target pair creation (shift by 1)
- [ ] 8.3 Implement full training step: forward → loss → backward → clip → step
- [ ] 8.4 Implement `TrainingMetrics` recording (loss, lr, tokens/sec, grad_norm)
- [ ] 8.5 Write overfitting test: loss decreases on repeated same data

## Checkpoint
- [ ] 9.1 Implement `CheckpointManager.Save(model, optimizer, step, path)`
- [ ] 9.2 Implement `CheckpointManager.Load(path)` → restored model + optimizer + step
- [ ] 9.3 Write tests: save, load, verify identical model output after restore

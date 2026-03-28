# Training — Technical Design

## Gradient Infrastructure

Each trainable tensor needs a companion gradient tensor. Two approaches:

**Chosen approach:** Wrap parameters in a `Parameter` class:

```csharp
public class Parameter
{
    public Tensor Data { get; set; }
    public Tensor Grad { get; set; }  // same shape as Data, zeroed before each step

    public void ZeroGrad() => Grad = TensorFactory.Zeros(Data.Shape);
}
```

Every layer exposes its parameters as `IEnumerable<Parameter>`.

## Backward Pass Strategy

Manual chain rule — no autograd tape. Each layer implements:

```csharp
public interface ILayer
{
    Tensor Forward(Tensor input);
    Tensor Backward(Tensor gradOutput);  // returns gradInput
    IEnumerable<Parameter> Parameters();
}
```

During `Backward`, each layer:
1. Computes gradients w.r.t. its own parameters (accumulated into `Parameter.Grad`)
2. Computes gradient w.r.t. its input (returned to the previous layer)

The model's `Backward` chains these in reverse order.

## Cross-Entropy Loss

```
softmax_output = softmax(logits, axis=-1)   // [batch, seq, vocab]
loss = -mean( log(softmax_output[b, s, target[b,s]]) )  // scalar

grad_logits = softmax_output
grad_logits[b, s, target[b,s]] -= 1.0
grad_logits /= (batch * seq)
```

## AdamW Implementation

```csharp
public class AdamW
{
    private Dictionary<Parameter, (Tensor m, Tensor v)> _state;
    private float _lr, _beta1, _beta2, _eps, _weightDecay;
    private int _step;

    public void Step(IEnumerable<Parameter> parameters)
    {
        _step++;
        foreach (var p in parameters)
        {
            var (m, v) = _state[p];
            // m = beta1 * m + (1 - beta1) * grad
            // v = beta2 * v + (1 - beta2) * grad^2
            // m_hat = m / (1 - beta1^step)
            // v_hat = v / (1 - beta2^step)
            // p.Data -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p.Data)
        }
    }
}
```

## Training Loop

```
for step in 0..total_steps:
    batch = sample_sequences(tokens, batch_size, seq_len)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model.Forward(inputs)
    loss = cross_entropy(logits, targets)

    model.ZeroGrad()
    grad = loss_backward(logits, targets)
    model.Backward(grad)

    clip_grad_norm(model.Parameters(), max_norm)
    lr = schedule.GetLR(step)
    optimizer.Step(model.Parameters())

    report(step, loss, lr, grad_norm)
```

## Checkpoint Format (Binary)

```
[magic: "PCKP" 4 bytes]
[version: uint32]
[step: int64]
[num_parameters: int32]
for each parameter:
    [name_len: int32][name: utf8 bytes]
    [shape_rank: int32][shape: int32[]]
    [data: float[]]
[optimizer_state_follows: bool]
if true:
    for each parameter:
        [m: float[]][v: float[]]
```

## Project Location

`src/PicoLLM.Training/`:
- `Parameter.cs`
- `ILayer.cs`
- `CrossEntropyLoss.cs`
- `AdamW.cs`
- `GradientClipper.cs`
- `LearningRateSchedule.cs`
- `TrainingLoop.cs`
- `TrainingMetrics.cs`
- `CheckpointManager.cs`

# BUG FIX: Ollama Cannot Load PicoLLM GGUF — Missing ffn_gate.weight

## Problem

Ollama returns `500 Internal Server Error: unable to load model` when running `ollama create picollm -f Modelfile`. The GGUF file passes structural validation (gguf-dump shows correct metadata and 35 tensors), but llama.cpp crashes during model loading because it expects tensor `blk.N.ffn_gate.weight` in every block and PicoLLM does not write it.

## Root Cause

PicoLLM declares `general.architecture = "llama"` in the GGUF metadata, but implements a **GPT-2 style FFN** (two linear layers with GELU activation). The real LLaMA architecture uses a **SwiGLU gated FFN** with three linear layers. When llama.cpp loads a model tagged as `llama`, it requires three FFN tensors per block:

- `blk.N.ffn_gate.weight` — gate projection (MISSING)
- `blk.N.ffn_up.weight` — up projection (present)
- `blk.N.ffn_down.weight` — down projection (present)

## Fix

Change PicoLLM's `FeedForward` layer from GPT-2 style to LLaMA style, and update the GGUF exporter to write the gate tensor. This requires retraining any existing model.

---

## Changes Required

### Change 1: Replace FeedForward with SwiGLU gated FFN

**File:** `src/PicoLLM.Core/Layers/FeedForward.cs`

**Current implementation (WRONG for llama architecture):**
```csharp
// GPT-2 style: two layers with GELU
// output = Linear_down(GELU(Linear_up(x)))
public class FeedForward
{
    private LinearLayer _up;    // [embed_dim, ff_dim]
    private LinearLayer _down;  // [ff_dim, embed_dim]

    public Tensor Forward(Tensor x)
    {
        var hidden = Activations.GELU(_up.Forward(x));
        return _down.Forward(hidden);
    }
}
```

**Replace with (CORRECT for llama architecture):**
```csharp
/// <summary>
/// SwiGLU gated feedforward network as used in LLaMA.
/// Computes: output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
/// 
/// This uses three linear projections instead of two, and SiLU (Sigmoid Linear Unit)
/// instead of GELU. The "gate" controls how much of the "up" projection passes through,
/// which is why it's called a gated linear unit.
/// 
/// Reference: Shazeer (2020) "GLU Variants Improve Transformer"
/// </summary>
public class FeedForward
{
    private LinearLayer _gate;  // [embed_dim, ff_dim] — gate projection
    private LinearLayer _up;    // [embed_dim, ff_dim] — up projection
    private LinearLayer _down;  // [ff_dim, embed_dim] — down projection

    public FeedForward(int embedDim, int ffDim, int? seed = null)
    {
        _gate = new LinearLayer(embedDim, ffDim, useBias: false, seed: seed);
        _up = new LinearLayer(embedDim, ffDim, useBias: false, seed: seed.HasValue ? seed.Value + 1 : null);
        _down = new LinearLayer(ffDim, embedDim, useBias: false, seed: seed.HasValue ? seed.Value + 2 : null);
    }

    /// <summary>
    /// Forward pass: SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    /// where ⊙ is element-wise multiplication and SiLU(x) = x * sigmoid(x)
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        var gateOutput = Activations.SiLU(_gate.Forward(x));  // SiLU(x @ W_gate)
        var upOutput = _up.Forward(x);                         // x @ W_up
        var hidden = TensorMath.Multiply(gateOutput, upOutput); // element-wise multiply
        return _down.Forward(hidden);                           // project back down
    }

    /// <summary>
    /// Backward pass through SwiGLU.
    /// Given dL/dOutput, compute gradients for W_gate, W_up, W_down, and dL/dInput.
    /// </summary>
    public Tensor Backward(Tensor gradOutput)
    {
        // Backward through _down: get gradient for hidden
        var gradHidden = _down.Backward(gradOutput);

        // gradHidden flows into element-wise multiply of gateOutput and upOutput
        // grad w.r.t. gateOutput = gradHidden * upOutput (stored from forward pass)
        // grad w.r.t. upOutput = gradHidden * gateOutput (stored from forward pass)
        var gradGateOutput = TensorMath.Multiply(gradHidden, _lastUpOutput);
        var gradUpOutput = TensorMath.Multiply(gradHidden, _lastGateOutput);

        // Backward through SiLU for gate path
        // SiLU(x) = x * sigmoid(x)
        // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        var gradGatePreAct = TensorMath.Multiply(gradGateOutput, SiLUDerivative(_lastGatePreAct));
        var gradInputFromGate = _gate.Backward(gradGatePreAct);

        // Backward through up path (linear only, no activation)
        var gradInputFromUp = _up.Backward(gradUpOutput);

        // Input receives gradients from both paths
        return TensorMath.Add(gradInputFromGate, gradInputFromUp);
    }

    public IEnumerable<Parameter> Parameters()
    {
        return _gate.Parameters()
            .Concat(_up.Parameters())
            .Concat(_down.Parameters());
    }

    // Store intermediate values during forward pass for backward pass
    private Tensor _lastGatePreAct;  // x @ W_gate (before SiLU)
    private Tensor _lastGateOutput;  // SiLU(x @ W_gate)
    private Tensor _lastUpOutput;    // x @ W_up
}
```

**Important:** The forward pass must cache `_lastGatePreAct`, `_lastGateOutput`, and `_lastUpOutput` for the backward pass. Update the forward method to store these before returning.

### Change 2: Add SiLU activation function

**File:** `src/PicoLLM.Core/Activations/SiLU.cs` (new file)

```csharp
/// <summary>
/// SiLU (Sigmoid Linear Unit), also called Swish.
/// SiLU(x) = x * sigmoid(x)
/// 
/// Used in LLaMA's SwiGLU feedforward network as the gating activation.
/// Unlike ReLU, SiLU is smooth and non-monotonic, which helps gradient flow.
/// </summary>
public static class SiLU
{
    /// <summary>
    /// Compute SiLU(x) = x * sigmoid(x) element-wise.
    /// </summary>
    public static Tensor Forward(Tensor x)
    {
        // SiLU(x) = x * (1 / (1 + exp(-x)))
        var result = TensorFactory.Zeros(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            float xi = x.Data[i];
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-xi));
            result.Data[i] = xi * sigmoid;
        }
        return result;
    }

    /// <summary>
    /// Compute the derivative of SiLU for backpropagation.
    /// SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    /// </summary>
    public static Tensor Derivative(Tensor x)
    {
        var result = TensorFactory.Zeros(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            float xi = x.Data[i];
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-xi));
            result.Data[i] = sigmoid * (1.0f + xi * (1.0f - sigmoid));
        }
        return result;
    }
}
```

Also add a static helper to the `Activations` class (or wherever ReLU/GELU are accessed):
```csharp
public static Tensor SiLU(Tensor x) => Activations.SiLU.Forward(x);
```

### Change 3: Update GGUF exporter to write ffn_gate.weight

**File:** `src/PicoLLM.Gguf/TensorNameMapper.cs`

Add the gate tensor mapping. The tensor name map must include:
```csharp
["Block.{i}.Ffn.Gate.Weights"] = "blk.{i}.ffn_gate.weight",
```

Make sure `GgufExporter` iterates the `FeedForward` parameters in this order: gate, up, down. Verify that `FeedForward.Parameters()` returns them in that order (gate first, then up, then down).

**File:** `src/PicoLLM.Gguf/GgufExporter.cs`

Update the tensor count. Previously 35 tensors (9 per block × 4 blocks - 1 + token_embd + output_norm + output = 35). Now there are 4 more tensors (one ffn_gate.weight per block), so **39 tensors total**.

Verify the exporter writes tensors in this order per block:
1. `blk.N.attn_norm.weight` — [128]
2. `blk.N.attn_q.weight` — [128, 128]
3. `blk.N.attn_k.weight` — [128, 128]
4. `blk.N.attn_v.weight` — [128, 128]
5. `blk.N.attn_output.weight` — [128, 128]
6. `blk.N.ffn_norm.weight` — [128]
7. `blk.N.ffn_gate.weight` — [128, 512] **(NEW)**
8. `blk.N.ffn_up.weight` — [128, 512]
9. `blk.N.ffn_down.weight` — [512, 128]

### Change 4: Update FeedForward backward pass in training

**File:** `src/PicoLLM.Training/` (wherever backward passes are wired)

The backward pass for `FeedForward` now has three paths instead of two. Make sure the training infrastructure calls `FeedForward.Backward()` which handles all three internally (gate, up, down), and that all three linear layers' parameters are included in the optimizer's parameter list.

### Change 5: Update ModelConfig parameter count

**File:** `src/PicoLLM.Core/Model/PicoLLMModel.cs` or `ModelConfig.cs`

The `TotalParameters()` method must account for the new gate weights. Per block, the FFN now has 3 weight matrices instead of 2:
- Old: `embed_dim × ff_dim + ff_dim × embed_dim` = 128×512 + 512×128 = 131,072
- New: `embed_dim × ff_dim × 3 + ff_dim × embed_dim` = 128×512 + 128×512 + 512×128 = 196,608

Wait — actually: gate is [embed_dim, ff_dim], up is [embed_dim, ff_dim], down is [ff_dim, embed_dim]. So new FFN params per block = 128×512 + 128×512 + 512×128 = 196,608 (was 131,072). Total model parameter increase = 65,536 × 4 blocks = 262,144 additional parameters.

### Change 6: Delete or deprecate GELU usage in FFN

Since `FeedForward` no longer uses GELU (it uses SiLU via SwiGLU gating), make sure no code path still calls GELU inside the feedforward layer. GELU can remain in the Activations library for educational reference but should not be used in the active model architecture.

---

## Tests to Write or Update

1. **SiLU activation test:** Verify `SiLU(0) = 0`, `SiLU(large positive) ≈ large positive`, `SiLU(large negative) ≈ 0`. Verify derivative values against numerically computed derivatives.

2. **FeedForward forward test:** Verify output shape is [batch, seq, embed_dim] for input [batch, seq, embed_dim]. Verify the gate path produces different results than the old GELU path.

3. **FeedForward backward test:** Verify all three parameter sets (gate, up, down) receive non-zero gradients. Numerical gradient check: perturb each weight by epsilon, compute loss, compare analytical gradient.

4. **GGUF export test:** Export a model and verify gguf-dump shows exactly 39 tensors. Verify `blk.N.ffn_gate.weight` appears for N = 0, 1, 2, 3. Verify gate tensor shape is [128, 512].

5. **Ollama integration test (manual):** After changes, run:
   ```
   ollama create picollm -f Modelfile
   ollama run picollm "Hello"
   ```
   Verify no 500 error. Output will be incoherent (expected for a tiny model) but the model should load and generate tokens.

---

## Task Checklist

- [ ] 1. Create `SiLU.cs` in `PicoLLM.Core/Activations/` with Forward and Derivative methods
- [ ] 2. Add `SiLU` static helper to the Activations class
- [ ] 3. Write SiLU unit tests (values + derivative numerical check)
- [ ] 4. Modify `FeedForward` constructor to create three LinearLayer instances: _gate, _up, _down
- [ ] 5. Modify `FeedForward.Forward()` to implement SwiGLU: `(SiLU(x @ W_gate) * (x @ W_up)) @ W_down`
- [ ] 6. Cache intermediate tensors in Forward for Backward pass
- [ ] 7. Modify `FeedForward.Backward()` to compute gradients through all three paths
- [ ] 8. Modify `FeedForward.Parameters()` to return gate, up, down parameters in that order
- [ ] 9. Write FeedForward forward/backward unit tests
- [ ] 10. Add `"Block.{i}.Ffn.Gate.Weights" = "blk.{i}.ffn_gate.weight"` to TensorNameMapper
- [ ] 11. Update GgufExporter tensor count from 35 to 39
- [ ] 12. Update GgufExporter to write ffn_gate.weight tensors in correct position
- [ ] 13. Write GGUF export test: verify 39 tensors, verify ffn_gate.weight present and correct shape
- [ ] 14. Update `TotalParameters()` to account for gate weights
- [ ] 15. Remove GELU usage from FeedForward (keep GELU class for educational reference)
- [ ] 16. Run full `dotnet build` and `dotnet test` — all must pass
- [ ] 17. Retrain model (existing checkpoint is incompatible due to architecture change)
- [ ] 18. Export new GGUF and test with `ollama create picollm -f Modelfile` then `ollama run picollm "test"`

# PicoLLM — Specification Index

> An educational LLM built entirely from scratch in C# with a Lynx-style browser, transformer training, and GGUF export.
> Supersedes the TinyLLM prototype. **Every class is built fresh — nothing is pre-existing.**

## Quick Start for Claude Code

1. Copy this entire `picollm-specs/` folder into your project root
2. Rename it or keep it as-is — Claude Code reads `CLAUDE.md` at the repo root first
3. Start a Claude Code session and say: _"Read CLAUDE.md and begin building capability 1 (core-tensor)"_

## Master Documents

| File | Purpose |
|------|---------|
| [`CLAUDE.md`](CLAUDE.md) | **Repo-root instructions for Claude Code** — read this first. Contains the TinyLLM→PicoLLM class mapping and build-from-scratch directive. |
| [`openspec/project.md`](openspec/project.md) | **Master project context** — tech stack, constraints, architecture, build order, TinyLLM traceability table |
| [`openspec/AGENTS.md`](openspec/AGENTS.md) | **Agent workflow instructions** — how to use the specs |

---

## TinyLLM → PicoLLM Traceability

Every class from the TinyLLM prototype is superseded by one or more PicoLLM classes. This table is the authoritative cross-reference. **All PicoLLM classes MUST be built.**

| TinyLLM (prototype) | PicoLLM (production) | Capability Spec | What Changed |
|---------------------|---------------------|-----------------|--------------|
| `Tensor` | `Tensor`, `TensorMath`, `TensorFactory` | core-tensor | Added 4D tensors, batched matmul, masking, strides |
| `VocabTokenizer` | `BpeTokenizer`, `BpeTrainer` | tokenizer | Upgraded from word-level to full Byte-Pair Encoding |
| `EmbeddingLayer` | `TokenEmbedding`, `PositionalEncoding`, `EmbeddingLayer` | embedding | Added sinusoidal positional encoding, batched input |
| `DenseLayer` | `LinearLayer` | transformer | Added optional bias, used for Q/K/V/O projections + FFN |
| `Activations` | `ReLU`, `GELU`, `Softmax`, `Sigmoid`, `Tanh` | core-tensor | Added GELU, Sigmoid, Tanh; each has derivative for backprop |
| `Loss` (CrossEntropy) | `CrossEntropyLoss` | training | Added numerically stable log-softmax |
| `Loss` (MSE) | `MseLoss` | training | Standalone class with gradient |
| `Backprop` | Per-layer `Backward()` methods | training | Backprop through every layer type including attention |
| `Trainer` (SGD) | `TrainingLoop`, `AdamW`, `LearningRateSchedule`, `GradientClipper` | training | Upgraded SGD→AdamW, added LR warmup+cosine decay, grad clipping |
| `DataLoader` | Sequence sampling in `TrainingLoop` | training | Sliding window next-token prediction from flat corpus |
| `TinyNetwork` | `PicoLLMModel` | transformer | Full transformer decoder: Embedding→N×DecoderBlock→Norm→Head |
| `ModelSerializer` | `CheckpointManager`, `GgufExporter` | training, gguf-export | Binary checkpoint for resume + GGUF v3 for Ollama/LM Studio |

**New in PicoLLM (not in TinyLLM):**

| Class | Capability Spec | Purpose |
|-------|-----------------|---------|
| `LayerNorm` | transformer | Normalize activations (mean=0, var=1) with learnable scale/shift |
| `MultiHeadAttention` | transformer | Scaled dot-product attention with causal masking |
| `FeedForward` | transformer | Two-layer FFN with GELU: up-project → GELU → down-project |
| `DecoderBlock` | transformer | Pre-norm residual: Attention + FFN with skip connections |
| `ModelConfig` | transformer | Hyperparameter record (vocab, embed, heads, layers, ff, ctx) |
| `Parameter` | training | Wraps Tensor + Gradient for optimizer tracking |
| `TrainingMetrics` | training | Loss, LR, tokens/sec, gradient norm per step |
| `HttpFetcher` | browser-engine | HTTP/HTTPS client with User-Agent, redirects, timeout |
| `BrowseSession` | browser-engine | Tracks pages + images across a browsing session |
| `RobotsTxtParser` | browser-engine | Respects robots.txt Disallow rules |
| `HtmlParser` | html-parser | AngleSharp-based DOM walk → clean text + images + links |
| `ParsedPage` | html-parser | Result record: title, text, images, links, meta |
| `GgufWriter` | gguf-export | Low-level GGUF v3 binary writer (header, metadata, tensors) |
| `GgufExporter` | gguf-export | High-level: model + tokenizer → .gguf file |
| `TensorNameMapper` | gguf-export | PicoLLM internal names → llama.cpp tensor names |
| `IComputeProvider` | gpu-acceleration | Abstraction for CPU vs GPU matrix math |
| `CudaComputeProvider` | gpu-acceleration | ILGPU-based NVIDIA GPU offload |
| `GpuDetector` | gpu-acceleration | Runtime NVIDIA GPU detection |
| `PicoOrchestrator` | orchestrator | Pipeline: browse → tokenize → train → checkpoint → export |
| `OrchestratorEvent` | orchestrator | Progress events for UI subscription |
| Desktop UI | ui-shell | Browser pane + training dashboard + loss chart |

---

## Capability Specifications (Build Order)

Each capability folder contains three files:
- **`spec.md`** — Requirements (WHAT). Uses SHALL/MUST. Contains testable GIVEN/WHEN/THEN scenarios.
- **`design.md`** — Technical design (HOW). Data structures, algorithms, class layouts.
- **`tasks.md`** — Implementation checklist. Complete top-to-bottom. ~130 tasks total.

| # | Capability | Folder | Project | Key Deliverables |
|---|-----------|--------|---------|-----------------|
| 1 | **Core Tensor** | [`openspec/specs/core-tensor/`](openspec/specs/core-tensor/) | PicoLLM.Core | Tensor, TensorMath, TensorFactory, all activations |
| 2 | **Tokenizer** | [`openspec/specs/tokenizer/`](openspec/specs/tokenizer/) | PicoLLM.Tokenizer | BpeTokenizer, BpeTrainer, persistence |
| 3 | **Embedding** | [`openspec/specs/embedding/`](openspec/specs/embedding/) | PicoLLM.Core | TokenEmbedding, sinusoidal PositionalEncoding |
| 4 | **Transformer** | [`openspec/specs/transformer/`](openspec/specs/transformer/) | PicoLLM.Core | LinearLayer, LayerNorm, MultiHeadAttention, FFN, DecoderBlock, PicoLLMModel |
| 5 | **Training** | [`openspec/specs/training/`](openspec/specs/training/) | PicoLLM.Training | CrossEntropy, backprop for ALL layers, AdamW, LR schedule, checkpoints |
| 6 | **Browser Engine** | [`openspec/specs/browser-engine/`](openspec/specs/browser-engine/) | PicoLLM.Browser | HTTP fetcher, robots.txt, image downloader |
| 7 | **HTML Parser** | [`openspec/specs/html-parser/`](openspec/specs/html-parser/) | PicoLLM.Browser | HTML → clean text + images + links (AngleSharp) |
| 8 | **GGUF Export** | [`openspec/specs/gguf-export/`](openspec/specs/gguf-export/) | PicoLLM.Gguf | GGUF v3 binary writer, Ollama/LM Studio compatible |
| 9 | **GPU Acceleration** | [`openspec/specs/gpu-acceleration/`](openspec/specs/gpu-acceleration/) | PicoLLM.Gpu | Optional ILGPU CUDA offload for matmul |
| 10 | **Orchestrator** | [`openspec/specs/orchestrator/`](openspec/specs/orchestrator/) | PicoLLM.App | Browse → tokenize → train → checkpoint → export pipeline |
| 11 | **UI Shell** | [`openspec/specs/ui-shell/`](openspec/specs/ui-shell/) | PicoLLM.App | Desktop app: browser pane + training dashboard |

---

## Dependency Graph

```
core-tensor ──► embedding ──► transformer ──► training ──┐
                                                          │
tokenizer ────────────────────────────────────────────────┤
                                                          │
browser-engine ──► html-parser ───────────────────────────┤
                                                          │
gguf-export ──────────────────────────────────────────────┤
                                                          │
gpu-acceleration (optional) ──────────────────────────────┤
                                                          ▼
                                                    orchestrator ──► ui-shell
```

---

## Complete Class Inventory (50+ classes)

For reference, here is every class that will exist when the project is complete:

**PicoLLM.Core** (ZERO dependencies):
`Tensor`, `TensorMath`, `TensorFactory`, `ReLU`, `GELU`, `Softmax`, `Sigmoid`, `Tanh`, `TokenEmbedding`, `PositionalEncoding`, `EmbeddingLayer`, `LinearLayer`, `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `DecoderBlock`, `PicoLLMModel`, `ModelConfig`, `IComputeProvider`, `CpuComputeProvider`

**PicoLLM.Tokenizer** (ZERO dependencies):
`BpeTokenizer`, `BpeTrainer`, `TokenizerConfig`

**PicoLLM.Training**:
`Parameter`, `ILayer`, `CrossEntropyLoss`, `MseLoss`, `AdamW`, `GradientClipper`, `LearningRateSchedule`, `TrainingLoop`, `TrainingMetrics`, `CheckpointManager`

**PicoLLM.Browser**:
`HttpFetcher`, `BrowseSession`, `BrowseResult`, `BrowseStatus`, `ImageDownload`, `ImageDownloader`, `RobotsTxtParser`, `UrlResolver`, `HtmlParser`, `ParsedPage`, `ImageReference`, `LinkReference`, `TextExtractor`, `ElementFilter`

**PicoLLM.Gguf**:
`GgufWriter`, `GgufExporter`, `GgufConstants`, `GgufDataType`, `GgufValueType`, `TensorNameMapper`, `GgufValidator`

**PicoLLM.Gpu** (optional):
`CudaComputeProvider`, `GpuDetector`, `GpuInfo`, `MatMulKernel`

**PicoLLM.App**:
`PicoOrchestrator`, `OrchestratorConfig`, `TrainingConfig`, `OrchestratorEvent` (+ 7 event subtypes), MainPage, BrowserPane, DashboardPanel, ModelInfoPanel

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML framework | None (pure C#) | Educational: learner can step through every operation |
| Architecture | GPT-2 style decoder-only | Simplest modern LLM; maps to llama.cpp tensor layout |
| Tokenizer | Byte-Pair Encoding | Industry standard; learner sees merge algorithm |
| Export format | GGUF v3 | Ollama + LM Studio compatibility |
| UI framework | MAUI (or Avalonia) | Cross-platform Windows + macOS |
| GPU library | ILGPU (optional) | Pure .NET, no CUDA SDK install required |
| HTML parser | AngleSharp | Standard .NET HTML parser; not ML code |
| Attention | Pre-norm residual | Stable training, matches modern practice |
| Optimizer | AdamW | Standard for transformer training |
| Positional encoding | Sinusoidal (fixed) | Simpler than RoPE; educational clarity |
| Checkpoint format | Custom binary | Fast save/load for training resume |
| Export format | GGUF v3 F32 | Ollama/LM Studio import; quantization is stretch goal |

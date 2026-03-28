# PicoLLM — An Educational LLM Built From Scratch in C#

## Purpose

PicoLLM is a cross-platform (Windows / macOS) desktop application that teaches people how large language models are built and how they learn. It consists of two integrated subsystems:

1. **PicoBrowser** — A text-first web browser (inspired by Lynx) that fetches web pages, strips them to HTML text + images (JPEG/PNG/GIF) + links, and discards everything else (JavaScript, CSS, ads, tracking, video, audio, iframes).
2. **PicoLLM Engine** — A from-scratch transformer-based language model implemented purely in C# with zero ML library dependencies. The engine ingests content from the browser, tokenizes it, and trains (or continues training) an actual transformer model. At the end of a browsing session, the model is serialized to GGUF format for use in Ollama or LM Studio.

## Educational Philosophy

Every component of the LLM pipeline — tokenization, embedding, attention, feedforward, layer normalization, backpropagation, optimizer, GGUF serialization — is implemented in plain C# so a reader can step through the code and understand exactly what happens at each stage. No PyTorch, no TensorFlow, no ML.NET, no ONNX. The browser and UI layers may use standard .NET libraries freely.

## Prior Art (TinyLLM) — MUST BE BUILT FROM SCRATCH

A previous project called TinyLLM prototyped the following classes in C#. PicoLLM is a **complete superset** — every one of these classes MUST be implemented fresh (improved and production-quality) as part of the PicoLLM specs. Do NOT assume any of this code already exists. Build it all.

The TinyLLM classes and where they land in PicoLLM:

| TinyLLM Class | PicoLLM Equivalent | Spec Capability | Notes |
|---------------|-------------------|-----------------|-------|
| `Tensor` | `Tensor` | core-tensor | N-dimensional float array with shape/strides. PicoLLM adds batched matmul, 4D support, masking. |
| `VocabTokenizer` | `BpeTokenizer` | tokenizer | TinyLLM used simple word-level. PicoLLM upgrades to full BPE with merge rules. |
| `EmbeddingLayer` | `TokenEmbedding` | embedding | Lookup table mapping token IDs to dense vectors. PicoLLM adds sinusoidal positional encoding. |
| `DenseLayer` | `LinearLayer` | transformer | Fully connected layer: y = x @ W + b. PicoLLM adds optional bias, Xavier init. Used for Q/K/V projections, FFN, and output head. |
| `Activations` (ReLU, Softmax) | `Activations` (ReLU, GELU, Softmax, Sigmoid, Tanh) | core-tensor | PicoLLM adds GELU (used in FFN), Sigmoid, Tanh. Each activation also needs a derivative for backprop. |
| `Loss` (CrossEntropy) | `CrossEntropyLoss` | training | PicoLLM adds numerically stable log-softmax, MSE as secondary loss. |
| `Loss` (MSE) | `MseLoss` | training | Mean squared error loss with gradient. |
| `Backprop` (manual chain rule) | Per-layer `Backward()` methods | training | TinyLLM had a single backprop function. PicoLLM implements backward on every layer: LinearLayer, LayerNorm, MultiHeadAttention, FeedForward, DecoderBlock, TokenEmbedding. |
| `Trainer` (SGD loop) | `TrainingLoop` + `AdamW` | training | TinyLLM used simple SGD. PicoLLM upgrades to AdamW with LR schedule, gradient clipping, metrics. |
| `DataLoader` (sliding window) | Sequence sampling in `TrainingLoop` | training | Token sequences sampled from flat corpus, shifted by 1 for next-token prediction. |
| `TinyNetwork` (layer composition) | `PicoLLMModel` | transformer | TinyLLM stacked dense layers. PicoLLM composes: Embedding → N × DecoderBlock → LayerNorm → LinearHead. |
| `ModelSerializer` (JSON save/load) | `CheckpointManager` (binary) + `GgufExporter` (GGUF) | training, gguf-export | TinyLLM used JSON. PicoLLM uses binary for checkpoints (faster) and GGUF v3 for export to Ollama/LM Studio. |

**CRITICAL**: The specs for `core-tensor`, `embedding`, `transformer`, and `training` explicitly require building ALL of the above. Claude Code should NOT skip any class because it was "already built in TinyLLM." Nothing is pre-built. Everything is built from scratch as part of PicoLLM.

## Tech Stack

- **Language:** C# 12+ / .NET 9+
- **UI Framework:** .NET MAUI (cross-platform Windows + macOS) OR WinForms + Mac Catalyst (builder's choice)
- **LLM Engine:** Pure C# — no ML libraries whatsoever
- **Browser HTTP:** `HttpClient` (standard .NET)
- **HTML Parsing:** AngleSharp (NuGet) — allowed because it is browser infrastructure, not ML
- **Image Handling:** `System.Drawing` / `SkiaSharp` — for decoding JPEG/PNG/GIF
- **GPU (optional):** ILGPU (NuGet) for NVIDIA CUDA acceleration — allowed because it is compute infrastructure
- **Serialization:** Custom GGUF binary writer (pure C#)
- **Testing:** xUnit + FluentAssertions
- **Target frameworks:** `net9.0-windows` and `net9.0-macos` (or `net9.0` with MAUI)

## Architecture Patterns

- Solution uses a layered project structure similar to Pyxis Aviate's Active Record approach
- Projects: `PicoLLM.Core` (tensor, layers, model), `PicoLLM.Tokenizer`, `PicoLLM.Training`, `PicoLLM.Gguf`, `PicoLLM.Browser`, `PicoLLM.Gpu` (optional), `PicoLLM.App`, `PicoLLM.Tests`
- All LLM math lives in `PicoLLM.Core` — this project SHALL have zero NuGet dependencies
- Public API surfaces use interfaces for testability
- Model state is immutable during inference; mutable only during training
- The training pipeline is explicitly sequential (no hidden parallelism) so learners can follow the flow

## Code Standards

- C# 12 features: primary constructors, collection expressions, pattern matching
- Nullable reference types enabled everywhere
- XML doc comments on all public types and methods — these serve as the educational narrative
- No `dynamic`, no `reflection` in the core engine
- Naming: PascalCase for types/methods, camelCase for locals, `_camelCase` for private fields
- One class per file, file name matches class name
- Use `ReadOnlySpan<T>` and `Span<T>` for hot-path tensor operations where beneficial

## GGUF Export Requirements

The model MUST be exportable to GGUF v3 format with:
- Magic: "GGUF" (4 bytes ASCII)
- Version: 3 (uint32 LE)
- All metadata keys per the llama.cpp specification (general.architecture, general.name, context_length, embedding_length, etc.)
- Tensor data: F32 (full precision) — quantization is a stretch goal
- Tensor naming: follows llama.cpp convention (e.g., `blk.0.attn_q.weight`)
- The exported .gguf file SHALL be loadable by `ollama create` with a Modelfile

## Build Order for Claude Code

The specs are organized so that each capability can be built and tested in isolation before the next begins. The recommended build order is:

1. **core-tensor** — Tensor math foundation (build from scratch: Tensor, TensorMath, TensorFactory, all activations)
2. **tokenizer** — BPE tokenizer (train + encode + decode)
3. **embedding** — Token + positional embeddings
4. **transformer** — Multi-head attention, feedforward, layer norm, decoder block
5. **training** — Loss, backprop through transformer, AdamW optimizer, training loop
6. **browser-engine** — HTTP fetcher + content pipeline
7. **html-parser** — HTML → clean text + images + links
8. **gguf-export** — GGUF v3 binary writer + metadata
9. **gpu-acceleration** — Optional ILGPU CUDA offload for matrix multiply
10. **orchestrator** — Wires browser → tokenizer → training → model persistence
11. **ui-shell** — MAUI or WinForms desktop app with browser pane + training dashboard

Each spec folder contains a `spec.md` with requirements and scenarios, plus a `design.md` with implementation guidance and a `tasks.md` with an ordered checklist.

## Key Constraints

- The model will be small by design (educational, not production). Typical config: 4–8 attention heads, 4–6 transformer layers, 128–256 embedding dimensions, 512–2048 context window.
- Training happens on CPU by default. GPU acceleration is opt-in and detected at runtime.
- The browser deliberately ignores JavaScript, CSS, iframes, video, audio, and tracking scripts. It fetches raw HTML over HTTP/HTTPS and extracts only: text content, images (JPEG/PNG/GIF as `<img>` tags), and hyperlinks (`<a href>`).
- Image alt-text is extracted and fed to the tokenizer as text. Image pixel data is NOT used for training (this is a text-only LLM).
- The model file on disk uses PicoLLM's own format for checkpointing during training, and GGUF for final export.

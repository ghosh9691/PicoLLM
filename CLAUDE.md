# CLAUDE.md — PicoLLM

## What This Project Is

PicoLLM is an educational LLM built **entirely from scratch** in C#. It includes a Lynx-style text browser (PicoBrowser) that feeds web content into a hand-built transformer model. The trained model exports to GGUF format for use in Ollama or LM Studio.

## CRITICAL: Build Everything From Scratch

This project supersedes a previous prototype called **TinyLLM**. Every class from TinyLLM MUST be reimplemented as part of PicoLLM — nothing is pre-built, nothing should be skipped. The specs explicitly require every component.

Here is the complete mapping of TinyLLM → PicoLLM classes. **All of these MUST be built:**

| TinyLLM (old) | PicoLLM (build this) | Which Spec |
|---------------|---------------------|------------|
| `Tensor` | `Tensor` + `TensorMath` + `TensorFactory` | core-tensor |
| `VocabTokenizer` | `BpeTokenizer` + `BpeTrainer` | tokenizer |
| `EmbeddingLayer` | `TokenEmbedding` + `PositionalEncoding` + `EmbeddingLayer` | embedding |
| `DenseLayer` | `LinearLayer` | transformer |
| `Activations` (ReLU, Softmax) | `ReLU`, `GELU`, `Softmax`, `Sigmoid`, `Tanh` (each with derivative) | core-tensor |
| `Loss` (CrossEntropy, MSE) | `CrossEntropyLoss` + `MseLoss` | training |
| `Backprop` | `ILayer.Backward()` on every layer type | training |
| `Trainer` (SGD) | `TrainingLoop` + `AdamW` + `LearningRateSchedule` + `GradientClipper` | training |
| `DataLoader` | Sequence sampling in `TrainingLoop` | training |
| `TinyNetwork` | `PicoLLMModel` (Embedding → N×DecoderBlock → LayerNorm → LinearHead) | transformer |
| `ModelSerializer` (JSON) | `CheckpointManager` (binary) + `GgufExporter` (GGUF v3) | training, gguf-export |

**Plus these NEW classes not in TinyLLM:**

| New Class | Which Spec |
|-----------|------------|
| `LayerNorm` | transformer |
| `MultiHeadAttention` | transformer |
| `FeedForward` | transformer |
| `DecoderBlock` | transformer |
| `ModelConfig` | transformer |
| `Parameter` (wraps Tensor + Grad) | training |
| `TrainingMetrics` | training |
| `HttpFetcher`, `BrowseSession` | browser-engine |
| `RobotsTxtParser`, `UrlResolver` | browser-engine |
| `HtmlParser`, `ParsedPage` | html-parser |
| `GgufWriter`, `GgufExporter`, `TensorNameMapper` | gguf-export |
| `IComputeProvider`, `CudaComputeProvider`, `GpuDetector` | gpu-acceleration |
| `PicoOrchestrator` | orchestrator |
| Desktop UI (browser pane + training dashboard) | ui-shell |

## How to Build This Project

This project uses **spec-driven development** with OpenSpec. All specifications are pre-written.

### Step 1: Read the master spec
```
openspec/project.md
```

### Step 2: Read the agent instructions
```
openspec/AGENTS.md
```

### Step 3: Build capabilities in order
Each capability has three files in `openspec/specs/<capability>/`:
- `spec.md` — WHAT to build (requirements + testable scenarios)
- `design.md` — HOW to build it (architecture + data structures + algorithms)
- `tasks.md` — Checklist (complete top-to-bottom, every task is a deliverable)

**Build order (strict — each depends on the previous):**

| # | Capability | Project | What Gets Built |
|---|-----------|---------|-----------------|
| 1 | `core-tensor` | PicoLLM.Core | `Tensor`, `TensorMath`, `TensorFactory`, `ReLU`, `GELU`, `Softmax`, `Sigmoid`, `Tanh` |
| 2 | `tokenizer` | PicoLLM.Tokenizer | `BpeTokenizer`, `BpeTrainer`, `TokenizerConfig` |
| 3 | `embedding` | PicoLLM.Core | `TokenEmbedding`, `PositionalEncoding`, `EmbeddingLayer` |
| 4 | `transformer` | PicoLLM.Core | `LinearLayer`, `LayerNorm`, `MultiHeadAttention`, `FeedForward`, `DecoderBlock`, `PicoLLMModel`, `ModelConfig` |
| 5 | `training` | PicoLLM.Training | `Parameter`, `ILayer`, `CrossEntropyLoss`, `MseLoss`, backward passes for ALL layers, `AdamW`, `GradientClipper`, `LearningRateSchedule`, `TrainingLoop`, `TrainingMetrics`, `CheckpointManager` |
| 6 | `browser-engine` | PicoLLM.Browser | `HttpFetcher`, `BrowseSession`, `BrowseResult`, `ImageDownloader`, `RobotsTxtParser`, `UrlResolver` |
| 7 | `html-parser` | PicoLLM.Browser | `HtmlParser`, `ParsedPage`, `TextExtractor`, `ElementFilter` |
| 8 | `gguf-export` | PicoLLM.Gguf | `GgufWriter`, `GgufExporter`, `GgufConstants`, `TensorNameMapper`, `GgufValidator` |
| 9 | `gpu-acceleration` | PicoLLM.Gpu | `IComputeProvider`, `CpuComputeProvider`, `CudaComputeProvider`, `GpuDetector` |
| 10 | `orchestrator` | PicoLLM.App | `PicoOrchestrator`, `OrchestratorConfig`, `OrchestratorEvent` hierarchy |
| 11 | `ui-shell` | PicoLLM.App | Main window, browser pane, training dashboard, loss chart, model info panel |

### Step 4: For each capability
```
1. Read openspec/specs/<capability>/spec.md
2. Read openspec/specs/<capability>/design.md
3. Work through openspec/specs/<capability>/tasks.md top-to-bottom
4. Write xUnit tests for each task
5. Ensure `dotnet build` and `dotnet test` pass
6. Move to the next capability
```

## Dependency Rules

- **PicoLLM.Core has ZERO NuGet dependencies.** All tensor math, all layers, all activations — pure C#.
- **PicoLLM.Tokenizer has ZERO NuGet dependencies.** Only System.Text.Json (BCL).
- **PicoLLM.Browser may use AngleSharp** (HTML parsing is browser infra, not ML).
- **PicoLLM.Gpu may use ILGPU** (GPU compute is infra, not ML).
- **XML doc comments on every public type and method.** This is an educational project.
- **One class per file.** File name = class name.
- **.NET 9+, C# 12, nullable reference types enabled everywhere.**

## Solution Structure

```
PicoLLM.sln
├── src/
│   ├── PicoLLM.Core/           # Tensors, activations, layers, model (ZERO dependencies)
│   │   ├── Tensors/            # Tensor.cs, TensorMath.cs, TensorFactory.cs
│   │   ├── Activations/        # ReLU.cs, GELU.cs, Softmax.cs, Sigmoid.cs, Tanh.cs
│   │   ├── Layers/             # LinearLayer.cs, LayerNorm.cs, MultiHeadAttention.cs,
│   │   │                       # FeedForward.cs, DecoderBlock.cs, TokenEmbedding.cs,
│   │   │                       # PositionalEncoding.cs, EmbeddingLayer.cs
│   │   └── Model/              # PicoLLMModel.cs, ModelConfig.cs
│   │
│   ├── PicoLLM.Tokenizer/     # BpeTokenizer.cs, BpeTrainer.cs, TokenizerConfig.cs
│   │
│   ├── PicoLLM.Training/      # Parameter.cs, ILayer.cs, CrossEntropyLoss.cs, MseLoss.cs,
│   │                           # AdamW.cs, GradientClipper.cs, LearningRateSchedule.cs,
│   │                           # TrainingLoop.cs, TrainingMetrics.cs, CheckpointManager.cs
│   │
│   ├── PicoLLM.Browser/       # HttpFetcher.cs, BrowseSession.cs, BrowseResult.cs,
│   │   │                       # ImageDownloader.cs, RobotsTxtParser.cs, UrlResolver.cs
│   │   └── Parsing/            # HtmlParser.cs, ParsedPage.cs, TextExtractor.cs, ElementFilter.cs
│   │
│   ├── PicoLLM.Gguf/          # GgufWriter.cs, GgufExporter.cs, GgufConstants.cs,
│   │                           # TensorNameMapper.cs, GgufValidator.cs
│   │
│   ├── PicoLLM.Gpu/           # IComputeProvider.cs, CpuComputeProvider.cs,
│   │                           # CudaComputeProvider.cs, GpuDetector.cs, GpuInfo.cs
│   │
│   └── PicoLLM.App/           # Desktop UI + orchestrator
│       ├── Orchestration/      # PicoOrchestrator.cs, OrchestratorConfig.cs, OrchestratorEvent.cs
│       └── Views/              # MainPage, BrowserPane, DashboardPanel, ModelInfoPanel
│
├── tests/
│   └── PicoLLM.Tests/         # xUnit tests for all projects
│
├── openspec/                   # Specifications (read these first!)
│   ├── project.md              # Master project context
│   ├── AGENTS.md               # Workflow instructions for AI agents
│   └── specs/                  # 11 capability specs (spec.md + design.md + tasks.md each)
│       ├── core-tensor/
│       ├── tokenizer/
│       ├── embedding/
│       ├── transformer/
│       ├── training/
│       ├── browser-engine/
│       ├── html-parser/
│       ├── gguf-export/
│       ├── gpu-acceleration/
│       ├── orchestrator/
│       └── ui-shell/
│
├── CLAUDE.md                   # ← YOU ARE HERE
└── INDEX.md                    # Human-readable table of contents
```

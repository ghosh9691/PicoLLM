# AGENTS.md — Instructions for AI Coding Assistants

## Project Context

Always read `openspec/project.md` before starting any implementation work. It contains the full tech stack, constraints, architecture patterns, and build order.

## Build Workflow

This project uses OpenSpec for spec-driven development. The specs are pre-written and live in `openspec/specs/<capability>/`. Each capability folder contains:

- `spec.md` — Requirements (WHAT the system must do). Uses SHALL/MUST keywords.
- `design.md` — Technical design (HOW to implement). Includes class diagrams, data structures, algorithms.
- `tasks.md` — Ordered implementation checklist. Complete tasks top-to-bottom.

## Build Order

Complete capabilities in this exact order. Each builds on the previous:

```
1. core-tensor       → PicoLLM.Core (Tensor, math ops)
2. tokenizer         → PicoLLM.Tokenizer (BPE)
3. embedding         → PicoLLM.Core (EmbeddingLayer, PositionalEncoding)
4. transformer       → PicoLLM.Core (Attention, FFN, LayerNorm, DecoderBlock)
5. training          → PicoLLM.Training (Loss, Backprop, AdamW, TrainingLoop)
6. browser-engine    → PicoLLM.Browser (HTTP fetching, content pipeline)
7. html-parser       → PicoLLM.Browser (HTML → text + images + links)
8. gguf-export       → PicoLLM.Gguf (GGUF v3 binary writer)
9. gpu-acceleration  → PicoLLM.Gpu (optional ILGPU CUDA)
10. orchestrator     → PicoLLM.App (wiring: browse → tokenize → train → save)
11. ui-shell         → PicoLLM.App (desktop UI)
```

## Implementation Rules

1. **Read the spec first.** Open `openspec/specs/<capability>/spec.md` before writing any code.
2. **Follow tasks.md.** Complete checklist items in order. Mark each done as you go.
3. **PicoLLM.Core has ZERO NuGet dependencies.** All tensor math, all layers, all attention — pure C#.
4. **Write tests alongside code.** Each task should have corresponding xUnit tests in `PicoLLM.Tests`.
5. **XML doc comments on every public member.** These are the educational narrative.
6. **One class per file.** File name matches class name.
7. **Use the project structure from project.md.** Don't reorganize unless a spec explicitly says to.

## When Starting a New Capability

```
1. Read openspec/specs/<capability>/spec.md
2. Read openspec/specs/<capability>/design.md
3. Open openspec/specs/<capability>/tasks.md
4. Implement tasks in order
5. Run tests after each task
6. Move to next capability
```

## Code Quality Gates

Before marking a capability complete:
- [ ] All requirements in spec.md are implemented
- [ ] All scenarios in spec.md pass as tests
- [ ] All tasks in tasks.md are checked off
- [ ] `dotnet build` succeeds with zero warnings
- [ ] `dotnet test` passes all tests for this capability
- [ ] XML doc comments present on all public types and methods

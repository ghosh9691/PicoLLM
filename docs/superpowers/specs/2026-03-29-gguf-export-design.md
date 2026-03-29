# GGUF Export — Design

**Date:** 2026-03-29
**Capability:** 8 — gguf-export
**Project:** PicoLLM.Gguf (new)

## Scope

New `PicoLLM.Gguf` project referencing `PicoLLM.Core` and `PicoLLM.Tokenizer`. Pure BCL — no extra NuGet packages. Test project gains a reference to Gguf.

## Architecture

| File | Responsibility |
|------|----------------|
| `GgufConstants.cs` | Magic bytes, version (3), alignment (32), value type enum, metadata key string constants |
| `GgufWriter.cs` | Low-level `BinaryWriter` wrapper — all GGUF primitive writes |
| `TensorNameMapper.cs` | PicoLLM internal tensor names → llama.cpp conventions |
| `GgufExporter.cs` | High-level orchestrator: model + tokenizer → `.gguf` file |
| `GgufValidator.cs` | Read-back verifier: checks magic, version, tensor count |

## File Layout

```
┌─────────────────────────────────┐
│ HEADER (24 bytes)               │
│   Magic "GGUF" (4 bytes)        │
│   Version: 3 (uint32 LE)        │
│   Tensor count (uint64 LE)      │
│   Metadata KV count (uint64 LE) │
├─────────────────────────────────┤
│ METADATA KV PAIRS               │
│   architecture, name, dims...   │
│   tokenizer vocab + types       │
├─────────────────────────────────┤
│ TENSOR INFO ARRAY               │
│   name, n_dims, dims[], type,   │
│   data offset (from data start) │
├─────────────────────────────────┤
│ PADDING to 32-byte boundary     │
├─────────────────────────────────┤
│ TENSOR DATA (F32, LE)           │
│   each tensor 32-byte aligned   │
└─────────────────────────────────┘
```

## String Encoding

GGUF strings: `uint64` length (LE) followed by UTF-8 bytes — no null terminator.

## Tensor Name Mapping

PicoLLM internal names → llama.cpp names (layer index substituted for `{i}`):

| Internal | llama.cpp |
|----------|-----------|
| `Embedding.TokenEmbedding.Weights` | `token_embd.weight` |
| `Block.{i}.Attention.QueryProj.Weights` | `blk.{i}.attn_q.weight` |
| `Block.{i}.Attention.KeyProj.Weights` | `blk.{i}.attn_k.weight` |
| `Block.{i}.Attention.ValueProj.Weights` | `blk.{i}.attn_v.weight` |
| `Block.{i}.Attention.OutputProj.Weights` | `blk.{i}.attn_output.weight` |
| `Block.{i}.AttnNorm.Gamma` | `blk.{i}.attn_norm.weight` |
| `Block.{i}.Ffn.Up.Weights` | `blk.{i}.ffn_up.weight` |
| `Block.{i}.Ffn.Down.Weights` | `blk.{i}.ffn_down.weight` |
| `Block.{i}.FfnNorm.Gamma` | `blk.{i}.ffn_norm.weight` |
| `FinalNorm.Gamma` | `output_norm.weight` |
| `LmHead.Weights` | `output.weight` |

## Architecture Tag

Uses `"llama"` identifier for Ollama/llama.cpp compatibility. Note: PicoLLM uses standard `LayerNorm` (mean+variance), not `RMSNorm`. This is documented in `GgufExporter` as a known tradeoff — full inference compatibility may require switching to RMSNorm.

## Tests

- `GgufWriter`: binary output matches expected bytes for known inputs (header, string, uint32, padding)
- `TensorNameMapper`: all model tensor names map to valid llama.cpp names
- `GgufExporter` integration: export a small model, verify file structure
- `GgufValidator`: export then validate round-trip passes

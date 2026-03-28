# GGUF Export — Technical Design

## File Layout

```
┌──────────────────────────────────┐
│ HEADER (24 bytes)                │
│  Magic: "GGUF" (4 bytes)        │
│  Version: 3 (uint32 LE)         │
│  Tensor Count (uint64 LE)       │
│  Metadata KV Count (uint64 LE)  │
├──────────────────────────────────┤
│ METADATA KV PAIRS               │
│  key: gguf_string                │
│  value_type: uint32              │
│  value: type-specific encoding   │
├──────────────────────────────────┤
│ TENSOR INFO ARRAY                │
│  For each tensor:                │
│    name: gguf_string             │
│    n_dims: uint32                │
│    dims: uint64[n_dims]          │
│    type: uint32 (0 = F32)       │
│    offset: uint64 (from data)   │
├──────────────────────────────────┤
│ PADDING to 32-byte alignment     │
├──────────────────────────────────┤
│ TENSOR DATA                      │
│  Tensors written sequentially    │
│  Each aligned to 32 bytes        │
└──────────────────────────────────┘
```

## GGUF Value Types

```
UINT8   = 0    UINT32  = 4    FLOAT64 = 10
INT8    = 1    INT32   = 5    BOOL    = 7
UINT16  = 2    FLOAT32 = 6    STRING  = 8
INT16   = 3    UINT64  = 9    ARRAY   = 9
                INT64   = 10
```

## GGUF String Format

```csharp
void WriteGgufString(BinaryWriter w, string s)
{
    var bytes = Encoding.UTF8.GetBytes(s);
    w.Write((ulong)bytes.Length);  // uint64 LE
    w.Write(bytes);                // raw UTF-8, no null terminator
}
```

## Tensor Name Mapping

Map PicoLLM internal names to llama.cpp conventions:

```csharp
Dictionary<string, string> _nameMap = new()
{
    ["Embedding.TokenEmbedding.Weights"]     = "token_embd.weight",
    ["Block.{i}.AttnNorm.Gamma"]             = "blk.{i}.attn_norm.weight",
    ["Block.{i}.Attention.QueryProj.Weights"] = "blk.{i}.attn_q.weight",
    ["Block.{i}.Attention.KeyProj.Weights"]   = "blk.{i}.attn_k.weight",
    ["Block.{i}.Attention.ValueProj.Weights"] = "blk.{i}.attn_v.weight",
    ["Block.{i}.Attention.OutputProj.Weights"]= "blk.{i}.attn_output.weight",
    ["Block.{i}.FfnNorm.Gamma"]              = "blk.{i}.ffn_norm.weight",
    ["Block.{i}.Ffn.Up.Weights"]             = "blk.{i}.ffn_up.weight",
    ["Block.{i}.Ffn.Down.Weights"]           = "blk.{i}.ffn_down.weight",
    ["FinalNorm.Gamma"]                       = "output_norm.weight",
    ["LmHead.Weights"]                        = "output.weight",
};
```

## Architecture Choice: "llama"

We use the "llama" architecture identifier because:
- llama.cpp and Ollama natively support it
- The GPT-2 style decoder-only architecture maps cleanly to llama's expected tensor layout
- Pre-norm (RMSNorm in llama, LayerNorm in ours) is close enough for inference

Note: We use standard LayerNorm (mean+variance) not RMSNorm. For full Ollama inference compatibility, we may need to switch to RMSNorm or use a custom architecture tag. Document this tradeoff in the code.

## Writer Class

```csharp
public class GgufWriter : IDisposable
{
    private BinaryWriter _writer;

    public void WriteHeader(int tensorCount, int kvCount) { ... }
    public void WriteMetadataString(string key, string value) { ... }
    public void WriteMetadataUint32(string key, uint value) { ... }
    public void WriteMetadataFloat32(string key, float value) { ... }
    public void WriteMetadataStringArray(string key, string[] values) { ... }
    public void WriteMetadataInt32Array(string key, int[] values) { ... }
    public void WriteTensorInfo(string name, int[] shape, GgufDataType type, ulong offset) { ... }
    public void WriteTensorData(Tensor tensor) { ... }
    public void Pad(int alignment) { ... }
}
```

## Project Location

`src/PicoLLM.Gguf/`:
- `GgufWriter.cs` — low-level binary writer
- `GgufExporter.cs` — high-level: takes PicoLLMModel + BpeTokenizer, writes .gguf
- `GgufConstants.cs` — magic, version, type enums, metadata keys
- `TensorNameMapper.cs` — internal name → llama.cpp name mapping

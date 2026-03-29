namespace PicoLLM.Gguf;

/// <summary>GGUF metadata value types (encodes what follows the key in a KV pair).</summary>
public enum GgufValueType : uint
{
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

/// <summary>GGUF tensor data types (stored in tensor info entries).</summary>
public enum GgufDataType : uint
{
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
}

/// <summary>
/// Fixed constants and metadata key strings for the GGUF v3 format.
/// PicoLLM exports using the "llama" architecture identifier for Ollama compatibility.
/// </summary>
/// <remarks>
/// <b>Known tradeoff:</b> PicoLLM uses standard LayerNorm (mean + variance normalization),
/// while the llama architecture in llama.cpp assumes RMSNorm. The exported file will be
/// structurally valid GGUF but may produce slightly different inference results than a
/// natively-trained LLaMA model. To achieve full inference compatibility, migrate the
/// model to use RMSNorm.
/// </remarks>
public static class GgufConstants
{
    /// <summary>4-byte file magic: ASCII "GGUF".</summary>
    public static readonly byte[] Magic = "GGUF"u8.ToArray();

    /// <summary>GGUF format version we produce.</summary>
    public const uint Version = 3;

    /// <summary>Tensor data section alignment boundary in bytes.</summary>
    public const int Alignment = 32;

    // ── General metadata keys ────────────────────────────────────────────────
    public const string KeyArchitecture      = "general.architecture";
    public const string KeyName              = "general.name";
    public const string KeyFileType          = "general.file_type";

    // ── Llama architecture metadata keys ────────────────────────────────────
    public const string KeyContextLength     = "llama.context_length";
    public const string KeyEmbeddingLength   = "llama.embedding_length";
    public const string KeyBlockCount        = "llama.block_count";
    public const string KeyFeedForwardLength = "llama.feed_forward_length";
    public const string KeyHeadCount         = "llama.attention.head_count";
    public const string KeyHeadCountKv       = "llama.attention.head_count_kv";
    public const string KeyRopeDimCount      = "llama.rope.dimension_count";
    public const string KeyLayerNormEps      = "llama.attention.layer_norm_rms_epsilon";

    // ── Tokenizer metadata keys ──────────────────────────────────────────────
    public const string KeyTokenizerModel    = "tokenizer.ggml.model";
    public const string KeyTokenizerTokens   = "tokenizer.ggml.tokens";
    public const string KeyTokenizerTypes    = "tokenizer.ggml.token_type";
    public const string KeyBosTokenId        = "tokenizer.ggml.bos_token_id";
    public const string KeyEosTokenId        = "tokenizer.ggml.eos_token_id";
    public const string KeyPadTokenId        = "tokenizer.ggml.padding_token_id";

    /// <summary>Total number of metadata KV pairs written by GgufExporter.</summary>
    public const int MetadataKvCount = 17;
}

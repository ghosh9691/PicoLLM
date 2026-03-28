using System.Text.Json.Serialization;

namespace PicoLLM.Tokenizer;

/// <summary>
/// Serializable representation of a trained BPE tokenizer.
/// Written to / read from <c>tokenizer.json</c>.
/// </summary>
public sealed class TokenizerConfig
{
    /// <summary>File format version — bump if the schema changes.</summary>
    [JsonPropertyName("version")]
    public int Version { get; set; } = 1;

    /// <summary>Total vocabulary size (special + byte + merge tokens).</summary>
    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }

    /// <summary>
    /// Special token name → token ID.
    /// Canonical entries: &lt;|pad|&gt;=0, &lt;|unk|&gt;=1, &lt;|bos|&gt;=2, &lt;|eos|&gt;=3.
    /// </summary>
    [JsonPropertyName("special_tokens")]
    public Dictionary<string, int> SpecialTokens { get; set; } = new();

    /// <summary>
    /// Byte token ID → byte value as a 1-element list.
    /// IDs 4–259 cover bytes 0–255.
    /// Stored as a dictionary with string keys for JSON compatibility.
    /// </summary>
    [JsonPropertyName("byte_tokens")]
    public Dictionary<string, List<int>> ByteTokens { get; set; } = new();

    /// <summary>
    /// Ordered merge rules as [leftId, rightId, mergedId] triples.
    /// The order determines priority: earlier = higher priority.
    /// </summary>
    [JsonPropertyName("merges")]
    public List<List<int>> Merges { get; set; } = new();
}

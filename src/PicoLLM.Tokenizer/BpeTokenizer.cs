using System.Text;
using System.Text.Json;

namespace PicoLLM.Tokenizer;

/// <summary>
/// Byte-Pair Encoding tokenizer.
/// Encodes strings into token ID sequences and decodes them back.
/// Load a trained tokenizer with <see cref="Load"/>, or build one from a <see cref="TokenizerConfig"/>
/// returned by <see cref="BpeTrainer"/>.
/// </summary>
public sealed class BpeTokenizer
{
    // ── Special token IDs ────────────────────────────────────────────────────

    /// <summary>Padding token ID.</summary>
    public const int PadId = 0;

    /// <summary>Unknown token ID.</summary>
    public const int UnkId = 1;

    /// <summary>Beginning-of-sequence token ID.</summary>
    public const int BosId = 2;

    /// <summary>End-of-sequence token ID.</summary>
    public const int EosId = 3;

    private const int ByteOffset = 4;

    // ── Internal state ───────────────────────────────────────────────────────

    private readonly Dictionary<int, byte[]> _vocab;           // token ID → byte sequence
    private readonly List<(int Left, int Right)> _mergeRules;  // ordered merge priorities
    private readonly Dictionary<(int, int), int> _mergeToId;   // pair → merged token ID
    private readonly int _vocabSize;

    // ── Construction ─────────────────────────────────────────────────────────

    private BpeTokenizer(
        Dictionary<int, byte[]> vocab,
        List<(int, int)> mergeRules,
        Dictionary<(int, int), int> mergeToId,
        int vocabSize)
    {
        _vocab = vocab;
        _mergeRules = mergeRules;
        _mergeToId = mergeToId;
        _vocabSize = vocabSize;
    }

    /// <summary>The total number of tokens in the vocabulary.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Number of merge rules learned during training.</summary>
    public int MergeCount => _mergeRules.Count;

    // ── Factory: build from config ───────────────────────────────────────────

    /// <summary>
    /// Constructs a <see cref="BpeTokenizer"/> from a <see cref="TokenizerConfig"/>
    /// produced by <see cref="BpeTrainer"/>.
    /// </summary>
    public static BpeTokenizer FromConfig(TokenizerConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        var vocab = new Dictionary<int, byte[]>();

        // Special tokens (empty byte sequences — they are markers, not decoded)
        foreach (var (_, id) in config.SpecialTokens)
            vocab[id] = [];

        // Byte tokens
        foreach (var (idStr, bytesAsInts) in config.ByteTokens)
        {
            int id = int.Parse(idStr);
            vocab[id] = bytesAsInts.Select(b => (byte)b).ToArray();
        }

        // Reconstruct byte sequences for merged tokens from the merge rules
        // (left bytes + right bytes = merged bytes)
        var mergeRules = new List<(int, int)>(config.Merges.Count);
        var mergeToId = new Dictionary<(int, int), int>(config.Merges.Count);
        foreach (var triple in config.Merges)
        {
            int left = triple[0], right = triple[1], merged = triple[2];
            mergeRules.Add((left, right));
            mergeToId[(left, right)] = merged;
            if (!vocab.ContainsKey(merged) && vocab.TryGetValue(left, out var lb) && vocab.TryGetValue(right, out var rb))
                vocab[merged] = lb.Concat(rb).ToArray();
        }

        return new BpeTokenizer(vocab, mergeRules, mergeToId, config.VocabSize);
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes a string into a sequence of token IDs.
    /// The sequence is wrapped with BOS and EOS tokens.
    /// </summary>
    /// <param name="text">UTF-8 text to encode.</param>
    /// <param name="addSpecialTokens">
    ///   When <c>true</c> (default), prepend BOS and append EOS.
    /// </param>
    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Convert to UTF-8 bytes → initial token sequence
        var bytes = Encoding.UTF8.GetBytes(text);
        var tokens = new List<int>(bytes.Length);
        foreach (var b in bytes)
            tokens.Add(b + ByteOffset);

        // Apply merge rules in priority order (first rule = highest priority)
        ApplyMerges(tokens);

        if (addSpecialTokens)
        {
            tokens.Insert(0, BosId);
            tokens.Add(EosId);
        }

        return tokens.ToArray();
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes a sequence of token IDs back into a UTF-8 string.
    /// Special tokens (BOS, EOS, PAD, UNK) are silently skipped.
    /// </summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        var bytes = new List<byte>();
        foreach (var id in tokenIds)
        {
            // Skip special tokens
            if (id == PadId || id == BosId || id == EosId || id == UnkId) continue;
            if (_vocab.TryGetValue(id, out var tokenBytes))
                bytes.AddRange(tokenBytes);
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// <summary>Saves the tokenizer to a JSON file at <paramref name="path"/>.</summary>
    public void Save(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        var config = BuildConfig();
        var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json, Encoding.UTF8);
    }

    /// <summary>Loads a tokenizer from a JSON file previously saved with <see cref="Save"/>.</summary>
    public static BpeTokenizer Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        var json = File.ReadAllText(path, Encoding.UTF8);
        var config = JsonSerializer.Deserialize<TokenizerConfig>(json)
            ?? throw new InvalidDataException($"Failed to deserialize tokenizer from '{path}'.");
        return FromConfig(config);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private void ApplyMerges(List<int> tokens)
    {
        // Iterate merge rules in priority order
        foreach (var (left, right) in _mergeRules)
        {
            int i = 0;
            while (i < tokens.Count - 1)
            {
                if (tokens[i] == left && tokens[i + 1] == right)
                {
                    tokens[i] = _mergeToId[(left, right)];
                    tokens.RemoveAt(i + 1);
                }
                else i++;
            }
        }
    }

    private TokenizerConfig BuildConfig()
    {
        var specialTokens = new Dictionary<string, int>
        {
            ["<|pad|>"] = PadId,
            ["<|unk|>"] = UnkId,
            ["<|bos|>"] = BosId,
            ["<|eos|>"] = EosId,
        };

        var byteTokens = new Dictionary<string, List<int>>();
        for (int b = 0; b < 256; b++)
            byteTokens[(b + ByteOffset).ToString()] = [b];

        var merges = _mergeRules
            .Select((pair, idx) => new List<int> { pair.Left, pair.Right, _mergeToId[pair] })
            .ToList();

        return new TokenizerConfig
        {
            Version = 1,
            VocabSize = _vocabSize,
            SpecialTokens = specialTokens,
            ByteTokens = byteTokens,
            Merges = merges,
        };
    }
}

namespace PicoLLM.Tokenizer;

/// <summary>
/// Trains a Byte-Pair Encoding vocabulary from a raw text corpus.
/// After training, hand the result to <see cref="BpeTokenizer"/> for encode/decode.
/// </summary>
/// <remarks>
/// Algorithm:
/// <list type="number">
///   <item>Convert the corpus to UTF-8 bytes; each byte becomes its own token (IDs 4–259).</item>
///   <item>Count all adjacent pairs in the current token sequence.</item>
///   <item>Find the most frequent pair; merge it into a new token; record the merge rule.</item>
///   <item>Repeat until the target vocabulary size is reached.</item>
/// </list>
/// </remarks>
public sealed class BpeTrainer
{
    // Special token IDs — must match BpeTokenizer constants.
    private const int PadId = 0;
    private const int UnkId = 1;
    private const int BosId = 2;
    private const int EosId = 3;
    private const int ByteOffset = 4; // byte 0 → token ID 4

    /// <summary>
    /// Trains a BPE tokenizer from the given corpus text.
    /// </summary>
    /// <param name="corpus">The raw training text (UTF-8).</param>
    /// <param name="targetVocabSize">
    ///   Total vocabulary size including special tokens and byte tokens.
    ///   Minimum 260 (4 special + 256 bytes).
    /// </param>
    /// <returns>A <see cref="TokenizerConfig"/> ready to save or pass to <see cref="BpeTokenizer"/>.</returns>
    public TokenizerConfig Train(string corpus, int targetVocabSize)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (targetVocabSize < 260)
            throw new ArgumentOutOfRangeException(nameof(targetVocabSize),
                "targetVocabSize must be at least 260 (4 special + 256 byte tokens).");

        // ── 1. Build initial vocabulary ──────────────────────────────────────
        // vocab[id] = byte sequence this token represents
        var vocab = new Dictionary<int, byte[]>();

        // Special tokens
        var specialTokens = new Dictionary<string, int>
        {
            ["<|pad|>"] = PadId,
            ["<|unk|>"] = UnkId,
            ["<|bos|>"] = BosId,
            ["<|eos|>"] = EosId,
        };
        vocab[PadId] = [];
        vocab[UnkId] = [];
        vocab[BosId] = [];
        vocab[EosId] = [];

        // Byte tokens: byte b → ID (b + ByteOffset)
        for (int b = 0; b < 256; b++)
            vocab[b + ByteOffset] = [(byte)b];

        int nextId = ByteOffset + 256; // 260

        // ── 2. Tokenise corpus to initial byte-level sequence ────────────────
        var corpusBytes = System.Text.Encoding.UTF8.GetBytes(corpus);
        var sequence = new List<int>(corpusBytes.Length);
        foreach (var b in corpusBytes)
            sequence.Add(b + ByteOffset);

        // ── 3. Iterative merge loop ──────────────────────────────────────────
        var merges = new List<(int Left, int Right, int Merged)>();
        int mergeRules = targetVocabSize - nextId; // how many merges to perform

        for (int step = 0; step < mergeRules && sequence.Count > 1; step++)
        {
            // Count adjacent pairs
            var pairCounts = CountPairs(sequence);
            if (pairCounts.Count == 0) break;

            // Find the most frequent pair (tie-break by pair value for determinism)
            var best = FindBestPair(pairCounts);

            // Create a new token for this pair
            var newBytes = vocab[best.Left].Concat(vocab[best.Right]).ToArray();
            vocab[nextId] = newBytes;
            merges.Add((best.Left, best.Right, nextId));

            // Replace all occurrences of (left, right) in the sequence
            ApplyMerge(sequence, best.Left, best.Right, nextId);

            nextId++;
        }

        // ── 4. Build config ──────────────────────────────────────────────────
        var byteTokens = new Dictionary<string, List<int>>();
        for (int b = 0; b < 256; b++)
            byteTokens[(b + ByteOffset).ToString()] = [b];

        return new TokenizerConfig
        {
            Version = 1,
            VocabSize = nextId,
            SpecialTokens = specialTokens,
            ByteTokens = byteTokens,
            Merges = merges.Select(m => new List<int> { m.Left, m.Right, m.Merged }).ToList(),
        };
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private static Dictionary<(int, int), int> CountPairs(List<int> sequence)
    {
        var counts = new Dictionary<(int, int), int>();
        for (int i = 0; i < sequence.Count - 1; i++)
        {
            var pair = (sequence[i], sequence[i + 1]);
            counts[pair] = counts.GetValueOrDefault(pair) + 1;
        }
        return counts;
    }

    private static (int Left, int Right) FindBestPair(Dictionary<(int, int), int> counts)
    {
        (int, int) best = default;
        int bestCount = -1;

        foreach (var (pair, count) in counts)
        {
            if (count > bestCount || (count == bestCount && pair.CompareTo(best) < 0))
            {
                bestCount = count;
                best = pair;
            }
        }

        return best;
    }

    private static void ApplyMerge(List<int> sequence, int left, int right, int merged)
    {
        int i = 0;
        while (i < sequence.Count - 1)
        {
            if (sequence[i] == left && sequence[i + 1] == right)
            {
                sequence[i] = merged;
                sequence.RemoveAt(i + 1);
                // Don't advance i — the new token may be part of another merge on the left
            }
            else
            {
                i++;
            }
        }
    }
}

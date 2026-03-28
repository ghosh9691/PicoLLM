# Tokenizer — Technical Design

## Algorithm: Byte-Pair Encoding (BPE)

### Training Phase
1. Convert all text to bytes (UTF-8).
2. Initialize vocabulary with 256 single-byte tokens + 4 special tokens.
3. Count all adjacent token pairs in the corpus.
4. Find the most frequent pair.
5. Merge that pair into a new token, add to vocabulary and merge rules.
6. Repeat steps 3–5 until vocab size target is reached.

### Encoding Phase
1. Convert input string to UTF-8 bytes → initial token sequence (one token per byte).
2. Apply merge rules in priority order (most frequent first):
   - Scan for the highest-priority pair in the current sequence.
   - Replace all occurrences of that pair with the merged token.
   - Repeat until no more merges apply.
3. Prepend `<|bos|>` and append `<|eos|>`.

### Decoding Phase
1. For each token ID, look up its byte sequence.
2. Concatenate all byte sequences.
3. Decode UTF-8.

## Data Structures

```csharp
public class BpeTokenizer
{
    private readonly Dictionary<int, byte[]> _vocab;           // token ID → bytes
    private readonly List<(int, int)> _mergeRules;             // ordered pairs to merge
    private readonly Dictionary<(int, int), int> _mergeToId;   // pair → merged token ID
    private readonly Dictionary<string, int> _specialTokens;
    private int _nextId;
}
```

## Persistence Format (tokenizer.json)

```json
{
    "version": 1,
    "vocab_size": 512,
    "special_tokens": { "<|pad|>": 0, "<|unk|>": 1, "<|bos|>": 2, "<|eos|>": 3 },
    "byte_tokens": { "4": [0], "5": [1], ... },
    "merges": [ [260, 261, 262], ... ]
}
```

## Project Location

`src/PicoLLM.Tokenizer/`:
- `BpeTokenizer.cs` — Core tokenizer
- `BpeTrainer.cs` — Training logic (separate from inference for clarity)
- `TokenizerConfig.cs` — Serializable config

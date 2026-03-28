# Tokenizer — Implementation Tasks

## Setup
- [ ] 1.1 Create `PicoLLM.Tokenizer` project (net9.0 class library)
- [ ] 1.2 Add `System.Text.Json` for serialization (standard library, not NuGet)

## Core Types
- [ ] 2.1 Implement `TokenizerConfig` — vocab size, special tokens map
- [ ] 2.2 Implement special token constants (PAD=0, UNK=1, BOS=2, EOS=3)

## BPE Training
- [ ] 3.1 Implement byte-level initial vocabulary (256 entries, IDs 4–259)
- [ ] 3.2 Implement pair frequency counting across the corpus
- [ ] 3.3 Implement single merge step: find top pair, create new token, update corpus
- [ ] 3.4 Implement full training loop with target vocab size
- [ ] 3.5 Write tests: train on known corpus, verify merge count, verify vocab size

## Encoding
- [ ] 4.1 Implement UTF-8 string → byte token sequence conversion
- [ ] 4.2 Implement iterative merge application (priority order)
- [ ] 4.3 Implement BOS/EOS wrapping
- [ ] 4.4 Write tests: encode known strings, verify token count reduces with merges

## Decoding
- [ ] 5.1 Implement token ID → byte sequence lookup
- [ ] 5.2 Implement byte concatenation → UTF-8 string
- [ ] 5.3 Write tests: round-trip encode/decode on ASCII, Unicode, emoji

## Persistence
- [ ] 6.1 Implement `Save(path)` — write tokenizer.json
- [ ] 6.2 Implement `Load(path)` — read and reconstruct tokenizer
- [ ] 6.3 Write tests: save, load, verify identical encode/decode results

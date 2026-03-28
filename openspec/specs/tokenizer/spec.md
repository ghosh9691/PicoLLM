# Tokenizer Specification

## Purpose

Implement a Byte-Pair Encoding (BPE) tokenizer that can be trained on a corpus of text and then used to encode/decode text for the transformer model.

### Requirement: BPE Training

The system SHALL train a BPE tokenizer from a text corpus by iteratively merging the most frequent adjacent byte/token pairs until a target vocabulary size is reached.

#### Scenario: Train on small corpus
- **GIVEN** a corpus "the cat sat on the mat" and target vocab size of 280
- **WHEN** the tokenizer is trained
- **THEN** it produces a vocabulary with 256 byte-level tokens + up to 24 merge rules
- **AND** the merge rules are ordered by training frequency

### Requirement: Special Tokens

The system SHALL support special tokens: `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`.

#### Scenario: Special token IDs
- **GIVEN** a trained tokenizer
- **WHEN** special token IDs are queried
- **THEN** `<|pad|>` = 0, `<|unk|>` = 1, `<|bos|>` = 2, `<|eos|>` = 3
- **AND** byte-level tokens start at ID 4

### Requirement: Encoding

The system SHALL encode a string into a sequence of token IDs by applying learned merge rules greedily from highest to lowest priority.

#### Scenario: Encode known text
- **GIVEN** a trained tokenizer and input "hello"
- **WHEN** the text is encoded
- **THEN** the output is a list of integer token IDs
- **AND** decoding those IDs back produces "hello"

### Requirement: Decoding

The system SHALL decode a sequence of token IDs back into a UTF-8 string.

#### Scenario: Round-trip encode-decode
- **GIVEN** any valid UTF-8 string
- **WHEN** it is encoded then decoded
- **THEN** the original string is recovered exactly

### Requirement: Vocabulary Persistence

The system SHALL save and load the tokenizer vocabulary and merge rules to/from a JSON file.

#### Scenario: Save and reload
- **GIVEN** a trained tokenizer
- **WHEN** saved to `tokenizer.json` and loaded into a new instance
- **THEN** the new instance produces identical encode/decode results

### Requirement: Vocab Size Configuration

The system SHALL allow configuring the target vocabulary size (minimum 256 for byte-level, typical range 256–8192 for educational use).

#### Scenario: Small vocabulary
- **GIVEN** a target vocab size of 512
- **WHEN** training completes
- **THEN** the vocabulary contains exactly 512 tokens (256 bytes + 4 special + 252 merges)

# GGUF Export Specification

## Purpose

Serialize the trained PicoLLM model to GGUF v3 binary format so it can be loaded by Ollama (`ollama create`) or LM Studio.

### Requirement: GGUF v3 Header

The system SHALL write a valid GGUF v3 header: 4-byte magic "GGUF", uint32 version (3), uint64 tensor count, uint64 metadata KV count.

#### Scenario: Header validation
- **GIVEN** a trained model
- **WHEN** exported to GGUF
- **THEN** the first 4 bytes are ASCII "GGUF"
- **AND** bytes 4–7 are uint32 LE value 3
- **AND** the tensor count matches the model's actual tensor count

### Requirement: Metadata Key-Value Pairs

The system SHALL write all required metadata keys per the GGUF specification.

#### Scenario: Required metadata
- **GIVEN** a PicoLLM model with config (vocab=512, embed=128, heads=4, layers=4, ctx=512)
- **WHEN** exported
- **THEN** metadata includes:
  - `general.architecture` = "llama" (string)
  - `general.name` = "PicoLLM" (string)
  - `general.file_type` = 0 (uint32, F32)
  - `llama.context_length` = 512 (uint32)
  - `llama.embedding_length` = 128 (uint32)
  - `llama.block_count` = 4 (uint32)
  - `llama.feed_forward_length` = 512 (uint32)
  - `llama.attention.head_count` = 4 (uint32)
  - `llama.attention.head_count_kv` = 4 (uint32)
  - `llama.rope.dimension_count` = 32 (uint32) (embed/heads)
  - `llama.attention.layer_norm_rms_epsilon` = 1e-5 (float32)

### Requirement: Tokenizer Metadata

The system SHALL include tokenizer vocabulary and special token metadata in the GGUF file.

#### Scenario: Tokenizer in GGUF
- **GIVEN** a trained BPE tokenizer with vocab_size 512
- **WHEN** exported
- **THEN** metadata includes:
  - `tokenizer.ggml.model` = "gpt2" (string, BPE type)
  - `tokenizer.ggml.tokens` — string array of all token representations
  - `tokenizer.ggml.token_type` — int array of token types (normal=1, special=3, byte=6)
  - `tokenizer.ggml.bos_token_id` = 2 (uint32)
  - `tokenizer.ggml.eos_token_id` = 3 (uint32)
  - `tokenizer.ggml.padding_token_id` = 0 (uint32)

### Requirement: Tensor Data

The system SHALL write all model tensors in F32 format with names following llama.cpp conventions.

#### Scenario: Tensor naming
- **GIVEN** a 4-layer model
- **WHEN** tensors are written
- **THEN** tensor names follow the pattern:
  - `token_embd.weight` — token embedding
  - `blk.{i}.attn_q.weight` — query projection for layer i
  - `blk.{i}.attn_k.weight` — key projection
  - `blk.{i}.attn_v.weight` — value projection
  - `blk.{i}.attn_output.weight` — output projection
  - `blk.{i}.ffn_up.weight` — feedforward up projection
  - `blk.{i}.ffn_down.weight` — feedforward down projection
  - `blk.{i}.attn_norm.weight` — attention layer norm gamma
  - `blk.{i}.ffn_norm.weight` — feedforward layer norm gamma
  - `output_norm.weight` — final layer norm
  - `output.weight` — lm_head projection

### Requirement: Tensor Alignment

The system SHALL align tensor data to 32-byte boundaries as specified by GGUF.

#### Scenario: Alignment padding
- **GIVEN** tensor info section ends at byte offset 1234
- **WHEN** tensor data section begins
- **THEN** it starts at the next 32-byte aligned offset (1248)
- **AND** the gap is filled with 0x00 bytes

### Requirement: Little-Endian Byte Order

The system SHALL write all values in little-endian byte order.

#### Scenario: Endianness
- **GIVEN** a uint32 value of 3
- **WHEN** written to file
- **THEN** bytes are [0x03, 0x00, 0x00, 0x00]

### Requirement: GGUF String Encoding

The system SHALL encode strings as: uint64 length (LE), followed by UTF-8 bytes (no null terminator).

#### Scenario: String encoding
- **GIVEN** the string "llama"
- **WHEN** encoded
- **THEN** bytes are: [5, 0, 0, 0, 0, 0, 0, 0, 'l', 'l', 'a', 'm', 'a']

### Requirement: Ollama Compatibility

The exported GGUF file SHALL be usable with `ollama create` using a Modelfile.

#### Scenario: Ollama import
- **GIVEN** an exported `picollm.gguf` file
- **WHEN** a Modelfile contains `FROM ./picollm.gguf`
- **AND** `ollama create picollm -f Modelfile` is run
- **THEN** Ollama accepts the model without errors

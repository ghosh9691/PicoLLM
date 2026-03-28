# GGUF Export — Implementation Tasks

## Setup
- [ ] 1.1 Create `PicoLLM.Gguf` project, reference PicoLLM.Core and PicoLLM.Tokenizer

## Constants and Types
- [ ] 2.1 Implement `GgufConstants` — magic bytes, version, alignment, metadata keys
- [ ] 2.2 Implement `GgufDataType` enum (F32=0, F16=1, Q4_0=2, etc. — we only use F32)
- [ ] 2.3 Implement `GgufValueType` enum (UINT8=0 through ARRAY=9)

## Low-Level Writer
- [ ] 3.1 Implement `GgufWriter` wrapping BinaryWriter with little-endian
- [ ] 3.2 Implement `WriteGgufString(string)` — uint64 length + UTF-8 bytes
- [ ] 3.3 Implement `WriteHeader(tensorCount, kvCount)`
- [ ] 3.4 Implement `WriteMetadataString(key, value)`
- [ ] 3.5 Implement `WriteMetadataUint32(key, value)`
- [ ] 3.6 Implement `WriteMetadataFloat32(key, value)`
- [ ] 3.7 Implement `WriteMetadataStringArray(key, values)`
- [ ] 3.8 Implement `WriteMetadataInt32Array(key, values)`
- [ ] 3.9 Implement `WriteTensorInfo(name, shape, type, offset)`
- [ ] 3.10 Implement `Pad(alignment)` — write 0x00 bytes to next boundary
- [ ] 3.11 Implement `WriteTensorData(float[])` — raw LE float bytes
- [ ] 3.12 Write tests: verify binary output matches expected bytes for known inputs

## Tensor Name Mapping
- [ ] 4.1 Implement `TensorNameMapper` — PicoLLM internal names → llama.cpp convention
- [ ] 4.2 Handle layer index substitution (Block.{i}.*)
- [ ] 4.3 Write tests: verify all model tensors map to valid llama.cpp names

## High-Level Exporter
- [ ] 5.1 Implement `GgufExporter.Export(model, tokenizer, outputPath)`
- [ ] 5.2 Write model metadata (architecture, dimensions, context length)
- [ ] 5.3 Write tokenizer metadata (vocab tokens, token types, special token IDs)
- [ ] 5.4 Compute tensor offsets (sequential, 32-byte aligned)
- [ ] 5.5 Write tensor info array
- [ ] 5.6 Write padding between info and data sections
- [ ] 5.7 Write all tensor data sequentially with alignment
- [ ] 5.8 Write integration test: export a small model, verify file structure

## Validation
- [ ] 6.1 Implement `GgufValidator.Validate(path)` — read back and check header/metadata/tensors
- [ ] 6.2 Write test: export then validate round-trip
- [ ] 6.3 Document manual Ollama import test procedure in README

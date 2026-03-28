# Orchestrator — Implementation Tasks

## Setup
- [ ] 1.1 Create orchestration classes in `PicoLLM.App` project
- [ ] 1.2 Reference all other projects (Core, Tokenizer, Training, Browser, Gguf, optionally Gpu)

## Event System
- [ ] 2.1 Implement all `OrchestratorEvent` record types
- [ ] 2.2 Implement event emission via `Action<OrchestratorEvent>` delegate

## Configuration
- [ ] 3.1 Implement `OrchestratorConfig` with defaults
- [ ] 3.2 Implement `TrainingConfig` with defaults
- [ ] 3.3 Support loading config from JSON file (optional, can hardcode defaults initially)

## State Management
- [ ] 4.1 Implement checkpoint detection and loading at session start
- [ ] 4.2 Implement tokenizer detection and loading at session start
- [ ] 4.3 Implement fresh model initialization when no checkpoint exists
- [ ] 4.4 Write tests: load existing state, init fresh state

## Browse-to-Train Pipeline
- [ ] 5.1 Implement `ProcessUrlAsync(url)` — fetch → parse → accumulate text → train
- [ ] 5.2 Implement `ProcessUrlsAsync(urls)` — iterate with error resilience
- [ ] 5.3 Implement corpus accumulation (concatenate page texts)
- [ ] 5.4 Implement tokenizer training on first session (if no tokenizer.json)
- [ ] 5.5 Implement training step execution (N steps per page)
- [ ] 5.6 Wire progress events at each stage
- [ ] 5.7 Write integration test: mock HTTP → full pipeline → verify model updated

## Session End
- [ ] 6.1 Implement `EndSession(ggufPath)` — save checkpoint + export GGUF
- [ ] 6.2 Emit checkpoint and export events with file paths
- [ ] 6.3 Write test: end session produces both files on disk

## Error Handling
- [ ] 7.1 Implement per-URL try/catch with skip-and-continue
- [ ] 7.2 Emit `SessionErrorEvent` for failed URLs
- [ ] 7.3 Write test: one bad URL in a batch doesn't stop the rest

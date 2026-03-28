# Orchestrator Specification

## Purpose

Wire the browser, parser, tokenizer, trainer, and model persistence into a single pipeline: browse pages → extract text → tokenize → train → checkpoint → export GGUF. This is the "brain" that coordinates all subsystems.

### Requirement: Browse-to-Train Pipeline

The system SHALL accept a list of URLs (or allow interactive browsing), fetch and parse each page, accumulate text into a corpus, tokenize it, and run training steps on the model.

#### Scenario: Train from a single page
- **GIVEN** a URL "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
- **WHEN** the orchestrator processes it
- **THEN** the page is fetched, parsed to clean text, tokenized, and used for one or more training steps
- **AND** model weights are updated

#### Scenario: Train from multiple pages in a session
- **GIVEN** a browsing session visiting 5 pages
- **WHEN** the session ends
- **THEN** text from all 5 pages has been incorporated into training
- **AND** a checkpoint is saved

### Requirement: Incremental Learning

The system SHALL load an existing model checkpoint (if available) before training on new content, so knowledge accumulates across sessions.

#### Scenario: Resume from prior session
- **GIVEN** a checkpoint file from a previous session exists at the configured path
- **WHEN** a new browsing session starts
- **THEN** the model and optimizer state are loaded from the checkpoint
- **AND** training continues from where it left off

#### Scenario: First session (no checkpoint)
- **GIVEN** no checkpoint file exists
- **WHEN** a new session starts
- **THEN** a fresh model is initialized from scratch using the configured hyperparameters

### Requirement: Tokenizer Training vs Reuse

The system SHALL train a new BPE tokenizer from the first session's corpus if no tokenizer exists, and reuse the existing tokenizer for subsequent sessions.

#### Scenario: First session tokenizer
- **GIVEN** no saved tokenizer exists
- **WHEN** the first batch of pages is fetched
- **THEN** a BPE tokenizer is trained on the accumulated text
- **AND** saved to `tokenizer.json` for future sessions

#### Scenario: Subsequent sessions
- **GIVEN** a saved `tokenizer.json` exists
- **WHEN** a new session starts
- **THEN** the existing tokenizer is loaded and used without retraining

### Requirement: GGUF Export on Session End

The system SHALL export the current model to GGUF format when the user ends a browsing/training session.

#### Scenario: Session end export
- **GIVEN** a session with at least one training step completed
- **WHEN** the user ends the session
- **THEN** the model is exported to `picollm.gguf`
- **AND** a checkpoint is also saved for future incremental training
- **AND** the user is notified of the export file location

### Requirement: Training Configuration

The system SHALL accept a configuration for training hyperparameters.

#### Scenario: Default config
- **GIVEN** no user overrides
- **WHEN** training begins
- **THEN** defaults are used: batch_size=4, seq_len=128, lr=1e-4, warmup_steps=50, steps_per_page=100, model config per project.md defaults

### Requirement: Progress Reporting

The system SHALL emit progress events that the UI can subscribe to: page fetch status, training step metrics (loss, lr, tokens/sec), and export status.

#### Scenario: Progress events
- **GIVEN** the orchestrator is processing a page
- **WHEN** each stage completes
- **THEN** events are emitted: PageFetched, PageParsed, TokenizationComplete, TrainingStepComplete(loss, lr), ExportComplete(path)

### Requirement: Error Resilience

The system SHALL skip pages that fail to fetch or parse and continue with the remaining pages in the session.

#### Scenario: One page fails
- **GIVEN** a session with 5 URLs where URL #3 returns a 404
- **WHEN** the session runs
- **THEN** URLs 1, 2, 4, 5 are processed normally
- **AND** URL #3 is reported as skipped with an error message

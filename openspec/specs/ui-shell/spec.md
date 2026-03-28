# UI Shell Specification

## Purpose

Provide a desktop application with two panes: a Lynx-style text browser on the left and a training dashboard on the right, allowing the user to browse the web and watch the LLM learn in real time.

### Requirement: Address Bar

The system SHALL provide a URL input field where the user can type a URL and press Enter (or click Go) to navigate.

#### Scenario: Navigate to URL
- **GIVEN** the user types "https://en.wikipedia.org/wiki/Neural_network" in the address bar
- **WHEN** Enter is pressed
- **THEN** the page is fetched, parsed, and displayed in the browser pane
- **AND** training begins on the parsed content

### Requirement: Browser Pane (Lynx-style)

The system SHALL display parsed web content as styled text: headings in bold/larger font, body text in normal font, links as underlined/colored text, and image alt-text in brackets.

#### Scenario: Text rendering
- **GIVEN** a parsed page with headings, paragraphs, links, and images
- **WHEN** displayed in the browser pane
- **THEN** headings appear bold, paragraphs are separated by whitespace
- **AND** links are underlined and clickable (navigating to their target URL)
- **AND** images show "[Image: alt text]" inline

### Requirement: Link Navigation

The system SHALL allow clicking on links in the browser pane to navigate to that URL, adding the new page to the training session.

#### Scenario: Click a link
- **GIVEN** the browser pane shows a page with a link to "/about"
- **WHEN** the user clicks the link
- **THEN** the address bar updates to the resolved absolute URL
- **AND** the new page is fetched, parsed, displayed, and trained on

### Requirement: Training Dashboard

The system SHALL display real-time training metrics: current loss, learning rate, tokens processed per second, gradient norm, total steps, pages processed, and GPU status.

#### Scenario: Dashboard updates during training
- **GIVEN** training is in progress
- **WHEN** each training step completes
- **THEN** the dashboard updates loss, lr, tokens/sec, and grad norm values
- **AND** a loss chart shows loss over time

### Requirement: Loss Chart

The system SHALL display a simple line chart of loss over training steps.

#### Scenario: Loss trend visible
- **GIVEN** 100 training steps have completed
- **WHEN** the loss chart is viewed
- **THEN** it shows a line plot with step on x-axis and loss on y-axis
- **AND** a general downward trend is visible (assuming the model is learning)

### Requirement: Session Controls

The system SHALL provide Start Session, End Session, and Export GGUF buttons.

#### Scenario: End session and export
- **GIVEN** the user has browsed several pages and training is in progress
- **WHEN** the "End Session" button is clicked
- **THEN** training stops, a checkpoint is saved, GGUF is exported
- **AND** a dialog shows the export file path

### Requirement: Model Info Panel

The system SHALL display model configuration: parameter count, vocab size, embed dimension, layer count, head count, context length, and GPU availability.

#### Scenario: Model info visible
- **GIVEN** a model is initialized
- **WHEN** the model info panel is viewed
- **THEN** it shows "Parameters: 1.2M, Vocab: 512, Embed: 128, Layers: 4, Heads: 4, Context: 512"

### Requirement: Status Bar

The system SHALL display a status bar at the bottom showing: current activity (Idle / Fetching / Parsing / Training / Exporting), session duration, and total pages browsed.

#### Scenario: Status during fetch
- **GIVEN** a page is being fetched
- **WHEN** the status bar is viewed
- **THEN** it shows "Fetching https://example.com..."

### Requirement: Cross-Platform

The system SHALL run on Windows and macOS using .NET MAUI or an alternative cross-platform UI framework.

#### Scenario: Windows and macOS
- **GIVEN** the application
- **WHEN** built for Windows
- **THEN** it runs as a native Windows desktop app
- **WHEN** built for macOS
- **THEN** it runs as a native macOS desktop app

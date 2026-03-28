# UI Shell — Implementation Tasks

## Setup
- [ ] 1.1 Configure `PicoLLM.App` as MAUI app (or Avalonia — builder's choice)
- [ ] 1.2 Set target frameworks: net9.0-windows, net9.0-maccatalyst (MAUI) or net9.0 (Avalonia)
- [ ] 1.3 Reference all project libraries

## Main Layout
- [ ] 2.1 Create main window with split pane layout (browser left, dashboard right)
- [ ] 2.2 Create address bar with URL text input and Go/Stop buttons
- [ ] 2.3 Create status bar at bottom

## Browser Pane
- [ ] 3.1 Implement text rendering for parsed page content
- [ ] 3.2 Render headings as bold/larger text
- [ ] 3.3 Render links as underlined/colored clickable text
- [ ] 3.4 Render image alt text as gray italic bracketed text
- [ ] 3.5 Implement click handler for links → navigate to URL
- [ ] 3.6 Implement scrollable content area

## Training Dashboard
- [ ] 4.1 Implement metric labels: loss, LR, tokens/sec, grad norm, step, pages
- [ ] 4.2 Subscribe to orchestrator progress events
- [ ] 4.3 Update metrics on UI thread
- [ ] 4.4 Implement loss chart (simple line plot, last 500 points)

## Model Info Panel
- [ ] 5.1 Display parameter count, vocab size, embed dim, layers, heads, context
- [ ] 5.2 Display GPU detection result (name or "CPU only")

## Session Controls
- [ ] 6.1 Implement "Start Session" button → init orchestrator
- [ ] 6.2 Implement "End Session & Export" button → stop training, save checkpoint, export GGUF
- [ ] 6.3 Show file path dialog/notification on export complete
- [ ] 6.4 Implement Stop/Cancel with CancellationToken

## Address Bar Navigation
- [ ] 7.1 Handle Enter key → process URL through orchestrator
- [ ] 7.2 Handle Go button click
- [ ] 7.3 Update address bar when link is clicked in browser pane

## Status Bar
- [ ] 8.1 Show current activity (Idle / Fetching / Parsing / Training / Exporting)
- [ ] 8.2 Show session duration timer
- [ ] 8.3 Show total pages browsed count

## Threading
- [ ] 9.1 Run orchestrator on background thread
- [ ] 9.2 Dispatch UI updates to main thread
- [ ] 9.3 Test: UI remains responsive during training

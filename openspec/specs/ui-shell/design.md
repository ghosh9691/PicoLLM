# UI Shell — Technical Design

## Framework Choice

**Option A (Recommended): .NET MAUI** — native cross-platform for Windows + macOS. Single codebase.

**Option B: Avalonia UI** — if MAUI proves problematic on macOS, Avalonia is a mature alternative with similar XAML-based development.

**Option C: WinForms (Windows only)** — simplest, but no macOS support. Use only if cross-platform is deprioritized.

The builder should choose based on their environment. The orchestrator and all business logic are UI-framework-agnostic.

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  [Address Bar: URL input]                          [Go] [Stop]   │
├────────────────────────────────┬─────────────────────────────────┤
│                                │  Training Dashboard             │
│  Browser Pane                  │  ┌───────────────────────────┐  │
│  (Lynx-style text view)       │  │ Loss: 3.245               │  │
│                                │  │ LR: 0.0001               │  │
│  # Page Title                  │  │ Tokens/s: 1,234           │  │
│                                │  │ Grad Norm: 0.82           │  │
│  This is paragraph text with   │  │ Step: 347 / 500           │  │
│  [links] shown in color.       │  │ Pages: 3                  │  │
│                                │  ├───────────────────────────┤  │
│  [Image: A diagram showing     │  │ Loss Chart                │  │
│   neural network layers]       │  │  ╲                        │  │
│                                │  │   ╲__                     │  │
│  ## Subheading                 │  │      ╲___╱╲___            │  │
│                                │  │                ╲____      │  │
│  More text content here...     │  ├───────────────────────────┤  │
│                                │  │ Model Info                │  │
│                                │  │ Params: 1.2M              │  │
│                                │  │ Vocab: 512 | Embed: 128   │  │
│                                │  │ Layers: 4 | Heads: 4      │  │
│                                │  │ Context: 512              │  │
│                                │  │ GPU: NVIDIA RTX 3060 ✓    │  │
│                                │  └───────────────────────────┘  │
├────────────────────────────────┴─────────────────────────────────┤
│  Status: Training on page 3/5 | Step 347 | Session: 00:12:34    │
│  [Start Session]  [End Session & Export GGUF]                    │
└──────────────────────────────────────────────────────────────────┘
```

## Browser Pane Rendering

Use a `ScrollView` containing a `FormattedText` or `RichTextBox` equivalent:
- Headings: bold, larger font size
- Body text: normal font
- Links: blue, underlined, clickable (mouse event → navigate)
- Image alt text: gray italic, bracketed `[Image: ...]`
- Paragraph breaks: vertical spacing between blocks

For MAUI: use a `CollectionView` or `Label` with `FormattedString` and `Span` elements.
For Avalonia: use `TextBlock` with `Inlines`.

## Training Dashboard

Subscribe to `PicoOrchestrator.OnProgress` events and update UI on the main thread.

For the loss chart:
- MAUI: Use a simple `GraphicsView` with custom draw logic (line chart)
- Avalonia: Use OxyPlot.Avalonia or custom SkiaSharp rendering

Keep the last 500 data points in a ring buffer for display.

## Threading Model

- Browser pane and dashboard run on the UI thread
- Orchestrator (fetch + parse + train) runs on a background thread
- Progress events are dispatched to UI thread via `Dispatcher.Dispatch()` or equivalent
- Use `CancellationToken` for the Stop button

## Project Location

`src/PicoLLM.App/`:
- `MainPage.xaml` + `MainPage.xaml.cs` (or equivalent)
- `ViewModels/MainViewModel.cs`
- `Views/BrowserPane.cs` — text rendering
- `Views/DashboardPanel.cs` — metrics + chart
- `Views/ModelInfoPanel.cs`
- `Converters/` — value converters for binding

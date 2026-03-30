# UI Shell — Design Document
**Date:** 2026-03-30
**Capability:** 11 — ui-shell
**Project:** PicoLLM.App (Avalonia desktop)

---

## Framework & Platform

- **UI Framework:** Avalonia UI (NuGet — no workload required)
- **Target frameworks:** `net10.0-windows` and `net10.0-macos`
- **Theme:** Avalonia Fluent theme
- **Architecture pattern:** MVVM-Lite (`CommunityToolkit.Mvvm` source generators)
- `PicoLLM.Tests` stays on `net10.0` — references `Orchestration/` types only, no Avalonia dependency

---

## Project Structure

`PicoLLM.App` is converted from a plain `Microsoft.NET.Sdk` class library to an Avalonia app project in-place. The existing `Orchestration/` folder is untouched.

```
src/PicoLLM.App/
├── PicoLLM.App.csproj              ← Avalonia app, net10.0-windows + net10.0-macos
├── App.axaml + App.axaml.cs        ← Avalonia application entry point
├── MainWindow.axaml + .cs          ← top-level window, wires ViewModel
├── ViewModels/
│   ├── MainViewModel.cs            ← all live UI state + commands
│   └── SettingsViewModel.cs        ← data dir + model config, persisted to JSON
├── Views/
│   ├── BrowserPane.axaml           ← Lynx-style text renderer, back/forward
│   ├── DashboardPanel.axaml        ← metrics grid + loss chart
│   ├── LossChartControl.cs         ← custom Avalonia Control with DrawingContext render
│   ├── ModelInfoPanel.axaml        ← static model config display
│   └── SettingsPanel.axaml         ← modal dialog: data dir + hyperparams
└── Orchestration/                  ← unchanged (PicoOrchestrator, config, events)
```

**NuGet packages added to `PicoLLM.App.csproj`:**
- `Avalonia`
- `Avalonia.Desktop`
- `Avalonia.Themes.Fluent`
- `CommunityToolkit.Mvvm`

---

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  [Address Bar: URL input]                          [Go] [Stop]   │
├────────────────────────────────┬─────────────────────────────────┤
│  ← →  en.wikipedia.org/…       │  Training Dashboard             │
│  ─────────────────────────     │  ┌───────────────────────────┐  │
│                                │  │ Loss  3.245  LR  0.0001   │  │
│  # Page Title                  │  │ Tok/s 1,234  ‖∇‖  0.82   │  │
│                                │  │ Step  347    Pages  3     │  │
│  Body text in near-black       │  ├───────────────────────────┤  │
│  #111111 on white background.  │  │ Loss Chart (custom canvas)│  │
│                                │  │  ╲___╱╲___╲____           │  │
│  Unvisited links blue #0000EE  │  ├───────────────────────────┤  │
│  Visited links purple #551A8B  │  │ Model Info                │  │
│                                │  │ Params: 1.2M              │  │
│  [Image: alt text in italic    │  │ Vocab: 1024 | Embed: 128  │  │
│   grey with left border]       │  │ Layers: 4   | Heads: 4    │  │
│                                │  │ Context: 512              │  │
│                                │  │ GPU: NVIDIA RTX 3060 ✓    │  │
│                                │  └───────────────────────────┘  │
├────────────────────────────────┴─────────────────────────────────┤
│  ● Training on page 3 | Step 347 | Session: 00:12:34             │
│  [Start Session]  [End Session & Export GGUF]                    │
└──────────────────────────────────────────────────────────────────┘
```

- Browser pane: ~60% width, white background, scrollable
- Dashboard: ~40% width, dark background panels, scrollable if needed
- Address bar: full width across the top
- Status bar + session buttons: full width across the bottom

---

## MainViewModel

All live UI state lives in one `MainViewModel`. `CommunityToolkit.Mvvm` source generators emit `INotifyPropertyChanged` boilerplate.

### Observable Properties

```csharp
// Training metrics
[ObservableProperty] float _loss;
[ObservableProperty] float _learningRate;
[ObservableProperty] float _tokensPerSec;
[ObservableProperty] float _gradNorm;
[ObservableProperty] int _step;
[ObservableProperty] int _pagesProcessed;

// Session state
[ObservableProperty] string _statusText = "Idle";
[ObservableProperty] bool _isSessionActive;
[ObservableProperty] TimeSpan _sessionDuration;

// Browser navigation
[ObservableProperty] string _addressBarUrl = "";
[ObservableProperty] FormattedPageContent? _currentPage;
[ObservableProperty] bool _canGoBack;
[ObservableProperty] bool _canGoForward;

// Loss chart (ring buffer, last 500 points)
public ObservableCollection<float> LossHistory { get; } = new();
```

### Commands

```csharp
[RelayCommand] Task NavigateAsync(string url)    // address bar Enter / Go button / link click
[RelayCommand] void GoBack()
[RelayCommand] void GoForward()
[RelayCommand] Task StartSessionAsync()          // opens SettingsPanel if not yet configured
[RelayCommand] Task EndSessionAsync()            // checkpoint + GGUF export + save file dialog
[RelayCommand] void StopNavigation()             // cancels in-flight fetch via CancellationToken
```

### Navigation History

```csharp
private readonly Stack<string> _backStack  = new();
private readonly Stack<string> _fwdStack   = new();
private readonly HashSet<string> _visitedUrls = new();
```

Navigating a new URL clears `_fwdStack`. `_visitedUrls` drives the blue/purple link colour in the browser pane.

---

## SettingsViewModel

Persisted to `settings.json` in the **fixed platform AppData dir** (`%APPDATA%\PicoLLM\settings.json` / `~/Library/Application Support/PicoLLM/settings.json`), not inside `DataDirectory`. This avoids a chicken-and-egg problem where the data directory itself is a setting. Loaded on app startup; written on "Save & Start Session".

```csharp
string DataDirectory       // default: platform AppData/PicoLLM
int    VocabSize           // default: 1024
int    EmbedDim            // default: 128
int    NumHeads            // default: 4
int    NumLayers           // default: 4
int    MaxSeqLen           // default: 512
int    StepsPerPage        // default: 100
float  LearningRate        // default: 1e-4f
int    BatchSize           // default: 4
int    SeqLen              // default: 128
int    WarmupSteps         // default: 50
```

**Platform defaults:**
- Windows: `%APPDATA%\PicoLLM`
- macOS: `~/Library/Application Support/PicoLLM`

The settings panel opens as a modal dialog when "Start Session" is clicked and no session is active. It pre-fills all defaults on first run.

---

## Browser Pane

### Render Model

`HtmlParser` produces a `ParsedPage`. The browser pane does not render `CleanText` directly — `MainViewModel.NavigateAsync` builds a `FormattedPageContent` by parsing the `ParsedPage`:

```csharp
public record FormattedPageContent(IReadOnlyList<IPageElement> Elements);

public interface IPageElement { }
public record HeadingElement(string Text, int Level) : IPageElement;       // H1=1, H2=2, H3=3
public record ParagraphElement(IReadOnlyList<TextRun> Runs) : IPageElement;
public record ImageAltElement(string AltText) : IPageElement;

public record TextRun(string Text, bool IsLink, string? Href, bool IsVisited);
```

**Builder strategy** (`FormattedPageContentBuilder.Build(ParsedPage)`):

1. Split `CleanText` on `\n\n` into blocks.
2. A block starting with `# ` / `## ` / `### ` etc. → `HeadingElement` (strip the prefix, level = number of `#`).
3. A block matching `[Image: ...]` → `ImageAltElement`.
4. All other blocks → `ParagraphElement`. Within each paragraph, scan for anchor text from `ParsedPage.Links` (longest match first) and split into `TextRun` spans: matching spans become `IsLink=true` with the corresponding `Href`; remainder become plain `TextRun`. Where the same anchor text maps to multiple hrefs, use the first matching `LinkReference`.

`BrowserPane` renders each `IPageElement` using an Avalonia `ItemsControl` with `DataTemplates` — one template per element type.

### Text & Colour Spec

| Element | Font | Colour |
|---------|------|--------|
| H1 | 20px Bold | `#111111` |
| H2 | 16px Bold + bottom rule | `#111111` |
| H3 | 14px Bold | `#222222` |
| Body text | 13px Regular | `#111111` |
| Unvisited link | 13px Underline | `#0000EE` |
| Visited link | 13px Underline | `#551A8B` |
| Image alt text | 12px Italic | `#555555`, left-border accent |
| Source metadata | 11px Regular | `#888888` |
| Background | — | `#FFFFFF` |

### Back / Forward

Back and Forward buttons live inside the browser pane header row (alongside the current URL display). Clicking a link calls `NavigateAsync` which pushes to `_backStack` before navigating.

---

## Training Dashboard

### Metrics Grid

Six values displayed in a 2×3 grid, updated on each `TrainingStepEvent`:
Loss, Learning Rate, Tokens/sec, Grad Norm, Step, Pages.

### Loss Chart — Custom Canvas Control

`LossChartControl` extends Avalonia `Control` and overrides `Render(DrawingContext context)`.

- Reads `LossHistory` (`ObservableCollection<float>`) from the binding context
- Subscribes to `CollectionChanged` → calls `InvalidateVisual()` to trigger a redraw
- Draws: background fill, X/Y axes, auto-scaled Y axis (min/max of visible window), polyline of loss points
- Ring buffer: keeps last 500 points; older points are discarded

No NuGet charting library required.

### Model Info Panel

Static display populated once when the session starts, from `OrchestratorConfig.Model`:
parameter count (computed), vocab size, embed dim, layers, heads, context length, GPU name (from `GpuDetector`).

---

## Threading Model

- `PicoOrchestrator.ProcessUrlsAsync` is called via `Task.Run()` from `MainViewModel`
- All `OnProgress` event handlers marshal to the UI thread: `Dispatcher.UIThread.Post(() => { ... })`
- `CancellationTokenSource` is held in `MainViewModel`; Stop button calls `.Cancel()`
- Session duration timer: `DispatcherTimer` on the UI thread, ticks every second

---

## Session Flow

1. **Cold start:** App opens, all controls disabled except "Start Session" and address bar (greyed out)
2. **Start Session:** Opens `SettingsPanel` modal (pre-filled defaults). "Save & Start Session" initialises `PicoOrchestrator`, enables address bar and controls
3. **Browse:** User types URL → `NavigateAsync` → fetch → parse → display in browser pane → train → emit events → update dashboard
4. **End Session:** `EndSessionAsync` → `PicoOrchestrator.EndSession(path)` → save file dialog for GGUF path → emits `CheckpointSavedEvent` + `GgufExportedEvent` → status bar shows export path
5. **Between sessions:** Browser pane and dashboard remain visible (read-only). Navigation controls (address bar, back/forward, Go) are disabled. "Start Session" re-opens settings to begin a new session.

---

## Error Handling

- Failed fetches (`BrowseStatus.Error`): status bar shows `"Error fetching <url>"`, browser pane shows inline error message, training continues on previously seen content
- Orchestrator exceptions: caught in `NavigateAsync`, displayed in status bar, session remains active
- End session with no trained model (no pages browsed): "End Session" button is disabled until at least one page has been successfully processed

---

## File Map

| File | Responsibility |
|------|---------------|
| `PicoLLM.App.csproj` | Avalonia app project, net10.0-windows + net10.0-macos |
| `App.axaml` | Avalonia app resources, theme |
| `MainWindow.axaml` | Root layout: address bar, split pane, status bar |
| `ViewModels/MainViewModel.cs` | All live state + commands |
| `ViewModels/SettingsViewModel.cs` | Config persistence |
| `Views/BrowserPane.axaml` | Lynx-style page renderer |
| `Views/DashboardPanel.axaml` | Metrics grid + loss chart + model info |
| `Views/LossChartControl.cs` | Custom DrawingContext chart |
| `Views/ModelInfoPanel.axaml` | Static model config display |
| `Views/SettingsPanel.axaml` | Modal settings dialog |
| `Models/FormattedPageContent.cs` | `IPageElement` hierarchy + `TextRun` |
| `Models/FormattedPageContentBuilder.cs` | Builds `FormattedPageContent` from `ParsedPage` |

using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PicoLLM.App.Models;
using PicoLLM.App.Orchestration;
using PicoLLM.Browser;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.App.ViewModels;

/// <summary>
/// Central view model for the PicoLLM desktop application.
/// Owns all live UI state and commands. Runs the orchestrator on a background thread
/// and marshals progress events back to the UI thread via the injected dispatcher.
/// </summary>
public partial class MainViewModel : ObservableObject
{
    // ── Dispatcher ───────────────────────────────────────────────────────────
    // Injected so tests can provide a synchronous dispatch (no Avalonia runtime needed).
    private readonly Action<Action> _dispatch;

    // ── Navigation state ─────────────────────────────────────────────────────
    private readonly Stack<string> _backStack = new();
    private readonly Stack<string> _fwdStack = new();
    private readonly HashSet<string> _visitedUrls = new();
    private string? _currentUrl;

    // ── Orchestrator ──────────────────────────────────────────────────────────
    private PicoOrchestrator? _orchestrator;
    private CancellationTokenSource? _cts;

    // Shared fetcher for standalone browsing (when no training session is active).
    private readonly HttpFetcher _httpFetcher = new();

    // ── Observable properties ─────────────────────────────────────────────────

    /// <summary>Text shown in the address bar.</summary>
    [ObservableProperty] private string _addressBarUrl = "";

    /// <summary>The rendered page currently displayed in the browser pane.</summary>
    [ObservableProperty] private FormattedPageContent? _currentPage;

    /// <summary>True when a previous page is available.</summary>
    [ObservableProperty] private bool _canGoBack;

    /// <summary>True when a forward page is available (after going back).</summary>
    [ObservableProperty] private bool _canGoForward;

    /// <summary>Training loss from the latest step.</summary>
    [ObservableProperty] private float _loss;

    /// <summary>Current learning rate.</summary>
    [ObservableProperty] private float _learningRate;

    /// <summary>Training throughput in tokens per second.</summary>
    [ObservableProperty] private float _tokensPerSec;

    /// <summary>Global gradient L2 norm before clipping.</summary>
    [ObservableProperty] private float _gradNorm;

    /// <summary>Total training steps completed this session.</summary>
    [ObservableProperty] private int _step;

    /// <summary>Number of pages processed this session.</summary>
    [ObservableProperty] private int _pagesProcessed;

    /// <summary>Human-readable status line for the status bar.</summary>
    [ObservableProperty] private string _statusText = "Idle";

    /// <summary>True when a session is active (orchestrator initialised).</summary>
    [ObservableProperty] private bool _isSessionActive;

    /// <summary>Elapsed session duration shown in the status bar.</summary>
    [ObservableProperty] private TimeSpan _sessionDuration;

    /// <summary>Parameter count (set once when session starts).</summary>
    [ObservableProperty] private string _parameterCount = "—";

    /// <summary>Model info summary string (set once when session starts).</summary>
    [ObservableProperty] private string _modelInfo = "No session active";

    /// <summary>GPU name or "CPU only".</summary>
    [ObservableProperty] private string _gpuStatus = "—";

    /// <summary>Number of pages currently waiting in the background training queue.</summary>
    [ObservableProperty] private int _trainingQueueDepth;

    /// <summary>Loss values for the chart — last 500 points.</summary>
    public ObservableCollection<float> LossHistory { get; } = new();

    /// <summary>Settings used for the current/next session.</summary>
    public SettingsViewModel Settings { get; } = new();

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a <see cref="MainViewModel"/>.
    /// </summary>
    /// <param name="dispatch">
    /// Action that schedules work on the UI thread.
    /// Defaults to <c>Avalonia.Threading.Dispatcher.UIThread.Post</c> when null.
    /// Pass <c>a => a()</c> in tests to execute synchronously.
    /// </param>
    public MainViewModel(Action<Action>? dispatch = null)
    {
        _dispatch = dispatch ?? (a => Avalonia.Threading.Dispatcher.UIThread.Post(a));
        Settings.Load();
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    /// <summary>Returns true if <paramref name="url"/> has been visited this session.</summary>
    public bool IsVisited(string url) => _visitedUrls.Contains(url);

    /// <summary>Navigate to a URL (address bar Enter, Go button, or link click).</summary>
    [RelayCommand]
    private async Task NavigateAsync(string url)
    {
        if (string.IsNullOrWhiteSpace(url)) return;

        // Prepend https:// when no scheme is present
        if (!url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) &&
            !url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            url = "https://" + url;
        }

        // Push current to back stack before navigating
        if (_currentUrl is not null)
        {
            _backStack.Push(_currentUrl);
            _fwdStack.Clear();
        }

        _currentUrl = url;
        _visitedUrls.Add(url);
        _dispatch(() =>
        {
            AddressBarUrl = url;
            CanGoBack = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count > 0;
        });

        await FetchAndRenderAsync(url).ConfigureAwait(false);
    }

    /// <summary>Navigate to the previous page.</summary>
    [RelayCommand(CanExecute = nameof(CanGoBack))]
    private async Task GoBackAsync()
    {
        if (_backStack.Count == 0) return;
        if (_currentUrl is not null) _fwdStack.Push(_currentUrl);
        _currentUrl = _backStack.Pop();
        _dispatch(() =>
        {
            AddressBarUrl = _currentUrl;
            CanGoBack = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count > 0;
        });
        await FetchAndRenderAsync(_currentUrl).ConfigureAwait(false);
    }

    /// <summary>Navigate forward (after going back).</summary>
    [RelayCommand(CanExecute = nameof(CanGoForward))]
    private async Task GoForwardAsync()
    {
        if (_fwdStack.Count == 0) return;
        if (_currentUrl is not null) _backStack.Push(_currentUrl);
        _currentUrl = _fwdStack.Pop();
        _dispatch(() =>
        {
            AddressBarUrl = _currentUrl;
            CanGoBack = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count > 0;
        });
        await FetchAndRenderAsync(_currentUrl).ConfigureAwait(false);
    }

    /// <summary>
    /// Fetches and renders a URL in the browser pane. Uses the orchestrator when a
    /// training session is active; falls back to a standalone fetch otherwise.
    /// Does NOT manipulate the navigation stacks.
    /// </summary>
    private async Task FetchAndRenderAsync(string url)
    {
        _dispatch(() => StatusText = $"Fetching {url}…");
        _cts ??= new CancellationTokenSource();
        try
        {
            if (_orchestrator is not null)
            {
                // Session active: orchestrator handles fetch + parse + training pipeline.
                await _orchestrator.BrowseAsync(url, _cts.Token).ConfigureAwait(false);
            }
            else
            {
                // No session: fetch and render without training.
                var result = await _httpFetcher.FetchAsync(url, _cts.Token).ConfigureAwait(false);
                bool success = result.Status is BrowseStatus.Success or BrowseStatus.Redirect;
                if (success && !string.IsNullOrWhiteSpace(result.HtmlContent))
                {
                    var page = await HtmlParser
                        .ParseAsync(result.HtmlContent, url, _cts.Token)
                        .ConfigureAwait(false);
                    var content = FormattedPageContentBuilder.Build(page, _visitedUrls);
                    _dispatch(() =>
                    {
                        CurrentPage = content;
                        StatusText = $"Loaded {url}";
                    });
                }
                else
                {
                    _dispatch(() => StatusText = $"Failed to load {url}: {result.ErrorMessage}");
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _dispatch(() => StatusText = $"Error: {ex.Message}");
        }
    }

    /// <summary>Cancels the current in-flight fetch or training step.</summary>
    [RelayCommand]
    private void StopNavigation()
    {
        _cts?.Cancel();
        _cts = null;
        _dispatch(() => StatusText = "Stopped.");
    }

    // ── Session ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Initialises the orchestrator with current settings. Call after the settings
    /// dialog has been confirmed by the user.
    /// </summary>
    /// <param name="httpOverride">
    /// Optional HTTP fetcher override — pass a fake in tests to avoid real network calls.
    /// </param>
    public void StartSession(Func<string, CancellationToken, Task<BrowseResult>>? httpOverride = null)
    {
        Settings.Save();
        var config = Settings.ToOrchestratorConfig();
        _orchestrator = new PicoOrchestrator(config, httpOverride);
        _orchestrator.OnProgress += OnOrchestratorProgress;
        _cts = new CancellationTokenSource();
        _dispatch(() =>
        {
            IsSessionActive = true;
            StatusText = "Session started. Browse a page to begin training.";
            Step = 0;
            PagesProcessed = 0;
            LossHistory.Clear();
        });
    }

    /// <summary>
    /// Ends the session: saves a checkpoint and exports the model to GGUF.
    /// </summary>
    /// <param name="ggufOutputPath">Destination path for the .gguf file chosen by the user.</param>
    public void EndSession(string ggufOutputPath)
    {
        if (_orchestrator is null) return;
        _dispatch(() => StatusText = "Saving checkpoint and exporting GGUF…");
        _orchestrator.EndSession(ggufOutputPath);
        _orchestrator.OnProgress -= OnOrchestratorProgress;
        _orchestrator = null;
        _cts?.Cancel();
        _cts = null;
        _dispatch(() =>
        {
            IsSessionActive = false;
            StatusText = $"Exported: {ggufOutputPath}";
        });
    }

    // ── Orchestrator event handler ────────────────────────────────────────────

    private void OnOrchestratorProgress(OrchestratorEvent ev)
    {
        switch (ev)
        {
            case TrainingStepEvent ts:
                _dispatch(() =>
                {
                    Loss = ts.Loss;
                    LearningRate = ts.Lr;
                    TokensPerSec = ts.TokensPerSec;
                    GradNorm = ts.GradNorm;
                    Step = ts.Step;
                    StatusText = $"Training… step {ts.Step}, loss {ts.Loss:F3}";
                    if (LossHistory.Count >= 500) LossHistory.RemoveAt(0);
                    LossHistory.Add(ts.Loss);
                });
                break;

            case PageFetchedEvent pf when pf.Success:
                _dispatch(() => StatusText = $"Parsing {pf.Url}…");
                break;

            case PageFetchedEvent pf when !pf.Success:
                _dispatch(() => StatusText = $"Error fetching {pf.Url}: {pf.Error}");
                break;

            case PageParsedEvent pp:
                var content = FormattedPageContentBuilder.Build(pp.Page, _visitedUrls);
                _dispatch(() =>
                {
                    CurrentPage = content;
                    PagesProcessed++;
                    StatusText = $"Training on page {PagesProcessed}…";
                });
                break;

            case CheckpointSavedEvent cp:
                _dispatch(() => StatusText = $"Checkpoint saved: {cp.Path}");
                break;

            case GgufExportedEvent ge:
                _dispatch(() => StatusText = $"GGUF exported: {ge.Path} ({ge.FileSizeBytes / 1024:N0} KB)");
                break;

            case TrainingQueueChangedEvent tq:
                _dispatch(() => TrainingQueueDepth = tq.QueueDepth);
                break;

            case SessionErrorEvent se:
                _dispatch(() => StatusText = $"Skipped {se.Url}: {se.Error}");
                break;
        }
    }
}

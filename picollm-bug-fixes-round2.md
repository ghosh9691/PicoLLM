# Delta for Link Navigation and Background Training

## MODIFIED Requirements

### Requirement: Clickable Link Navigation (ui-shell)

Links rendered in the browser pane SHALL be clickable. Clicking a link SHALL navigate to that URL, fetch and parse the new page, display it in the browser pane, and queue its content for training.

#### Scenario: Click a link navigates to new page
- **GIVEN** the browser pane displays a parsed page containing a link to "/wiki/Transformer"
- **WHEN** the user clicks on that link text
- **THEN** the URL is resolved to an absolute URL (e.g., "https://en.wikipedia.org/wiki/Transformer")
- **AND** the address bar updates to show the new URL
- **AND** the new page is fetched, parsed, and displayed in the browser pane
- **AND** the new page's text is queued for background training

#### Scenario: Link click while training is running
- **GIVEN** background training is in progress on a previous page
- **WHEN** the user clicks a link
- **THEN** the new page is fetched and displayed immediately (navigation is NOT blocked by training)
- **AND** the new page's content is added to the training queue

#### Implementation guidance — diagnosing why links don't work

The problem is almost certainly that the rendered text in the browser pane has no click event handlers attached to link spans. Check these in order:

1. **Links not rendered as interactive elements**: The `TextExtractor` or browser pane renderer is outputting links as plain text (same style as body text) with no click handler. Each link span needs:
   - A mouse click event handler (or `TapGestureRecognizer` in MAUI, or `PointerPressed` in Avalonia)
   - The target URL stored as metadata on the span/element (e.g., `Tag`, `CommandParameter`, or a data attribute)
   - Visual styling: underline + distinct color (blue or theme accent)

2. **Event handler exists but doesn't trigger navigation**: The click handler fires but doesn't call back into the orchestrator. Wire it to:
   ```csharp
   async void OnLinkClicked(string href)
   {
       var absoluteUrl = UrlResolver.Resolve(currentPageUrl, href);
       AddressBar.Text = absoluteUrl;
       await _orchestrator.ProcessUrlAsync(absoluteUrl);
   }
   ```

3. **Relative URLs not resolved**: The click handler passes the raw `href` (e.g., "/wiki/Transformer") to the fetcher without resolving against the current page's base URL. Use `UrlResolver.Resolve(currentPageUrl, href)`.

4. **Links extracted but not passed to the renderer**: `ParsedPage.Links` is populated by `HtmlParser` but the browser pane renderer ignores it, rendering only `CleanText` as a flat string. The renderer needs to parse the text and interleave link spans at the correct positions — OR the `HtmlParser`/`TextExtractor` should return a structured list of `ContentBlock` objects (text blocks, link blocks, heading blocks, image-alt blocks) instead of a flat string, making rendering trivial.

**Recommended fix (option A — minimal change):**

Change `ParsedPage` to include inline link positions:

```csharp
public record InlineLink(int StartIndex, int Length, string Href, string Text);

public record ParsedPage(
    string Url,
    string Title,
    string? MetaDescription,
    string CleanText,
    List<InlineLink> InlineLinks,    // NEW: link positions within CleanText
    List<ImageReference> Images,
    List<LinkReference> Links);
```

The `TextExtractor` tracks character offsets as it builds `CleanText` and records each link's start position and length. The browser pane renderer then creates clickable spans at those offsets.

**Recommended fix (option B — cleaner but larger change):**

Replace `CleanText` (flat string) with a structured content model:

```csharp
public abstract record ContentBlock;
public record TextBlock(string Text) : ContentBlock;
public record HeadingBlock(string Text, int Level) : ContentBlock;
public record LinkBlock(string Text, string Href) : ContentBlock;
public record ImageAltBlock(string AltText) : ContentBlock;

public record ParsedPage(
    string Url,
    string Title,
    string? MetaDescription,
    List<ContentBlock> Content,      // replaces CleanText
    List<ImageReference> Images,
    List<LinkReference> Links);
```

The browser pane renderer walks `Content` and creates the appropriate UI element for each block — plain text, bold heading, clickable link, or gray italic image alt. This makes rendering trivially correct and link clicking easy to wire.

---

### Requirement: Background Training Thread (orchestrator, ui-shell)

Training SHALL run on a background thread. The UI thread SHALL remain responsive at all times, allowing the user to navigate pages, click links, and interact with the dashboard while training proceeds.

#### Scenario: Navigate while training
- **GIVEN** training is running on content from page 1
- **WHEN** the user enters a new URL or clicks a link
- **THEN** the new page is fetched and displayed without waiting for training to complete
- **AND** the new page's content is added to a training queue
- **AND** the training loop processes queued content in FIFO order

#### Scenario: Training queue
- **GIVEN** the user browses 3 pages rapidly
- **WHEN** training on page 1 is still in progress
- **THEN** page 2 and page 3 content is queued
- **AND** after page 1 training completes, page 2 training begins automatically
- **AND** the dashboard shows which page is currently being trained on

#### Scenario: UI responsiveness during training
- **GIVEN** training is actively running (forward pass, backward pass, optimizer step)
- **WHEN** the user scrolls the browser pane, clicks a link, or types in the address bar
- **THEN** the UI responds instantly (no freezing, no lag)

#### Implementation guidance

The current architecture likely calls `_orchestrator.ProcessUrlAsync()` which does fetch → parse → train synchronously. The fix is to separate fetching/display from training:

```csharp
public class PicoOrchestrator
{
    private readonly ConcurrentQueue<TrainingItem> _trainingQueue = new();
    private readonly CancellationTokenSource _cts = new();
    private Task? _trainingTask;

    // Called from UI thread — fetches and displays, then queues for training
    public async Task<ParsedPage> BrowseAsync(string url)
    {
        var result = await _fetcher.FetchAsync(url);
        var page = _parser.Parse(result.HtmlContent, url);
        
        // Queue content for background training (non-blocking)
        _trainingQueue.Enqueue(new TrainingItem(url, page.CleanText));
        
        // Start training loop if not already running
        _trainingTask ??= Task.Run(() => TrainingLoopAsync(_cts.Token));
        
        return page; // UI can render immediately
    }

    // Runs on background thread — processes queue forever
    private async Task TrainingLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            if (_trainingQueue.TryDequeue(out var item))
            {
                var tokens = _tokenizer.Encode(item.Text);
                for (int step = 0; step < _config.StepsPerPage; step++)
                {
                    if (ct.IsCancellationRequested) break;
                    // ... training step ...
                    OnProgress?.Invoke(new TrainingStepEvent(...));
                }
            }
            else
            {
                await Task.Delay(100, ct); // idle wait for new content
            }
        }
    }

    public void EndSession(string ggufPath)
    {
        _cts.Cancel();
        _trainingTask?.Wait();
        // save checkpoint + export GGUF
    }
}
```

Key points:
- `BrowseAsync` returns the `ParsedPage` immediately for UI display — does NOT wait for training
- Training runs in `Task.Run()` on a thread pool thread
- `ConcurrentQueue<TrainingItem>` bridges UI and training threads safely
- Progress events are still emitted — UI dispatches them to the main thread as before
- `EndSession` cancels the training loop, waits for it to finish, then saves

**Also update the dashboard** to show:
- "Training: page 2 of 5" (current item in queue)
- "Queue: 3 pages waiting"
- Training can be paused/resumed if desired (stretch goal)

---

## Tasks

- [ ] 1. Decide on link rendering approach: Option A (InlineLink offsets) or Option B (ContentBlock model). Option B is cleaner — recommend that.
- [ ] 2. Update `HtmlParser`/`TextExtractor` to produce structured `ContentBlock` list (or `InlineLink` positions)
- [ ] 3. Update `ParsedPage` record with new content model
- [ ] 4. Update browser pane renderer to create clickable link elements from structured content
- [ ] 5. Wire link click handler: resolve relative URL → update address bar → call orchestrator → display new page
- [ ] 6. Write test: click a link navigates to resolved URL and displays new content
- [ ] 7. Refactor `PicoOrchestrator` to separate browsing (fetch+parse+display) from training
- [ ] 8. Implement `ConcurrentQueue<TrainingItem>` for training queue
- [ ] 9. Move training loop to `Task.Run()` background thread
- [ ] 10. Implement `CancellationToken` support for clean shutdown
- [ ] 11. Update `EndSession` to drain queue, cancel training, then save checkpoint + export GGUF
- [ ] 12. Update dashboard to show current training page and queue depth
- [ ] 13. Write test: UI thread is not blocked during training (simulate navigate while training runs)
- [ ] 14. Write test: multiple pages queued, processed in FIFO order

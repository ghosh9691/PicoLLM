# Delta for Browser Usability Fixes

## MODIFIED Requirements

### Requirement: URL Auto-Protocol (browser-engine)

The system SHALL automatically prepend `https://` to URLs entered without a protocol scheme.

#### Scenario: User enters bare domain
- **GIVEN** the user types "en.wikipedia.org/wiki/Neural_network" in the address bar
- **WHEN** the URL is submitted
- **THEN** the system prepends "https://" and navigates to "https://en.wikipedia.org/wiki/Neural_network"

#### Scenario: User enters URL with protocol
- **GIVEN** the user types "https://example.com"
- **WHEN** the URL is submitted
- **THEN** no modification is made — the URL is used as-is

#### Scenario: User enters http explicitly
- **GIVEN** the user types "http://example.com"
- **WHEN** the URL is submitted
- **THEN** the http:// scheme is preserved (not forced to https)

#### Implementation guidance
In the address bar handler (or `UrlResolver`), before passing the URL to `HttpFetcher`:
```csharp
if (!url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) &&
    !url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
{
    url = "https://" + url;
}
```
Update the address bar display to show the resolved URL after navigation.

---

### Requirement: Browser Pane Must Display Parsed Page Text (ui-shell)

The system SHALL display the parsed text content of each fetched page in the browser pane. The browser pane MUST NOT be empty after a successful page fetch.

#### Scenario: Page fetched and displayed
- **GIVEN** a URL is successfully fetched and parsed
- **WHEN** the browser pane is updated
- **THEN** the full `ParsedPage.CleanText` is rendered in the browser pane
- **AND** headings appear bold/larger
- **AND** links appear underlined and clickable
- **AND** image alt text appears as bracketed gray italic text

#### Scenario: Verify text rendering pipeline
- **GIVEN** `HtmlParser.Parse()` returns a `ParsedPage` with non-empty `CleanText`
- **WHEN** the orchestrator emits a `PageParsedEvent`
- **THEN** the UI handler receives the event, extracts the `ParsedPage`, and renders `CleanText` into the browser pane's text control

#### Debugging checklist (for Claude Code)
The text not showing is likely one of these issues — check in order:
1. **Orchestrator not passing ParsedPage to UI**: Verify `PageParsedEvent` includes the `ParsedPage` object (not just metadata)
2. **UI not subscribing to the event**: Verify the browser pane's event handler is wired to `OnProgress`
3. **UI update on wrong thread**: Verify `Dispatcher.Dispatch()` (or equivalent) wraps the text control update
4. **Text control not bound or not scrollable**: Verify the text control is actually in the visual tree and has its `Text`/`Content` property set
5. **ParsedPage.CleanText is empty**: Add a debug log after `HtmlParser.Parse()` to verify text extraction works — if empty, the `TextExtractor` DOM walker or `ElementFilter` may be too aggressive

---

## Tasks

- [ ] 1. Fix URL auto-protocol: add `https://` prepend logic in address bar handler or UrlResolver
- [ ] 2. Write test: bare domain gets https:// prepended, explicit http:// preserved, explicit https:// unchanged
- [ ] 3. Debug browser pane text display: trace from HtmlParser.Parse() → PageParsedEvent → UI handler → text control
- [ ] 4. Fix the root cause (whichever item from the debugging checklist applies)
- [ ] 5. Write test: after processing a URL, browser pane text control contains non-empty content matching ParsedPage.CleanText
- [ ] 6. Update address bar to show resolved URL (with protocol) after navigation

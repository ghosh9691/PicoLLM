# Browser Engine (Capability 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PicoBrowser's network layer — HTTP/HTTPS fetching with robots.txt compliance, redirect following, image downloading, and session tracking.

**Architecture:** Single shared `HttpClient` per session (best practice). `HttpFetcher` handles fetch + error wrapping. `RobotsTxtParser` caches per domain. `UrlResolver` uses `System.Uri`. `BrowseSession` is the top-level session object. All network errors are caught and returned as `BrowseResult` error states — no exceptions propagate to caller.

**Tech Stack:** C# 12 / .NET 9, AngleSharp (declared now, used in capability 7), xUnit + FluentAssertions.

---

## File Map

**New project:**
- `src/PicoLLM.Browser/PicoLLM.Browser.csproj` — references AngleSharp
- `src/PicoLLM.Browser/BrowseResult.cs` — record + BrowseStatus enum
- `src/PicoLLM.Browser/ImageDownload.cs` — image download record
- `src/PicoLLM.Browser/BrowseSession.cs` — session tracker
- `src/PicoLLM.Browser/HttpFetcher.cs` — main HTTP fetch logic
- `src/PicoLLM.Browser/UrlResolver.cs` — relative URL resolution
- `src/PicoLLM.Browser/RobotsTxtParser.cs` — fetch + cache + check

**Test files:**
- `tests/PicoLLM.Tests/Browser/BrowseResultTests.cs`
- `tests/PicoLLM.Tests/Browser/UrlResolverTests.cs`
- `tests/PicoLLM.Tests/Browser/RobotsTxtParserTests.cs`
- `tests/PicoLLM.Tests/Browser/HttpFetcherTests.cs`
- `tests/PicoLLM.Tests/Browser/BrowseSessionTests.cs`

---

### Task 1: Project Setup

- [ ] Create `src/PicoLLM.Browser/PicoLLM.Browser.csproj` with AngleSharp
- [ ] Add to `PicoLLM.slnx`
- [ ] Add reference to test project
- [ ] `dotnet build` — pass

### Task 2: Core Types

- [ ] `BrowseResult.cs` — record + `BrowseStatus` enum (Success, Error, Blocked, UnsupportedContentType, Timeout)
- [ ] `ImageDownload.cs` — record
- [ ] `BrowseSession.cs` — tracks pages + images
- [ ] Write basic construction tests
- [ ] Run tests — pass

### Task 3: UrlResolver

- [ ] Implement `UrlResolver.Resolve(string baseUrl, string relativeOrAbsolute)` → `string`
- [ ] Handle: absolute URL (return as-is), relative path (/foo), relative segment (foo), protocol-relative (//host/path)
- [ ] Write tests for all four cases
- [ ] Run tests — pass

### Task 4: HttpFetcher

- [ ] Implement `HttpFetcher(TimeSpan? timeout)` — configures HttpClient with UA header, auto-redirect ≤5, gzip decompression
- [ ] `FetchAsync(string url, CancellationToken ct)` → `BrowseResult`: wraps all exceptions, checks content-type (only text/html), records elapsed time
- [ ] User-Agent: `"PicoBrowser/1.0 (Educational; +https://github.com/ghosh9691/picollm)"`
- [ ] Write tests: success with mock handler, wrong content type, timeout, DNS error
- [ ] Run tests — pass

### Task 5: RobotsTxtParser

- [ ] `RobotsTxtParser(HttpFetcher fetcher)` — domain cache
- [ ] `IsAllowedAsync(string url, CancellationToken ct)` → `bool`: fetch robots.txt once per domain, parse User-agent/Disallow, check path
- [ ] If robots.txt fetch fails → allow (fail-open)
- [ ] Write tests: Disallow /private/ blocks /private/page, allows /public/, missing robots.txt allows all
- [ ] Run tests — pass

### Task 6: ImageDownloader + BrowseSession integration

- [ ] `ImageDownloader(HttpFetcher fetcher)` — `DownloadAsync(string absoluteUrl, string? altText, CancellationToken ct)` → `ImageDownload?`; only accept image/jpeg, image/png, image/gif
- [ ] `BrowseSession.BrowseAsync(string url, CancellationToken ct)` — checks robots, fetches page, adds to Pages list; returns `BrowseResult`
- [ ] Write tests: session tracks multiple pages, image content-type filtering
- [ ] Run all tests — pass
- [ ] `git commit`

# Browser Engine — Implementation Tasks

## Setup
- [ ] 1.1 Create `PicoLLM.Browser` project (net9.0 class library)
- [ ] 1.2 Add AngleSharp NuGet (for HTML parsing in next capability — declare dependency now)

## Core Types
- [ ] 2.1 Implement `BrowseResult` record
- [ ] 2.2 Implement `BrowseStatus` enum
- [ ] 2.3 Implement `ImageDownload` record
- [ ] 2.4 Implement `BrowseSession` with page/image tracking

## HTTP Fetching
- [ ] 3.1 Implement `HttpFetcher` with shared HttpClient instance
- [ ] 3.2 Set User-Agent header
- [ ] 3.3 Configure redirect following (max 5)
- [ ] 3.4 Configure timeout (default 30s, configurable)
- [ ] 3.5 Implement content-type check (accept only text/html)
- [ ] 3.6 Implement graceful error handling (no exceptions to caller)
- [ ] 3.7 Write tests: mock HTTP responses for success, redirect, timeout, error

## URL Resolution
- [ ] 4.1 Implement `UrlResolver` using System.Uri
- [ ] 4.2 Handle relative paths, protocol-relative, and absolute URLs
- [ ] 4.3 Write tests: various relative URL patterns

## Robots.txt
- [ ] 5.1 Implement `RobotsTxtParser` — fetch and parse /robots.txt
- [ ] 5.2 Cache per domain for session duration
- [ ] 5.3 Check Disallow rules before fetching a page
- [ ] 5.4 Write tests: parse known robots.txt, verify allow/disallow decisions

## Image Downloading
- [ ] 6.1 Implement `ImageDownloader` — download by absolute URL
- [ ] 6.2 Filter by content type (only jpeg, png, gif)
- [ ] 6.3 Extract alt text from img tag attributes
- [ ] 6.4 Write tests: mock image download, content type filtering

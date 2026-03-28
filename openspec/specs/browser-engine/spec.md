# Browser Engine Specification

## Purpose

Fetch web pages over HTTP/HTTPS and deliver raw HTML content to the parser. This is the "network layer" of PicoBrowser.

### Requirement: HTTP Fetching

The system SHALL fetch web pages via HTTP and HTTPS using standard .NET HttpClient.

#### Scenario: Fetch a web page
- **GIVEN** a valid URL "https://example.com"
- **WHEN** the browser fetches it
- **THEN** the raw HTML content is returned as a string
- **AND** the HTTP status code is available

#### Scenario: Follow redirects
- **GIVEN** a URL that returns a 301/302 redirect
- **WHEN** the browser fetches it
- **THEN** redirects are followed (up to a configurable limit, default 5)

### Requirement: User-Agent Header

The system SHALL send a User-Agent header identifying itself as PicoBrowser.

#### Scenario: User agent
- **GIVEN** any HTTP request
- **WHEN** sent to a server
- **THEN** the User-Agent header is "PicoBrowser/1.0 (Educational; +https://github.com/ghosh9691/picollm)"

### Requirement: Image Downloading

The system SHALL download images (JPEG, PNG, GIF) referenced by `<img>` tags, resolving relative URLs against the page's base URL.

#### Scenario: Download images
- **GIVEN** an HTML page with `<img src="/images/photo.jpg">`
- **WHEN** images are resolved
- **THEN** the absolute URL is constructed and the image bytes are downloaded
- **AND** only JPEG, PNG, and GIF content types are accepted

### Requirement: Timeout and Error Handling

The system SHALL enforce a configurable timeout (default 30 seconds) and gracefully handle network errors.

#### Scenario: Timeout
- **GIVEN** a server that does not respond within 30 seconds
- **WHEN** the request times out
- **THEN** a `BrowseResult` with error status and message is returned (no exception thrown to caller)

#### Scenario: DNS failure
- **GIVEN** a non-existent domain
- **WHEN** fetch is attempted
- **THEN** a `BrowseResult` with error status is returned

### Requirement: Content Type Filtering

The system SHALL only process responses with `text/html` content type. Non-HTML responses SHALL be reported as unsupported.

#### Scenario: PDF response
- **GIVEN** a URL that returns `application/pdf`
- **WHEN** fetched
- **THEN** the result indicates unsupported content type

### Requirement: Robots.txt Respect

The system SHALL check robots.txt before fetching a page and respect Disallow rules for the PicoBrowser user agent.

#### Scenario: Disallowed path
- **GIVEN** robots.txt contains "User-agent: * Disallow: /private/"
- **WHEN** a page under /private/ is requested
- **THEN** the fetch is skipped and a "blocked by robots.txt" result is returned

### Requirement: Browse Session

The system SHALL track all pages visited in a session, providing the list of URLs and their fetch status for the orchestrator.

#### Scenario: Multi-page session
- **GIVEN** a session where 5 pages are browsed
- **WHEN** the session is queried
- **THEN** it returns a list of 5 BrowseResult objects with URL, status, and content

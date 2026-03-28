# Browser Engine — Technical Design

## Core Types

```csharp
public record BrowseResult(
    string Url,
    BrowseStatus Status,      // Success, Redirect, Error, Blocked, UnsupportedContentType
    string? HtmlContent,
    string? ErrorMessage,
    int HttpStatusCode,
    TimeSpan ElapsedTime);

public enum BrowseStatus { Success, Redirect, Error, Blocked, UnsupportedContentType, Timeout }

public record ImageDownload(
    string SourceUrl,
    string AbsoluteUrl,
    byte[] Data,
    string ContentType,        // image/jpeg, image/png, image/gif
    string? AltText);

public class BrowseSession
{
    public List<BrowseResult> Pages { get; }
    public List<ImageDownload> Images { get; }
    public DateTime StartedAt { get; }
}
```

## HttpClient Configuration

```csharp
var handler = new HttpClientHandler
{
    AllowAutoRedirect = true,
    MaxAutomaticRedirections = 5,
    AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate
};

var client = new HttpClient(handler)
{
    Timeout = TimeSpan.FromSeconds(30)
};
client.DefaultRequestHeaders.UserAgent.ParseAdd("PicoBrowser/1.0 (Educational)");
```

Use a single `HttpClient` instance across the session (best practice for .NET).

## URL Resolution

Use `new Uri(baseUri, relativeUri)` for resolving relative URLs from `<img src>` and `<a href>`.

## Robots.txt

Simple parser: fetch `/robots.txt`, parse `User-agent` and `Disallow` lines. Cache per domain for the session lifetime.

## Project Location

`src/PicoLLM.Browser/`:
- `BrowseResult.cs`
- `BrowseSession.cs`
- `HttpFetcher.cs`
- `ImageDownloader.cs`
- `RobotsTxtParser.cs`
- `UrlResolver.cs`

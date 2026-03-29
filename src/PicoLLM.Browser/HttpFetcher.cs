using System.Diagnostics;
using System.Net;

namespace PicoLLM.Browser;

/// <summary>
/// Low-level HTTP fetcher. Wraps a single <see cref="HttpClient"/> instance
/// (best practice) with PicoBrowser headers, automatic redirect following,
/// configurable timeout, robots.txt enforcement, and content-type filtering.
/// All errors are captured and returned as <see cref="BrowseResult"/> values
/// — callers never receive raw exceptions.
/// </summary>
public sealed class HttpFetcher : IDisposable
{
    /// <summary>User-Agent sent with every request.</summary>
    public const string UserAgent =
        "PicoBrowser/1.0 (Educational; +https://github.com/ghosh9691/picollm)";

    private readonly HttpClient _client;
    private readonly RobotsTxtParser _robotsParser;
    private bool _disposed;

    /// <summary>
    /// Creates a fetcher with the given per-request timeout.
    /// </summary>
    /// <param name="timeoutSeconds">Seconds before a request is abandoned (default 30).</param>
    public HttpFetcher(int timeoutSeconds = 30)
        : this(timeoutSeconds, null) { }

    /// <summary>
    /// Creates a fetcher with a custom <see cref="HttpMessageHandler"/> (for testing).
    /// </summary>
    internal HttpFetcher(int timeoutSeconds, HttpMessageHandler? handler)
    {
        var effectiveHandler = handler ?? new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 5,
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate
        };

        _client = new HttpClient(effectiveHandler)
        {
            Timeout = TimeSpan.FromSeconds(timeoutSeconds)
        };

        _client.DefaultRequestHeaders.UserAgent.ParseAdd(UserAgent);
        _robotsParser = new RobotsTxtParser(_client);
    }

    /// <summary>
    /// Fetches the page at <paramref name="url"/> and returns a result record.
    /// Robots.txt is checked first; the content type is validated before reading the body.
    /// </summary>
    public async Task<BrowseResult> FetchAsync(string url, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        // Validate URL
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            return new BrowseResult(url, BrowseStatus.Error, null, "Invalid URL format", 0, sw.Elapsed);
        }

        // Check robots.txt
        try
        {
            bool allowed = await _robotsParser.IsAllowedAsync(uri, cancellationToken).ConfigureAwait(false);
            if (!allowed)
            {
                return new BrowseResult(url, BrowseStatus.Blocked, null,
                    "Blocked by robots.txt", 0, sw.Elapsed);
            }
        }
        catch (Exception ex)
        {
            // robots.txt failure is non-fatal — proceed with fetch
            _ = ex;
        }

        try
        {
            using var response = await _client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead,
                cancellationToken).ConfigureAwait(false);

            int statusCode = (int)response.StatusCode;

            // Content-type check
            string? contentType = response.Content.Headers.ContentType?.MediaType;
            if (contentType is null || !contentType.StartsWith("text/html", StringComparison.OrdinalIgnoreCase))
            {
                return new BrowseResult(
                    response.RequestMessage?.RequestUri?.ToString() ?? url,
                    BrowseStatus.UnsupportedContentType,
                    null,
                    $"Unsupported content type: {contentType ?? "(none)"}",
                    statusCode,
                    sw.Elapsed);
            }

            string html = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            string finalUrl = response.RequestMessage?.RequestUri?.ToString() ?? url;

            var status = finalUrl != url ? BrowseStatus.Redirect : BrowseStatus.Success;
            return new BrowseResult(finalUrl, status, html, null, statusCode, sw.Elapsed);
        }
        catch (TaskCanceledException ex) when (!cancellationToken.IsCancellationRequested || ex.InnerException is TimeoutException)
        {
            return new BrowseResult(url, BrowseStatus.Timeout, null, "Request timed out", 0, sw.Elapsed);
        }
        catch (HttpRequestException ex)
        {
            return new BrowseResult(url, BrowseStatus.Error, null, ex.Message, 0, sw.Elapsed);
        }
        catch (Exception ex)
        {
            return new BrowseResult(url, BrowseStatus.Error, null, ex.Message, 0, sw.Elapsed);
        }
    }

    /// <summary>Returns the underlying <see cref="HttpClient"/> for use by other components.</summary>
    internal HttpClient Client => _client;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _client.Dispose();
        _disposed = true;
    }
}

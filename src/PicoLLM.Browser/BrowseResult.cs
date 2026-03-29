namespace PicoLLM.Browser;

/// <summary>
/// The result of a single page fetch operation, including status, content, and timing.
/// </summary>
/// <param name="Url">The URL that was fetched (final URL after any redirects).</param>
/// <param name="Status">The outcome of the fetch.</param>
/// <param name="HtmlContent">Raw HTML content; null if the fetch did not succeed.</param>
/// <param name="ErrorMessage">Human-readable error detail; null on success.</param>
/// <param name="HttpStatusCode">HTTP status code; 0 if no response was received.</param>
/// <param name="ElapsedTime">Wall-clock time taken to complete the fetch.</param>
public record BrowseResult(
    string Url,
    BrowseStatus Status,
    string? HtmlContent,
    string? ErrorMessage,
    int HttpStatusCode,
    TimeSpan ElapsedTime);

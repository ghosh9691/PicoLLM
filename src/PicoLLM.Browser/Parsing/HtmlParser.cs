using AngleSharp;

namespace PicoLLM.Browser.Parsing;

/// <summary>
/// Parses raw HTML into a <see cref="ParsedPage"/> using AngleSharp for DOM construction
/// and <see cref="TextExtractor"/> for content extraction.
/// </summary>
public static class HtmlParser
{
    /// <summary>
    /// Parses <paramref name="html"/> and returns a <see cref="ParsedPage"/> with clean text,
    /// image references, link references, page title, and meta description.
    /// </summary>
    /// <param name="html">Raw HTML string.</param>
    /// <param name="url">The URL the HTML was fetched from (stored on the result).</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    public static async Task<ParsedPage> ParseAsync(
        string html,
        string url,
        CancellationToken cancellationToken = default)
    {
        var config = Configuration.Default;
        var context = BrowsingContext.New(config);
        var document = await context
            .OpenAsync(req => req.Content(html), cancellationToken)
            .ConfigureAwait(false);

        string title = document.Title ?? string.Empty;

        string? metaDescription = document
            .QuerySelector("meta[name='description']")
            ?.GetAttribute("content");

        var root = document.Body ?? document.DocumentElement;
        var (cleanText, images, links) = TextExtractor.Extract(root);

        return new ParsedPage(url, title, metaDescription, cleanText, images, links);
    }
}

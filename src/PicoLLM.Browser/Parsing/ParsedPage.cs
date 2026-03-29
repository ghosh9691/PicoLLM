namespace PicoLLM.Browser.Parsing;

/// <summary>
/// The fully-parsed result of an HTML page: clean text, images, links, title, and meta description.
/// </summary>
/// <param name="Url">The page URL (as provided to the parser).</param>
/// <param name="Title">Contents of the &lt;title&gt; element; empty string if absent.</param>
/// <param name="MetaDescription">Contents of &lt;meta name="description"&gt;; null if absent.</param>
/// <param name="CleanText">Visible text with paragraph breaks as \n\n and heading markers.</param>
/// <param name="Images">All &lt;img&gt; elements with src and optional alt text.</param>
/// <param name="Links">All &lt;a href&gt; elements that passed the link filter.</param>
public record ParsedPage(
    string Url,
    string Title,
    string? MetaDescription,
    string CleanText,
    List<ImageReference> Images,
    List<LinkReference> Links);

/// <summary>An image reference extracted from an &lt;img&gt; element.</summary>
/// <param name="SourceUrl">The value of the src attribute.</param>
/// <param name="AltText">The value of the alt attribute; null if absent.</param>
public record ImageReference(string SourceUrl, string? AltText);

/// <summary>A link extracted from an &lt;a href&gt; element.</summary>
/// <param name="Href">The href attribute value.</param>
/// <param name="AnchorText">The visible text content of the anchor element.</param>
public record LinkReference(string Href, string AnchorText);

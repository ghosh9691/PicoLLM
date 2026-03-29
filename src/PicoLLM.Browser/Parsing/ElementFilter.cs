namespace PicoLLM.Browser.Parsing;

/// <summary>
/// Classifies HTML element tags for the parsing pipeline.
/// Provides a skip-list for non-content elements and helpers for block/heading detection.
/// </summary>
public static class ElementFilter
{
    private static readonly HashSet<string> Blocklist = new(StringComparer.OrdinalIgnoreCase)
    {
        "script", "style", "noscript", "iframe", "object", "embed",
        "video", "audio", "canvas", "svg", "form", "input", "select",
        "textarea", "button"
    };

    private static readonly HashSet<string> BlockElements = new(StringComparer.OrdinalIgnoreCase)
    {
        "p", "div", "article", "section", "main", "header", "footer",
        "li", "td", "th", "blockquote", "pre", "br", "hr", "figure", "figcaption"
    };

    /// <summary>Returns true if the element's content should be entirely discarded.</summary>
    public static bool ShouldSkip(string tagName) => Blocklist.Contains(tagName);

    /// <summary>Returns true if the element is a block-level element (triggers paragraph breaks).</summary>
    public static bool IsBlock(string tagName) => BlockElements.Contains(tagName);

    /// <summary>Returns true if the tag is a heading element (h1–h6).</summary>
    public static bool IsHeading(string tagName) =>
        tagName.Length == 2
        && (tagName[0] == 'h' || tagName[0] == 'H')
        && tagName[1] >= '1' && tagName[1] <= '6';

    /// <summary>Returns the markdown prefix for a heading tag (e.g. "## " for h2).</summary>
    public static string HeadingPrefix(string tagName) =>
        tagName[1] switch
        {
            '1' => "# ",
            '2' => "## ",
            '3' => "### ",
            '4' => "#### ",
            '5' => "##### ",
            _   => "###### "
        };
}

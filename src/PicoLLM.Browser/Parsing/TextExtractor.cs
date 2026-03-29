using System.Text;
using System.Text.RegularExpressions;
using AngleSharp.Dom;

namespace PicoLLM.Browser.Parsing;

/// <summary>
/// Walks an AngleSharp DOM tree depth-first, collecting visible text, image references,
/// and link references. Non-content elements (script, style, etc.) are skipped.
/// Block elements are separated by double newlines; headings receive markdown-style prefixes.
/// </summary>
public static class TextExtractor
{
    /// <summary>
    /// Extracts clean text, images, and links from the given root element.
    /// </summary>
    /// <param name="root">The root DOM element to walk (typically document.Body).</param>
    /// <returns>A tuple of (normalised clean text, image list, link list).</returns>
    public static (string Text, List<ImageReference> Images, List<LinkReference> Links)
        Extract(IElement root)
    {
        var sb = new StringBuilder();
        var images = new List<ImageReference>();
        var links = new List<LinkReference>();

        Walk(root, sb, images, links);

        string text = NormalizeText(sb.ToString());
        return (text, images, links);
    }

    private static void Walk(
        INode node,
        StringBuilder sb,
        List<ImageReference> images,
        List<LinkReference> links)
    {
        if (node is IText textNode)
        {
            sb.Append(textNode.Text);
            return;
        }

        if (node is not IElement element) return;

        string tag = element.TagName.ToLowerInvariant();

        if (ElementFilter.ShouldSkip(tag)) return;

        if (tag == "img")
        {
            string? src = element.GetAttribute("src");
            if (src is not null)
            {
                string? alt = element.GetAttribute("alt");
                images.Add(new ImageReference(src, string.IsNullOrEmpty(alt) ? null : alt));
                if (!string.IsNullOrWhiteSpace(alt))
                    sb.Append($" [Image: {alt}] ");
            }
            return;
        }

        if (tag == "a")
        {
            string? href = element.GetAttribute("href");
            if (href is not null && IsValidHref(href))
            {
                string anchorText = element.TextContent.Trim();
                links.Add(new LinkReference(href, anchorText));
            }
            // still recurse into anchor children for text
        }

        bool isBlock = ElementFilter.IsBlock(tag);
        bool isHeading = ElementFilter.IsHeading(tag);

        if (isBlock || isHeading) sb.Append('\n');

        if (isHeading) sb.Append(ElementFilter.HeadingPrefix(tag));

        foreach (var child in element.ChildNodes)
            Walk(child, sb, images, links);

        if (isBlock || isHeading) sb.Append('\n');
    }

    private static bool IsValidHref(string href)
    {
        if (string.IsNullOrWhiteSpace(href)) return false;
        if (href.StartsWith("javascript:", StringComparison.OrdinalIgnoreCase)) return false;
        if (href.StartsWith("mailto:", StringComparison.OrdinalIgnoreCase)) return false;
        if (href.StartsWith("tel:", StringComparison.OrdinalIgnoreCase)) return false;
        if (href.StartsWith('#')) return false;
        return true;
    }

    private static string NormalizeText(string raw)
    {
        // Split on newlines, collapse whitespace within each line, rejoin
        var lines = raw.Split('\n');
        var normalized = lines
            .Select(l => Regex.Replace(l.Trim(), @"\s+", " "))
            .ToArray();

        string joined = string.Join("\n", normalized);

        // Collapse 3+ consecutive newlines to exactly 2
        joined = Regex.Replace(joined, @"\n{3,}", "\n\n");

        return joined.Trim();
    }
}

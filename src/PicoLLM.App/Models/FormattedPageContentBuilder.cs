using System.Text.RegularExpressions;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.App.Models;

/// <summary>
/// Builds a <see cref="FormattedPageContent"/> from a <see cref="ParsedPage"/>
/// by parsing the <c>CleanText</c> into structured page elements.
/// </summary>
public static class FormattedPageContentBuilder
{
    private static readonly Regex ImageAltRegex =
        new(@"^\[Image:\s*(.+?)\s*\]$", RegexOptions.Compiled);

    /// <summary>
    /// Converts a <see cref="ParsedPage"/> into a <see cref="FormattedPageContent"/>
    /// suitable for rendering in the browser pane.
    /// </summary>
    /// <param name="page">The parsed page from HtmlParser.</param>
    /// <param name="visitedUrls">
    ///   Set of URLs already visited this session; used to mark links as visited.
    /// </param>
    public static FormattedPageContent Build(
        ParsedPage page,
        HashSet<string>? visitedUrls = null)
    {
        visitedUrls ??= [];
        var elements = new List<IPageElement>();

        var blocks = page.CleanText.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

        foreach (var block in blocks)
        {
            var trimmed = block.Trim();
            if (string.IsNullOrWhiteSpace(trimmed)) continue;

            // Heading detection: # / ## / ###
            var headingLevel = DetectHeadingLevel(trimmed);
            if (headingLevel > 0)
            {
                // skip "# " (2 chars for h1), "## " (3 chars for h2), "### " (4 chars for h3)
                var headingText = trimmed[(headingLevel + 1)..].Trim();
                elements.Add(new HeadingElement(headingText, headingLevel));
                continue;
            }

            // Image alt text detection: [Image: ...]
            var imageMatch = ImageAltRegex.Match(trimmed);
            if (imageMatch.Success)
            {
                elements.Add(new ImageAltElement(imageMatch.Groups[1].Value));
                continue;
            }

            // Paragraph with inline link matching
            var runs = BuildTextRuns(trimmed, page.Links, visitedUrls);
            if (runs.Count > 0)
                elements.Add(new ParagraphElement(runs));
        }

        return new FormattedPageContent(elements);
    }

    private static int DetectHeadingLevel(string block)
    {
        if (block.StartsWith("### ")) return 3;
        if (block.StartsWith("## "))  return 2;
        if (block.StartsWith("# "))   return 1;
        return 0;
    }

    private static List<TextRun> BuildTextRuns(
        string text,
        IEnumerable<LinkReference> links,
        HashSet<string> visitedUrls)
    {
        // Find positions of all link anchor texts in the paragraph text.
        // Longer matches are preferred to avoid partial matches.
        var spans = new List<(int Start, int End, string Href, bool Visited)>();
        foreach (var link in links.OrderByDescending(l => l.AnchorText.Length))
        {
            int idx = text.IndexOf(link.AnchorText, StringComparison.Ordinal);
            if (idx < 0) continue;

            // Skip if this span overlaps an already-claimed range.
            bool overlaps = spans.Any(s =>
                idx < s.End && idx + link.AnchorText.Length > s.Start);
            if (overlaps) continue;

            spans.Add((idx, idx + link.AnchorText.Length,
                link.Href, visitedUrls.Contains(link.Href)));
        }

        spans.Sort((a, b) => a.Start.CompareTo(b.Start));

        var runs = new List<TextRun>();
        int pos = 0;
        foreach (var span in spans)
        {
            if (span.Start > pos)
                runs.Add(new TextRun(text[pos..span.Start], false, null, false));
            runs.Add(new TextRun(text[span.Start..span.End], true, span.Href, span.Visited));
            pos = span.End;
        }
        if (pos < text.Length)
            runs.Add(new TextRun(text[pos..], false, null, false));

        return runs.Count > 0 ? runs : [new TextRun(text, false, null, false)];
    }
}

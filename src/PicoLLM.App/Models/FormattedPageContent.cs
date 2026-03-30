namespace PicoLLM.App.Models;

/// <summary>The full structured content of a parsed page ready for browser-pane rendering.</summary>
/// <param name="Elements">Ordered list of page elements (headings, paragraphs, images).</param>
public record FormattedPageContent(IReadOnlyList<IPageElement> Elements);

/// <summary>Marker interface for all browser-pane page elements.</summary>
public interface IPageElement { }

/// <summary>A heading element (h1–h3).</summary>
/// <param name="Text">Heading text with markup stripped.</param>
/// <param name="Level">Heading level: 1 = h1, 2 = h2, 3 = h3.</param>
public record HeadingElement(string Text, int Level) : IPageElement;

/// <summary>A paragraph composed of styled text runs (plain text and links).</summary>
/// <param name="Runs">Ordered list of text runs that make up this paragraph.</param>
public record ParagraphElement(IReadOnlyList<TextRun> Runs) : IPageElement;

/// <summary>An image alt-text element rendered as italic grey bracketed text.</summary>
/// <param name="AltText">The image alt text.</param>
public record ImageAltElement(string AltText) : IPageElement;

/// <summary>A single styled run of text within a paragraph.</summary>
/// <param name="Text">The visible text of this run.</param>
/// <param name="IsLink">True if this run is a hyperlink.</param>
/// <param name="Href">The link target URL; null for plain text runs.</param>
/// <param name="IsVisited">True if this URL has been visited in the current session.</param>
public record TextRun(string Text, bool IsLink, string? Href, bool IsVisited);

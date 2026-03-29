# Capability 7 — HTML Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HTML parsing pipeline to `PicoLLM.Browser` that converts raw HTML into clean text, image references, and link references via an AngleSharp DOM walk.

**Architecture:** Four focused files in `src/PicoLLM.Browser/Parsing/`. `ElementFilter` decides which tags to skip or treat as blocks. `TextExtractor` performs the depth-first DOM walk and produces raw text plus image/link lists. `HtmlParser` orchestrates AngleSharp, calls `TextExtractor`, and returns a `ParsedPage`. `ParsedPage` and its supporting records are plain C# records.

**Tech Stack:** AngleSharp 1.1.2 (already in `PicoLLM.Browser.csproj`), xUnit, FluentAssertions, .NET 9 / C# 12.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/PicoLLM.Browser/Parsing/ParsedPage.cs` | Result records: `ParsedPage`, `ImageReference`, `LinkReference` |
| Create | `src/PicoLLM.Browser/Parsing/ElementFilter.cs` | Tag blacklist; block/heading classification |
| Create | `src/PicoLLM.Browser/Parsing/TextExtractor.cs` | Depth-first DOM walk; text/image/link collection |
| Create | `src/PicoLLM.Browser/Parsing/HtmlParser.cs` | AngleSharp bootstrap; title/meta extraction; calls TextExtractor |
| Create | `tests/PicoLLM.Tests/Browser/ElementFilterTests.cs` | Unit tests for ElementFilter |
| Create | `tests/PicoLLM.Tests/Browser/TextExtractorTests.cs` | Unit tests for TextExtractor |
| Create | `tests/PicoLLM.Tests/Browser/HtmlParserTests.cs` | Integration tests for HtmlParser |

---

## Task 1: Result Records

**Files:**
- Create: `src/PicoLLM.Browser/Parsing/ParsedPage.cs`

- [ ] **Step 1: Create the file**

```csharp
// src/PicoLLM.Browser/Parsing/ParsedPage.cs
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
```

- [ ] **Step 2: Build to verify no errors**

```
dotnet build src/PicoLLM.Browser/PicoLLM.Browser.csproj
```
Expected: Build succeeded, 0 errors.

- [ ] **Step 3: Commit**

```bash
git add src/PicoLLM.Browser/Parsing/ParsedPage.cs
git commit -m "feat(cap7): add ParsedPage, ImageReference, LinkReference records"
```

---

## Task 2: Element Filter

**Files:**
- Create: `src/PicoLLM.Browser/Parsing/ElementFilter.cs`
- Create: `tests/PicoLLM.Tests/Browser/ElementFilterTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Browser/ElementFilterTests.cs
using FluentAssertions;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.Tests.Browser;

public class ElementFilterTests
{
    [Theory]
    [InlineData("script")]
    [InlineData("SCRIPT")]
    [InlineData("style")]
    [InlineData("noscript")]
    [InlineData("iframe")]
    [InlineData("object")]
    [InlineData("embed")]
    [InlineData("video")]
    [InlineData("audio")]
    [InlineData("canvas")]
    [InlineData("svg")]
    [InlineData("form")]
    [InlineData("input")]
    [InlineData("select")]
    [InlineData("textarea")]
    [InlineData("button")]
    public void ShouldSkip_BlacklistedTag_ReturnsTrue(string tag)
    {
        ElementFilter.ShouldSkip(tag).Should().BeTrue();
    }

    [Theory]
    [InlineData("p")]
    [InlineData("div")]
    [InlineData("span")]
    [InlineData("article")]
    [InlineData("a")]
    [InlineData("img")]
    public void ShouldSkip_ContentTag_ReturnsFalse(string tag)
    {
        ElementFilter.ShouldSkip(tag).Should().BeFalse();
    }

    [Theory]
    [InlineData("h1")]
    [InlineData("h2")]
    [InlineData("h3")]
    [InlineData("h4")]
    [InlineData("h5")]
    [InlineData("h6")]
    public void IsHeading_HeadingTag_ReturnsTrue(string tag)
    {
        ElementFilter.IsHeading(tag).Should().BeTrue();
    }

    [Theory]
    [InlineData("p")]
    [InlineData("div")]
    [InlineData("h7")]
    public void IsHeading_NonHeadingTag_ReturnsFalse(string tag)
    {
        ElementFilter.IsHeading(tag).Should().BeFalse();
    }

    [Theory]
    [InlineData("h1", "# ")]
    [InlineData("h2", "## ")]
    [InlineData("h3", "### ")]
    [InlineData("h4", "#### ")]
    [InlineData("h5", "##### ")]
    [InlineData("h6", "###### ")]
    public void HeadingPrefix_ReturnsCorrectMarker(string tag, string expected)
    {
        ElementFilter.HeadingPrefix(tag).Should().Be(expected);
    }

    [Theory]
    [InlineData("p")]
    [InlineData("div")]
    [InlineData("article")]
    [InlineData("section")]
    [InlineData("li")]
    [InlineData("td")]
    [InlineData("th")]
    [InlineData("blockquote")]
    [InlineData("pre")]
    public void IsBlock_BlockTag_ReturnsTrue(string tag)
    {
        ElementFilter.IsBlock(tag).Should().BeTrue();
    }

    [Theory]
    [InlineData("span")]
    [InlineData("a")]
    [InlineData("strong")]
    [InlineData("em")]
    public void IsBlock_InlineTag_ReturnsFalse(string tag)
    {
        ElementFilter.IsBlock(tag).Should().BeFalse();
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure (type not defined yet)**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~ElementFilterTests"
```
Expected: Build error — `ElementFilter` does not exist.

- [ ] **Step 3: Implement ElementFilter**

```csharp
// src/PicoLLM.Browser/Parsing/ElementFilter.cs
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
```

- [ ] **Step 4: Run tests — expect all pass**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~ElementFilterTests"
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Browser/Parsing/ElementFilter.cs tests/PicoLLM.Tests/Browser/ElementFilterTests.cs
git commit -m "feat(cap7): add ElementFilter with blacklist and block/heading classification"
```

---

## Task 3: Text Extractor

**Files:**
- Create: `src/PicoLLM.Browser/Parsing/TextExtractor.cs`
- Create: `tests/PicoLLM.Tests/Browser/TextExtractorTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Browser/TextExtractorTests.cs
using AngleSharp;
using AngleSharp.Dom;
using FluentAssertions;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.Tests.Browser;

public class TextExtractorTests
{
    // Helper: parse HTML fragment and return the body element
    private static async Task<IElement> GetBodyAsync(string html)
    {
        var context = BrowsingContext.New(Configuration.Default);
        var document = await context.OpenAsync(req => req.Content(html));
        return document.Body ?? document.DocumentElement;
    }

    [Fact]
    public async Task Extract_TwoParagraphs_JoinedWithDoubleNewline()
    {
        var body = await GetBodyAsync("<p>Hello</p><p>World</p>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("Hello");
        text.Should().Contain("World");
        // Paragraph boundary between them
        var helloIndex = text.IndexOf("Hello");
        var worldIndex = text.IndexOf("World");
        text[helloIndex..(worldIndex)].Should().Contain("\n\n");
    }

    [Fact]
    public async Task Extract_ScriptAndStyleContent_Discarded()
    {
        var body = await GetBodyAsync(
            "<script>alert('x')</script><style>.x{}</style><p>VisibleText</p>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("VisibleText");
        text.Should().NotContain("alert");
        text.Should().NotContain(".x{}");
    }

    [Fact]
    public async Task Extract_HeadingH1_PrefixedWithHash()
    {
        var body = await GetBodyAsync("<h1>My Title</h1>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("# My Title");
    }

    [Fact]
    public async Task Extract_HeadingH2_PrefixedWithDoubleHash()
    {
        var body = await GetBodyAsync("<h2>Subtitle</h2>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("## Subtitle");
    }

    [Fact]
    public async Task Extract_ImageWithAlt_InlineAltTextAndImageReference()
    {
        var body = await GetBodyAsync("<img src=\"photo.jpg\" alt=\"A cat sitting\">");
        var (text, images, _) = TextExtractor.Extract(body);
        text.Should().Contain("[Image: A cat sitting]");
        images.Should().ContainSingle();
        images[0].SourceUrl.Should().Be("photo.jpg");
        images[0].AltText.Should().Be("A cat sitting");
    }

    [Fact]
    public async Task Extract_ImageWithoutSrc_NotCollected()
    {
        var body = await GetBodyAsync("<img alt=\"no src\">");
        var (_, images, _) = TextExtractor.Extract(body);
        images.Should().BeEmpty();
    }

    [Fact]
    public async Task Extract_ImageWithoutAlt_NoInlineText()
    {
        var body = await GetBodyAsync("<img src=\"photo.jpg\">");
        var (text, images, _) = TextExtractor.Extract(body);
        text.Should().NotContain("[Image:");
        images.Should().ContainSingle();
        images[0].AltText.Should().BeNull();
    }

    [Fact]
    public async Task Extract_ValidLink_CollectedWithAnchorText()
    {
        var body = await GetBodyAsync("<a href=\"https://example.com\">Click here</a>");
        var (_, _, links) = TextExtractor.Extract(body);
        links.Should().ContainSingle();
        links[0].Href.Should().Be("https://example.com");
        links[0].AnchorText.Should().Be("Click here");
    }

    [Fact]
    public async Task Extract_RelativeLink_Collected()
    {
        var body = await GetBodyAsync("<a href=\"/about\">About Us</a>");
        var (_, _, links) = TextExtractor.Extract(body);
        links.Should().ContainSingle();
        links[0].Href.Should().Be("/about");
    }

    [Theory]
    [InlineData("javascript:void(0)")]
    [InlineData("mailto:user@example.com")]
    [InlineData("tel:+1234567890")]
    [InlineData("#section")]
    public async Task Extract_NonHttpLink_Excluded(string href)
    {
        var body = await GetBodyAsync($"<a href=\"{href}\">Link</a>");
        var (_, _, links) = TextExtractor.Extract(body);
        links.Should().BeEmpty();
    }

    [Fact]
    public async Task Extract_MultipleSpacesInsideParagraph_Collapsed()
    {
        var body = await GetBodyAsync("<p>  Hello    world  </p>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("Hello world");
        text.Should().NotContain("  ");
    }

    [Fact]
    public async Task Extract_HtmlEntities_Decoded()
    {
        var body = await GetBodyAsync("<p>Tom &amp; Jerry &lt;3</p>");
        var (text, _, _) = TextExtractor.Extract(body);
        text.Should().Contain("Tom & Jerry <3");
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~TextExtractorTests"
```
Expected: Build error — `TextExtractor` does not exist.

- [ ] **Step 3: Implement TextExtractor**

```csharp
// src/PicoLLM.Browser/Parsing/TextExtractor.cs
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
```

- [ ] **Step 4: Run tests — expect all pass**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~TextExtractorTests"
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.Browser/Parsing/TextExtractor.cs tests/PicoLLM.Tests/Browser/TextExtractorTests.cs
git commit -m "feat(cap7): add TextExtractor — DOM walker with text/image/link extraction"
```

---

## Task 4: HTML Parser (Orchestrator)

**Files:**
- Create: `src/PicoLLM.Browser/Parsing/HtmlParser.cs`
- Create: `tests/PicoLLM.Tests/Browser/HtmlParserTests.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// tests/PicoLLM.Tests/Browser/HtmlParserTests.cs
using FluentAssertions;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.Tests.Browser;

public class HtmlParserTests
{
    [Fact]
    public async Task ParseAsync_ExtractsTitle()
    {
        const string html = "<html><head><title>My Page</title></head><body><p>Hello</p></body></html>";
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.Title.Should().Be("My Page");
    }

    [Fact]
    public async Task ParseAsync_NoTitle_ReturnsEmptyString()
    {
        const string html = "<html><body><p>Hello</p></body></html>";
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.Title.Should().BeEmpty();
    }

    [Fact]
    public async Task ParseAsync_ExtractsMetaDescription()
    {
        const string html = """
            <html>
              <head>
                <meta name="description" content="A great page about cats">
              </head>
              <body><p>Content</p></body>
            </html>
            """;
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.MetaDescription.Should().Be("A great page about cats");
    }

    [Fact]
    public async Task ParseAsync_NoMetaDescription_ReturnsNull()
    {
        const string html = "<html><body><p>Hello</p></body></html>";
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.MetaDescription.Should().BeNull();
    }

    [Fact]
    public async Task ParseAsync_PreservesUrl()
    {
        const string html = "<html><body><p>Hello</p></body></html>";
        var result = await HtmlParser.ParseAsync(html, "https://example.com/page");
        result.Url.Should().Be("https://example.com/page");
    }

    [Fact]
    public async Task ParseAsync_ReturnsCleanText()
    {
        const string html = """
            <html><body>
              <h1>Title</h1>
              <p>First paragraph.</p>
              <script>alert('ignored')</script>
              <p>Second paragraph.</p>
            </body></html>
            """;
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.CleanText.Should().Contain("# Title");
        result.CleanText.Should().Contain("First paragraph.");
        result.CleanText.Should().Contain("Second paragraph.");
        result.CleanText.Should().NotContain("alert");
    }

    [Fact]
    public async Task ParseAsync_ReturnsImages()
    {
        const string html = """
            <html><body>
              <img src="cat.jpg" alt="A fluffy cat">
              <img src="dog.jpg" alt="A happy dog">
            </body></html>
            """;
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.Images.Should().HaveCount(2);
        result.Images[0].SourceUrl.Should().Be("cat.jpg");
        result.Images[1].AltText.Should().Be("A happy dog");
    }

    [Fact]
    public async Task ParseAsync_ReturnsLinks()
    {
        const string html = """
            <html><body>
              <a href="https://example.com/about">About</a>
              <a href="mailto:x@y.com">Email</a>
            </body></html>
            """;
        var result = await HtmlParser.ParseAsync(html, "https://example.com");
        result.Links.Should().ContainSingle();
        result.Links[0].Href.Should().Be("https://example.com/about");
        result.Links[0].AnchorText.Should().Be("About");
    }

    [Fact]
    public async Task ParseAsync_RealisticPage_ProducesCompleteResult()
    {
        const string html = """
            <!DOCTYPE html>
            <html>
              <head>
                <title>Neural Networks 101</title>
                <meta name="description" content="Learn the basics of neural networks">
                <style>body { font-family: sans-serif; }</style>
              </head>
              <body>
                <nav><a href="#skip">Skip to content</a></nav>
                <h1>Neural Networks</h1>
                <p>A neural network is a computational model.</p>
                <h2>Architecture</h2>
                <p>It consists of <strong>layers</strong>.</p>
                <img src="diagram.png" alt="Diagram of neural network architecture">
                <script>console.log('analytics')</script>
                <a href="https://example.com/deep-learning">Learn more</a>
              </body>
            </html>
            """;

        var result = await HtmlParser.ParseAsync(html, "https://example.com/nn101");

        result.Title.Should().Be("Neural Networks 101");
        result.MetaDescription.Should().Be("Learn the basics of neural networks");
        result.CleanText.Should().Contain("# Neural Networks");
        result.CleanText.Should().Contain("## Architecture");
        result.CleanText.Should().Contain("A neural network is a computational model.");
        result.CleanText.Should().Contain("[Image: Diagram of neural network architecture]");
        result.CleanText.Should().NotContain("analytics");
        result.Images.Should().ContainSingle(i => i.SourceUrl == "diagram.png");
        result.Links.Should().ContainSingle(l => l.Href == "https://example.com/deep-learning");
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~HtmlParserTests"
```
Expected: Build error — `HtmlParser` does not exist.

- [ ] **Step 3: Implement HtmlParser**

```csharp
// src/PicoLLM.Browser/Parsing/HtmlParser.cs
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
```

- [ ] **Step 4: Run all Capability 7 tests**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~HtmlParserTests|FullyQualifiedName~TextExtractorTests|FullyQualifiedName~ElementFilterTests"
```
Expected: All tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```
dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```
Expected: All existing tests continue to pass.

- [ ] **Step 6: Commit**

```bash
git add src/PicoLLM.Browser/Parsing/HtmlParser.cs tests/PicoLLM.Tests/Browser/HtmlParserTests.cs
git commit -m "feat(cap7): add HtmlParser — AngleSharp orchestrator, completes capability 7"
```

---

## Self-Review

**Spec coverage check:**
- [x] Text extraction with paragraph boundaries → Task 3 (two-paragraph test)
- [x] Ignore script and style → Task 3 (`ScriptAndStyleContent_Discarded`)
- [x] Element filtering blacklist → Task 2 (full blacklist theory test)
- [x] Heading extraction with level markers → Task 3 (h1/h2 tests), Task 4 (realistic page)
- [x] Image reference extraction (src + alt) → Task 3 (image tests)
- [x] Alt text inline in text → Task 3 (`[Image: ...]` test)
- [x] Link extraction → Task 3 (valid/relative link tests)
- [x] Filter non-HTTP links → Task 3 (Theory with javascript/mailto/tel/#)
- [x] HTML entity decoding → Task 3 (`HtmlEntities_Decoded`) — AngleSharp decodes natively; the test confirms it
- [x] Whitespace normalisation → Task 3 (`MultipleSpacesInsideParagraph_Collapsed`)
- [x] ParsedPage with all fields → Task 4 (all fields verified in `RealisticPage` test)
- [x] Title extraction → Task 4
- [x] Meta description extraction → Task 4

**Placeholder scan:** None found.

**Type consistency:** `ParsedPage`, `ImageReference`, `LinkReference` defined in Task 1 and used consistently in Tasks 3 and 4. `ElementFilter` methods defined in Task 2 and called in Task 3 with matching names (`ShouldSkip`, `IsBlock`, `IsHeading`, `HeadingPrefix`). `TextExtractor.Extract` signature is consistent throughout.

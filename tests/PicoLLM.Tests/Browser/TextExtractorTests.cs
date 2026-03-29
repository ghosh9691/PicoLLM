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

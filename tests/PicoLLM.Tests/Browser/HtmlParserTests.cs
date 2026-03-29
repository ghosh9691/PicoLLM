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

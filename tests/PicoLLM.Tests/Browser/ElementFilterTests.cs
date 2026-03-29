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

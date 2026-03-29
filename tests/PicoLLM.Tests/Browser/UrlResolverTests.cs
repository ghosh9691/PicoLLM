using FluentAssertions;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Browser;

public class UrlResolverTests
{
    [Fact]
    public void Resolve_AbsoluteUrl_ReturnsUnchanged()
    {
        string? result = UrlResolver.Resolve("https://example.com/page", "https://other.com/img.png");
        result.Should().Be("https://other.com/img.png");
    }

    [Fact]
    public void Resolve_RelativePath_ResolvesAgainstBase()
    {
        string? result = UrlResolver.Resolve("https://example.com/news/article.html", "images/photo.jpg");
        result.Should().Be("https://example.com/news/images/photo.jpg");
    }

    [Fact]
    public void Resolve_RootRelativePath_ResolvesFromHost()
    {
        string? result = UrlResolver.Resolve("https://example.com/news/article.html", "/images/photo.jpg");
        result.Should().Be("https://example.com/images/photo.jpg");
    }

    [Fact]
    public void Resolve_ProtocolRelative_InheritsScheme()
    {
        string? result = UrlResolver.Resolve("https://example.com/page", "//cdn.example.com/img.png");
        result.Should().Be("https://cdn.example.com/img.png");
    }

    [Fact]
    public void Resolve_EmptyString_ReturnsNull()
    {
        string? result = UrlResolver.Resolve("https://example.com/page", "");
        result.Should().BeNull();
    }

    [Fact]
    public void Resolve_NullLikeWhitespace_ReturnsNull()
    {
        string? result = UrlResolver.Resolve("https://example.com/page", "   ");
        result.Should().BeNull();
    }

    [Fact]
    public void Resolve_InvalidBase_ReturnsNull()
    {
        string? result = UrlResolver.Resolve("not-a-url", "relative/path.jpg");
        result.Should().BeNull();
    }

    [Fact]
    public void Resolve_ParentDirectoryRelative_ResolvesCorrectly()
    {
        string? result = UrlResolver.Resolve("https://example.com/a/b/c.html", "../img.jpg");
        result.Should().Be("https://example.com/a/img.jpg");
    }

    [Fact]
    public void Resolve_HttpsAbsoluteIsPassedThrough()
    {
        string base_ = "https://foo.com/";
        string abs = "https://bar.com/baz.html";
        UrlResolver.Resolve(base_, abs).Should().Be(abs);
    }
}

using FluentAssertions;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Browser;

public class RobotsTxtParserTests
{
    [Fact]
    public void Parse_WildcardDisallow_ReturnsPath()
    {
        const string robots = """
            User-agent: *
            Disallow: /private/
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().Contain("/private/");
    }

    [Fact]
    public void Parse_MultipleDisallows_ReturnsAll()
    {
        const string robots = """
            User-agent: *
            Disallow: /admin/
            Disallow: /secret/
            Disallow: /tmp/
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().BeEquivalentTo(["/admin/", "/secret/", "/tmp/"]);
    }

    [Fact]
    public void Parse_PicoBrowserDisallow_IsRespected()
    {
        const string robots = """
            User-agent: PicoBrowser
            Disallow: /restricted/
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().Contain("/restricted/");
    }

    [Fact]
    public void Parse_OtherAgentDisallow_IsIgnored()
    {
        const string robots = """
            User-agent: Googlebot
            Disallow: /google-only/

            User-agent: *
            Disallow: /everyone/
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().Contain("/everyone/");
        paths.Should().NotContain("/google-only/");
    }

    [Fact]
    public void Parse_EmptyDisallow_IsIgnored()
    {
        const string robots = """
            User-agent: *
            Disallow:
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().BeEmpty();
    }

    [Fact]
    public void Parse_CommentsIgnored()
    {
        const string robots = """
            # This is a comment
            User-agent: * # inline comment
            Disallow: /private/ # block this
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().Contain("/private/");
        paths.Should().HaveCount(1);
    }

    [Fact]
    public void Parse_EmptyContent_ReturnsEmpty()
    {
        var paths = RobotsTxtParser.Parse("");
        paths.Should().BeEmpty();
    }

    [Fact]
    public void Parse_AllowAll_ReturnsEmpty()
    {
        const string robots = """
            User-agent: *
            Allow: /
            """;

        var paths = RobotsTxtParser.Parse(robots);
        paths.Should().BeEmpty();
    }
}

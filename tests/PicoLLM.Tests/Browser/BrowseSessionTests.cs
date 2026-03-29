using System.Net;
using System.Net.Http;
using FluentAssertions;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Browser;

public class BrowseSessionTests
{
    [Fact]
    public void NewSession_HasEmptyPagesAndImages()
    {
        var session = new BrowseSession();
        session.Pages.Should().BeEmpty();
        session.Images.Should().BeEmpty();
    }

    [Fact]
    public void NewSession_StartedAt_IsUtcAndRecent()
    {
        var before = DateTime.UtcNow.AddSeconds(-1);
        var session = new BrowseSession();
        session.StartedAt.Should().BeAfter(before).And.BeBefore(DateTime.UtcNow.AddSeconds(1));
    }

    [Fact]
    public void BrowseResult_Record_CanBeConstructed()
    {
        var result = new BrowseResult(
            Url: "https://example.com",
            Status: BrowseStatus.Success,
            HtmlContent: "<html/>",
            ErrorMessage: null,
            HttpStatusCode: 200,
            ElapsedTime: TimeSpan.FromMilliseconds(50));

        result.Url.Should().Be("https://example.com");
        result.Status.Should().Be(BrowseStatus.Success);
        result.HtmlContent.Should().Be("<html/>");
        result.ErrorMessage.Should().BeNull();
        result.HttpStatusCode.Should().Be(200);
        result.ElapsedTime.Should().Be(TimeSpan.FromMilliseconds(50));
    }

    [Fact]
    public void BrowseStatus_AllValuesAccessible()
    {
        var values = Enum.GetValues<BrowseStatus>();
        values.Should().Contain(BrowseStatus.Success);
        values.Should().Contain(BrowseStatus.Redirect);
        values.Should().Contain(BrowseStatus.Error);
        values.Should().Contain(BrowseStatus.Blocked);
        values.Should().Contain(BrowseStatus.UnsupportedContentType);
        values.Should().Contain(BrowseStatus.Timeout);
    }

    [Fact]
    public void ImageDownload_Record_CanBeConstructed()
    {
        byte[] data = [1, 2, 3];
        var img = new ImageDownload(
            SourceUrl: "/img.jpg",
            AbsoluteUrl: "https://example.com/img.jpg",
            Data: data,
            ContentType: "image/jpeg",
            AltText: "A photo");

        img.SourceUrl.Should().Be("/img.jpg");
        img.AbsoluteUrl.Should().Be("https://example.com/img.jpg");
        img.Data.Should().BeEquivalentTo(data);
        img.ContentType.Should().Be("image/jpeg");
        img.AltText.Should().Be("A photo");
    }
}

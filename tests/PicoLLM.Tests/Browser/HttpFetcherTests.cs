using System.Net;
using System.Net.Http;
using FluentAssertions;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Browser;

/// <summary>
/// Tests for <see cref="HttpFetcher"/> using a fake <see cref="HttpMessageHandler"/>
/// so no real network calls are made.
/// </summary>
public class HttpFetcherTests
{
    // ── Helpers ──────────────────────────────────────────────────────────────

    private static HttpFetcher MakeFetcher(Func<HttpRequestMessage, HttpResponseMessage> handler)
        => new(timeoutSeconds: 5, new FakeHandler(handler));

    private static HttpResponseMessage HtmlResponse(string body, HttpStatusCode status = HttpStatusCode.OK)
    {
        var resp = new HttpResponseMessage(status)
        {
            Content = new StringContent(body, System.Text.Encoding.UTF8, "text/html")
        };
        return resp;
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [Fact]
    public async Task FetchAsync_SuccessHtml_ReturnsSuccessResult()
    {
        var fetcher = MakeFetcher(_ => HtmlResponse("<html>hello</html>"));

        var result = await fetcher.FetchAsync("https://example.com/page");

        result.Status.Should().Be(BrowseStatus.Success);
        result.HtmlContent.Should().Contain("hello");
        result.HttpStatusCode.Should().Be(200);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public async Task FetchAsync_InvalidUrl_ReturnsError()
    {
        var fetcher = MakeFetcher(_ => HtmlResponse("<html/>"));

        var result = await fetcher.FetchAsync("not-a-valid-url");

        result.Status.Should().Be(BrowseStatus.Error);
        result.HtmlContent.Should().BeNull();
    }

    [Fact]
    public async Task FetchAsync_NonHtmlContentType_ReturnsUnsupported()
    {
        var fetcher = MakeFetcher(_ =>
        {
            var resp = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent([])
            };
            resp.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/pdf");
            return resp;
        });

        var result = await fetcher.FetchAsync("https://example.com/doc.pdf");

        result.Status.Should().Be(BrowseStatus.UnsupportedContentType);
        result.HtmlContent.Should().BeNull();
        result.ErrorMessage.Should().Contain("application/pdf");
    }

    [Fact]
    public async Task FetchAsync_NetworkError_ReturnsErrorResult()
    {
        var fetcher = new HttpFetcher(5, new ThrowingHandler());

        var result = await fetcher.FetchAsync("https://example.com/");

        result.Status.Should().Be(BrowseStatus.Error);
        result.HtmlContent.Should().BeNull();
        result.ErrorMessage.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task FetchAsync_UserAgentHeader_IsSent()
    {
        string? capturedAgent = null;
        var fetcher = MakeFetcher(req =>
        {
            capturedAgent = req.Headers.UserAgent.ToString();
            return HtmlResponse("<html/>");
        });

        await fetcher.FetchAsync("https://example.com/");

        capturedAgent.Should().Contain("PicoBrowser");
    }

    [Fact]
    public async Task FetchAsync_ElapsedTime_IsPositive()
    {
        var fetcher = MakeFetcher(_ => HtmlResponse("<html/>"));
        var result = await fetcher.FetchAsync("https://example.com/");
        result.ElapsedTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    // ── Fake handlers ─────────────────────────────────────────────────────────

    private sealed class FakeHandler(Func<HttpRequestMessage, HttpResponseMessage> respond)
        : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(respond(request));
    }

    private sealed class ThrowingHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
            => throw new HttpRequestException("Simulated network failure");
    }
}

using System.Net;
using System.Net.Http;
using FluentAssertions;
using PicoLLM.Browser;

namespace PicoLLM.Tests.Browser;

public class ImageDownloaderTests
{
    private static readonly byte[] FakeImageBytes = [0xFF, 0xD8, 0xFF, 0xE0]; // minimal JPEG header

    private static HttpFetcher MakeFetcher(Func<HttpRequestMessage, HttpResponseMessage> handler)
        => new(timeoutSeconds: 5, new FakeHandler(handler));

    private static HttpResponseMessage ImageResponse(byte[] data, string mediaType)
    {
        var resp = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new ByteArrayContent(data)
        };
        resp.Content.Headers.ContentType =
            new System.Net.Http.Headers.MediaTypeHeaderValue(mediaType);
        return resp;
    }

    [Fact]
    public async Task DownloadAsync_Jpeg_ReturnsImageDownload()
    {
        var fetcher = MakeFetcher(_ => ImageResponse(FakeImageBytes, "image/jpeg"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync(
            "https://example.com/photo.jpg", "/photo.jpg", "A photo");

        result.Should().NotBeNull();
        result!.ContentType.Should().Be("image/jpeg");
        result.Data.Should().BeEquivalentTo(FakeImageBytes);
        result.AltText.Should().Be("A photo");
        result.SourceUrl.Should().Be("/photo.jpg");
        result.AbsoluteUrl.Should().Be("https://example.com/photo.jpg");
    }

    [Fact]
    public async Task DownloadAsync_UnsupportedContentType_ReturnsNull()
    {
        var fetcher = MakeFetcher(_ => ImageResponse([0x25, 0x50, 0x44, 0x46], "application/pdf"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync("https://example.com/doc.pdf", "/doc.pdf");
        result.Should().BeNull();
    }

    [Fact]
    public async Task DownloadAsync_Png_IsAccepted()
    {
        var fetcher = MakeFetcher(_ => ImageResponse([0x89, 0x50, 0x4E, 0x47], "image/png"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync("https://example.com/img.png", "img.png");
        result.Should().NotBeNull();
        result!.ContentType.Should().Be("image/png");
    }

    [Fact]
    public async Task DownloadAsync_Gif_IsAccepted()
    {
        var fetcher = MakeFetcher(_ => ImageResponse([0x47, 0x49, 0x46, 0x38], "image/gif"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync("https://example.com/anim.gif", "anim.gif");
        result.Should().NotBeNull();
        result!.ContentType.Should().Be("image/gif");
    }

    [Fact]
    public async Task DownloadAsync_InvalidUrl_ReturnsNull()
    {
        var fetcher = MakeFetcher(_ => ImageResponse(FakeImageBytes, "image/jpeg"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync("not-a-url", "not-a-url");
        result.Should().BeNull();
    }

    [Fact]
    public async Task DownloadAsync_NoAltText_AltTextIsNull()
    {
        var fetcher = MakeFetcher(_ => ImageResponse(FakeImageBytes, "image/jpeg"));
        var downloader = new ImageDownloaderAccessor(fetcher);

        var result = await downloader.DownloadAsync("https://example.com/x.jpg", "x.jpg");
        result!.AltText.Should().BeNull();
    }

    // ── Fake handler ─────────────────────────────────────────────────────────

    private sealed class FakeHandler(Func<HttpRequestMessage, HttpResponseMessage> respond)
        : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(respond(request));
    }

    /// <summary>
    /// Thin wrapper that exposes <see cref="ImageDownloader"/> construction from a
    /// test-controlled <see cref="HttpFetcher"/> and delegates <c>DownloadAsync</c>.
    /// </summary>
    private sealed class ImageDownloaderAccessor(HttpFetcher fetcher)
    {
        private readonly ImageDownloader _inner = new(fetcher);

        public Task<ImageDownload?> DownloadAsync(
            string absoluteUrl, string sourceUrl, string? altText = null,
            CancellationToken ct = default)
            => _inner.DownloadAsync(absoluteUrl, sourceUrl, altText, ct);
    }
}

namespace PicoLLM.Browser;

/// <summary>
/// Tracks all pages and images encountered during a browsing session.
/// A session uses a single <see cref="HttpFetcher"/> so the underlying HttpClient
/// and robots.txt cache are shared across all requests.
/// </summary>
public sealed class BrowseSession
{
    private readonly HttpFetcher _fetcher;
    private readonly ImageDownloader _imageDownloader;
    private readonly List<BrowseResult> _pages = [];
    private readonly List<ImageDownload> _images = [];

    /// <summary>All page fetch results recorded in this session, in order.</summary>
    public IReadOnlyList<BrowseResult> Pages => _pages;

    /// <summary>All downloaded images recorded in this session, in order.</summary>
    public IReadOnlyList<ImageDownload> Images => _images;

    /// <summary>UTC time when this session was created.</summary>
    public DateTime StartedAt { get; } = DateTime.UtcNow;

    /// <summary>
    /// Initialises a new session with an optional timeout override.
    /// </summary>
    /// <param name="timeoutSeconds">Per-request timeout in seconds (default 30).</param>
    public BrowseSession(int timeoutSeconds = 30)
    {
        _fetcher = new HttpFetcher(timeoutSeconds);
        _imageDownloader = new ImageDownloader(_fetcher);
    }

    /// <summary>
    /// Fetches <paramref name="url"/>, records the result, and returns it.
    /// Never throws — all errors are captured in the returned <see cref="BrowseResult"/>.
    /// </summary>
    public async Task<BrowseResult> BrowseAsync(string url, CancellationToken cancellationToken = default)
    {
        var result = await _fetcher.FetchAsync(url, cancellationToken).ConfigureAwait(false);
        _pages.Add(result);
        return result;
    }

    /// <summary>
    /// Downloads an image at <paramref name="absoluteUrl"/> and records it if successful.
    /// Returns null if the URL does not point to a supported image type or the download fails.
    /// </summary>
    /// <param name="absoluteUrl">Fully-resolved URL of the image.</param>
    /// <param name="sourceUrl">Original src attribute value (for provenance).</param>
    /// <param name="altText">Alt text from the img tag, if any.</param>
    public async Task<ImageDownload?> DownloadImageAsync(
        string absoluteUrl,
        string sourceUrl,
        string? altText = null,
        CancellationToken cancellationToken = default)
    {
        var image = await _imageDownloader.DownloadAsync(absoluteUrl, sourceUrl, altText, cancellationToken)
            .ConfigureAwait(false);
        if (image is not null)
            _images.Add(image);
        return image;
    }
}

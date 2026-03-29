namespace PicoLLM.Browser;

/// <summary>
/// Downloads images by absolute URL using the session's shared <see cref="HttpFetcher"/>.
/// Only JPEG, PNG, and GIF content types are accepted.
/// </summary>
public sealed class ImageDownloader
{
    private static readonly HashSet<string> SupportedTypes =
        ["image/jpeg", "image/png", "image/gif", "image/jpg"];

    private readonly HttpClient _client;

    internal ImageDownloader(HttpFetcher fetcher)
    {
        _client = fetcher.Client;
    }

    /// <summary>
    /// Downloads the image at <paramref name="absoluteUrl"/>.
    /// Returns null if the URL is invalid, the content type is unsupported,
    /// or any network error occurs.
    /// </summary>
    /// <param name="absoluteUrl">Fully-resolved URL of the image resource.</param>
    /// <param name="sourceUrl">Original src attribute value (for provenance).</param>
    /// <param name="altText">Alt attribute from the img tag, if any.</param>
    /// <param name="cancellationToken">Optional cancellation.</param>
    public async Task<ImageDownload?> DownloadAsync(
        string absoluteUrl,
        string sourceUrl,
        string? altText = null,
        CancellationToken cancellationToken = default)
    {
        if (!Uri.TryCreate(absoluteUrl, UriKind.Absolute, out _))
            return null;

        try
        {
            using var response = await _client.GetAsync(absoluteUrl,
                HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
                return null;

            string? contentType = response.Content.Headers.ContentType?.MediaType?.ToLowerInvariant();
            if (contentType is null || !SupportedTypes.Contains(contentType))
                return null;

            byte[] data = await response.Content.ReadAsByteArrayAsync(cancellationToken).ConfigureAwait(false);

            return new ImageDownload(sourceUrl, absoluteUrl, data, contentType, altText);
        }
        catch
        {
            return null;
        }
    }
}

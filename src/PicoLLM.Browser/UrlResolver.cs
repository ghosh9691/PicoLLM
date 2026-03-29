namespace PicoLLM.Browser;

/// <summary>
/// Resolves URLs relative to a base URL using <see cref="System.Uri"/> semantics.
/// Handles absolute URLs, protocol-relative URLs (//host/path), and relative paths.
/// </summary>
public static class UrlResolver
{
    /// <summary>
    /// Resolves <paramref name="relativeOrAbsolute"/> against <paramref name="baseUrl"/>.
    /// Returns null if the inputs are invalid or the resolution fails.
    /// </summary>
    /// <param name="baseUrl">The URL of the page that contains the link.</param>
    /// <param name="relativeOrAbsolute">The href/src attribute value to resolve.</param>
    public static string? Resolve(string baseUrl, string relativeOrAbsolute)
    {
        if (string.IsNullOrWhiteSpace(relativeOrAbsolute))
            return null;

        // Protocol-relative: //host/path — inherit scheme from base
        if (relativeOrAbsolute.StartsWith("//", StringComparison.Ordinal))
        {
            if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var baseUri))
                return null;
            relativeOrAbsolute = baseUri.Scheme + ":" + relativeOrAbsolute;
        }

        // Already absolute
        if (Uri.TryCreate(relativeOrAbsolute, UriKind.Absolute, out var absolute))
            return absolute.ToString();

        // Relative — resolve against base
        if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var baseUriForRelative))
            return null;

        if (!Uri.TryCreate(baseUriForRelative, relativeOrAbsolute, out var resolved))
            return null;

        return resolved.ToString();
    }
}

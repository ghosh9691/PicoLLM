namespace PicoLLM.Browser;

/// <summary>Status codes for a browse operation.</summary>
public enum BrowseStatus
{
    /// <summary>Page was fetched successfully.</summary>
    Success,

    /// <summary>A redirect was followed and the final destination was returned.</summary>
    Redirect,

    /// <summary>A network or HTTP error occurred.</summary>
    Error,

    /// <summary>The URL was blocked by robots.txt.</summary>
    Blocked,

    /// <summary>The server returned a non-HTML content type.</summary>
    UnsupportedContentType,

    /// <summary>The request timed out.</summary>
    Timeout
}

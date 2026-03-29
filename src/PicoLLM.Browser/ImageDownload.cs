namespace PicoLLM.Browser;

/// <summary>
/// Represents a successfully downloaded image resource.
/// </summary>
/// <param name="SourceUrl">The raw src attribute value from the img tag.</param>
/// <param name="AbsoluteUrl">The fully-resolved absolute URL used to download the image.</param>
/// <param name="Data">Raw image bytes.</param>
/// <param name="ContentType">MIME type, e.g. image/jpeg, image/png, image/gif.</param>
/// <param name="AltText">Alt attribute text from the img tag; null if absent.</param>
public record ImageDownload(
    string SourceUrl,
    string AbsoluteUrl,
    byte[] Data,
    string ContentType,
    string? AltText);

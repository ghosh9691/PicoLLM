# HTML Parser — Technical Design

## Library Choice

Use **AngleSharp** (NuGet) for HTML parsing. It provides a full DOM, handles malformed HTML gracefully, and is the standard .NET HTML parser. This is browser infrastructure, not ML code, so external libraries are permitted.

## Core Types

```csharp
public record ParsedPage(
    string Url,
    string Title,
    string? MetaDescription,
    string CleanText,               // paragraphs joined with \n\n
    List<ImageReference> Images,
    List<LinkReference> Links);

public record ImageReference(string SourceUrl, string? AltText);
public record LinkReference(string Href, string AnchorText);
```

## Parsing Pipeline

```
Raw HTML string
    │
    ▼
AngleSharp DOM parse
    │
    ├─► Extract <title>
    ├─► Extract <meta name="description">
    │
    ▼
Walk DOM tree (depth-first)
    │
    ├─► Skip filtered elements (script, style, iframe, etc.)
    ├─► Collect <img> → ImageReference list
    ├─► Collect <a href> → LinkReference list
    ├─► Collect text nodes → clean text with paragraph breaks
    │
    ▼
Post-processing
    ├─► Decode HTML entities
    ├─► Normalize whitespace
    ├─► Collapse blank lines
    │
    ▼
ParsedPage
```

## Heading Formatting

When a heading element (h1–h6) is encountered, prefix its text with markdown-style markers:
- `<h1>` → `# `
- `<h2>` → `## `
- etc.

This preserves document structure in the text corpus without requiring a separate metadata channel.

## Image Alt Text Integration

Alt text from images is inserted inline at the image's DOM position, wrapped in brackets: `[Image: alt text here]`. This ensures image descriptions become part of the training corpus in their natural document position.

## Link Filtering

Only retain links where:
- `href` starts with `http://`, `https://`, or `/` (relative)
- Not `javascript:`, `mailto:`, `tel:`, `#`, or empty

## Project Location

`src/PicoLLM.Browser/Parsing/`:
- `HtmlParser.cs` — main parser orchestrating AngleSharp
- `ParsedPage.cs` — result types
- `TextExtractor.cs` — DOM walk and text collection
- `ElementFilter.cs` — tag whitelist/blacklist

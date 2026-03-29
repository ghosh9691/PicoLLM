# HTML Parser — Design

**Date:** 2026-03-29
**Capability:** 7 — html-parser
**Project:** PicoLLM.Browser (existing)

## Scope

Add a `Parsing/` subfolder to the existing `PicoLLM.Browser` project. AngleSharp is already referenced. No new project or NuGet packages required.

## Architecture

Four files, each with one responsibility:

| File | Responsibility |
|------|----------------|
| `ParsedPage.cs` | Result records: `ParsedPage`, `ImageReference`, `LinkReference` |
| `ElementFilter.cs` | Tag blacklist — decides which DOM nodes to skip |
| `TextExtractor.cs` | AngleSharp DOM walker — emits clean text, heading markers, inline alt text |
| `HtmlParser.cs` | Orchestrator: parse → extract → normalize → return `ParsedPage` |

## Data Flow

```
Raw HTML string
    │
    ▼ AngleSharp IBrowsingContext.OpenAsync()
DOM IDocument
    │
    ├─► <title> → ParsedPage.Title
    ├─► <meta name="description"> → ParsedPage.MetaDescription
    │
    ▼ TextExtractor (depth-first DOM walk)
    │
    ├─► ElementFilter.ShouldSkip() → skip script/style/iframe/etc.
    ├─► <h1>–<h6> → "# text\n\n" (markdown-style prefix)
    ├─► <img> → inline "[Image: alt text]" + add to Images list
    ├─► <a href> → add to Links list (filtered)
    ├─► text nodes → accumulate with block-element paragraph breaks
    │
    ▼ Post-processing
    ├─► HTML entity decode (AngleSharp handles this natively)
    ├─► Collapse internal whitespace within runs
    ├─► Collapse multiple blank lines to double newline
    │
    ▼ ParsedPage
```

## Key Decisions

- **Heading format:** Markdown-style (`# `, `## `, etc.) — preserves hierarchy in training corpus without a separate metadata channel.
- **Image alt text:** Inserted inline as `[Image: alt text]` at DOM position — ensures image descriptions enter the corpus naturally.
- **Link filtering:** Only `http://`, `https://`, or `/`-relative hrefs are kept. `javascript:`, `mailto:`, `tel:`, `#`, and empty hrefs are discarded.
- **Element blacklist:** `script`, `style`, `noscript`, `iframe`, `object`, `embed`, `video`, `audio`, `canvas`, `svg`, `form`, `input`, `select`, `textarea`, `button`.

## Tests

- `ElementFilter`: blacklisted tags skipped, whitelisted retained
- `TextExtractor`: known HTML → expected text (paragraphs, headings, entities, whitespace)
- Image extraction: src/alt pairs collected correctly
- Link extraction: filter rules applied
- `HtmlParser` integration: full realistic page → complete `ParsedPage`

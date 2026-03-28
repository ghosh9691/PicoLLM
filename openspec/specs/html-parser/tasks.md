# HTML Parser — Implementation Tasks

## Setup
- [ ] 1.1 Add AngleSharp NuGet package to PicoLLM.Browser
- [ ] 1.2 Create `Parsing/` subfolder in PicoLLM.Browser

## Result Types
- [ ] 2.1 Implement `ParsedPage`, `ImageReference`, `LinkReference` records

## Element Filtering
- [ ] 3.1 Implement `ElementFilter` with blacklist (script, style, iframe, object, embed, video, audio, canvas, svg, noscript, form, input, select, textarea, button)
- [ ] 3.2 Write tests: blacklisted elements skipped, whitelisted elements retained

## Text Extraction
- [ ] 4.1 Implement DOM walker (depth-first traversal)
- [ ] 4.2 Insert paragraph breaks (\n\n) between block elements
- [ ] 4.3 Format headings with markdown-style prefixes (# ## ### etc.)
- [ ] 4.4 Insert alt text inline as [Image: alt text]
- [ ] 4.5 Decode HTML entities
- [ ] 4.6 Normalize whitespace (collapse multiple spaces, trim)
- [ ] 4.7 Write tests: known HTML → expected text output

## Image Extraction
- [ ] 5.1 Collect all <img> elements with src and alt attributes
- [ ] 5.2 Skip images without src
- [ ] 5.3 Write tests: various img tag patterns

## Link Extraction
- [ ] 6.1 Collect all <a href> elements with href and text content
- [ ] 6.2 Filter out javascript:, mailto:, tel:, #, empty hrefs
- [ ] 6.3 Write tests: link filtering, anchor text extraction

## Full Parser
- [ ] 7.1 Implement `HtmlParser.Parse(html, url)` → ParsedPage
- [ ] 7.2 Extract <title> and <meta description>
- [ ] 7.3 Write integration test with a realistic HTML page

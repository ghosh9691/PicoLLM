# HTML Parser Specification

## Purpose

Parse raw HTML into clean text, extract image references (with alt text), and extract hyperlinks. Discard everything else: JavaScript, CSS, ads, tracking, video, audio, iframes, and non-semantic markup.

### Requirement: Text Extraction

The system SHALL extract visible text content from HTML, preserving paragraph boundaries as double newlines and ignoring non-visible elements.

#### Scenario: Extract text from a page
- **GIVEN** HTML with `<p>Hello</p><p>World</p>`
- **WHEN** text is extracted
- **THEN** the result is "Hello\n\nWorld"

#### Scenario: Ignore script and style
- **GIVEN** HTML with `<script>alert('x')</script><style>.x{}</style><p>Text</p>`
- **WHEN** text is extracted
- **THEN** the result is "Text" (script and style content is discarded)

### Requirement: Element Filtering

The system SHALL discard content from these elements: `script`, `style`, `noscript`, `iframe`, `object`, `embed`, `video`, `audio`, `canvas`, `svg`, `form`, `input`, `select`, `textarea`, `button`, `nav` (optional), `footer` (optional).

#### Scenario: Filter non-content elements
- **GIVEN** HTML with a mix of content and non-content elements
- **WHEN** parsed
- **THEN** only text from content elements (p, h1-h6, div, span, article, section, main, li, td, th, blockquote, pre, code, a) is retained

### Requirement: Heading Extraction

The system SHALL preserve heading hierarchy as text markers.

#### Scenario: Heading levels
- **GIVEN** `<h1>Title</h1><h2>Subtitle</h2><p>Content</p>`
- **WHEN** extracted
- **THEN** output includes heading text with level indicators (e.g., "# Title\n## Subtitle\nContent")

### Requirement: Image Reference Extraction

The system SHALL extract all `<img>` tags, producing a list of (src URL, alt text) pairs.

#### Scenario: Extract images
- **GIVEN** `<img src="photo.jpg" alt="A cat sitting">`
- **WHEN** images are extracted
- **THEN** the result includes (src="photo.jpg", alt="A cat sitting")

#### Scenario: Alt text fed to tokenizer
- **GIVEN** an image with alt="Diagram of neural network architecture"
- **WHEN** the page is processed for training
- **THEN** the alt text "Diagram of neural network architecture" is included in the text corpus

### Requirement: Link Extraction

The system SHALL extract all `<a href>` tags, producing a list of (URL, anchor text) pairs.

#### Scenario: Extract links
- **GIVEN** `<a href="/about">About Us</a>`
- **WHEN** links are extracted
- **THEN** the result includes (href="/about", text="About Us")

#### Scenario: Filter non-HTTP links
- **GIVEN** links with href values starting with `javascript:`, `mailto:`, `tel:`, or `#`
- **WHEN** links are extracted
- **THEN** those links are excluded from the result

### Requirement: HTML Entity Decoding

The system SHALL decode HTML entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&#x27;`, numeric entities) to their text equivalents.

#### Scenario: Entities decoded
- **GIVEN** text "Tom &amp; Jerry &lt;3"
- **WHEN** decoded
- **THEN** the result is "Tom & Jerry <3"

### Requirement: Whitespace Normalization

The system SHALL collapse multiple whitespace characters (spaces, tabs, newlines within elements) into single spaces, while preserving paragraph boundaries between block elements.

#### Scenario: Normalize whitespace
- **GIVEN** `<p>  Hello    world  </p>`
- **WHEN** text is extracted
- **THEN** the result is "Hello world"

### Requirement: ParsedPage Result

The system SHALL return a `ParsedPage` object containing: clean text, list of image references, list of links, page title, and meta description.

#### Scenario: Complete parse
- **GIVEN** a full HTML page
- **WHEN** parsed
- **THEN** a `ParsedPage` is returned with all extracted content

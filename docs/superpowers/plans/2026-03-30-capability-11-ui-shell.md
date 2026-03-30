# Capability 11 — UI Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the PicoLLM desktop application — an Avalonia UI shell with a Lynx-style browser pane, live training dashboard, and session controls wired to `PicoOrchestrator`.

**Architecture:** MVVM-Lite with one `MainViewModel` (all live state + commands) and one `SettingsViewModel` (persisted config). Views are Avalonia AXAML with `ItemsControl` data templates. The orchestrator runs on a background thread; all event handlers marshal to `Dispatcher.UIThread.Post`. The browser pane renders `FormattedPageContent` built from `ParsedPage` by splitting `CleanText` on heading markers and matching link anchor text.

**Tech Stack:** .NET 10 / C# 12, Avalonia 11.2.5, CommunityToolkit.Mvvm 8.4.0, xUnit + FluentAssertions. Targets `net10.0-windows` and `net10.0-macos`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/PicoLLM.App/PicoLLM.App.csproj` | Convert to Avalonia app; net10.0-windows + net10.0-macos |
| Modify | `tests/PicoLLM.Tests/PicoLLM.Tests.csproj` | Change to net10.0-windows |
| Modify | `PicoLLM.slnx` | Add PicoLLM.App to solution |
| Create | `src/PicoLLM.App/Program.cs` | Avalonia entry point |
| Create | `src/PicoLLM.App/App.axaml` | Application resources + Fluent theme |
| Create | `src/PicoLLM.App/App.axaml.cs` | Application lifecycle |
| Create | `src/PicoLLM.App/MainWindow.axaml` | Root layout: address bar, split grid, status bar |
| Create | `src/PicoLLM.App/MainWindow.axaml.cs` | Code-behind: wires file pickers to ViewModel |
| Create | `src/PicoLLM.App/Models/FormattedPageContent.cs` | `IPageElement` hierarchy + `TextRun` |
| Create | `src/PicoLLM.App/Models/FormattedPageContentBuilder.cs` | Builds render model from `ParsedPage` |
| Create | `src/PicoLLM.App/ViewModels/SettingsViewModel.cs` | Config records + JSON persistence |
| Create | `src/PicoLLM.App/ViewModels/MainViewModel.cs` | All live UI state + commands |
| Create | `src/PicoLLM.App/Views/BrowserPane.axaml` | Lynx-style page renderer with back/forward |
| Create | `src/PicoLLM.App/Views/BrowserPane.axaml.cs` | Link click → NavigateCommand |
| Create | `src/PicoLLM.App/Views/LossChartControl.cs` | Custom `Control` with `DrawingContext` chart |
| Create | `src/PicoLLM.App/Views/DashboardPanel.axaml` | Metrics grid + LossChartControl |
| Create | `src/PicoLLM.App/Views/DashboardPanel.axaml.cs` | Code-behind (minimal) |
| Create | `src/PicoLLM.App/Views/ModelInfoPanel.axaml` | Static model config display |
| Create | `src/PicoLLM.App/Views/SettingsPanel.axaml` | Modal settings dialog |
| Create | `src/PicoLLM.App/Views/SettingsPanel.axaml.cs` | Folder picker + Save logic |
| Create | `tests/PicoLLM.Tests/UiShell/FormattedPageContentBuilderTests.cs` | Builder logic tests |
| Create | `tests/PicoLLM.Tests/UiShell/SettingsViewModelTests.cs` | JSON persistence tests |
| Create | `tests/PicoLLM.Tests/UiShell/MainViewModelNavigationTests.cs` | Back/forward/visited tests |

---

## Task 1: Project Scaffold

**Files:**
- Modify: `src/PicoLLM.App/PicoLLM.App.csproj`
- Modify: `tests/PicoLLM.Tests/PicoLLM.Tests.csproj`
- Modify: `PicoLLM.slnx`
- Create: `src/PicoLLM.App/Program.cs`
- Create: `src/PicoLLM.App/App.axaml`
- Create: `src/PicoLLM.App/App.axaml.cs`
- Create: `src/PicoLLM.App/MainWindow.axaml`
- Create: `src/PicoLLM.App/MainWindow.axaml.cs`

- [ ] **Step 1: Rewrite `src/PicoLLM.App/PicoLLM.App.csproj`**

Replace the entire file:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFrameworks>net10.0-windows;net10.0-macos</TargetFrameworks>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <BuiltInComInteropSupport>true</BuiltInComInteropSupport>
    <ApplicationManifest>app.manifest</ApplicationManifest>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="../PicoLLM.Core/PicoLLM.Core.csproj" />
    <ProjectReference Include="../PicoLLM.Tokenizer/PicoLLM.Tokenizer.csproj" />
    <ProjectReference Include="../PicoLLM.Training/PicoLLM.Training.csproj" />
    <ProjectReference Include="../PicoLLM.Browser/PicoLLM.Browser.csproj" />
    <ProjectReference Include="../PicoLLM.Gguf/PicoLLM.Gguf.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Avalonia" Version="11.2.5" />
    <PackageReference Include="Avalonia.Desktop" Version="11.2.5" />
    <PackageReference Include="Avalonia.Themes.Fluent" Version="11.2.5" />
    <PackageReference Include="CommunityToolkit.Mvvm" Version="8.4.0" />
  </ItemGroup>

</Project>
```

> Note: `ApplicationManifest` is Windows-only and is safe to leave as-is on macOS (the SDK ignores it).

- [ ] **Step 2: Update `tests/PicoLLM.Tests/PicoLLM.Tests.csproj` — change target framework**

Change `<TargetFramework>net9.0</TargetFramework>` to:

```xml
<TargetFramework>net10.0-windows</TargetFramework>
```

This is required because `PicoLLM.App` now targets `net10.0-windows;net10.0-macos`. The test project runs on Windows; `net10.0-windows` satisfies both the platform and .NET 10 requirement.

- [ ] **Step 3: Add `PicoLLM.App` to `PicoLLM.slnx`**

Open `PicoLLM.slnx` and add the App project inside the `<Folder Name="/src/">` element:

```xml
<Project Path="src/PicoLLM.App/PicoLLM.App.csproj" />
```

- [ ] **Step 4: Create `src/PicoLLM.App/Program.cs`**

```csharp
using Avalonia;
using PicoLLM.App;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
```

- [ ] **Step 5: Create `src/PicoLLM.App/App.axaml`**

```xml
<Application xmlns="https://github.com/avaloniaui"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             x:Class="PicoLLM.App.App"
             RequestedThemeVariant="Light">
  <Application.Styles>
    <FluentTheme />
  </Application.Styles>
</Application>
```

- [ ] **Step 6: Create `src/PicoLLM.App/App.axaml.cs`**

```csharp
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using PicoLLM.App.Views;

namespace PicoLLM.App;

/// <summary>Avalonia application entry point. Bootstraps the main window.</summary>
public partial class App : Application
{
    /// <inheritdoc/>
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc/>
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = new MainWindow();

        base.OnFrameworkInitializationCompleted();
    }
}
```

- [ ] **Step 7: Create stub `src/PicoLLM.App/MainWindow.axaml`**

```xml
<Window xmlns="https://github.com/avaloniaui"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        x:Class="PicoLLM.App.MainWindow"
        Title="PicoLLM"
        Width="1100" Height="700"
        MinWidth="800" MinHeight="500">
  <TextBlock Text="PicoLLM — loading…" HorizontalAlignment="Center" VerticalAlignment="Center"/>
</Window>
```

- [ ] **Step 8: Create stub `src/PicoLLM.App/MainWindow.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace PicoLLM.App;

/// <summary>Root application window.</summary>
public partial class MainWindow : Window
{
    /// <summary>Initializes the main window.</summary>
    public MainWindow() => AvaloniaXamlLoader.Load(this);
}
```

- [ ] **Step 9: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded. 0 Warning(s) 0 Error(s)`

- [ ] **Step 10: Verify tests still pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```

Expected: All existing 327 tests pass.

- [ ] **Step 11: Commit**

```bash
cd d:/source/ghosh9691/PicoLLM
git add src/PicoLLM.App/PicoLLM.App.csproj tests/PicoLLM.Tests/PicoLLM.Tests.csproj PicoLLM.slnx src/PicoLLM.App/Program.cs src/PicoLLM.App/App.axaml src/PicoLLM.App/App.axaml.cs src/PicoLLM.App/MainWindow.axaml src/PicoLLM.App/MainWindow.axaml.cs
git commit -m "feat(cap11): scaffold Avalonia app — Program, App, stub MainWindow"
```

---

## Task 2: FormattedPageContent Models

**Files:**
- Create: `src/PicoLLM.App/Models/FormattedPageContent.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Models/FormattedPageContent.cs`**

```csharp
namespace PicoLLM.App.Models;

/// <summary>The full structured content of a parsed page ready for browser-pane rendering.</summary>
/// <param name="Elements">Ordered list of page elements (headings, paragraphs, images).</param>
public record FormattedPageContent(IReadOnlyList<IPageElement> Elements);

/// <summary>Marker interface for all browser-pane page elements.</summary>
public interface IPageElement { }

/// <summary>A heading element (h1–h3).</summary>
/// <param name="Text">Heading text with markup stripped.</param>
/// <param name="Level">Heading level: 1 = h1, 2 = h2, 3 = h3.</param>
public record HeadingElement(string Text, int Level) : IPageElement;

/// <summary>A paragraph composed of styled text runs (plain text and links).</summary>
/// <param name="Runs">Ordered list of text runs that make up this paragraph.</param>
public record ParagraphElement(IReadOnlyList<TextRun> Runs) : IPageElement;

/// <summary>An image alt-text element rendered as italic grey bracketed text.</summary>
/// <param name="AltText">The image alt text.</param>
public record ImageAltElement(string AltText) : IPageElement;

/// <summary>A single styled run of text within a paragraph.</summary>
/// <param name="Text">The visible text of this run.</param>
/// <param name="IsLink">True if this run is a hyperlink.</param>
/// <param name="Href">The link target URL; null for plain text runs.</param>
/// <param name="IsVisited">True if this URL has been visited in the current session.</param>
public record TextRun(string Text, bool IsLink, string? Href, bool IsVisited);
```

- [ ] **Step 2: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 3: Commit**

```bash
git add src/PicoLLM.App/Models/FormattedPageContent.cs
git commit -m "feat(cap11): add FormattedPageContent render model types"
```

---

## Task 3: FormattedPageContentBuilder (TDD)

**Files:**
- Create: `src/PicoLLM.App/Models/FormattedPageContentBuilder.cs`
- Create: `tests/PicoLLM.Tests/UiShell/FormattedPageContentBuilderTests.cs`

- [ ] **Step 1: Write failing tests**

Create `tests/PicoLLM.Tests/UiShell/FormattedPageContentBuilderTests.cs`:

```csharp
using FluentAssertions;
using PicoLLM.App.Models;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.Tests.UiShell;

public class FormattedPageContentBuilderTests
{
    private static ParsedPage MakePage(string cleanText,
        List<LinkReference>? links = null,
        List<ImageReference>? images = null) =>
        new("https://example.com", "Test", null, cleanText,
            images ?? [], links ?? []);

    [Fact]
    public void Build_H1_ProducesHeadingElementLevel1()
    {
        var page = MakePage("# Neural Network");
        var result = FormattedPageContentBuilder.Build(page);
        result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<HeadingElement>()
            .Which.Should().BeEquivalentTo(new HeadingElement("Neural Network", 1));
    }

    [Fact]
    public void Build_H2_ProducesHeadingElementLevel2()
    {
        var page = MakePage("## Architecture");
        var result = FormattedPageContentBuilder.Build(page);
        result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<HeadingElement>()
            .Which.Should().BeEquivalentTo(new HeadingElement("Architecture", 2));
    }

    [Fact]
    public void Build_H3_ProducesHeadingElementLevel3()
    {
        var page = MakePage("### Backpropagation");
        var result = FormattedPageContentBuilder.Build(page);
        result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<HeadingElement>()
            .Which.Should().BeEquivalentTo(new HeadingElement("Backpropagation", 3));
    }

    [Fact]
    public void Build_ImageAltText_ProducesImageAltElement()
    {
        var page = MakePage("[Image: A diagram of neural layers]");
        var result = FormattedPageContentBuilder.Build(page);
        result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<ImageAltElement>()
            .Which.Should().BeEquivalentTo(new ImageAltElement("A diagram of neural layers"));
    }

    [Fact]
    public void Build_PlainParagraph_ProducesSinglePlainRun()
    {
        var page = MakePage("Some plain body text.");
        var result = FormattedPageContentBuilder.Build(page);
        var para = result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<ParagraphElement>().Subject;
        para.Runs.Should().ContainSingle()
            .Which.Should().BeEquivalentTo(new TextRun("Some plain body text.", false, null, false));
    }

    [Fact]
    public void Build_ParagraphWithLink_SplitsIntoRuns()
    {
        var links = new List<LinkReference> { new("https://en.wikipedia.org/wiki/Neuron", "neuron") };
        var page = MakePage("A neuron processes inputs.", links);
        var result = FormattedPageContentBuilder.Build(page);
        var para = result.Elements.Should().ContainSingle()
            .Which.Should().BeOfType<ParagraphElement>().Subject;

        para.Runs.Should().HaveCount(3);
        para.Runs[0].Should().BeEquivalentTo(new TextRun("A ", false, null, false));
        para.Runs[1].Should().BeEquivalentTo(new TextRun("neuron", true, "https://en.wikipedia.org/wiki/Neuron", false));
        para.Runs[2].Should().BeEquivalentTo(new TextRun(" processes inputs.", false, null, false));
    }

    [Fact]
    public void Build_VisitedLink_MarkedIsVisitedTrue()
    {
        var links = new List<LinkReference> { new("https://example.com/about", "About") };
        var visited = new HashSet<string> { "https://example.com/about" };
        var page = MakePage("See the About page.", links);
        var result = FormattedPageContentBuilder.Build(page, visited);
        var para = result.Elements.OfType<ParagraphElement>().Single();
        para.Runs.Single(r => r.IsLink).IsVisited.Should().BeTrue();
    }

    [Fact]
    public void Build_MultipleElements_PreservesOrder()
    {
        var cleanText = "# Title\n\nFirst paragraph.\n\n## Subtitle\n\nSecond paragraph.";
        var page = MakePage(cleanText);
        var result = FormattedPageContentBuilder.Build(page);

        result.Elements.Should().HaveCount(4);
        result.Elements[0].Should().BeOfType<HeadingElement>().Which.Level.Should().Be(1);
        result.Elements[1].Should().BeOfType<ParagraphElement>();
        result.Elements[2].Should().BeOfType<HeadingElement>().Which.Level.Should().Be(2);
        result.Elements[3].Should().BeOfType<ParagraphElement>();
    }

    [Fact]
    public void Build_EmptyCleanText_ReturnsEmpty()
    {
        var page = MakePage("");
        var result = FormattedPageContentBuilder.Build(page);
        result.Elements.Should().BeEmpty();
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~FormattedPageContentBuilder"
```

Expected: Build error — `FormattedPageContentBuilder` does not exist.

- [ ] **Step 3: Create `src/PicoLLM.App/Models/FormattedPageContentBuilder.cs`**

```csharp
using System.Text.RegularExpressions;
using PicoLLM.Browser.Parsing;

namespace PicoLLM.App.Models;

/// <summary>
/// Builds a <see cref="FormattedPageContent"/> from a <see cref="ParsedPage"/>
/// by parsing the <c>CleanText</c> into structured page elements.
/// </summary>
public static class FormattedPageContentBuilder
{
    private static readonly Regex ImageAltRegex =
        new(@"^\[Image:\s*(.+?)\s*\]$", RegexOptions.Compiled);

    /// <summary>
    /// Converts a <see cref="ParsedPage"/> into a <see cref="FormattedPageContent"/>
    /// suitable for rendering in the browser pane.
    /// </summary>
    /// <param name="page">The parsed page from <see cref="HtmlParser"/>.</param>
    /// <param name="visitedUrls">
    ///   Set of URLs already visited this session; used to mark links as visited.
    /// </param>
    public static FormattedPageContent Build(
        ParsedPage page,
        HashSet<string>? visitedUrls = null)
    {
        visitedUrls ??= [];
        var elements = new List<IPageElement>();

        var blocks = page.CleanText.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

        foreach (var block in blocks)
        {
            var trimmed = block.Trim();
            if (string.IsNullOrWhiteSpace(trimmed)) continue;

            // Heading detection: # / ## / ###
            var headingLevel = DetectHeadingLevel(trimmed);
            if (headingLevel > 0)
            {
                var headingText = trimmed[(headingLevel + 1)..].Trim(); // skip "# " prefix
                elements.Add(new HeadingElement(headingText, headingLevel));
                continue;
            }

            // Image alt text detection: [Image: ...]
            var imageMatch = ImageAltRegex.Match(trimmed);
            if (imageMatch.Success)
            {
                elements.Add(new ImageAltElement(imageMatch.Groups[1].Value));
                continue;
            }

            // Paragraph with inline link matching
            var runs = BuildTextRuns(trimmed, page.Links, visitedUrls);
            if (runs.Count > 0)
                elements.Add(new ParagraphElement(runs));
        }

        return new FormattedPageContent(elements);
    }

    private static int DetectHeadingLevel(string block)
    {
        if (block.StartsWith("### ")) return 3;
        if (block.StartsWith("## "))  return 2;
        if (block.StartsWith("# "))   return 1;
        return 0;
    }

    private static List<TextRun> BuildTextRuns(
        string text,
        IEnumerable<LinkReference> links,
        HashSet<string> visitedUrls)
    {
        // Find positions of all link anchor texts in the paragraph text.
        // Longer matches are preferred to avoid partial matches.
        var spans = new List<(int Start, int End, string Href, bool Visited)>();
        foreach (var link in links.OrderByDescending(l => l.AnchorText.Length))
        {
            int idx = text.IndexOf(link.AnchorText, StringComparison.Ordinal);
            if (idx < 0) continue;

            // Skip if this span overlaps an already-claimed range
            bool overlaps = spans.Any(s =>
                idx < s.End && idx + link.AnchorText.Length > s.Start);
            if (overlaps) continue;

            spans.Add((idx, idx + link.AnchorText.Length,
                link.Href, visitedUrls.Contains(link.Href)));
        }

        spans.Sort((a, b) => a.Start.CompareTo(b.Start));

        var runs = new List<TextRun>();
        int pos = 0;
        foreach (var span in spans)
        {
            if (span.Start > pos)
                runs.Add(new TextRun(text[pos..span.Start], false, null, false));
            runs.Add(new TextRun(text[span.Start..span.End], true, span.Href, span.Visited));
            pos = span.End;
        }
        if (pos < text.Length)
            runs.Add(new TextRun(text[pos..], false, null, false));

        return runs.Count > 0 ? runs : [new TextRun(text, false, null, false)];
    }
}
```

- [ ] **Step 4: Run tests — expect all pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~FormattedPageContentBuilder"
```

Expected: All 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.App/Models/FormattedPageContentBuilder.cs tests/PicoLLM.Tests/UiShell/FormattedPageContentBuilderTests.cs
git commit -m "feat(cap11): add FormattedPageContentBuilder with TDD — parses ParsedPage into render elements"
```

---

## Task 4: SettingsViewModel (TDD)

**Files:**
- Create: `src/PicoLLM.App/ViewModels/SettingsViewModel.cs`
- Create: `tests/PicoLLM.Tests/UiShell/SettingsViewModelTests.cs`

- [ ] **Step 1: Write failing tests**

Create `tests/PicoLLM.Tests/UiShell/SettingsViewModelTests.cs`:

```csharp
using FluentAssertions;
using PicoLLM.App.ViewModels;

namespace PicoLLM.Tests.UiShell;

public class SettingsViewModelTests : IDisposable
{
    private readonly string _dir =
        Path.Combine(Path.GetTempPath(), "pico_settings_" + Guid.NewGuid().ToString("N"));

    public SettingsViewModelTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void DefaultValues_AreCorrect()
    {
        var vm = new SettingsViewModel(_dir);
        vm.DataDirectory.Should().NotBeNullOrEmpty();
        vm.VocabSize.Should().Be(1024);
        vm.EmbedDim.Should().Be(128);
        vm.NumHeads.Should().Be(4);
        vm.NumLayers.Should().Be(4);
        vm.MaxSeqLen.Should().Be(512);
        vm.StepsPerPage.Should().Be(100);
        vm.LearningRate.Should().BeApproximately(1e-4f, 1e-7f);
        vm.BatchSize.Should().Be(4);
        vm.SeqLen.Should().Be(128);
        vm.WarmupSteps.Should().Be(50);
    }

    [Fact]
    public void SaveAndLoad_RoundTripsAllValues()
    {
        var vm = new SettingsViewModel(_dir)
        {
            DataDirectory = "/custom/path",
            VocabSize     = 2048,
            EmbedDim      = 256,
            NumHeads      = 8,
            NumLayers     = 6,
            MaxSeqLen     = 1024,
            StepsPerPage  = 200,
            LearningRate  = 3e-4f,
            BatchSize     = 8,
            SeqLen        = 256,
            WarmupSteps   = 100
        };
        vm.Save();

        var vm2 = new SettingsViewModel(_dir);
        vm2.Load();

        vm2.DataDirectory.Should().Be("/custom/path");
        vm2.VocabSize.Should().Be(2048);
        vm2.EmbedDim.Should().Be(256);
        vm2.NumHeads.Should().Be(8);
        vm2.NumLayers.Should().Be(6);
        vm2.MaxSeqLen.Should().Be(1024);
        vm2.StepsPerPage.Should().Be(200);
        vm2.LearningRate.Should().BeApproximately(3e-4f, 1e-6f);
        vm2.BatchSize.Should().Be(8);
        vm2.SeqLen.Should().Be(256);
        vm2.WarmupSteps.Should().Be(100);
    }

    [Fact]
    public void Load_WhenNoFile_KeepsDefaults()
    {
        var vm = new SettingsViewModel(_dir);
        vm.Load(); // no settings.json exists yet
        vm.VocabSize.Should().Be(1024);
    }

    [Fact]
    public void ToOrchestratorConfig_ReflectsAllSettings()
    {
        var vm = new SettingsViewModel(_dir) { DataDirectory = _dir };
        var config = vm.ToOrchestratorConfig();
        config.Model.VocabSize.Should().Be(vm.VocabSize);
        config.Model.EmbedDim.Should().Be(vm.EmbedDim);
        config.Model.NumHeads.Should().Be(vm.NumHeads);
        config.Model.NumLayers.Should().Be(vm.NumLayers);
        config.Model.MaxSeqLen.Should().Be(vm.MaxSeqLen);
        config.StepsPerPage.Should().Be(vm.StepsPerPage);
        config.Training.LearningRate.Should().BeApproximately(vm.LearningRate, 1e-7f);
        config.DataDirectory.Should().Be(_dir);
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~SettingsViewModelTests"
```

Expected: Build error — `SettingsViewModel` does not exist.

- [ ] **Step 3: Create `src/PicoLLM.App/ViewModels/SettingsViewModel.cs`**

```csharp
using System.Text.Json;
using System.Text.Json.Serialization;
using CommunityToolkit.Mvvm.ComponentModel;
using PicoLLM.App.Orchestration;
using PicoLLM.Core.Model;

namespace PicoLLM.App.ViewModels;

/// <summary>
/// Persisted application settings including data directory and model/training hyperparameters.
/// Saved as JSON to the platform AppData directory, not to <see cref="DataDirectory"/>
/// (which avoids a chicken-and-egg problem when DataDirectory itself changes).
/// </summary>
public partial class SettingsViewModel : ObservableObject
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly string _settingsDir;

    private string SettingsFilePath => Path.Combine(_settingsDir, "settings.json");

    // ── Data directory ────────────────────────────────────────────────────────

    /// <summary>Directory for tokenizer, checkpoints, and GGUF exports.</summary>
    [ObservableProperty] private string _dataDirectory = GetPlatformDefaultDataDir();

    // ── Model hyperparameters ─────────────────────────────────────────────────

    /// <summary>Target BPE vocabulary size.</summary>
    [ObservableProperty] private int _vocabSize = 1024;

    /// <summary>Token embedding dimension.</summary>
    [ObservableProperty] private int _embedDim = 128;

    /// <summary>Number of attention heads per layer.</summary>
    [ObservableProperty] private int _numHeads = 4;

    /// <summary>Number of transformer decoder layers.</summary>
    [ObservableProperty] private int _numLayers = 4;

    /// <summary>Maximum context length in tokens.</summary>
    [ObservableProperty] private int _maxSeqLen = 512;

    /// <summary>Training steps to run per fetched page.</summary>
    [ObservableProperty] private int _stepsPerPage = 100;

    // ── Training hyperparameters ──────────────────────────────────────────────

    /// <summary>Peak learning rate for AdamW.</summary>
    [ObservableProperty] private float _learningRate = 1e-4f;

    /// <summary>Number of sequences per training batch.</summary>
    [ObservableProperty] private int _batchSize = 4;

    /// <summary>Input sequence length for training (tokens).</summary>
    [ObservableProperty] private int _seqLen = 128;

    /// <summary>Linear LR warmup steps.</summary>
    [ObservableProperty] private int _warmupSteps = 50;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a <see cref="SettingsViewModel"/> that persists to <paramref name="settingsDir"/>.
    /// Defaults to the platform AppData directory when <paramref name="settingsDir"/> is null.
    /// </summary>
    public SettingsViewModel(string? settingsDir = null)
    {
        _settingsDir = settingsDir ?? GetPlatformAppDataDir();
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// <summary>Loads settings from <c>settings.json</c>. Silently keeps defaults if file absent or corrupt.</summary>
    public void Load()
    {
        var path = SettingsFilePath;
        if (!File.Exists(path)) return;
        try
        {
            var dto = JsonSerializer.Deserialize<SettingsDto>(File.ReadAllText(path), JsonOptions);
            if (dto is null) return;
            DataDirectory = dto.DataDirectory ?? DataDirectory;
            VocabSize     = dto.VocabSize;
            EmbedDim      = dto.EmbedDim;
            NumHeads      = dto.NumHeads;
            NumLayers     = dto.NumLayers;
            MaxSeqLen     = dto.MaxSeqLen;
            StepsPerPage  = dto.StepsPerPage;
            LearningRate  = dto.LearningRate;
            BatchSize     = dto.BatchSize;
            SeqLen        = dto.SeqLen;
            WarmupSteps   = dto.WarmupSteps;
        }
        catch { /* corrupt file — use defaults */ }
    }

    /// <summary>Saves current settings to <c>settings.json</c>.</summary>
    public void Save()
    {
        Directory.CreateDirectory(_settingsDir);
        var dto = new SettingsDto(DataDirectory, VocabSize, EmbedDim, NumHeads, NumLayers,
            MaxSeqLen, StepsPerPage, LearningRate, BatchSize, SeqLen, WarmupSteps);
        File.WriteAllText(SettingsFilePath, JsonSerializer.Serialize(dto, JsonOptions));
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// <summary>Builds an <see cref="OrchestratorConfig"/> from the current settings.</summary>
    public OrchestratorConfig ToOrchestratorConfig() =>
        new(
            Model: new ModelConfig(VocabSize, EmbedDim, NumHeads, NumLayers, 4, MaxSeqLen),
            Training: new TrainingConfig(BatchSize, SeqLen, LearningRate, WarmupSteps, 1.0f),
            DataDirectory: DataDirectory,
            StepsPerPage: StepsPerPage);

    // ── Platform helpers ──────────────────────────────────────────────────────

    private static string GetPlatformAppDataDir() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "PicoLLM");

    private static string GetPlatformDefaultDataDir() =>
        Path.Combine(GetPlatformAppDataDir(), "sessions");

    // ── DTO ───────────────────────────────────────────────────────────────────

    private record SettingsDto(
        string? DataDirectory,
        int     VocabSize,
        int     EmbedDim,
        int     NumHeads,
        int     NumLayers,
        int     MaxSeqLen,
        int     StepsPerPage,
        float   LearningRate,
        int     BatchSize,
        int     SeqLen,
        int     WarmupSteps);
}
```

- [ ] **Step 4: Run tests — expect all pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~SettingsViewModelTests"
```

Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.App/ViewModels/SettingsViewModel.cs tests/PicoLLM.Tests/UiShell/SettingsViewModelTests.cs
git commit -m "feat(cap11): add SettingsViewModel with JSON persistence and TDD"
```

---

## Task 5: MainViewModel — Navigation Logic (TDD)

**Files:**
- Create: `src/PicoLLM.App/ViewModels/MainViewModel.cs` (navigation portion)
- Create: `tests/PicoLLM.Tests/UiShell/MainViewModelNavigationTests.cs`

- [ ] **Step 1: Write failing tests**

Create `tests/PicoLLM.Tests/UiShell/MainViewModelNavigationTests.cs`:

```csharp
using FluentAssertions;
using PicoLLM.App.ViewModels;

namespace PicoLLM.Tests.UiShell;

public class MainViewModelNavigationTests
{
    // Synchronous dispatcher for tests (no Avalonia runtime needed)
    private static MainViewModel MakeVm() => new(dispatch: a => a());

    [Fact]
    public void InitialState_CanGoBackAndForwardAreFalse()
    {
        var vm = MakeVm();
        vm.CanGoBack.Should().BeFalse();
        vm.CanGoForward.Should().BeFalse();
    }

    [Fact]
    public async Task NavigateAsync_SetsAddressBarUrl()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://example.com");
        vm.AddressBarUrl.Should().Be("https://example.com");
    }

    [Fact]
    public async Task NavigateAsync_SecondUrl_EnablesCanGoBack()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://example.com");
        await vm.NavigateCommand.ExecuteAsync("https://example.com/about");
        vm.CanGoBack.Should().BeTrue();
    }

    [Fact]
    public async Task NavigateAsync_NewUrl_ClearsForwardStack()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://a.com");
        await vm.NavigateCommand.ExecuteAsync("https://b.com");
        vm.GoBackCommand.Execute(null);
        vm.CanGoForward.Should().BeTrue();

        await vm.NavigateCommand.ExecuteAsync("https://c.com");
        vm.CanGoForward.Should().BeFalse();
    }

    [Fact]
    public async Task GoBack_RestoresPreviousUrl()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://a.com");
        await vm.NavigateCommand.ExecuteAsync("https://b.com");
        vm.GoBackCommand.Execute(null);
        vm.AddressBarUrl.Should().Be("https://a.com");
    }

    [Fact]
    public async Task GoForward_RestoresNextUrl()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://a.com");
        await vm.NavigateCommand.ExecuteAsync("https://b.com");
        vm.GoBackCommand.Execute(null);
        vm.GoForwardCommand.Execute(null);
        vm.AddressBarUrl.Should().Be("https://b.com");
    }

    [Fact]
    public async Task Navigate_MarksUrlAsVisited()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://example.com");
        vm.IsVisited("https://example.com").Should().BeTrue();
    }

    [Fact]
    public void IsVisited_UnvisitedUrl_ReturnsFalse()
    {
        var vm = MakeVm();
        vm.IsVisited("https://never-visited.com").Should().BeFalse();
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~MainViewModelNavigationTests"
```

Expected: Build error — `MainViewModel` does not exist.

- [ ] **Step 3: Create `src/PicoLLM.App/ViewModels/MainViewModel.cs`**

```csharp
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PicoLLM.App.Models;
using PicoLLM.App.Orchestration;

namespace PicoLLM.App.ViewModels;

/// <summary>
/// Central view model for the PicoLLM desktop application.
/// Owns all live UI state and commands. Runs the orchestrator on a background thread
/// and marshals progress events back to the UI thread via the injected dispatcher.
/// </summary>
public partial class MainViewModel : ObservableObject
{
    // ── Dispatcher ───────────────────────────────────────────────────────────
    // Injected so tests can provide a synchronous dispatch (no Avalonia runtime needed).
    private readonly Action<Action> _dispatch;

    // ── Navigation state ─────────────────────────────────────────────────────
    private readonly Stack<string> _backStack  = new();
    private readonly Stack<string> _fwdStack   = new();
    private readonly HashSet<string> _visitedUrls = new();
    private string? _currentUrl;

    // ── Orchestrator ──────────────────────────────────────────────────────────
    private PicoOrchestrator? _orchestrator;
    private CancellationTokenSource? _cts;

    // ── Observable properties ─────────────────────────────────────────────────

    /// <summary>Text shown in the address bar.</summary>
    [ObservableProperty] private string _addressBarUrl = "";

    /// <summary>The rendered page currently displayed in the browser pane.</summary>
    [ObservableProperty] private FormattedPageContent? _currentPage;

    /// <summary>True when a previous page is available.</summary>
    [ObservableProperty] private bool _canGoBack;

    /// <summary>True when a forward page is available (after going back).</summary>
    [ObservableProperty] private bool _canGoForward;

    /// <summary>Training loss from the latest step.</summary>
    [ObservableProperty] private float _loss;

    /// <summary>Current learning rate.</summary>
    [ObservableProperty] private float _learningRate;

    /// <summary>Training throughput in tokens per second.</summary>
    [ObservableProperty] private float _tokensPerSec;

    /// <summary>Global gradient L2 norm before clipping.</summary>
    [ObservableProperty] private float _gradNorm;

    /// <summary>Total training steps completed this session.</summary>
    [ObservableProperty] private int _step;

    /// <summary>Number of pages processed this session.</summary>
    [ObservableProperty] private int _pagesProcessed;

    /// <summary>Human-readable status line for the status bar.</summary>
    [ObservableProperty] private string _statusText = "Idle";

    /// <summary>True when a session is active (orchestrator initialised).</summary>
    [ObservableProperty] private bool _isSessionActive;

    /// <summary>Elapsed session duration shown in the status bar.</summary>
    [ObservableProperty] private TimeSpan _sessionDuration;

    /// <summary>Parameter count (set once when session starts).</summary>
    [ObservableProperty] private string _parameterCount = "—";

    /// <summary>Model info summary string (set once when session starts).</summary>
    [ObservableProperty] private string _modelInfo = "No session active";

    /// <summary>GPU name or "CPU only".</summary>
    [ObservableProperty] private string _gpuStatus = "—";

    /// <summary>Loss values for the chart — last 500 points.</summary>
    public ObservableCollection<float> LossHistory { get; } = new();

    /// <summary>Settings used for the current/next session.</summary>
    public SettingsViewModel Settings { get; } = new();

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a <see cref="MainViewModel"/>.
    /// </summary>
    /// <param name="dispatch">
    /// Action that schedules work on the UI thread.
    /// Defaults to <c>Avalonia.Threading.Dispatcher.UIThread.Post</c> when null.
    /// Pass <c>a => a()</c> in tests to execute synchronously.
    /// </param>
    public MainViewModel(Action<Action>? dispatch = null)
    {
        _dispatch = dispatch ?? (a => Avalonia.Threading.Dispatcher.UIThread.Post(a));
        Settings.Load();
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    /// <summary>Returns true if <paramref name="url"/> has been visited this session.</summary>
    public bool IsVisited(string url) => _visitedUrls.Contains(url);

    /// <summary>Navigate to a URL (address bar Enter, Go button, or link click).</summary>
    [RelayCommand]
    private async Task NavigateAsync(string url)
    {
        if (string.IsNullOrWhiteSpace(url)) return;

        // Push current to back stack before navigating
        if (_currentUrl is not null)
        {
            _backStack.Push(_currentUrl);
            _fwdStack.Clear();
        }

        _currentUrl = url;
        _visitedUrls.Add(url);
        _dispatch(() =>
        {
            AddressBarUrl = url;
            CanGoBack    = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count  > 0;
        });

        if (_orchestrator is null) return; // no session yet — navigation history still tracked

        _dispatch(() => StatusText = $"Fetching {url}…");
        try
        {
            _cts ??= new CancellationTokenSource();
            await _orchestrator.ProcessUrlAsync(url, _cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _dispatch(() => StatusText = $"Error: {ex.Message}");
        }
    }

    /// <summary>Navigate to the previous page.</summary>
    [RelayCommand(CanExecute = nameof(CanGoBack))]
    private void GoBack()
    {
        if (_backStack.Count == 0) return;
        if (_currentUrl is not null) _fwdStack.Push(_currentUrl);
        _currentUrl = _backStack.Pop();
        _dispatch(() =>
        {
            AddressBarUrl = _currentUrl;
            CanGoBack    = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count  > 0;
        });
    }

    /// <summary>Navigate forward (after going back).</summary>
    [RelayCommand(CanExecute = nameof(CanGoForward))]
    private void GoForward()
    {
        if (_fwdStack.Count == 0) return;
        if (_currentUrl is not null) _backStack.Push(_currentUrl);
        _currentUrl = _fwdStack.Pop();
        _dispatch(() =>
        {
            AddressBarUrl = _currentUrl;
            CanGoBack    = _backStack.Count > 0;
            CanGoForward = _fwdStack.Count  > 0;
        });
    }

    /// <summary>Cancels the current in-flight fetch or training step.</summary>
    [RelayCommand]
    private void StopNavigation()
    {
        _cts?.Cancel();
        _cts = null;
        _dispatch(() => StatusText = "Stopped.");
    }

    // ── Session ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Initialises the orchestrator with current settings. Call after the settings
    /// dialog has been confirmed by the user.
    /// </summary>
    public void StartSession()
    {
        Settings.Save();
        var config = Settings.ToOrchestratorConfig();
        _orchestrator = new PicoOrchestrator(config);
        _orchestrator.OnProgress += OnOrchestratorProgress;
        _cts = new CancellationTokenSource();
        _dispatch(() =>
        {
            IsSessionActive = true;
            StatusText      = "Session started. Browse a page to begin training.";
            Step            = 0;
            PagesProcessed  = 0;
            LossHistory.Clear();
        });
    }

    /// <summary>
    /// Ends the session: saves a checkpoint and exports the model to GGUF.
    /// </summary>
    /// <param name="ggufOutputPath">Destination path for the .gguf file chosen by the user.</param>
    public void EndSession(string ggufOutputPath)
    {
        if (_orchestrator is null) return;
        _dispatch(() => StatusText = "Saving checkpoint and exporting GGUF…");
        _orchestrator.EndSession(ggufOutputPath);
        _orchestrator.OnProgress -= OnOrchestratorProgress;
        _orchestrator = null;
        _cts?.Cancel();
        _cts = null;
        _dispatch(() =>
        {
            IsSessionActive = false;
            StatusText      = $"Exported: {ggufOutputPath}";
        });
    }

    // ── Orchestrator event handler ────────────────────────────────────────────

    private void OnOrchestratorProgress(OrchestratorEvent ev)
    {
        switch (ev)
        {
            case TrainingStepEvent ts:
                _dispatch(() =>
                {
                    Loss         = ts.Loss;
                    LearningRate = ts.Lr;
                    TokensPerSec = ts.TokensPerSec;
                    GradNorm     = ts.GradNorm;
                    Step         = ts.Step;
                    StatusText   = $"Training… step {ts.Step}, loss {ts.Loss:F3}";
                    if (LossHistory.Count >= 500) LossHistory.RemoveAt(0);
                    LossHistory.Add(ts.Loss);
                });
                break;

            case PageFetchedEvent pf when pf.Success:
                _dispatch(() => StatusText = $"Parsing {pf.Url}…");
                break;

            case PageFetchedEvent pf when !pf.Success:
                _dispatch(() => StatusText = $"Error fetching {pf.Url}: {pf.Error}");
                break;

            case PageParsedEvent pp:
                _dispatch(() =>
                {
                    PagesProcessed++;
                    StatusText = $"Training on page {PagesProcessed}…";
                });
                break;

            case CheckpointSavedEvent cp:
                _dispatch(() => StatusText = $"Checkpoint saved: {cp.Path}");
                break;

            case GgufExportedEvent ge:
                _dispatch(() => StatusText = $"GGUF exported: {ge.Path} ({ge.FileSizeBytes / 1024:N0} KB)");
                break;

            case SessionErrorEvent se:
                _dispatch(() => StatusText = $"Skipped {se.Url}: {se.Error}");
                break;
        }
    }
}
```

- [ ] **Step 4: Run navigation tests — expect all pass**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --filter "FullyQualifiedName~MainViewModelNavigationTests"
```

Expected: All 8 tests pass.

- [ ] **Step 5: Run full suite — no regressions**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/PicoLLM.App/ViewModels/MainViewModel.cs tests/PicoLLM.Tests/UiShell/MainViewModelNavigationTests.cs
git commit -m "feat(cap11): add MainViewModel with navigation history, session management, orchestrator wiring"
```

---

## Task 6: LossChartControl

**Files:**
- Create: `src/PicoLLM.App/Views/LossChartControl.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Views/LossChartControl.cs`**

```csharp
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace PicoLLM.App.Views;

/// <summary>
/// Custom Avalonia control that renders a loss-over-steps line chart using
/// <see cref="DrawingContext"/>. Redraws automatically when <see cref="LossHistory"/> changes.
/// No charting library dependencies.
/// </summary>
public class LossChartControl : Control
{
    /// <summary>Avalonia styled property for the loss data collection.</summary>
    public static readonly StyledProperty<ObservableCollection<float>?> LossHistoryProperty =
        AvaloniaProperty.Register<LossChartControl, ObservableCollection<float>?>(
            nameof(LossHistory));

    private static readonly IBrush BackgroundBrush = new SolidColorBrush(Color.Parse("#1a1a1a"));
    private static readonly IBrush AxisBrush       = new SolidColorBrush(Color.Parse("#444444"));
    private static readonly IBrush LineBrush       = new SolidColorBrush(Color.Parse("#ff6b6b"));
    private static readonly IBrush LabelBrush      = new SolidColorBrush(Color.Parse("#666666"));
    private static readonly IPen   AxisPen         = new Pen(AxisBrush, 1);
    private static readonly IPen   LinePen         = new Pen(LineBrush, 1.5);

    /// <summary>The loss values to display. Bind to <c>MainViewModel.LossHistory</c>.</summary>
    public ObservableCollection<float>? LossHistory
    {
        get => GetValue(LossHistoryProperty);
        set => SetValue(LossHistoryProperty, value);
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property != LossHistoryProperty) return;

        if (change.OldValue is ObservableCollection<float> old)
            old.CollectionChanged -= OnCollectionChanged;

        if (change.NewValue is ObservableCollection<float> @new)
            @new.CollectionChanged += OnCollectionChanged;

        InvalidateVisual();
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        => InvalidateVisual();

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        var bounds = Bounds;
        if (bounds.Width < 10 || bounds.Height < 10) return;

        const double padLeft = 36, padRight = 8, padTop = 8, padBottom = 20;
        double chartW = bounds.Width  - padLeft - padRight;
        double chartH = bounds.Height - padTop  - padBottom;

        // Background
        context.FillRectangle(BackgroundBrush, bounds);

        var data = LossHistory;
        if (data is null || data.Count < 2)
        {
            DrawAxes(context, padLeft, padTop, chartW, chartH);
            return;
        }

        float minLoss = data.Min();
        float maxLoss = data.Max();
        float range   = maxLoss - minLoss;
        if (range < 1e-6f) range = 1f; // avoid division by zero

        // Y axis labels
        DrawYLabel(context, maxLoss.ToString("F2"), padLeft, padTop,         LabelBrush);
        DrawYLabel(context, minLoss.ToString("F2"), padLeft, padTop + chartH, LabelBrush);

        DrawAxes(context, padLeft, padTop, chartW, chartH);

        // Loss line
        var geometry = new StreamGeometry();
        using (var ctx = geometry.Open())
        {
            for (int i = 0; i < data.Count; i++)
            {
                double x = padLeft + i / (double)(data.Count - 1) * chartW;
                double y = padTop  + (1.0 - (data[i] - minLoss) / range) * chartH;
                if (i == 0) ctx.BeginFigure(new Point(x, y), false);
                else        ctx.LineTo(new Point(x, y));
            }
        }
        context.DrawGeometry(null, LinePen, geometry);
    }

    private static void DrawAxes(DrawingContext ctx,
        double padLeft, double padTop, double chartW, double chartH)
    {
        // Y axis
        ctx.DrawLine(AxisPen,
            new Point(padLeft, padTop),
            new Point(padLeft, padTop + chartH));
        // X axis
        ctx.DrawLine(AxisPen,
            new Point(padLeft, padTop + chartH),
            new Point(padLeft + chartW, padTop + chartH));
    }

    private static void DrawYLabel(DrawingContext ctx, string text,
        double padLeft, double y, IBrush brush)
    {
        var ft = new FormattedText(
            text,
            System.Globalization.CultureInfo.CurrentCulture,
            FlowDirection.LeftToRight,
            new Typeface("Inter, Segoe UI, sans-serif"),
            9, brush);
        ctx.DrawText(ft, new Point(padLeft - ft.Width - 2, y - ft.Height / 2));
    }
}
```

- [ ] **Step 2: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 3: Commit**

```bash
git add src/PicoLLM.App/Views/LossChartControl.cs
git commit -m "feat(cap11): add LossChartControl — custom DrawingContext line chart"
```

---

## Task 7: BrowserPane

**Files:**
- Create: `src/PicoLLM.App/Views/BrowserPane.axaml`
- Create: `src/PicoLLM.App/Views/BrowserPane.axaml.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Views/BrowserPane.axaml`**

```xml
<UserControl xmlns="https://github.com/avaloniaui"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             xmlns:vm="using:PicoLLM.App.ViewModels"
             xmlns:m="using:PicoLLM.App.Models"
             xmlns:v="using:PicoLLM.App.Views"
             x:Class="PicoLLM.App.Views.BrowserPane"
             Background="White">

  <UserControl.DataTemplates>

    <!-- HeadingElement template -->
    <DataTemplate DataType="m:HeadingElement">
      <TextBlock Text="{Binding Text}"
                 FontWeight="Bold"
                 Margin="0,6,0,4">
        <TextBlock.Styles>
          <Style Selector="TextBlock">
            <Setter Property="FontSize" Value="{Binding Level,
              Converter={StaticResource HeadingFontSizeConverter}}"/>
          </Style>
        </TextBlock.Styles>
      </TextBlock>
    </DataTemplate>

    <!-- ParagraphElement template -->
    <DataTemplate DataType="m:ParagraphElement">
      <v:ParagraphView Runs="{Binding Runs}" Margin="0,0,0,8"/>
    </DataTemplate>

    <!-- ImageAltElement template -->
    <DataTemplate DataType="m:ImageAltElement">
      <Border BorderBrush="#CCCCCC" BorderThickness="0,0,0,0"
              Background="#F7F7F7" Padding="8,4" Margin="0,4,0,8"
              CornerRadius="2">
        <Border.BorderBrush>
          <SolidColorBrush Color="#CCCCCC"/>
        </Border.BorderBrush>
        <TextBlock Text="{Binding AltText, StringFormat='[Image: {0}]'}"
                   FontStyle="Italic" FontSize="12"
                   Foreground="#555555" TextWrapping="Wrap"/>
      </Border>
    </DataTemplate>

  </UserControl.DataTemplates>

  <DockPanel>

    <!-- Back / Forward bar -->
    <Border DockPanel.Dock="Top"
            Background="#F0F0F0" BorderBrush="#CCCCCC"
            BorderThickness="0,0,0,1" Padding="6,4">
      <DockPanel>
        <Button Content="←" Width="32"
                Command="{Binding GoBackCommand}"
                IsEnabled="{Binding CanGoBack}"
                Margin="0,0,4,0"/>
        <Button Content="→" Width="32"
                Command="{Binding GoForwardCommand}"
                IsEnabled="{Binding CanGoForward}"
                Margin="0,0,8,0"/>
        <TextBlock Text="{Binding AddressBarUrl}"
                   FontSize="11" Foreground="#777777"
                   VerticalAlignment="Center"
                   TextTrimming="CharacterEllipsis"/>
      </DockPanel>
    </Border>

    <!-- Page content -->
    <ScrollViewer HorizontalScrollBarVisibility="Disabled"
                  VerticalScrollBarVisibility="Auto">
      <ItemsControl ItemsSource="{Binding CurrentPage.Elements}"
                    Margin="16,12">
        <ItemsControl.ItemsPanel>
          <ItemsPanelTemplate>
            <StackPanel Orientation="Vertical"/>
          </ItemsPanelTemplate>
        </ItemsControl.ItemsPanel>
      </ItemsControl>
    </ScrollViewer>

  </DockPanel>
</UserControl>
```

- [ ] **Step 2: Create `src/PicoLLM.App/Views/BrowserPane.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace PicoLLM.App.Views;

/// <summary>Lynx-style browser pane. Renders a <see cref="FormattedPageContent"/> using data templates.</summary>
public partial class BrowserPane : UserControl
{
    /// <summary>Initializes the browser pane.</summary>
    public BrowserPane() => AvaloniaXamlLoader.Load(this);
}
```

- [ ] **Step 3: Create inline `ParagraphView` control for link-aware text runs**

Add `src/PicoLLM.App/Views/ParagraphView.cs`:

```csharp
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using PicoLLM.App.Models;
using PicoLLM.App.ViewModels;

namespace PicoLLM.App.Views;

/// <summary>
/// Renders a <see cref="ParagraphElement"/>'s text runs as a wrapping sequence of
/// styled <see cref="TextBlock"/> elements. Link runs are blue/purple and respond to clicks.
/// </summary>
public class ParagraphView : WrapPanel
{
    /// <summary>Styled property for the list of text runs to render.</summary>
    public static readonly StyledProperty<IReadOnlyList<TextRun>?> RunsProperty =
        AvaloniaProperty.Register<ParagraphView, IReadOnlyList<TextRun>?>(nameof(Runs));

    private static readonly IBrush UnvisitedBrush = new SolidColorBrush(Color.Parse("#0000EE"));
    private static readonly IBrush VisitedBrush   = new SolidColorBrush(Color.Parse("#551A8B"));
    private static readonly IBrush PlainBrush      = new SolidColorBrush(Color.Parse("#111111"));

    /// <summary>The text runs to render.</summary>
    public IReadOnlyList<TextRun>? Runs
    {
        get => GetValue(RunsProperty);
        set => SetValue(RunsProperty, value);
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == RunsProperty) Rebuild();
    }

    private void Rebuild()
    {
        Children.Clear();
        if (Runs is null) return;

        foreach (var run in Runs)
        {
            var tb = new TextBlock
            {
                Text       = run.Text,
                FontSize   = 13,
                Foreground = run.IsLink
                    ? (run.IsVisited ? VisitedBrush : UnvisitedBrush)
                    : PlainBrush,
                TextDecorations = run.IsLink ? TextDecorations.Underline : null,
                Cursor = run.IsLink ? new Cursor(StandardCursorType.Hand) : Cursor.Default,
                VerticalAlignment = Avalonia.Layout.VerticalAlignment.Top,
            };

            if (run.IsLink && run.Href is not null)
            {
                var href = run.Href; // capture for closure
                tb.PointerPressed += (_, e) =>
                {
                    if (e.GetCurrentPoint(tb).Properties.IsLeftButtonPressed)
                        OnLinkClicked(href);
                };
            }

            Children.Add(tb);
        }
    }

    private void OnLinkClicked(string href)
    {
        // Walk up the visual tree to find MainViewModel
        var vm = this.FindAncestorOfType<Control>()?.DataContext as MainViewModel
              ?? DataContext as MainViewModel;
        vm?.NavigateCommand.Execute(href);
    }
}
```

- [ ] **Step 4: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 5: Commit**

```bash
git add src/PicoLLM.App/Views/BrowserPane.axaml src/PicoLLM.App/Views/BrowserPane.axaml.cs src/PicoLLM.App/Views/ParagraphView.cs
git commit -m "feat(cap11): add BrowserPane with Lynx-style rendering and clickable links"
```

---

## Task 8: DashboardPanel and ModelInfoPanel

**Files:**
- Create: `src/PicoLLM.App/Views/DashboardPanel.axaml`
- Create: `src/PicoLLM.App/Views/DashboardPanel.axaml.cs`
- Create: `src/PicoLLM.App/Views/ModelInfoPanel.axaml`
- Create: `src/PicoLLM.App/Views/ModelInfoPanel.axaml.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Views/ModelInfoPanel.axaml`**

```xml
<UserControl xmlns="https://github.com/avaloniaui"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             x:Class="PicoLLM.App.Views.ModelInfoPanel">
  <Border Background="#1E1E1E" Padding="10" CornerRadius="4">
    <StackPanel>
      <TextBlock Text="Model Info" FontSize="10" Foreground="#888888"
                 FontWeight="SemiBold" TextTransform="Uppercase"
                 LetterSpacing="1" Margin="0,0,0,8"/>
      <TextBlock Text="{Binding ModelInfo}"
                 FontSize="11" Foreground="#CCCCCC" TextWrapping="Wrap"
                 LineHeight="18"/>
      <TextBlock Text="{Binding GpuStatus}"
                 FontSize="11" Foreground="#6BFF9E"
                 Margin="0,4,0,0"/>
    </StackPanel>
  </Border>
</UserControl>
```

- [ ] **Step 2: Create `src/PicoLLM.App/Views/ModelInfoPanel.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace PicoLLM.App.Views;

/// <summary>Displays static model configuration and GPU status.</summary>
public partial class ModelInfoPanel : UserControl
{
    /// <summary>Initializes the model info panel.</summary>
    public ModelInfoPanel() => AvaloniaXamlLoader.Load(this);
}
```

- [ ] **Step 3: Create `src/PicoLLM.App/Views/DashboardPanel.axaml`**

```xml
<UserControl xmlns="https://github.com/avaloniaui"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             xmlns:v="using:PicoLLM.App.Views"
             x:Class="PicoLLM.App.Views.DashboardPanel"
             Background="#121212">
  <ScrollViewer VerticalScrollBarVisibility="Auto">
    <StackPanel Margin="10" Spacing="10">

      <!-- Metrics grid -->
      <Border Background="#1E1E1E" Padding="10" CornerRadius="4">
        <StackPanel>
          <TextBlock Text="Training Metrics" FontSize="10" Foreground="#888888"
                     FontWeight="SemiBold" TextTransform="Uppercase"
                     LetterSpacing="1" Margin="0,0,0,10"/>
          <Grid ColumnDefinitions="*,*" RowDefinitions="Auto,Auto,Auto" RowSpacing="8">

            <!-- Loss -->
            <StackPanel Grid.Row="0" Grid.Column="0">
              <TextBlock Text="Loss" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding Loss, StringFormat={}{0:F3}}"
                         FontSize="18" FontWeight="Bold" Foreground="#FF6B6B"/>
            </StackPanel>

            <!-- LR -->
            <StackPanel Grid.Row="0" Grid.Column="1">
              <TextBlock Text="LR" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding LearningRate, StringFormat={}{0:F6}}"
                         FontSize="18" FontWeight="Bold" Foreground="#6BA3FF"/>
            </StackPanel>

            <!-- Tokens/s -->
            <StackPanel Grid.Row="1" Grid.Column="0">
              <TextBlock Text="Tokens/s" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding TokensPerSec, StringFormat={}{0:N0}}"
                         FontSize="18" FontWeight="Bold" Foreground="#6BFF9E"/>
            </StackPanel>

            <!-- Grad Norm -->
            <StackPanel Grid.Row="1" Grid.Column="1">
              <TextBlock Text="‖∇‖" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding GradNorm, StringFormat={}{0:F3}}"
                         FontSize="18" FontWeight="Bold" Foreground="#FFD06B"/>
            </StackPanel>

            <!-- Step -->
            <StackPanel Grid.Row="2" Grid.Column="0">
              <TextBlock Text="Step" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding Step}"
                         FontSize="14" FontWeight="Bold" Foreground="#E8E8E8"/>
            </StackPanel>

            <!-- Pages -->
            <StackPanel Grid.Row="2" Grid.Column="1">
              <TextBlock Text="Pages" FontSize="10" Foreground="#888888"/>
              <TextBlock Text="{Binding PagesProcessed}"
                         FontSize="14" FontWeight="Bold" Foreground="#E8E8E8"/>
            </StackPanel>

          </Grid>
        </StackPanel>
      </Border>

      <!-- Loss chart -->
      <Border Background="#1E1E1E" Padding="10" CornerRadius="4">
        <StackPanel>
          <TextBlock Text="Loss Chart" FontSize="10" Foreground="#888888"
                     FontWeight="SemiBold" TextTransform="Uppercase"
                     LetterSpacing="1" Margin="0,0,0,6"/>
          <v:LossChartControl Height="100"
                              LossHistory="{Binding LossHistory}"/>
        </StackPanel>
      </Border>

      <!-- Model info -->
      <v:ModelInfoPanel/>

    </StackPanel>
  </ScrollViewer>
</UserControl>
```

- [ ] **Step 4: Create `src/PicoLLM.App/Views/DashboardPanel.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace PicoLLM.App.Views;

/// <summary>Training metrics dashboard: live metrics grid, loss chart, and model info.</summary>
public partial class DashboardPanel : UserControl
{
    /// <summary>Initializes the dashboard panel.</summary>
    public DashboardPanel() => AvaloniaXamlLoader.Load(this);
}
```

- [ ] **Step 5: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 6: Commit**

```bash
git add src/PicoLLM.App/Views/DashboardPanel.axaml src/PicoLLM.App/Views/DashboardPanel.axaml.cs src/PicoLLM.App/Views/ModelInfoPanel.axaml src/PicoLLM.App/Views/ModelInfoPanel.axaml.cs
git commit -m "feat(cap11): add DashboardPanel (metrics grid + loss chart) and ModelInfoPanel"
```

---

## Task 9: SettingsPanel

**Files:**
- Create: `src/PicoLLM.App/Views/SettingsPanel.axaml`
- Create: `src/PicoLLM.App/Views/SettingsPanel.axaml.cs`

- [ ] **Step 1: Create `src/PicoLLM.App/Views/SettingsPanel.axaml`**

```xml
<Window xmlns="https://github.com/avaloniaui"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:vm="using:PicoLLM.App.ViewModels"
        x:Class="PicoLLM.App.Views.SettingsPanel"
        Title="Session Settings"
        Width="480" Height="540"
        CanResize="False"
        WindowStartupLocation="CenterOwner">

  <ScrollViewer>
    <StackPanel Margin="20" Spacing="16">

      <!-- Data directory -->
      <StackPanel Spacing="6">
        <TextBlock Text="DATA DIRECTORY" FontSize="10" Foreground="#888888"
                   FontWeight="SemiBold" LetterSpacing="1"/>
        <DockPanel>
          <Button Content="Browse…" DockPanel.Dock="Right"
                  Click="OnBrowseClick" Margin="8,0,0,0"/>
          <TextBox Text="{Binding DataDirectory}" FontSize="12"/>
        </DockPanel>
        <TextBlock Text="Tokenizer, checkpoints, and GGUF files are saved here."
                   FontSize="10" Foreground="#888888"/>
      </StackPanel>

      <!-- Model hyperparameters -->
      <StackPanel Spacing="6">
        <TextBlock Text="MODEL HYPERPARAMETERS" FontSize="10" Foreground="#888888"
                   FontWeight="SemiBold" LetterSpacing="1"/>
        <Grid ColumnDefinitions="*,20,*" RowDefinitions="Auto,Auto,Auto" RowSpacing="8">
          <StackPanel Grid.Row="0" Grid.Column="0" Spacing="3">
            <TextBlock Text="Vocab Size" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding VocabSize}" Minimum="260" Maximum="32000" Increment="256"/>
          </StackPanel>
          <StackPanel Grid.Row="0" Grid.Column="2" Spacing="3">
            <TextBlock Text="Embed Dim" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding EmbedDim}" Minimum="32" Maximum="1024" Increment="32"/>
          </StackPanel>
          <StackPanel Grid.Row="1" Grid.Column="0" Spacing="3">
            <TextBlock Text="Layers" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding NumLayers}" Minimum="1" Maximum="24" Increment="1"/>
          </StackPanel>
          <StackPanel Grid.Row="1" Grid.Column="2" Spacing="3">
            <TextBlock Text="Heads" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding NumHeads}" Minimum="1" Maximum="16" Increment="1"/>
          </StackPanel>
          <StackPanel Grid.Row="2" Grid.Column="0" Spacing="3">
            <TextBlock Text="Context Length" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding MaxSeqLen}" Minimum="64" Maximum="4096" Increment="64"/>
          </StackPanel>
          <StackPanel Grid.Row="2" Grid.Column="2" Spacing="3">
            <TextBlock Text="Steps / Page" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding StepsPerPage}" Minimum="1" Maximum="10000" Increment="10"/>
          </StackPanel>
        </Grid>
      </StackPanel>

      <!-- Training hyperparameters -->
      <StackPanel Spacing="6">
        <TextBlock Text="TRAINING HYPERPARAMETERS" FontSize="10" Foreground="#888888"
                   FontWeight="SemiBold" LetterSpacing="1"/>
        <Grid ColumnDefinitions="*,20,*" RowDefinitions="Auto,Auto" RowSpacing="8">
          <StackPanel Grid.Row="0" Grid.Column="0" Spacing="3">
            <TextBlock Text="Learning Rate" FontSize="11" Foreground="#777777"/>
            <TextBox Text="{Binding LearningRate, StringFormat={}{0:F6}}" FontSize="12"/>
          </StackPanel>
          <StackPanel Grid.Row="0" Grid.Column="2" Spacing="3">
            <TextBlock Text="Batch Size" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding BatchSize}" Minimum="1" Maximum="64" Increment="1"/>
          </StackPanel>
          <StackPanel Grid.Row="1" Grid.Column="0" Spacing="3">
            <TextBlock Text="Sequence Length" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding SeqLen}" Minimum="16" Maximum="2048" Increment="16"/>
          </StackPanel>
          <StackPanel Grid.Row="1" Grid.Column="2" Spacing="3">
            <TextBlock Text="Warmup Steps" FontSize="11" Foreground="#777777"/>
            <NumericUpDown Value="{Binding WarmupSteps}" Minimum="0" Maximum="1000" Increment="10"/>
          </StackPanel>
        </Grid>
      </StackPanel>

      <!-- Action buttons -->
      <StackPanel Orientation="Horizontal" HorizontalAlignment="Right" Spacing="10">
        <Button Content="Cancel" Click="OnCancelClick" MinWidth="80"/>
        <Button Content="Save &amp; Start Session" Click="OnSaveClick"
                Classes="accent" MinWidth="160"/>
      </StackPanel>

    </StackPanel>
  </ScrollViewer>
</Window>
```

- [ ] **Step 2: Create `src/PicoLLM.App/Views/SettingsPanel.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using PicoLLM.App.ViewModels;

namespace PicoLLM.App.Views;

/// <summary>Modal settings dialog. Bound to <see cref="SettingsViewModel"/>.</summary>
public partial class SettingsPanel : Window
{
    /// <summary>Initializes the settings panel.</summary>
    public SettingsPanel() => AvaloniaXamlLoader.Load(this);

    private async void OnBrowseClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select Data Directory",
            AllowMultiple = false
        });

        if (folders.Count > 0 && DataContext is SettingsViewModel vm)
            vm.DataDirectory = folders[0].Path.LocalPath;
    }

    private void OnCancelClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        => Close(false);

    private void OnSaveClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        => Close(true);
}
```

- [ ] **Step 3: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.App/Views/SettingsPanel.axaml src/PicoLLM.App/Views/SettingsPanel.axaml.cs
git commit -m "feat(cap11): add SettingsPanel modal dialog with folder picker"
```

---

## Task 10: MainWindow — Full Layout

**Files:**
- Modify: `src/PicoLLM.App/MainWindow.axaml`
- Modify: `src/PicoLLM.App/MainWindow.axaml.cs`
- Modify: `src/PicoLLM.App/App.axaml.cs`

- [ ] **Step 1: Replace `src/PicoLLM.App/MainWindow.axaml`**

```xml
<Window xmlns="https://github.com/avaloniaui"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:vm="using:PicoLLM.App.ViewModels"
        xmlns:v="using:PicoLLM.App.Views"
        x:Class="PicoLLM.App.MainWindow"
        x:DataType="vm:MainViewModel"
        Title="PicoLLM"
        Width="1100" Height="700"
        MinWidth="800" MinHeight="500">

  <DockPanel>

    <!-- Address bar -->
    <Border DockPanel.Dock="Top"
            Background="#F5F5F5" BorderBrush="#DDDDDD"
            BorderThickness="0,0,0,1" Padding="8,6">
      <DockPanel>
        <Button Content="Stop" DockPanel.Dock="Right"
                Command="{Binding StopNavigationCommand}"
                Margin="4,0,0,0" MinWidth="50"/>
        <Button Content="Go" DockPanel.Dock="Right"
                Click="OnGoClick"
                IsDefault="True"
                Margin="4,0,0,0" MinWidth="40"/>
        <TextBox x:Name="AddressBox"
                 Text="{Binding AddressBarUrl}"
                 FontSize="13"
                 KeyDown="OnAddressKeyDown"
                 Watermark="Enter URL…"
                 VerticalContentAlignment="Center"/>
      </DockPanel>
    </Border>

    <!-- Status bar + session buttons -->
    <Border DockPanel.Dock="Bottom"
            Background="#1A1A1A" Padding="10,6"
            BorderBrush="#333333" BorderThickness="0,1,0,0">
      <DockPanel>
        <StackPanel DockPanel.Dock="Right" Orientation="Horizontal" Spacing="8">
          <Button Content="Start Session"
                  Click="OnStartSessionClick"
                  IsEnabled="{Binding !IsSessionActive}"
                  Classes="accent"/>
          <Button Content="End Session &amp; Export GGUF"
                  Click="OnEndSessionClick"
                  IsEnabled="{Binding IsSessionActive}"/>
        </StackPanel>
        <TextBlock Text="{Binding StatusText}"
                   Foreground="#AAAAAA" FontSize="11"
                   VerticalAlignment="Center"/>
      </DockPanel>
    </Border>

    <!-- Main split: browser (left) | dashboard (right) -->
    <Grid>
      <Grid.ColumnDefinitions>
        <ColumnDefinition Width="3*"/>
        <ColumnDefinition Width="4"/>
        <ColumnDefinition Width="2*"/>
      </Grid.ColumnDefinitions>

      <!-- Browser pane -->
      <v:BrowserPane Grid.Column="0"/>

      <!-- Splitter -->
      <GridSplitter Grid.Column="1"
                    ResizeDirection="Columns"
                    Background="#333333"/>

      <!-- Dashboard -->
      <v:DashboardPanel Grid.Column="2"/>

    </Grid>

  </DockPanel>
</Window>
```

- [ ] **Step 2: Replace `src/PicoLLM.App/MainWindow.axaml.cs`**

```csharp
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using PicoLLM.App.ViewModels;
using PicoLLM.App.Views;

namespace PicoLLM.App;

/// <summary>Root application window. Thin code-behind — delegates all logic to <see cref="MainViewModel"/>.</summary>
public partial class MainWindow : Window
{
    private MainViewModel Vm => (MainViewModel)DataContext!;

    /// <summary>Initializes the main window.</summary>
    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this);
        DataContext = new MainViewModel();
    }

    private void OnAddressKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && sender is TextBox tb)
            Vm.NavigateCommand.Execute(tb.Text?.Trim());
    }

    private void OnGoClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var tb = this.FindControl<TextBox>("AddressBox");
        Vm.NavigateCommand.Execute(tb?.Text?.Trim());
    }

    private async void OnStartSessionClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var dialog = new SettingsPanel { DataContext = Vm.Settings };
        var confirmed = await dialog.ShowDialog<bool?>(this);
        if (confirmed is true)
            Vm.StartSession();
    }

    private async void OnEndSessionClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title           = "Export GGUF",
            SuggestedFileName = "model.gguf",
            FileTypeChoices = [new FilePickerFileType("GGUF Model") { Patterns = ["*.gguf"] }]
        });
        if (file is not null)
            Vm.EndSession(file.Path.LocalPath);
    }
}
```

- [ ] **Step 3: Build to verify**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: `Build succeeded.`

- [ ] **Step 4: Commit**

```bash
git add src/PicoLLM.App/MainWindow.axaml src/PicoLLM.App/MainWindow.axaml.cs
git commit -m "feat(cap11): wire MainWindow — address bar, split layout, session buttons"
```

---

## Task 11: Full Build, All Tests, Final Commit

**Files:**
- No new files.

- [ ] **Step 1: Full solution build**

```
cd d:/source/ghosh9691/PicoLLM && dotnet build
```

Expected: All projects build with 0 errors.

- [ ] **Step 2: Run all tests**

```
cd d:/source/ghosh9691/PicoLLM && dotnet test tests/PicoLLM.Tests/PicoLLM.Tests.csproj --logger "console;verbosity=minimal"
```

Expected: All tests pass (327 existing + new UiShell tests).

- [ ] **Step 3: Smoke-test the app launches**

```
cd d:/source/ghosh9691/PicoLLM && dotnet run --project src/PicoLLM.App/PicoLLM.App.csproj
```

Expected: The PicoLLM window opens showing the address bar, split browser/dashboard panes, and status bar. Close the window.

- [ ] **Step 4: Final capability commit**

```bash
cd d:/source/ghosh9691/PicoLLM
git add -A
git commit -m "feat: capability 11 — UI shell (Avalonia, BrowserPane, DashboardPanel, SettingsPanel, MainWindow)"
```

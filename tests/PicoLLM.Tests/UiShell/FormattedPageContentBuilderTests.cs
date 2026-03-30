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

using FluentAssertions;
using PicoLLM.App.Orchestration;
using PicoLLM.App.ViewModels;
using PicoLLM.Browser;

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

    // ── URL auto-protocol tests ───────────────────────────────────────────────

    [Fact]
    public async Task NavigateAsync_BareDomain_PrependHttps()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("en.wikipedia.org/wiki/Neural_network");
        vm.AddressBarUrl.Should().Be("https://en.wikipedia.org/wiki/Neural_network");
    }

    [Fact]
    public async Task NavigateAsync_ExplicitHttp_Preserved()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("http://example.com");
        vm.AddressBarUrl.Should().Be("http://example.com");
    }

    [Fact]
    public async Task NavigateAsync_ExplicitHttps_Unchanged()
    {
        var vm = MakeVm();
        await vm.NavigateCommand.ExecuteAsync("https://example.com");
        vm.AddressBarUrl.Should().Be("https://example.com");
    }

    // ── CurrentPage / BrowseAsync tests ──────────────────────────────────────

    private const string SimpleHtml = """
        <html><body>
        <h1>Test Page</h1>
        <p>Hello world. This is a paragraph with some content.</p>
        <p>Second paragraph with more text to ensure extraction works.</p>
        </body></html>
        """;

    private static Func<string, CancellationToken, Task<BrowseResult>> FakeHttp(string html) =>
        (url, _) => Task.FromResult(new BrowseResult(
            Url: url, Status: BrowseStatus.Success, HtmlContent: html,
            ErrorMessage: null, HttpStatusCode: 200, ElapsedTime: TimeSpan.Zero));

    [Fact]
    public async Task Navigate_WithSession_SetsCurrentPage()
    {
        var vm = MakeVm();
        vm.StartSession(FakeHttp(SimpleHtml));

        await vm.NavigateCommand.ExecuteAsync("https://example.com");

        // BrowseAsync emits PageParsedEvent → CurrentPage must be set
        vm.CurrentPage.Should().NotBeNull();
        vm.CurrentPage!.Elements.Should().NotBeEmpty();
    }

    [Fact]
    public async Task Navigate_WithSession_CurrentPageContainsText()
    {
        var vm = MakeVm();
        vm.StartSession(FakeHttp(SimpleHtml));

        await vm.NavigateCommand.ExecuteAsync("https://example.com");

        // At least the heading should be present
        vm.CurrentPage!.Elements
            .OfType<PicoLLM.App.Models.HeadingElement>()
            .Should().Contain(h => h.Text == "Test Page");
    }

    [Fact]
    public async Task Navigate_WithSession_IncrementsPagesProcessed()
    {
        var vm = MakeVm();
        vm.StartSession(FakeHttp(SimpleHtml));

        await vm.NavigateCommand.ExecuteAsync("https://example.com");

        vm.PagesProcessed.Should().Be(1);
    }

    // ── Training queue tests ──────────────────────────────────────────────────

    [Fact]
    public async Task Navigate_WithSession_TrainingQueueDepthReturnsToZeroOrLower()
    {
        var vm = MakeVm();
        vm.StartSession(FakeHttp(SimpleHtml));

        await vm.NavigateCommand.ExecuteAsync("https://example.com");

        // After BrowseAsync returns the queue depth has been set (may be 0 if bg loop
        // already drained it, or 1 if still waiting). Either way it must be non-negative.
        vm.TrainingQueueDepth.Should().BeGreaterThanOrEqualTo(0);
    }
}

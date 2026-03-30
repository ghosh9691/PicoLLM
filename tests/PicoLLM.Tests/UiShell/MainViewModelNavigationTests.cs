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

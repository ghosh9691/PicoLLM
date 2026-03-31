using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.VisualTree;
using PicoLLM.App.Models;
using PicoLLM.App.ViewModels;
using PicoLLM.Browser;

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
    private static readonly IBrush PlainBrush     = new SolidColorBrush(Color.Parse("#111111"));

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
                Text            = run.Text,
                FontSize        = 13,
                Foreground      = run.IsLink
                    ? (run.IsVisited ? VisitedBrush : UnvisitedBrush)
                    : PlainBrush,
                TextDecorations = run.IsLink ? TextDecorations.Underline : null,
                Cursor          = run.IsLink ? new Cursor(StandardCursorType.Hand) : Cursor.Default,
                VerticalAlignment = Avalonia.Layout.VerticalAlignment.Top,
            };

            if (run.IsLink && run.Href is not null)
            {
                var href = run.Href;
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
        // Walk the full visual tree upward until we find a control whose DataContext
        // is MainViewModel. Item containers have ParagraphElement as their DataContext,
        // so we cannot stop at the first ancestor — we must keep going.
        Control? current = this;
        MainViewModel? vm = null;
        while (current is not null)
        {
            if (current.DataContext is MainViewModel found)
            {
                vm = found;
                break;
            }
            current = current.GetVisualParent() as Control;
        }

        if (vm is null) return;

        // Resolve relative hrefs against the current page URL.
        var absoluteUrl = UrlResolver.Resolve(vm.AddressBarUrl, href) ?? href;
        vm.NavigateCommand.Execute(absoluteUrl);
    }
}

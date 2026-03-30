using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using PicoLLM.App.Models;

namespace PicoLLM.App.Views;

/// <summary>
/// Renders a <see cref="HeadingElement"/> with font size and styling appropriate to the heading level.
/// H1=20px bold, H2=16px bold with bottom rule, H3=14px bold.
/// </summary>
public class HeadingView : ContentControl
{
    private static readonly IBrush HeadingBrush = new SolidColorBrush(Color.Parse("#111111"));
    private static readonly IBrush H3Brush      = new SolidColorBrush(Color.Parse("#222222"));

    /// <inheritdoc/>
    protected override void OnDataContextChanged(EventArgs e)
    {
        base.OnDataContextChanged(e);
        Rebuild();
    }

    private void Rebuild()
    {
        if (DataContext is not HeadingElement h) { Content = null; return; }

        var tb = new TextBlock
        {
            Text         = h.Text,
            FontWeight   = FontWeight.Bold,
            Foreground   = h.Level == 3 ? H3Brush : HeadingBrush,
            TextWrapping = TextWrapping.Wrap,
        };

        switch (h.Level)
        {
            case 1:
                tb.FontSize = 20;
                Margin  = new Thickness(0, 6, 0, 4);
                Content = tb;
                break;
            case 2:
                tb.FontSize = 16;
                Margin  = new Thickness(0, 14, 0, 6);
                Content = new Border
                {
                    BorderBrush     = new SolidColorBrush(Color.Parse("#DDDDDD")),
                    BorderThickness = new Thickness(0, 0, 0, 1),
                    Padding         = new Thickness(0, 0, 0, 4),
                    Child           = tb
                };
                break;
            default: // H3
                tb.FontSize = 14;
                Margin  = new Thickness(0, 10, 0, 4);
                Content = tb;
                break;
        }
    }
}

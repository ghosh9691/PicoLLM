using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace PicoLLM.App;

/// <summary>Root application window.</summary>
public partial class MainWindow : Window
{
    /// <summary>Initializes the main window.</summary>
    public MainWindow() => AvaloniaXamlLoader.Load(this);
}

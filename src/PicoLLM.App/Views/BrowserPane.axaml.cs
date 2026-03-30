using Avalonia.Controls;

namespace PicoLLM.App.Views;

/// <summary>Lynx-style browser pane. Renders a <see cref="PicoLLM.App.Models.FormattedPageContent"/> using data templates.</summary>
public partial class BrowserPane : UserControl
{
    /// <summary>Initializes the browser pane.</summary>
    public BrowserPane() => InitializeComponent();
}

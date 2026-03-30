using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using PicoLLM.App.ViewModels;
using PicoLLM.App.Views;

namespace PicoLLM.App;

/// <summary>Root application window. Thin code-behind — delegates all logic to <see cref="MainViewModel"/>.</summary>
public partial class MainWindow : Window
{
    private MainViewModel Vm => (MainViewModel)DataContext!;

    /// <summary>Initializes the main window and sets up the view model.</summary>
    public MainWindow()
    {
        InitializeComponent();
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
            Title             = "Export GGUF",
            SuggestedFileName = "model.gguf",
            FileTypeChoices   = [new FilePickerFileType("GGUF Model") { Patterns = ["*.gguf"] }]
        });
        if (file is not null)
            Vm.EndSession(file.Path.LocalPath);
    }
}

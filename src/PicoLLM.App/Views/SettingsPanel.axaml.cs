using Avalonia.Controls;
using Avalonia.Platform.Storage;
using PicoLLM.App.ViewModels;

namespace PicoLLM.App.Views;

/// <summary>Modal settings dialog. Bound to <see cref="SettingsViewModel"/>.</summary>
public partial class SettingsPanel : Window
{
    /// <summary>Initializes the settings panel.</summary>
    public SettingsPanel() => InitializeComponent();

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

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
        DrawYLabel(context, maxLoss.ToString("F2"), padLeft, padTop,          LabelBrush);
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

using FluentAssertions;
using PicoLLM.Gpu;

namespace PicoLLM.Tests.Gpu;

public class GpuDetectorTests
{
    [Fact]
    public void DetectNvidia_NeverThrows()
    {
        var act = () => GpuDetector.DetectNvidia();
        act.Should().NotThrow();
    }

    [Fact]
    public void DetectNvidia_ReturnsNullOrValidInfo()
    {
        var info = GpuDetector.DetectNvidia();
        if (info is not null)
        {
            info.Name.Should().NotBeNullOrWhiteSpace();
            info.VramBytes.Should().BeGreaterThan(0);
            info.MultiprocessorCount.Should().BeGreaterThan(0);
        }
    }
}

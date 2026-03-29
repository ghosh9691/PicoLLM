using FluentAssertions;
using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;
using PicoLLM.Gpu;

namespace PicoLLM.Tests.Gpu;

/// <summary>
/// These tests require a real NVIDIA GPU with CUDA support.
/// They are skipped automatically when no GPU is detected.
/// </summary>
public class CudaComputeProviderTests
{
    private static bool GpuAvailable() => GpuDetector.DetectNvidia() is not null;

    [Fact]
    public void CudaComputeProvider_WhenNoGpu_ThrowsOnConstruction()
    {
        if (GpuAvailable()) return; // skip — GPU present

        var act = () => new CudaComputeProvider();
        act.Should().Throw<InvalidOperationException>()
           .WithMessage("*No CUDA*");
    }

    [Fact]
    public void CudaComputeProvider_MatMul_MatchesCpu_WithinTolerance()
    {
        if (!GpuAvailable()) return;

        using var gpu = new CudaComputeProvider();
        using var cpu = new CpuComputeProvider();

        var a = TensorFactory.Random([512, 256], seed: 1);
        var b = TensorFactory.Random([256, 512], seed: 2);

        var gpuResult = gpu.MatMul(a, b);
        var cpuResult = cpu.MatMul(a, b);

        gpuResult.Shape[0].Should().Be(512);
        gpuResult.Shape[1].Should().Be(512);

        for (int i = 0; i < gpuResult.Data.Length; i++)
            gpuResult.Data[i].Should().BeApproximately(cpuResult.Data[i], 1e-3f);
    }

    [Fact]
    public void CudaComputeProvider_SmallMatrix_UsesCpuFallback()
    {
        if (!GpuAvailable()) return;

        using var gpu = new CudaComputeProvider();
        using var cpu = new CpuComputeProvider();

        var a = TensorFactory.FromData([4, 4], [
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f
        ]); // identity
        var b = TensorFactory.Random([4, 4], seed: 7);

        var result = gpu.MatMul(a, b);
        var expected = cpu.MatMul(a, b);

        for (int i = 0; i < result.Data.Length; i++)
            result.Data[i].Should().BeApproximately(expected.Data[i], 1e-5f);
    }

    [Fact]
    public void CudaComputeProvider_Properties_AreCorrect()
    {
        if (!GpuAvailable()) return;

        using var gpu = new CudaComputeProvider();
        gpu.IsGpu.Should().BeTrue();
        gpu.Name.Should().StartWith("CUDA:");
    }
}

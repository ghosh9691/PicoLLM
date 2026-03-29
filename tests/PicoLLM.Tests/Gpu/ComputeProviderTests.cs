using FluentAssertions;
using PicoLLM.Core.Compute;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Tests.Gpu;

public class ComputeProviderTests
{
    [Fact]
    public void TensorFactory_FromData_CreatesCorrectTensor()
    {
        var data = new float[] { 1f, 2f, 3f, 4f };
        var t = TensorFactory.FromData([2, 2], data);
        t.Shape[0].Should().Be(2);
        t.Shape[1].Should().Be(2);
        t.Data[0].Should().Be(1f);
        t.Data[3].Should().Be(4f);
    }

    [Fact]
    public void CpuComputeProvider_Name_IsCpu()
    {
        using var provider = new CpuComputeProvider();
        provider.Name.Should().Be("CPU");
        provider.IsGpu.Should().BeFalse();
    }

    [Fact]
    public void CpuComputeProvider_MatMul_MatchesTensorMath()
    {
        // [2,3] @ [3,2] = [2,2]
        var a = TensorFactory.FromData([2, 3], [1f, 2f, 3f, 4f, 5f, 6f]);
        var b = TensorFactory.FromData([3, 2], [7f, 8f, 9f, 10f, 11f, 12f]);

        using var cpu = new CpuComputeProvider();
        var result = cpu.MatMul(a, b);

        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        result.Shape[0].Should().Be(2);
        result.Shape[1].Should().Be(2);
        result.Data[0].Should().BeApproximately(58f,  1e-4f);
        result.Data[1].Should().BeApproximately(64f,  1e-4f);
        result.Data[2].Should().BeApproximately(139f, 1e-4f);
        result.Data[3].Should().BeApproximately(154f, 1e-4f);
    }

    [Fact]
    public void CpuComputeProvider_BatchedMatMul_MatchesTensorMath()
    {
        // [1,1,2,2] @ [1,1,2,2] — batch=1, head=1, S=2, D=2
        var a = TensorFactory.FromData([1, 1, 2, 2], [1f, 0f, 0f, 1f]);  // identity
        var b = TensorFactory.FromData([1, 1, 2, 2], [1f, 2f, 3f, 4f]);

        using var cpu = new CpuComputeProvider();
        var result = cpu.BatchedMatMul(a, b);

        // identity @ B = B
        result.Data[0].Should().BeApproximately(1f, 1e-4f);
        result.Data[1].Should().BeApproximately(2f, 1e-4f);
        result.Data[2].Should().BeApproximately(3f, 1e-4f);
        result.Data[3].Should().BeApproximately(4f, 1e-4f);
    }

    [Fact]
    public void CpuComputeProvider_Dispose_DoesNotThrow()
    {
        var act = () =>
        {
            using var provider = new CpuComputeProvider();
            provider.MatMul(
                TensorFactory.FromData([2, 2], [1f, 0f, 0f, 1f]),
                TensorFactory.FromData([2, 2], [5f, 6f, 7f, 8f]));
        };
        act.Should().NotThrow();
    }

}

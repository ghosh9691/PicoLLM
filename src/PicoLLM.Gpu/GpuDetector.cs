using ILGPU;
using ILGPU.Runtime.Cuda;

namespace PicoLLM.Gpu;

/// <summary>
/// Detects NVIDIA CUDA-capable GPUs at runtime using ILGPU.
/// All methods are safe to call on machines without a GPU.
/// </summary>
public static class GpuDetector
{
    /// <summary>
    /// Attempts to find the first available NVIDIA CUDA GPU.
    /// </summary>
    /// <returns>
    /// A <see cref="GpuInfo"/> record if a CUDA GPU is detected; <c>null</c> if no
    /// CUDA device is available or if GPU initialization fails for any reason.
    /// </returns>
    public static GpuInfo? DetectNvidia()
    {
        try
        {
            using var context = Context.CreateDefault();
            var devices = context.GetCudaDevices();
            if (devices.Count == 0) return null;

            var device = devices[0];
            return new GpuInfo(
                Name:                device.Name,
                VramBytes:           device.MemorySize,
                MultiprocessorCount: device.NumMultiprocessors);
        }
        catch
        {
            return null;
        }
    }
}

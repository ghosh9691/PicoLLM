namespace PicoLLM.Gpu;

/// <summary>
/// Describes the NVIDIA GPU detected at runtime.
/// </summary>
/// <param name="Name">GPU display name (e.g. "NVIDIA GeForce RTX 4090").</param>
/// <param name="VramBytes">Total GPU memory in bytes.</param>
/// <param name="MultiprocessorCount">Number of CUDA multiprocessors (streaming multiprocessors).</param>
public record GpuInfo(string Name, long VramBytes, int MultiprocessorCount);

namespace PicoLLM.Gguf;

/// <summary>
/// Reads a GGUF file and verifies its header is structurally valid.
/// Useful for smoke-testing exports without loading the full model.
/// </summary>
public static class GgufValidator
{
    /// <summary>Result of a GGUF file validation check.</summary>
    /// <param name="IsValid">True if the file passed all checks.</param>
    /// <param name="TensorCount">Number of tensors declared in the header; 0 on failure.</param>
    /// <param name="KvCount">Number of KV pairs declared in the header; 0 on failure.</param>
    /// <param name="Error">Human-readable error description; null on success.</param>
    public record ValidationResult(bool IsValid, ulong TensorCount, ulong KvCount, string? Error);

    /// <summary>
    /// Validates the GGUF file at <paramref name="path"/>.
    /// Checks: magic bytes, version number, header completeness.
    /// </summary>
    public static ValidationResult Validate(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            using var reader = new BinaryReader(fs);

            // Check magic first (requires at least 4 bytes)
            if (fs.Length < 4)
                return new ValidationResult(false, 0, 0, "File too small to contain a GGUF magic");

            var magic = reader.ReadBytes(4);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
                return new ValidationResult(false, 0, 0,
                    $"Invalid magic bytes: expected GGUF, got {System.Text.Encoding.ASCII.GetString(magic)}");

            if (fs.Length < 24)
                return new ValidationResult(false, 0, 0, "File too small to contain a GGUF header");

            // Version
            uint version = reader.ReadUInt32();
            if (version != 3)
                return new ValidationResult(false, 0, 0,
                    $"Unsupported GGUF version: {version} (expected 3)");

            // Counts
            ulong tensorCount = reader.ReadUInt64();
            ulong kvCount     = reader.ReadUInt64();

            return new ValidationResult(true, tensorCount, kvCount, null);
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, 0, 0, ex.Message);
        }
    }
}

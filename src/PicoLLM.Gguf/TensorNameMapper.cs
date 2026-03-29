using System.Text.RegularExpressions;

namespace PicoLLM.Gguf;

/// <summary>
/// Translates PicoLLM internal tensor path identifiers to llama.cpp naming conventions
/// used by GGUF. Layer-indexed paths (Block.{i}.*) substitute the integer index at runtime.
/// </summary>
public static class TensorNameMapper
{
    // Maps suffix after "Block.{i}." to llama.cpp suffix after "blk.{i}."
    private static readonly Dictionary<string, string> BlockSuffixMap = new()
    {
        ["AttnNorm.Gamma"]               = "attn_norm.weight",
        ["Attention.QueryProj.Weights"]  = "attn_q.weight",
        ["Attention.KeyProj.Weights"]    = "attn_k.weight",
        ["Attention.ValueProj.Weights"]  = "attn_v.weight",
        ["Attention.OutputProj.Weights"] = "attn_output.weight",
        ["FfnNorm.Gamma"]                = "ffn_norm.weight",
        ["FFN.Up.Weights"]               = "ffn_up.weight",
        ["FFN.Down.Weights"]             = "ffn_down.weight",
    };

    private static readonly Dictionary<string, string> TopLevelMap = new()
    {
        ["Embedding.TokenEmbedding.Weights"] = "token_embd.weight",
        ["FinalNorm.Gamma"]                  = "output_norm.weight",
        ["LmHead.Weights"]                   = "output.weight",
    };

    private static readonly Regex BlockPattern =
        new(@"^Block\.(\d+)\.(.+)$", RegexOptions.Compiled);

    /// <summary>
    /// Converts a PicoLLM internal tensor path to its llama.cpp GGUF name.
    /// </summary>
    /// <param name="internalPath">
    ///   Either a top-level path (e.g. "Embedding.TokenEmbedding.Weights")
    ///   or a block-indexed path (e.g. "Block.2.Attention.QueryProj.Weights").
    /// </param>
    /// <exception cref="ArgumentException">Thrown if the path has no known mapping.</exception>
    public static string ToGgufName(string internalPath)
    {
        ArgumentNullException.ThrowIfNull(internalPath);

        if (TopLevelMap.TryGetValue(internalPath, out var topLevel))
            return topLevel;

        var match = BlockPattern.Match(internalPath);
        if (match.Success)
        {
            int layerIndex = int.Parse(match.Groups[1].Value);
            string suffix  = match.Groups[2].Value;
            if (BlockSuffixMap.TryGetValue(suffix, out var ggufSuffix))
                return $"blk.{layerIndex}.{ggufSuffix}";
        }

        throw new ArgumentException(
            $"No GGUF name mapping found for internal path: '{internalPath}'",
            nameof(internalPath));
    }
}

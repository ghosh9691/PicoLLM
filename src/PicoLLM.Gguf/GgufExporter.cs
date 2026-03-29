using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;
using PicoLLM.Tokenizer;

namespace PicoLLM.Gguf;

/// <summary>
/// Exports a trained <see cref="PicoLLMModel"/> plus its <see cref="BpeTokenizer"/>
/// to a GGUF v3 binary file.
/// </summary>
/// <remarks>
/// Uses a two-pass strategy:
/// <list type="number">
///   <item>Collect all named tensors and pre-compute data offsets.</item>
///   <item>Write header → metadata KVs → tensor info array → padding → tensor data.</item>
/// </list>
/// </remarks>
public static class GgufExporter
{
    /// <summary>
    /// Writes the model and tokenizer to <paramref name="outputPath"/> in GGUF v3 format.
    /// </summary>
    /// <param name="model">The trained PicoLLM model.</param>
    /// <param name="tokenizer">The matching BPE tokenizer.</param>
    /// <param name="outputPath">Destination file path (created or overwritten).</param>
    public static void Export(PicoLLMModel model, BpeTokenizer tokenizer, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(outputPath);

        var namedTensors = CollectNamedTensors(model);
        var offsets = ComputeOffsets(namedTensors);

        using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(namedTensors.Count, GgufConstants.MetadataKvCount);
        WriteMetadata(writer, model.Config, tokenizer);
        WriteTensorInfoArray(writer, namedTensors, offsets);
        writer.Pad(GgufConstants.Alignment);
        WriteTensorDataSection(writer, namedTensors);
    }

    // ── Tensor Collection ────────────────────────────────────────────────────

    private static List<(string GgufName, Tensor Data)> CollectNamedTensors(PicoLLMModel model)
    {
        var result = new List<(string, Tensor)>();

        // Token embedding
        result.Add(("token_embd.weight", model.Embedding.TokenEmbedding.Weights));

        // Decoder blocks
        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i];
            result.Add(($"blk.{i}.attn_norm.weight",   block.AttnNorm.Gamma));
            result.Add(($"blk.{i}.attn_q.weight",      block.Attention.QueryProj.Weights));
            result.Add(($"blk.{i}.attn_k.weight",      block.Attention.KeyProj.Weights));
            result.Add(($"blk.{i}.attn_v.weight",      block.Attention.ValueProj.Weights));
            result.Add(($"blk.{i}.attn_output.weight", block.Attention.OutputProj.Weights));
            result.Add(($"blk.{i}.ffn_norm.weight",    block.FfnNorm.Gamma));
            result.Add(($"blk.{i}.ffn_up.weight",      block.FFN.Up.Weights));
            result.Add(($"blk.{i}.ffn_down.weight",    block.FFN.Down.Weights));
        }

        // Final norm + LM head
        result.Add(("output_norm.weight", model.FinalNorm.Gamma));
        result.Add(("output.weight",      model.LmHead.Weights));

        return result;
    }

    // ── Offset Computation ────────────────────────────────────────────────────

    private static ulong[] ComputeOffsets(List<(string Name, Tensor Data)> tensors)
    {
        var offsets = new ulong[tensors.Count];
        ulong cursor = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = cursor;
            ulong byteSize = (ulong)(tensors[i].Data.Length * sizeof(float));
            cursor += byteSize;
            // Align to 32 bytes
            ulong remainder = cursor % GgufConstants.Alignment;
            if (remainder != 0) cursor += (ulong)GgufConstants.Alignment - remainder;
        }
        return offsets;
    }

    // ── Metadata Writing ─────────────────────────────────────────────────────

    private static void WriteMetadata(GgufWriter writer, ModelConfig config, BpeTokenizer tokenizer)
    {
        int ffDim = config.EmbedDim * config.FfMultiplier;
        int ropeDim = config.EmbedDim / config.NumHeads;

        writer.WriteMetadataString(GgufConstants.KeyArchitecture, "llama");
        writer.WriteMetadataString(GgufConstants.KeyName,         "PicoLLM");
        writer.WriteMetadataUint32(GgufConstants.KeyFileType,     0u);  // 0 = F32

        writer.WriteMetadataUint32(GgufConstants.KeyContextLength,     (uint)config.MaxSeqLen);
        writer.WriteMetadataUint32(GgufConstants.KeyEmbeddingLength,   (uint)config.EmbedDim);
        writer.WriteMetadataUint32(GgufConstants.KeyBlockCount,        (uint)config.NumLayers);
        writer.WriteMetadataUint32(GgufConstants.KeyFeedForwardLength, (uint)ffDim);
        writer.WriteMetadataUint32(GgufConstants.KeyHeadCount,         (uint)config.NumHeads);
        writer.WriteMetadataUint32(GgufConstants.KeyHeadCountKv,       (uint)config.NumHeads);
        writer.WriteMetadataUint32(GgufConstants.KeyRopeDimCount,      (uint)ropeDim);
        writer.WriteMetadataFloat32(GgufConstants.KeyLayerNormEps,     1e-5f);

        var (tokens, tokenTypes) = tokenizer.GetVocabRepresentations();
        writer.WriteMetadataString(GgufConstants.KeyTokenizerModel, "gpt2");
        writer.WriteMetadataStringArray(GgufConstants.KeyTokenizerTokens, tokens);
        writer.WriteMetadataInt32Array(GgufConstants.KeyTokenizerTypes,  tokenTypes);
        writer.WriteMetadataUint32(GgufConstants.KeyBosTokenId, (uint)BpeTokenizer.BosId);
        writer.WriteMetadataUint32(GgufConstants.KeyEosTokenId, (uint)BpeTokenizer.EosId);
        writer.WriteMetadataUint32(GgufConstants.KeyPadTokenId, (uint)BpeTokenizer.PadId);
    }

    // ── Tensor Info Array ────────────────────────────────────────────────────

    private static void WriteTensorInfoArray(
        GgufWriter writer,
        List<(string Name, Tensor Data)> tensors,
        ulong[] offsets)
    {
        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, tensor) = tensors[i];
            writer.WriteTensorInfo(name, tensor.GetShape(), GgufDataType.F32, offsets[i]);
        }
    }

    // ── Tensor Data Section ───────────────────────────────────────────────────

    private static void WriteTensorDataSection(
        GgufWriter writer,
        List<(string Name, Tensor Data)> tensors)
    {
        foreach (var (_, tensor) in tensors)
        {
            writer.WriteTensorData(tensor.Data.ToArray());
            writer.Pad(GgufConstants.Alignment);
        }
    }
}

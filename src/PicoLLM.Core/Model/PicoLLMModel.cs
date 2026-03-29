using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Model;

/// <summary>
/// Full GPT-style decoder-only transformer model.
/// </summary>
/// <remarks>
/// Architecture:
/// <code>
/// TokenEmbedding + PositionalEncoding  → [B, S, E]
/// N × DecoderBlock                     → [B, S, E]
/// Final LayerNorm                      → [B, S, E]
/// Linear(embed_dim → vocab_size)       → [B, S, VocabSize]   (logits)
/// </code>
/// Logits are unnormalized scores over the vocabulary.
/// Apply softmax to get a probability distribution for next-token prediction.
/// </remarks>
public sealed class PicoLLMModel
{
    private readonly EmbeddingLayer _embedding;
    private readonly DecoderBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly LinearLayer _lmHead;

    /// <summary>Model hyperparameters.</summary>
    public ModelConfig Config { get; }

    /// <summary>
    /// Initializes the model with the given configuration.
    /// </summary>
    /// <param name="config">Model hyperparameters.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public PicoLLMModel(ModelConfig config, int? seed = null)
    {
        Config = config;

        _embedding = new EmbeddingLayer(
            config.VocabSize, config.EmbedDim, config.MaxSeqLen, seed);

        _blocks = new DecoderBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new DecoderBlock(
                config.EmbedDim, config.NumHeads, config.FfMultiplier,
                seed: seed.HasValue ? seed + i * 10 : null);

        _finalNorm = new LayerNorm(config.EmbedDim);
        _lmHead = new LinearLayer(
            config.EmbedDim, config.VocabSize, useBias: true,
            seed: seed.HasValue ? seed + config.NumLayers * 10 : null);
    }

    /// <summary>
    /// Forward pass: token IDs → vocabulary logits.
    /// </summary>
    /// <param name="tokenIds">Token ID matrix [batch, seq]. All values must be in [0, VocabSize).</param>
    /// <returns>Logits tensor [batch, seq, vocab_size].</returns>
    public Tensor Forward(int[,] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        int batch = tokenIds.GetLength(0);
        int seqLen = tokenIds.GetLength(1);

        // Convert 2D array to jagged for EmbeddingLayer
        var jagged = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            jagged[b] = new int[seqLen];
            for (int s = 0; s < seqLen; s++)
                jagged[b][s] = tokenIds[b, s];
        }

        // Embedding: [batch, seq, embed_dim]
        Tensor x = _embedding.ForwardBatch(jagged);

        // Decoder stack
        foreach (var block in _blocks)
            x = block.Forward(x);

        // Final layer norm
        x = _finalNorm.Forward(x);

        // LM head: [batch, seq, vocab_size]
        return _lmHead.Forward(x);
    }

    /// <summary>
    /// Returns the total number of learnable scalar parameters across the entire model.
    /// </summary>
    public int TotalParameters() => GetAllParameters().Sum(p => p.Length);

    /// <summary>
    /// Returns a flat list of every learnable parameter tensor in the model.
    /// This is the list the optimizer iterates over.
    /// Order: token embedding → decoder blocks (in order) → final norm → lm head.
    /// </summary>
    public IReadOnlyList<Tensor> GetAllParameters()
    {
        var result = new List<Tensor>();

        // Token embedding weights (positional encoding is sinusoidal — no learnable params)
        result.Add(_embedding.TokenEmbedding.Weights);

        // Decoder blocks
        foreach (var block in _blocks)
            result.AddRange(block.Parameters());

        // Final layer norm
        result.AddRange(_finalNorm.Parameters());

        // LM head
        result.AddRange(_lmHead.Parameters());

        return result;
    }

    /// <summary>Zeros all accumulated gradients in the entire model.</summary>
    public void ZeroGrad()
    {
        _embedding.ZeroGrad();
        foreach (var block in _blocks) block.ZeroGrad();
        _finalNorm.ZeroGrad();
        _lmHead.ZeroGrad();
    }
}

using PicoLLM.Core.Compute;
using PicoLLM.Core.Layers;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

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
/// Call <see cref="Forward"/> for inference, then <see cref="Backward"/> to compute
/// gradients, then the optimizer to update weights.
/// </remarks>
public sealed class PicoLLMModel
{
    private readonly EmbeddingLayer _embedding;
    private readonly DecoderBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly LinearLayer _lmHead;

    // Cached for backward pass
    private int[][]? _lastJaggedIds;

    /// <summary>Model hyperparameters.</summary>
    public ModelConfig Config { get; }

    /// <summary>The combined token + positional embedding layer.</summary>
    public EmbeddingLayer Embedding => _embedding;

    /// <summary>The decoder blocks, one per transformer layer.</summary>
    public IReadOnlyList<DecoderBlock> Blocks => _blocks;

    /// <summary>The final layer normalisation applied before the LM head.</summary>
    public LayerNorm FinalNorm => _finalNorm;

    /// <summary>The language model head linear projection [embed_dim → vocab_size].</summary>
    public LinearLayer LmHead => _lmHead;

    /// <summary>
    /// Initializes the model with the given configuration.
    /// </summary>
    public PicoLLMModel(ModelConfig config, int? seed = null,
        IComputeProvider? computeProvider = null)
    {
        Config = config;

        _embedding = new EmbeddingLayer(
            config.VocabSize, config.EmbedDim, config.MaxSeqLen, seed);

        _blocks = new DecoderBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new DecoderBlock(
                config.EmbedDim, config.NumHeads, config.FfMultiplier,
                seed: seed.HasValue ? seed + i * 10 : null,
                computeProvider: computeProvider);

        _finalNorm = new LayerNorm(config.EmbedDim);
        _lmHead = new LinearLayer(
            config.EmbedDim, config.VocabSize, useBias: true,
            seed: seed.HasValue ? seed + config.NumLayers * 10 : null);
    }

    /// <summary>
    /// Forward pass: token IDs → vocabulary logits.
    /// Caches token IDs and all intermediate activations for <see cref="Backward"/>.
    /// </summary>
    /// <param name="tokenIds">Token ID matrix [batch, seq]. All values must be in [0, VocabSize).</param>
    /// <returns>Logits tensor [batch, seq, vocab_size].</returns>
    public Tensor Forward(int[,] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        int batch  = tokenIds.GetLength(0);
        int seqLen = tokenIds.GetLength(1);

        _lastJaggedIds = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            _lastJaggedIds[b] = new int[seqLen];
            for (int s = 0; s < seqLen; s++)
                _lastJaggedIds[b][s] = tokenIds[b, s];
        }

        Tensor x = _embedding.ForwardBatch(_lastJaggedIds);

        foreach (var block in _blocks)
            x = block.Forward(x);

        x = _finalNorm.Forward(x);
        return _lmHead.Forward(x);
    }

    /// <summary>
    /// Backward pass. Propagates gradients from logits through the entire model.
    /// Must be called after <see cref="Forward"/>.
    /// </summary>
    /// <param name="gradLogits">Gradient w.r.t. logits, shape [batch, seq, vocab_size].</param>
    public void Backward(Tensor gradLogits)
    {
        if (_lastJaggedIds is null)
            throw new InvalidOperationException("Forward() must be called before Backward().");

        // LM head backward → [B, S, E]
        var grad = _lmHead.Backward(gradLogits);

        // Final norm backward
        grad = _finalNorm.Backward(grad);

        // Decoder blocks in reverse order
        for (int i = _blocks.Length - 1; i >= 0; i--)
            grad = _blocks[i].Backward(grad);

        // Embedding backward (sparse — no return value needed)
        _embedding.BackwardBatch(grad, _lastJaggedIds);
    }

    /// <summary>
    /// Returns the total number of learnable scalar parameters across the entire model.
    /// </summary>
    public int TotalParameters() => GetAllParameters().Sum(p => p.Data.Length);

    /// <summary>
    /// Returns a flat list of every learnable <see cref="Parameter"/> in the model.
    /// This is the list the optimizer iterates over.
    /// Order: token embedding → decoder blocks (in order) → final norm → lm head.
    /// </summary>
    public IReadOnlyList<Parameter> GetAllParameters()
    {
        var result = new List<Parameter>();
        result.Add(_embedding.TokenEmbedding.WeightsParameter);
        foreach (var block in _blocks) result.AddRange(block.Parameters());
        result.AddRange(_finalNorm.Parameters());
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
        _lastJaggedIds = null;
    }
}

using PicoLLM.Core.Model;
using PicoLLM.Core.Tensors;
using PicoLLM.Core.Training;

namespace PicoLLM.Training;

/// <summary>
/// Saves and loads full training state (model weights, optimizer moments, step counter) to binary files.
/// </summary>
/// <remarks>
/// Binary format:
/// <code>
/// [magic: "PCKP" 4 bytes]
/// [version: uint32 = 1]
/// [step: int64]
/// [num_parameters: int32]
/// for each parameter:
///     [name_len: int32][name: utf8 bytes]
///     [shape_rank: int32][shape: int32[]]
///     [data: float[] × product(shape)]
/// [has_optimizer_state: bool (1 byte)]
/// if true:
///     for each parameter:
///         [m: float[]][v: float[]]
/// </code>
/// </remarks>
public static class CheckpointManager
{
    private static readonly byte[] Magic = "PCKP"u8.ToArray();
    private const uint Version = 1u;

    /// <summary>
    /// Saves the model and optimizer state to a binary file.
    /// </summary>
    /// <param name="model">The model whose weights to save.</param>
    /// <param name="optimizer">The optimizer whose moment state to save.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="path">Output file path.</param>
    public static void Save(PicoLLMModel model, AdamW optimizer, int step, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNullOrEmpty(path);

        var parameters = model.GetAllParameters();

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Header
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((long)step);
        writer.Write(parameters.Count);

        // Parameter data
        for (int i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            string name = $"param_{i}";
            var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            var shape = p.Data.GetShape();
            writer.Write(shape.Length);
            foreach (var dim in shape) writer.Write(dim);

            foreach (float v in p.Data.Data) writer.Write(v);
        }

        // Optimizer state
        writer.Write(true);
        var state = optimizer.State;
        foreach (var p in parameters)
        {
            if (state.TryGetValue(p, out var moments))
            {
                foreach (float mv in moments.m) writer.Write(mv);
                foreach (float vv in moments.v) writer.Write(vv);
            }
            else
            {
                // No state yet for this param — write zeros
                int n = p.Data.Length;
                for (int i = 0; i < n; i++) writer.Write(0f);
                for (int i = 0; i < n; i++) writer.Write(0f);
            }
        }
    }

    /// <summary>
    /// Loads training state from a checkpoint file into the provided model and optimizer.
    /// </summary>
    /// <param name="path">Checkpoint file path.</param>
    /// <param name="model">Model to restore weights into.</param>
    /// <param name="optimizer">Optimizer to restore moments into.</param>
    /// <returns>The saved step number.</returns>
    public static int Load(string path, PicoLLMModel model, AdamW optimizer)
    {
        ArgumentNullException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);

        // Validate magic
        var magic = reader.ReadBytes(4);
        if (!magic.SequenceEqual(Magic))
            throw new InvalidDataException("Not a valid PicoLLM checkpoint (bad magic bytes).");

        uint version = reader.ReadUInt32();
        if (version != Version)
            throw new InvalidDataException($"Unsupported checkpoint version {version}.");

        int step = (int)reader.ReadInt64();
        int numParams = reader.ReadInt32();

        var parameters = model.GetAllParameters();
        if (parameters.Count != numParams)
            throw new InvalidDataException(
                $"Checkpoint has {numParams} parameters but model has {parameters.Count}.");

        // Read parameter data
        for (int i = 0; i < numParams; i++)
        {
            int nameLen = reader.ReadInt32();
            reader.ReadBytes(nameLen); // name (ignored — using position order)

            int rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int d = 0; d < rank; d++) shape[d] = reader.ReadInt32();

            int n = 1; foreach (var s in shape) n *= s;
            var data = parameters[i].Data.MutableData;
            for (int j = 0; j < n; j++) data[j] = reader.ReadSingle();
        }

        // Optimizer state
        bool hasState = reader.ReadBoolean();
        if (hasState)
        {
            var moments = new List<(float[] m, float[] v)>();
            foreach (var p in parameters)
            {
                int n = p.Data.Length;
                var m = new float[n];
                var v = new float[n];
                for (int j = 0; j < n; j++) m[j] = reader.ReadSingle();
                for (int j = 0; j < n; j++) v[j] = reader.ReadSingle();
                moments.Add((m, v));
            }
            optimizer.LoadState(parameters, moments);
        }

        optimizer.SetStep(step);
        return step;
    }
}

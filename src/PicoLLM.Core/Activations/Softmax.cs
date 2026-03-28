using PicoLLM.Core.Tensors;

namespace PicoLLM.Core.Activations;

/// <summary>
/// Softmax activation function applied along a specified axis.
/// Converts a vector of real values into a probability distribution that sums to 1.
/// Delegates to <see cref="TensorMath.Softmax"/> for the numerically stable implementation.
/// </summary>
public static class Softmax
{
    /// <summary>
    /// Applies numerically stable softmax along <paramref name="axis"/>.
    /// Default axis is the last dimension (−1 interpreted as Rank−1).
    /// </summary>
    public static Tensor Forward(Tensor input, int axis = -1)
    {
        int resolvedAxis = axis < 0 ? input.Rank + axis : axis;
        return TensorMath.Softmax(input, resolvedAxis);
    }

    /// <summary>
    /// Softmax Jacobian-vector product for backprop.
    /// Given upstream gradient <paramref name="gradOutput"/> and softmax output <paramref name="output"/>,
    /// computes dL/dx = s ⊙ (dL/dy − (dL/dy · s) · 1) where s is the softmax output.
    /// </summary>
    public static Tensor Backward(Tensor output, Tensor gradOutput, int axis = -1)
    {
        int resolvedAxis = axis < 0 ? output.Rank + axis : axis;

        // dot = sum(gradOutput * output) along axis
        var dot = TensorMath.Sum(TensorMath.Multiply(gradOutput, output), resolvedAxis);

        // Expand dot back for broadcasting (simple loop implementation)
        int[] shape = output.GetShape();
        int axisSize = shape[resolvedAxis];
        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < resolvedAxis; i++) outerSize *= shape[i];
        for (int i = resolvedAxis + 1; i < output.Rank; i++) innerSize *= shape[i];

        var src_o = output.Data;
        var src_g = gradOutput.Data;
        var src_d = dot.Data;
        var result = new float[output.Length];

        for (int outer = 0; outer < outerSize; outer++)
            for (int a = 0; a < axisSize; a++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int idx = (outer * axisSize + a) * innerSize + inner;
                    result[idx] = src_o[idx] * (src_g[idx] - src_d[outer * innerSize + inner]);
                }

        return new Tensor(shape, result);
    }
}

# Core Tensor Specification

## Purpose

Provide the foundational N-dimensional tensor type and mathematical operations required by all subsequent layers. This extends the TinyLLM Tensor with operations needed for transformer architectures.

### Requirement: Tensor Storage

The system SHALL store tensor data as a contiguous `float[]` array with shape metadata, supporting 1D through 4D tensors.

#### Scenario: Create a 2D tensor
- **GIVEN** a shape of [3, 4]
- **WHEN** a Tensor is constructed
- **THEN** it allocates a float array of length 12 and stores shape [3, 4]

#### Scenario: Create a 4D tensor for batched attention
- **GIVEN** a shape of [batch, heads, seq_len, head_dim]
- **WHEN** a Tensor is constructed with shape [2, 4, 16, 32]
- **THEN** it allocates 2×4×16×32 = 4096 floats

### Requirement: Element Access

The system SHALL support element access by multi-dimensional index with bounds checking.

#### Scenario: Get/Set by index
- **GIVEN** a tensor of shape [3, 4]
- **WHEN** element [1, 2] is set to 5.0 and then retrieved
- **THEN** the returned value is 5.0

#### Scenario: Out-of-bounds access
- **GIVEN** a tensor of shape [3, 4]
- **WHEN** element [3, 0] is accessed
- **THEN** an IndexOutOfRangeException is thrown

### Requirement: Element-wise Operations

The system SHALL support element-wise add, subtract, multiply, divide, and scalar operations.

#### Scenario: Element-wise addition
- **GIVEN** two tensors A and B of identical shape [2, 3]
- **WHEN** C = A + B
- **THEN** each element C[i,j] = A[i,j] + B[i,j]

#### Scenario: Broadcasting scalar multiply
- **GIVEN** a tensor A of shape [2, 3] and scalar 2.0
- **WHEN** C = A * 2.0
- **THEN** each element C[i,j] = A[i,j] * 2.0

### Requirement: Matrix Multiplication

The system SHALL support 2D matrix multiplication and batched matrix multiplication for attention computation.

#### Scenario: Standard matmul
- **GIVEN** A of shape [M, K] and B of shape [K, N]
- **WHEN** C = MatMul(A, B)
- **THEN** C has shape [M, N] and C[i,j] = sum(A[i,k] * B[k,j]) for k in 0..K

#### Scenario: Batched matmul for attention
- **GIVEN** Q of shape [batch, heads, seq, head_dim] and K^T of shape [batch, heads, head_dim, seq]
- **WHEN** scores = BatchedMatMul(Q, K^T)
- **THEN** scores has shape [batch, heads, seq, seq]

### Requirement: Transpose

The system SHALL support transposing any two dimensions of a tensor.

#### Scenario: 2D transpose
- **GIVEN** A of shape [3, 4]
- **WHEN** B = Transpose(A, 0, 1)
- **THEN** B has shape [4, 3] and B[j,i] = A[i,j]

#### Scenario: Transpose last two dims for attention
- **GIVEN** K of shape [batch, heads, seq, head_dim]
- **WHEN** K_T = Transpose(K, 2, 3)
- **THEN** K_T has shape [batch, heads, head_dim, seq]

### Requirement: Reshape and View

The system SHALL support reshaping a tensor without copying data when the total element count is preserved.

#### Scenario: Reshape for multi-head split
- **GIVEN** a tensor of shape [batch, seq, embed_dim] where embed_dim = heads × head_dim
- **WHEN** reshaped to [batch, seq, heads, head_dim] then transposed to [batch, heads, seq, head_dim]
- **THEN** data is logically reorganized for per-head attention computation

### Requirement: Softmax

The system SHALL implement numerically stable softmax along a specified axis.

#### Scenario: Softmax along last dimension
- **GIVEN** a tensor of shape [2, 3] with row [1.0, 2.0, 3.0]
- **WHEN** Softmax is applied along axis 1
- **THEN** output row sums to 1.0 and values are exp(x - max) / sum(exp(x - max))

### Requirement: Tensor Fill and Initialization

The system SHALL support filling tensors with zeros, ones, constants, random normal, and random uniform distributions for weight initialization.

#### Scenario: Xavier initialization
- **GIVEN** a weight tensor of shape [fan_in, fan_out]
- **WHEN** initialized with Xavier uniform
- **THEN** values are drawn from U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))

### Requirement: Reduction Operations

The system SHALL support sum, mean, max, and argmax along a specified axis.

#### Scenario: Sum along axis
- **GIVEN** a tensor of shape [3, 4]
- **WHEN** Sum(axis=1) is computed
- **THEN** result has shape [3] where each element is the sum of that row

### Requirement: Masking

The system SHALL support applying a boolean mask to a tensor, setting masked positions to a specified value (typically negative infinity for attention masking).

#### Scenario: Causal attention mask
- **GIVEN** attention scores of shape [seq, seq]
- **WHEN** a causal mask is applied (upper triangle set to -inf)
- **THEN** after softmax, future positions have zero attention weight

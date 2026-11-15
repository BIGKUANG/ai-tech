# 融合 Linear / MLP / 激活函数 API

- 源码：
  - `flash_attn/ops/fused_dense.py`（CUDA 融合 Linear/MLP）
  - `flash_attn/ops/activations.py`（激活函数）
  - `flash_attn/ops/triton/linear.py`、`mlp.py`、`k_activations.py`（Triton 实现）

---

## 融合 Dense / MLP（fused_dense.py）

### fused_dense_func

```python
fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
)
```

融合 `Linear` + 可选张量并行（all-gather + matmul + reduce-scatter）。当输入 dtype 为 fp16/bf16（或 fp32 且开启 autocast）且在 CUDA 上时走融合内核，否则回退到 `F.linear`。

- `return_residual=True` 时额外返回输入 `x`（便于后续与残差融合）。

### FusedDense (nn.Module)

```python
FusedDense(in_features, out_features, bias=True, return_residual=False, process_group=None, sequence_parallel=True, device=None, dtype=None)
```

`forward(x, process_group=None)`，继承自 `nn.Linear`。

### ColumnParallelLinear

```python
ColumnParallelLinear(in_features, out_features, process_group, ...)
```

列并行线性层（张量并行切分输出维），`forward(x)`。可选 `sequence_parallel`。

### RowParallelLinear

```python
RowParallelLinear(in_features, out_features, process_group, ...)
```

行并行线性层（张量并行切分输入维），`forward(x)`。

### fused_mlp_func

```python
fused_mlp_func(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    activation: str = "gelu_approx",
    save_pre_act: bool = True,
    return_residual: bool = False,
    checkpoint_lvl: int = 0,
    heuristic: int = 0,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
)
```

融合两层 MLP：`activation(x @ W1^T + b1) @ W2^T + b2`。

- `activation`：`"gelu_approx"`、`"relu"`、`"sqrelu"` 之一。
- 若 `save_pre_act=True`，输入维需能被 128（relu）或 8（gelu）整除。
- `checkpoint_lvl`：见下方 `FusedMLP`。

### FusedMLP (nn.Module)

```python
FusedMLP(in_features, hidden_features=None, out_features=None,
         bias1=True, bias2=True, activation="gelu_approx",
         return_residual=False, checkpoint_lvl=0, heuristic="auto",
         device=None, dtype=None)
```

`forward(x, process_group=None)`。

- `checkpoint_lvl`（越大越慢但省显存）：
  - `0`：反向不重算
  - `1`：反向重算 `gelu_out`
  - `2`：反向重算 `pre_act` 与 `gelu_out`
- `heuristic`：`-1` 表示不融合 gemm+gelu（分开 kernel）；`0..4` 选择融合 gemm+gelu 的算法；`'auto'` 自动选择（CUDA ≥ 11.8 时 fp16/bf16 都用 0；H100 上 `gelu_approx` 用 -1，因为融合 cuBLASLt 反而更慢）。

### ParallelFusedMLP (nn.Module)

```python
ParallelFusedMLP(in_features, hidden_features=None, out_features=None,
                 activation="gelu_approx", process_group=None,
                 bias1=True, bias2=True, sequence_parallel=True,
                 checkpoint_lvl=0, heuristic="auto", device=None, dtype=None)
```

张量并行 + 序列并行的 MLP。`forward(x)`。内部使用 `ColumnParallelLinear` 与 `RowParallelLinear`。

---

## 激活函数（activations.py）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `bias_gelu(y, bias)` | `@torch.jit` 函数 | GELU（tanh 近似）+ bias，融合 |
| `bias_gelu_back(g, y, bias)` | `@torch.jit` 函数 | 上述的反向 |
| `GeLUFunction` / `bias_gelu_impl` | autograd.Function / 别名 | bias + GELU 可微算子 |
| `gelu_fwd(x)` / `gelu_bwd(g, x)` | `@torch.jit` 函数 | GELU（tanh 近似）前向/反向 |
| `FastGeLUFunction` / `fast_gelu_impl` | autograd.Function / 别名 | 无 bias 的 GELU 可微算子 |
| `relu_bwd(g, x)` | `@torch.jit` 函数 | ReLU 反向 |
| `sqrelu_fwd(x)` / `sqrelu_bwd(g, x)` | `@torch.jit` 函数 | Squared ReLU 前向/反向 |
| `swiglu_fwd` / `swiglu_bwd` | jiterator 函数 | SwiGLU 前向/反向 |
| `swiglu` | 别名 | `SwiGLUFunction.apply` |

**示例**

```python
from flash_attn.ops.activations import bias_gelu_impl

y = torch.randn(4, 512, device="cuda", dtype=torch.float16)
bias = torch.randn(512, device="cuda", dtype=torch.float16)
out = bias_gelu_impl(y, bias)
```

---

## Triton 算子（ops/triton/）

无需编译 CUDA 扩展的 Triton 实现。

### linear.py

| 名称 | 说明 |
| --- | --- |
| `triton_linear_act(x, weight, bias=None, activation=..., ...)` | 融合 Linear + 激活 |
| `triton_dgrad_act(grad, weight, activation=..., ...)` | 反向数据梯度 + 激活反向 |
| `kernel_fwd` / `kernel_bwd` | 底层 Triton kernel |

### mlp.py

| 名称 | 说明 |
| --- | --- |
| `FusedDenseSqreluDense` | 融合 `Linear + Squared ReLU + Linear` 的 `nn.Module` |
| `FusedDenseSqreluDenseFunc` | 对应 autograd.Function |

### k_activations.py

| 名称 | 说明 |
| --- | --- |
| `Activation` | 激活类型枚举 |
| `get_triton_activation_kernel(activation)` | 取前向 kernel |
| `get_triton_activation_bwd_kernel(activation)` | 取反向 kernel |
| `tanh`, `cosh`, `relu`, `relu_grad`, `squared_relu`, `squared_relu_grad`, `leaky_relu`, `leaky_relu_grad`, `gelu`, `gelu_grad`, `gelu_approx`, `gelu_approx_grad` | 激活原语 |

### cross_entropy.py

`cross_entropy_loss`（见 [api-losses.md](./api-losses.md)）。

### rotary.py

`apply_rotary`（见 [api-rotary.md](./api-rotary.md)）。

### layer_norm.py

见 [api-normalization.md](./api-normalization.md)。

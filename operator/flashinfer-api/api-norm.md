# 归一化层（`flashinfer.norm`）

> 源码：`flashinfer/norm/__init__.py`、`flashinfer/norm/utils.py`、`flashinfer/cake_rmsnorm_train.py`
> 上游文档：`docs/api/norm.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.norm` 提供 Transformer 网络中使用的高性能归一化算子，包括 RMSNorm、LayerNorm、
残差相加融合（Residual-Add Fusion），以及带 FP8/FP4 量化的变体。多数算子同时提供两套后端实现：
CuTe DSL 实现（默认，适用于受支持的架构）与 CUDA JIT 实现（回退路径）。当环境变量
`FLASHINFER_USE_CUDA_NORM=1`，或当前设备架构不被已安装的 CuTe DSL 支持时（例如 Rubin `sm_107`
对应的 DSL 不含 `sm_107a`），会自动切换到 CUDA JIT 实现；两套实现在功能上等价。

此外，`flashinfer.cake_rmsnorm_train` 提供面向训练的前向/反向 BF16 RMSNorm（SM100/SM103/SM107），
使用 FP32 统计与累加、每行保存一个 FP32 倒数 RMS、融合的 `dx` 与确定性 FP32 `dw` 反向，并支持可选
的残差相加融合与 PyTorch autograd 封装。

注意：`flashinfer/norm/utils.py` 仅包含 CuTe DSL 内核的内部辅助函数（FP8 转换、归约、布局计算等），
不属于本页文档化的公开 API。

## 导入

```python
import flashinfer

# 归一化层
from flashinfer.norm import (
    rmsnorm,
    rmsnorm_quant,
    fused_add_rmsnorm,
    fused_add_rmsnorm_quant,
    fused_add_rmsnorm_fp8_block_quant,
    gemma_rmsnorm,
    gemma_fused_add_rmsnorm,
    layernorm,
    layernorm_quant,
    fused_rmsnorm_silu,
    fused_qk_rmsnorm_rope,
    fused_dit_residual_layernorm_scale_shift,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_gate_residual_layernorm_gamma_beta,
)

# 训练用 Cake RMSNorm
from flashinfer.cake_rmsnorm_train import (
    cake_rmsnorm,
    cake_rmsnorm_train_forward,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm_train_backward_workspace,
    CakeRMSNormFunction,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `rmsnorm` | 函数 | Root Mean Square 归一化。 |
| `rmsnorm_quant` | 函数 | RMSNorm + FP8 量化。 |
| `fused_add_rmsnorm` | 函数 | 融合残差相加的 RMSNorm。 |
| `fused_add_rmsnorm_quant` | 函数 | 融合残差相加的 RMSNorm + FP8 量化。 |
| `fused_add_rmsnorm_fp8_block_quant` | 函数 | 融合残差相加 + RMSNorm + 1x128 FP8 分块量化。 |
| `gemma_rmsnorm` | 函数 | Gemma 风格 RMSNorm（`weight + 1`）。 |
| `gemma_fused_add_rmsnorm` | 函数 | Gemma 风格融合残差相加的 RMSNorm。 |
| `layernorm` | 函数 | Layer Normalization。 |
| `layernorm_quant` | 函数 | LayerNorm + FP8 量化。 |
| `fused_rmsnorm_silu` | 函数 | 融合 RMSNorm + SiLU 激活（支持 bf16 / FP8 / NVFP4 输出）。 |
| `fused_qk_rmsnorm_rope` | 函数 | 融合 QK RMSNorm + 3D RoPE + V 拷贝（视频生成 DIT）。 |
| `fused_dit_residual_layernorm_scale_shift` | 函数 | 融合残差 + LayerNorm + scale/shift（DIT）。 |
| `fused_dit_gate_residual_layernorm_scale_shift` | 函数 | 融合 gate + 残差 + LayerNorm + scale/shift（DIT）。 |
| `fused_dit_gate_residual_layernorm_gamma_beta` | 函数 | 融合 gate + 残差 + LayerNorm(gamma, beta)（DIT）。 |
| `cake_rmsnorm` | 函数 | 支持 autograd 的融合 BF16 RMSNorm（训练）。 |
| `cake_rmsnorm_train_forward` | 函数 | Cake RMSNorm 前向（FP32 统计）。 |
| `cake_rmsnorm_train_backward` | 函数 | Cake RMSNorm 反向（`dx` + 确定性 FP32 `dw`）。 |
| `cake_rmsnorm_train_backward_workspace_bytes` | 函数 | 计算反向工作区所需字节数。 |
| `cake_rmsnorm_train_backward_workspace` | 函数 | 分配零初始化的反向工作区。 |
| `CakeRMSNormFunction` | 类 | `torch.autograd.Function` 封装。 |

## API 详情

### `rmsnorm`

```python
rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：Root Mean Square 归一化，计算

```python
out[i] = (input[i] / RMS(input)) * weight[i]
```

**参数**

- `input`（`torch.Tensor`）：输入张量，二维形状 `(batch_size, hidden_size)` 或三维形状
  `(batch_size, num_heads, hidden_size)`。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：输出张量；若提供，内核将原地更新该张量。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用程序化依赖启动（Programmatic
  Dependent Launch，PDL）。为 `None` 时按设备能力自动决定。

**返回**

- `torch.Tensor`：归一化后的张量，二维形状 `(batch_size, hidden_size)` 或三维形状
  `(batch_size, num_heads, hidden_size)`。

---

### `rmsnorm_quant`

```python
rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None
```

**功能**：RMSNorm + FP8 量化，计算

```python
out[i] = ((input[i] / RMS(input)) * weight[i]).to(fp8)
```

**参数**

- `out`（`torch.Tensor`）：输出张量，内核会将输出量化到该张量的 dtype，必须为
  `float8_e4m3fn` 或 `float8_e5m2`。
- `input`（`torch.Tensor`）：输入张量，二维形状 `(batch_size, hidden_size)`。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `scale`（`torch.Tensor`）：量化缩放因子，形状 `(1,)`。传入 `float` 已废弃，将在未来版本移除。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `None`：结果原地写入 `out`。

---

### `fused_add_rmsnorm`

```python
fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None
```

**功能**：融合残差相加的 RMSNorm，分两步原地更新：

```python
# Step 1
residual[i] += input[i]
# Step 2
input[i] = (residual[i] / RMS(residual)) * weight[i]
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`，被原地更新为归一化结果。
- `residual`（`torch.Tensor`）：残差张量，形状 `(batch_size, hidden_size)`，被原地更新。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `None`：`input` 与 `residual` 均被原地修改。

---

### `fused_add_rmsnorm_quant`

```python
fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None
```

**功能**：融合残差相加的 RMSNorm + FP8 量化：

```python
# Step 1
residual[i] += input[i]
# Step 2
input[i] = ((residual[i] / RMS(residual)) * weight[i]).to(fp8)
```

**参数**

- `out`（`torch.Tensor`）：输出张量，量化到该张量 dtype，必须为 `float8_e4m3fn` 或
  `float8_e5m2`。
- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`。
- `residual`（`torch.Tensor`）：残差张量，形状 `(batch_size, hidden_size)`，被原地更新。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `scale`（`torch.Tensor`）：量化缩放因子，形状 `(1,)`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `None`：结果写入 `out`，`residual` 被原地更新。

---

### `fused_add_rmsnorm_fp8_block_quant`

```python
fused_add_rmsnorm_fp8_block_quant(
    out: torch.Tensor,
    block_scale: torch.Tensor,
    normed_out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None
```

**功能**：一次遍历内完成 残差相加 + RMSNorm + 1x128 FP8 分块量化。用于替代此前在 FP8
分块缩放 GEMM 之前使用的双内核生产者（先 `fused_add_rmsnorm`，再做逐 token 组的 1x128 FP8 量化）。
与 `fused_add_rmsnorm_quant`（标量逐张量 scale）不同，本函数输出 **动态的逐 1x128 块** FP32 scale，
且采用 `flashinfer.deep_gemm` 所需、列主序（MN-major、TMA 对齐）的布局。分三步：

```python
# Step 1
residual = input + residual          # 归一化前，原地写入
# Step 2
normed = RMSNorm(residual) * weight  # bf16，写入 normed_out
# Step 3
out, block_scale = quant_1x128_fp8(normed)  # 由 bf16 取整后的 normed 生成 fp8
```

**参数**

- `out`（`torch.Tensor`）：FP8 输出，形状 `(batch_size, hidden_size)`，dtype `float8_e4m3fn`。
- `block_scale`（`torch.Tensor`）：FP32 分块 scale，形状 `(hidden_size // 128, round_up(batch_size, 4))`，
  连续存储。逻辑形状为 `(batch_size, hidden_size // 128)` 的 scale 是其转置
  `block_scale.transpose(0, 1)[:batch_size]`。
- `normed_out`（`torch.Tensor`）：bf16/fp16 归一化输出，形状 `(batch_size, hidden_size)`。
- `input`、`residual`（`torch.Tensor`）：形状 `(batch_size, hidden_size)`；`residual` 被原地更新为
  `input + residual`。
- `weight`（`torch.Tensor`）：RMSNorm 权重，形状 `(hidden_size,)`。`hidden_size` 必须是 128 的倍数
  （且 `hidden_size<=8192` 时为 256 的倍数，`8192<hidden_size<=16384` 时为 512 的倍数）。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `None`：结果写入 `out`、`block_scale`、`normed_out`，`residual` 被原地更新。

**注意事项**

- `block_scale` 为 deep_gemm 期望的列主序（MN-major、TMA 对齐）缓冲区。

---

### `gemma_rmsnorm`

```python
gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：Gemma 风格 RMSNorm，权重偏置为 1：

```python
out[i] = (input[i] / RMS(input)) * (weight[i] + 1)
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：输出张量；若提供则原地更新。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `torch.Tensor`：Gemma 归一化后的张量，形状 `(batch_size, hidden_size)`。

---

### `gemma_fused_add_rmsnorm`

```python
gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None
```

**功能**：Gemma 风格融合残差相加的 RMSNorm：

```python
# Step 1
residual[i] += input[i]
# Step 2
input[i] = (residual[i] / RMS(residual)) * (weight + 1)
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`，被原地更新。
- `residual`（`torch.Tensor`）：残差张量，形状 `(batch_size, hidden_size)`，被原地更新。
- `weight`（`torch.Tensor`）：权重张量，形状 `(hidden_size,)`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `None`：`input` 与 `residual` 均被原地修改。

---

### `layernorm`

```python
layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor
```

**功能**：Layer Normalization。

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`，需为 bfloat16。
- `gemma`（`torch.Tensor`）：LayerNorm 权重，形状 `(hidden_size,)`，需为 float32。
- `beta`（`torch.Tensor`）：LayerNorm 偏置，形状 `(hidden_size,)`，需为 float32。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。

**返回**

- `torch.Tensor`：LayerNorm 后的张量，形状 `(batch_size, hidden_size)`，dtype 与 `input` 相同。

---

### `layernorm_quant`

```python
layernorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
) -> None
```

**功能**：LayerNorm + FP8 量化：

```python
out[i] = (((input[i] - E[input]) / sqrt(Var[input] + eps)) * gemma[i] + beta[i]) / scale
```

**参数**

- `out`（`torch.Tensor`）：输出张量，形状 `(batch_size, hidden_size)`，需连续。输出被量化到该张量
  dtype，必须为 `float8_e4m3fn` 或 `float8_e5m2`。
- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, hidden_size)`，需为 bfloat16 且连续。
- `gemma`（`torch.Tensor`）：LayerNorm 权重，形状 `(hidden_size,)`，需为 float32。
- `beta`（`torch.Tensor`）：LayerNorm 偏置，形状 `(hidden_size,)`，需为 float32。
- `scale`（`torch.Tensor`）：量化缩放因子，形状 `(1,)`；归一化输出在转 FP8 前除以该 scale。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。

**返回**

- `None`：结果原地写入 `out`。

**注意事项**

- 目前尚无 CuTe-DSL 版 Layernorm-quant 内核，始终使用 CUDA JIT 模块。

---

### `fused_rmsnorm_silu`

```python
fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    block_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple]
```

**功能**：融合 RMSNorm + SiLU 激活：

```python
out[i] = SiLU(RMSNorm(input[i], weight, eps))
# SiLU(x) = x / (1 + exp(-x))
```

针对 SM100（B200）上 WAN VAE decoder 的问题规模做优化。其他形状与架构（SM80+）使用保守回退启发式。

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(num_tokens, hidden_size)`，dtype `bfloat16`。
- `weight`（`torch.Tensor`）：scale（gamma）张量，形状 `(hidden_size,)`，dtype `bfloat16`。
- `eps`（`float`，默认 `1e-6`）：数值稳定性 epsilon。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：输出张量；`None` 时按 `input` 分配为 `bfloat16`。
  `out` 的 dtype 决定输出格式：
  - `torch.bfloat16`：形状 `(num_tokens, hidden_size)`。
  - `torch.float8_e4m3fn`：FP8 E4M3 输出，形状 `(num_tokens, hidden_size)`，需 SM89+。
  - `torch.float4_e2m1fn_x2`：NVFP4 分块缩放输出，形状 `(num_tokens, hidden_size // 2)`，
    需 SM100+ 且 `hidden_size` 能被 16 整除。
- `block_scale`（`Optional[torch.Tensor]`，默认 `None`）：预分配的分块 scale 输出（仅 NVFP4），
  形状 `(num_tokens, hidden_size // 16)`，dtype `torch.float8_e4m3fn`。为 `None` 且 `out` 为 NVFP4 时
  自动分配；bf16/fp8 输出时被忽略。

**返回**

- bf16/fp8：返回归一化并 SiLU 激活后的张量，形状 `(num_tokens, hidden_size)`。
- NVFP4：返回元组 `(y_fp4, block_scale)`，其中 `y_fp4` 形状 `(num_tokens, hidden_size // 2)`、dtype
  `float4_e2m1fn_x2`，`block_scale` 形状 `(num_tokens, hidden_size // 16)`、dtype `float8_e4m3fn`
  （每 16 个元素一个 E4M3 scale）。

**注意事项**

- 内核调优参数在 B200（SM100）上针对 WAN VAE decoder 规模做过 sweep 优化：`hidden_size` 属于
  `{64, 128, 160, 256, 320, 512, 640, 1024}`，`num_tokens` 属于
  `{1560, 6240, 24960, 99840, 399360}`。其他规模使用功能正确但可能达不到峰值的回退启发式；
  非 SM100 架构同样使用回退路径。

---

### `fused_qk_rmsnorm_rope`

```python
fused_qk_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float = 1e-6,
    base: float = 10000.0,
    interleave: bool = True,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    output_fp8: bool = False,
    output_quant_scale: float = 1.0,
    v_quant_scale: float = 1.0,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：融合 QK RMSNorm + 3D RoPE + V 拷贝，面向视频生成 DIT 自注意力。对 Q 和 K 应用跨头
RMSNorm，然后按 3D 空间分解（frame/height/width）应用旋转位置编码，并将 V 拷贝到连续输出缓冲区。
可选将所有输出量化为 FP8 E4M3。

**参数**

- `qkv`（`torch.Tensor`）：合并的 QKV 输入，BF16，连续。支持形状：
  - 三维：`[batch, seq_len, (num_heads_q+num_heads_k+num_heads_v)*head_dim]`。
  - 二维：`[num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]`，其中
    `num_tokens` 必须能被 `ppf*pph*ppw` 整除。
- `q_weight`（`torch.Tensor`）：Q 的 RMSNorm 权重 `[num_heads_q * head_dim]`，BF16。
- `k_weight`（`torch.Tensor`）：K 的 RMSNorm 权重 `[num_heads_k * head_dim]`，BF16。
- `ppf` / `pph` / `ppw`（`int`）：frame / height / width 维度的 patch 数量，
  `seq_len = ppf * pph * ppw`。
- `num_frame_channels` / `num_height_channels` / `num_width_channels`（`int`）：三个维度各自的 RoPE
  频率通道数（均须为偶数），之和 == `head_dim`。
- `num_heads_q` / `num_heads_k` / `num_heads_v`（`int`）：Q / K / V 头数。
- `head_dim`（`int`）：每头维度（必须为 64、128 或 256）。
- `eps`（`float`，默认 `1e-6`）：RMSNorm epsilon。
- `base`（`float`，默认 `10000.0`）：RoPE 基频率。
- `interleave`（`bool`，默认 `True`）：`True` 为交错 RoPE（非 NeoX），`False` 为 NeoX 风格。
- `factor`（`float`，默认 `1.0`）：YARN RoPE 缩放因子，`1.0` 关闭 YARN。
- `low` / `high`（`float`，默认 `0.0`）：YARN 低频 / 高频阈值。
- `attention_factor`（`float`，默认 `1.0`）：作用于 cos/sin 的 YARN attention factor；`factor` 为
  `1.0` 时必须为 `1.0`。
- `is_qk_norm`（`bool`，默认 `True`）：是否应用 RMSNorm（`False` 表示仅 RoPE，跳过归一化）。
- `output_fp8`（`bool`，默认 `False`）：是否将 Q/K/V 输出量化为 FP8 E4M3。
- `output_quant_scale`（`float`，默认 `1.0`）：Q 与 K 输出的 FP8 量化 scale。
- `v_quant_scale`（`float`，默认 `1.0`）：V 输出的 FP8 量化 scale。
- `q_out` / `k_out` / `v_out`（`Optional[torch.Tensor]`，默认 `None`）：预分配的 Q/K/V 输出张量
  （destination-passing 风格）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：`(q_out, k_out, v_out)`。若输入为三维，各自形状为
  `[batch, seq_len, num_heads_x, head_dim]`；若输入为二维，各自形状为
  `[num_tokens, num_heads_x, head_dim]`。

**注意事项**

- 支持计算能力：80、86、89、90、100、103、107、110、120、121。SM100/103（Blackwell）使用原生
  float2 打包数学（FFMA2）；SM89+ 使用原生 FP8 E4M3 转换；所有 SM100+/SM89+ 特性都有标量回退，
  故 SM80 是实际最低要求。

---

### `fused_dit_residual_layernorm_scale_shift`

```python
fused_dit_residual_layernorm_scale_shift(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    residual: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：单内核内计算融合残差 + LayerNorm + scale/shift：

```python
residual_out = residual + input          # residual 为 None 时即为 input
norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)
```

**参数**

- `input`（`torch.Tensor`）：输入张量 `[batch, num_rows, 3072]`，BF16，连续。
- `scale` / `shift`（`torch.Tensor`）：BF16，来自 `temb.chunk(6, dim=2)` 的 stride `6 * hidden_dim`。
- `residual`（`Optional[torch.Tensor]`，默认 `None`）：残差张量，BF16，连续；为 `None` 时不做残差相加。
- `scale_bias` / `shift_bias`（`Optional[torch.Tensor]`）：偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，默认 `1e-6`）：LayerNorm epsilon。
- `use_nvfp4` / `use_mxfp8`（`bool`，默认 `False`）：将归一化输出量化为 NVFP4 / MXFP8（仅 SM100+）。
- `global_scaling_factor`（`Optional[torch.Tensor]`）：NVFP4 输出的全局 scale `[1]`，FP32。
- `input_global_scaling_factor`（`Optional[torch.Tensor]`）：预量化 NVFP4 输入的 scale `[1]`，FP32。
- `residual_out` / `norm_out` / `sf_out`（`Optional[torch.Tensor]`）：预分配的残差输出、归一化输出、
  NVFP4/MXFP8 的 scale factor 输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。

**注意事项**

- 该内核仅面向 WAN 2.2 5B（`hidden_dim=3072`）。主要目标架构：SM90（Hopper）、SM100/SM103（Blackwell）。
  BF16 输出兼容 SM80+；NVFP4/MXFP8 需 SM100+。
- gate/scale/shift 张量必须使用 stride `6 * hidden_dim`（来自 `temb.chunk(6, dim=2)`）；传入连续张量
  会导致非法内存访问或错误结果，应使用 `temb.chunk(6, dim=2)[i].squeeze(2)` 获得正确 stride。

---

### `fused_dit_gate_residual_layernorm_scale_shift`

```python
fused_dit_gate_residual_layernorm_scale_shift(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：单内核内计算融合 gate + 残差 + LayerNorm + scale/shift：

```python
residual_out = residual + input * (gate + gate_bias)
norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)
```

**参数**

- `input`（`torch.Tensor`）：输入张量 `[batch, num_rows, 3072]`，BF16，连续。
- `residual`（`torch.Tensor`）：残差张量，与 input 同形状，BF16，连续。
- `gate`（`torch.Tensor`）：门控张量，BF16，stride `6 * hidden_dim`（来自 `temb.chunk(6, dim=2)`）。
- `scale` / `shift`（`torch.Tensor`）：BF16，stride 约定同 gate。
- `gate_bias` / `scale_bias` / `shift_bias`（`Optional[torch.Tensor]`）：偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，默认 `1e-6`）：LayerNorm epsilon。
- `use_nvfp4` / `use_mxfp8`（`bool`，默认 `False`）：量化为 NVFP4 / MXFP8（仅 SM100+）。
- `global_scaling_factor`（`Optional[torch.Tensor]`）：NVFP4 输出的全局 scale `[1]`，FP32。
- `input_global_scaling_factor`（`Optional[torch.Tensor]`）：预量化 NVFP4 输入的 scale `[1]`，FP32。
- `residual_out` / `norm_out` / `sf_out`（`Optional[torch.Tensor]`）：预分配的残差输出、归一化输出、
  scale factor 输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。

**注意事项**

- 仅面向 WAN 2.2 5B（`hidden_dim=3072`）。BF16 输出兼容 SM80+；NVFP4/MXFP8 需 SM100+。

---

### `fused_dit_gate_residual_layernorm_gamma_beta`

```python
fused_dit_gate_residual_layernorm_gamma_beta(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：单内核内计算融合 gate + 残差 + LayerNorm(gamma, beta)：

```python
residual_out = residual + input * (gate + gate_bias)
norm_out = LayerNorm(residual_out, gamma, beta)
```

**参数**

- `input`（`torch.Tensor`）：输入张量 `[batch, num_rows, 3072]`，BF16，连续。
- `residual`（`torch.Tensor`）：残差张量，与 input 同形状，BF16，连续。
- `gate`（`torch.Tensor`）：门控张量，BF16，stride `6 * hidden_dim`（来自 `temb.chunk(6, dim=2)`）。
- `gamma`（`torch.Tensor`）：LayerNorm 权重 `[3072]`，FP32。
- `beta`（`torch.Tensor`）：LayerNorm 偏置 `[3072]`，FP32。
- `gate_bias`（`Optional[torch.Tensor]`）：门控偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，默认 `1e-6`）：LayerNorm epsilon。
- `use_nvfp4` / `use_mxfp8`（`bool`，默认 `False`）：量化为 NVFP4 / MXFP8（仅 SM100+）。
- `global_scaling_factor`（`Optional[torch.Tensor]`）：NVFP4 输出的全局 scale `[1]`，FP32；
  `use_nvfp4=True` 时必填。
- `input_global_scaling_factor`（`Optional[torch.Tensor]`）：预量化 NVFP4 输入的 scale `[1]`，FP32。
- `residual_out` / `norm_out` / `sf_out`（`Optional[torch.Tensor]`）：预分配的残差输出、归一化输出、
  scale factor 输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。BF16 时二者均为 `[batch, num_rows, 3072]`
  BF16；NVFP4/MXFP8 时 `residual_out` 为 BF16，`norm_out` 为打包的 int32。

**注意事项**

- 仅面向 WAN 2.2 5B（`hidden_dim=3072`）。主要目标：SM90、SM100/SM103。BF16 兼容 SM80+；
  NVFP4/MXFP8 需 SM100+。

---

### `cake_rmsnorm`

```python
cake_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
)
```

**功能**：支持 autograd 的融合 BF16 RMSNorm（可选融合残差相加）。返回前向结果，反向自动调用
`cake_rmsnorm_train_backward` 的融合 `dx` 与确定性 `dw` 内核。

**参数**

- `x`（`torch.Tensor`）：BF16 `[T, H]` 输入，位于受支持的 CUDA 设备。
- `w`（`torch.Tensor`）：BF16 `[H]` RMSNorm 权重。
- `eps`（`float`，默认 `1e-6`）：方差 epsilon，内核中按 FP32 计算。
- `residual`（`Optional[torch.Tensor]`，默认 `None`）：BF16 `[T, H]` 更新量，在归一化前加到 `x` 上；
  提供时同时返回归一化输入 `h_new`。
- `deterministic`（`bool`，默认 `True`）：是否使用确定性固定顺序的权重梯度归约。仅支持 `True`；
  `False` 会在反向时被拒绝。

**返回**

- BF16 `y`，形状 `[T, H]`；当给出 `residual` 时返回 `(y, h_new)`。

**注意事项**

- 需要受支持的 CUDA 设备与 `hidden` 尺寸（见 `cake_rmsnorm_train_forward`）。

---

### `CakeRMSNormFunction`

```python
class CakeRMSNormFunction(torch.autograd.Function)
```

**功能**：autograd 封装，反向仅保存 `(x_or_h_new, w, rstd)`。

**方法**

- `forward(ctx, x, w, eps, residual=None, deterministic=True)`：返回 `y` 或 `(y, h_new)`。在残差模式下，
  `x` 与 `residual` 的梯度为同一张量（`dx + g_h_new`）。权重梯度由内核按 FP32 累加，并以 `w.dtype` 返回。
- `backward(ctx, grad_y, *grad_rest)`：调用 `cake_rmsnorm_train_backward` 计算 `dx`、`dw`，并返回
  对应输入的梯度。

**典型用法**

一般无需直接使用该类，直接调用 `cake_rmsnorm` 即可。

---

### `cake_rmsnorm_train_forward`

```python
cake_rmsnorm_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]
```

**功能**：融合 BF16 RMSNorm 前向，带 FP32 统计。启用残差时，归一化值为 `h_new = BF16(h + residual)`。

**参数**

- `x`（`torch.Tensor`）：BF16 `[T, H]` 输入，任意布局。具有连续特征维、32 字节对齐存储且行 stride
  为 32 字节倍数的行 stride 视图会被原地读取；其他布局（奇列偏移、奇行 stride、转置存储）会被物化一次。
  与 `residual` 一起给出时，`x` 是残差流 `h`。
- `w`（`torch.Tensor`）：BF16 `[H]` 权重（不连续时会被拷贝）。
- `eps`（`float`）：方差 epsilon（内核内按 FP32）。
- `residual`（`torch.Tensor`，可选）：BF16 `[T, H]` 更新量，在归一化前加到 `x` 上（布局规则同 `x`）。
  启用融合残差相加路径。
- `out`、`rstd`、`residual_out`（`torch.Tensor`，可选）：预分配输出——BF16 `[T, H]` 的 `y`、FP32 `[T]`
  的倒数 RMS，以及（仅残差时）BF16 `[T, H]` 的 `h_new`。连续对齐缓冲区会被直接写入；其他布局在
  launch 后接收一次拷贝。

**返回**

- `(y, rstd)`，或在给出 `residual` 时返回 `(y, rstd, h_new)`。`rstd` 是反向所需的 FP32 逐行
  `1 / sqrt(mean(h_new^2) + eps)`；不产生其他 `[T, H]` 临时张量。

**注意事项**

- 若当前架构/`hidden`/残差组合没有导出的 Cake RMSNorm 训练前向内核，会抛出 `ValueError`，
  并给出受支持的 hidden 尺寸。

---

### `cake_rmsnorm_train_backward`

```python
cake_rmsnorm_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    deterministic: bool = True,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

**功能**：融合 BF16 RMSNorm 反向：计算 `dx` 与 FP32 权重梯度 `dw`。

**参数**

- `g`（`torch.Tensor`）：BF16 `[T, H]`，`y` 的梯度，任意布局。具有连续特征维、16 字节对齐存储且
  行 stride 为 16 字节倍数的视图（包括 expanded 或 sliced 的 autograd 梯度）会被原地读取；其他布局
  物化一次。
- `x`（`torch.Tensor`）：被归一化的 BF16 `[T, H]` 张量（普通前向的 `x`，或残差前向的 `h_new`）；
  布局规则同 `g`。
- `w`（`torch.Tensor`）：BF16 `[H]` 权重（不连续时拷贝）。
- `rstd`（`torch.Tensor`）：前向返回的 FP32 `[T]` 倒数 RMS。
- `deterministic`（`bool`，默认 `True`）：仅导出确定性固定顺序的 `dw` 归约；`False` 被拒绝。
- `g_residual`（`torch.Tensor`，可选）：直接流经 `h_new` 的 BF16 `[T, H]` 梯度（残差前向），任意布局。
  它会被加一次，使返回的 `dx` 同时是残差相加两个输入的梯度。
- `workspace`（`torch.Tensor`，可选）：来自 `cake_rmsnorm_train_backward_workspace` 的调用方持有的
  `uint8` 缓冲区；省略时自动分配。
- `dx`、`dw`（`torch.Tensor`，可选）：预分配输出——BF16 `[T, H]` 与 FP32 `[H]`。连续对齐缓冲区直接写入；
  其他布局在 launch 后接收一次拷贝。

**返回**

- `(dx, dw)`，其中 `dw` 为 FP32；由调用方决定是否转换。对于固定的 token 数与架构，`dw` 是位级可复现的。

---

### `cake_rmsnorm_train_backward_workspace_bytes`

```python
cake_rmsnorm_train_backward_workspace_bytes(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> int
```

**功能**：返回宽度为 `hidden`、位于 `device` 的调用方持有的反向工作区所需字节数。工作区包含 FP32
逐 chunk 的 `dw` 部分和（大小按设备最大 chunk 数，独立于 `rows`），以及 `uint32` 完成计数器。
计数器在首次使用时必须为零（用 `torch.zeros` 分配或使用 `cake_rmsnorm_train_backward_workspace`）；
内核会让它们保持为零，因此同一缓冲区可跨调用、任意 token 数与 CUDA Graph 重放复用，无需主机写入。

**参数**

- `rows`（`int`）：token 数（须非负）。
- `hidden`（`int`）：隐藏维宽度。
- `device`（`torch.device`）：目标设备。
- `residual`（`bool`，默认 `False`）：是否为残差反向配置。

**返回**

- `int`：工作区总字节数。

---

### `cake_rmsnorm_train_backward_workspace`

```python
cake_rmsnorm_train_backward_workspace(
    rows: int, hidden: int, device: torch.device, *, residual: bool = False
) -> torch.Tensor
```

**功能**：分配零初始化的反向工作区（字节数见 `cake_rmsnorm_train_backward_workspace_bytes`）。

**参数**

- `rows` / `hidden` / `device` / `residual`：同 `cake_rmsnorm_train_backward_workspace_bytes`。

**返回**

- `torch.Tensor`：`uint8` 张量，零初始化，256 字节对齐以保证后续使用的对齐要求。

---

## 相关链接 / 参考

- 上游文档：`docs/api/norm.rst`
- 相关量化 API：`flashinfer.quantization.nvfp4_quantize`
- 相关 GEMM 消费者：`flashinfer.deep_gemm`

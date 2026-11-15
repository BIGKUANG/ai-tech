# 激活函数（`flashinfer.activation`）

> 源码：`flashinfer/activation.py`
> 上游文档：`docs/api/activation.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.activation` 提供一组面向 Transformer MLP 中 up/gate 层的激活算子。这些算子将输入的
最后一维分成两半，对前半部分应用激活函数，再与后半部分逐元素相乘（即 "and mul" 模式），在单次内核
调用内完成。另提供面向 Mixture-of-Experts（MoE）流水线的融合量化算子。所有算子均支持 PDL
（Programmatic Dependent Launch，程序化依赖启动），可通过 `enable_pdl` 显式控制。

## 导入

```python
import flashinfer

from flashinfer.activation import (
    silu_and_mul,
    gelu_tanh_and_mul,
    gelu_and_mul,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `silu_and_mul` | 函数 | 融合 SiLU + Mul。 |
| `gelu_tanh_and_mul` | 函数 | 融合 GeLU（tanh 近似）+ Mul。 |
| `gelu_and_mul` | 函数 | 融合 GeLU + Mul。 |
| `silu_and_mul_scaled_nvfp4_experts_quantize` | 函数 | 融合 SiLU + Mul + 逐专家 NVFP4 量化（带逐行 mask）。 |

## API 详情

### `silu_and_mul`

```python
silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：融合 SiLU 与 Mul 运算：

```python
silu(input[..., :hidden_size]) * input[..., hidden_size:]
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(..., 2 * hidden_size)`。最后一维必须为偶数
  （内核会选择能整除 `hidden_size` 的向量宽度，因此不再要求是 16 字节的倍数）。
- `out`（`torch.Tensor`，默认 `None`）：输出张量；若提供，内核将原地更新该张量。当 `out` 不为 `None`
  时，会校验其形状与 `input` 的 "and mul" 关系（`input.shape[:-1] == out.shape[:-1]` 且
  `input.shape[-1] == 2 * out.shape[-1]`）。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL；为 `None` 时按设备支持自动决定。

**返回**

- `torch.Tensor`：输出张量，形状 `(..., hidden_size)`。

---

### `gelu_tanh_and_mul`

```python
gelu_tanh_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：融合 GeLU（tanh 近似）与 Mul 运算：

```python
gelu(tanh(input[..., :hidden_size])) * input[..., hidden_size:]
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(..., 2 * hidden_size)`，最后一维必须为偶数。
- `out`（`torch.Tensor`，默认 `None`）：输出张量；若提供则原地更新，并进行形状校验。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `torch.Tensor`：输出张量，形状 `(..., hidden_size)`。

---

### `gelu_and_mul`

```python
gelu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：融合 GeLU 与 Mul 运算：

```python
gelu(input[..., :hidden_size]) * input[..., hidden_size:]
```

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(..., 2 * hidden_size)`，最后一维必须为偶数。
- `out`（`torch.Tensor`，默认 `None`）：输出张量；若提供则原地更新，并进行形状校验。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。

**返回**

- `torch.Tensor`：输出张量，形状 `(..., hidden_size)`。

---

### `silu_and_mul_scaled_nvfp4_experts_quantize`

```python
silu_and_mul_scaled_nvfp4_experts_quantize(
    a,
    mask,
    a_global_sf,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：融合 SiLU + Mul + 逐专家 NVFP4 量化，带逐行 mask。用于 Mixture-of-Experts（MoE）流水线，
将每个专家的 SiLU 门控激活与 NVFP4 量化融合，并通过 `mask` 跳过不属于当前专家的行。

**参数**

- `a`（`torch.Tensor`）：输入张量，形状 `[B, M, K]`，dtype 为 fp16/bf16。
- `mask`（`torch.Tensor`）：在量化前应用的 mask 张量（通常是专家分配 mask）。
- `a_global_sf`（`torch.Tensor`）：形状 `[1]` 的全局 scale factor，dtype `float32`。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`）：NVFP4 "4over6" scale 候选搜索。需要 fp16 / bf16 输入。
  - 省略（默认 `_UNSET`）：读取遗留的 `FLASHINFER_NVFP4_4OVER6*` 环境变量。
  - `None`：关闭 4over6，忽略环境变量。
  - `NVFP44Over6Config`：以该 recipe 开启，忽略环境变量。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`。其中 `x_q` 的逻辑形状为 `[M, K/2, B]`，dtype
  `FLOAT4_E2M1X2`（实现将 `[B, M, K/2]` 的物理布局置换为 batch 维在最后，以匹配 grouped-GEMM 期望）；
  `sf` 是逻辑形状 `[32, 4, padded_M // 128, 4, padded_K // 64, B]` 的 6D swizzled scale-factor 张量，
  按 `float8_e4m3fn` 查看。`padded_M` 将 `M` 向上取整到 128 的倍数，`padded_K` 将 `K // sf_vec_size`
  向上取整到 4 的倍数，此处 `sf_vec_size` 固定为 `16`（NVFP4），与
  `flashinfer.quantization.nvfp4_quantize` 一致。

**注意事项**

- 该算子通过 `get_fp4_quantization_module` 的 `silu_and_mul_scaled_nvfp4_experts_quantize_sm100` 实现，
  面向 SM100（Blackwell）。
- `nvfp4_4over6` 参数被追加在最后，以保持既有的位置参数构造方式继续可用。

## 相关链接 / 参考

- 上游文档：`docs/api/activation.rst`
- 相关：`flashinfer.fused_moe`、`flashinfer.quantization`

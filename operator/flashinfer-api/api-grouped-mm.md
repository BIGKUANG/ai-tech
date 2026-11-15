# 分组矩阵乘（`flashinfer.grouped_mm`）

> 源码：`flashinfer/grouped_mm/__init__.py`、`grouped_mm/core.py`、`grouped_mm/cudnn/`、`grouped_mm/cute_sm120_fp8_groupwise/`、`grouped_mm/cute_sm120_mxfp8_groupwise/`
> 上游文档：`docs/api/grouped_mm.rst`
> 适用版本：FlashInfer 0.7.2

面向 Mixture-of-Experts（MoE）层的分组矩阵乘 API，其中每个专家持有自己的权重矩阵，token
通过一个 `m_indptr` 累计计数张量路由到专家。本模块的函数镜像稠密的
`flashinfer.gemm.mm_*` API，当前分发到 cuDNN MoE 后端。`grouped_mm_bf16` 还接受
`backend="cake"`——面向 SM100 / SM103 / SM107 的实验性生成式 ragged BF16 后端，它在设备上
读取 `m_indptr` 且无需 padding。

## 导入

```python
from flashinfer.grouped_mm import (
    grouped_mm_bf16,
    grouped_mm_fp8,
    grouped_mm_mxfp8,
    grouped_mm_fp4,
    moe_gemm_fp8_nt_groupwise,
    moe_gemm_mxfp8_nt_groupwise,
)
```

## API 列表

### BF16 / FP16

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_mm_bf16` | 函数 | BF16/FP16 分组矩阵乘。 |

### FP8

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_mm_fp8` | 函数 | FP8 分组矩阵乘。 |

### MXFP8

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_mm_mxfp8` | 函数 | MXFP8 分组矩阵乘。 |

### FP4（NVFP4 / MXFP4）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_mm_fp4` | 函数 | FP4 分组矩阵乘。 |

### MXFP8 MoE GEMM（SM120 cute 后端）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `moe_gemm_mxfp8_nt_groupwise` | 函数 | MXFP8 零 padding 分组 GEMM（SM120）。 |

### FP8 MoE GEMM（SM120 cute 后端）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `moe_gemm_fp8_nt_groupwise` | 函数 | FP8 零 padding 分组 GEMM（SM120）。 |

## API 详情

### `grouped_mm_bf16`

```python
grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor
```

**功能**：BF16/FP16 数据类型的 MoE 分组矩阵乘（cuDNN MOE 后端，或实验性 Cake 后端）。在
专家间执行分组 GEMM，用于 Mixture-of-Experts 层。镜像 `flashinfer.mm_bf16` 但针对按专家
划分的输入：

```text
out[start:end] = a[start:end] × b[e]ᵀ   （对每个专家 e）
```

其中 `start, end = m_indptr[e], m_indptr[e+1]`。

**参数**

- `a`（`torch.Tensor`）：token 激活，形状 `(cum_m, k)`，bf16 或 fp16。
- `b`（`torch.Tensor`）：专家权重，形状 `(batch_size, n, k)`。
- `m_indptr`（`torch.Tensor`）：累计 token 计数，形状 `(batch_size + 1,)`，`int32`。
- `out`（`torch.Tensor`，可选）：预分配输出 `(m_out, n)`。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，`torch.bfloat16`（默认）或
  `torch.float16`、`torch.float32`。
- `backend`（`str`，默认 `"cudnn"`）：`"cudnn"`（默认），或实验性 `"cake"` 后端
  （`flashinfer.experimental.cake_moe_grouped_gemm`，SM100 / SM103 / SM107）：生成式 ragged
  BF16 程序，在设备上读取 `m_indptr`，无需 padding，仅产生 bfloat16 输出，且要求
  `n % 256 == 0`、`k % 64 == 0`。显式命名即为 opt-in，并发出一次 `ExperimentalWarning`。
- `tactic`（`int`，默认 `-1`）：cuDNN 执行计划索引。`-1` 使用启发式最优计划；非负值选择
  特定计划。Cake 后端每架构仅一个程序，仅接受 `-1`。

**返回**

- `torch.Tensor`：输出张量 `(m_out, n)`。

**示例**

```python
>>> import torch, flashinfer
>>> E, tpe, k, n = 8, 128, 4096, 2048
>>> a = torch.randn(E * tpe, k, dtype=torch.bfloat16, device="cuda")
>>> b = torch.randn(E, n, k, dtype=torch.bfloat16, device="cuda")
>>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
>>> out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr)
>>> out_cake = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")
```

---

### `grouped_mm_fp8`

```python
grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor
```

**功能**：FP8 数据类型的 MoE 分组矩阵乘（cuDNN MOE 后端）。在专家间执行分组 GEMM。镜像
`flashinfer.mm_fp8` 但针对按专家划分的输入：

```text
out[start:end] = a[start:end] × b[e]ᵀ   （对每个专家 e）
```

其中 `start, end = m_indptr[e], m_indptr[e+1]`。

**参数**

- `a`（`torch.Tensor`）：token 激活，形状 `(cum_m, k)`，e4m3 或 e5m2。
- `b`（`torch.Tensor`）：专家权重，形状 `(batch_size, n, k)`。
- `m_indptr`（`torch.Tensor`）：累计 token 计数，形状 `(batch_size + 1,)`，`int32`。
- `alpha`（`torch.Tensor`，可选）：输出缩放，形状 `(1,)`。
- `out`（`torch.Tensor`，可选）：预分配输出 `(m_out, n)`。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，`torch.bfloat16`（默认）或
  `torch.float16`、`torch.float32`。
- `backend`（`str`，默认 `"cudnn"`）：后端选择，目前仅支持 `"cudnn"`。
- `tactic`（`int`，默认 `-1`）：cuDNN 执行计划索引。

**返回**

- `torch.Tensor`：输出张量 `(m_out, n)`。

**示例**

```python
>>> import torch, flashinfer
>>> E, tpe, k, n = 8, 128, 4096, 2048
>>> a = torch.randn(E * tpe, k, dtype=torch.float8_e4m3, device="cuda")
>>> b = torch.randn(E, n, k, dtype=torch.float8_e4m3, device="cuda")
>>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
>>> out = flashinfer.grouped_mm.grouped_mm_fp8(a, b, m_indptr, alpha=torch.tensor(1.0, device="cuda"))
```

---

### `grouped_mm_mxfp8`

```python
grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor
```

**功能**：MXFP8 数据类型的 MoE 分组矩阵乘（cuDNN MOE 后端）。镜像 `flashinfer.mm_mxfp8`
但针对按专家划分的输入。

```text
out[start:end] = a[start:end] × b[e]ᵀ   （对每个专家 e）
```

其中 `start, end = m_indptr[e], m_indptr[e+1]`。

**参数**

- `a`（`torch.Tensor`）：token 激活，形状 `(cum_m, k)`，e4m3 或 e5m2。
- `b`（`torch.Tensor`）：专家权重，形状 `(batch_size, n, k)`，e4m3 或 e5m2。
- `a_descale`（`torch.Tensor`）：A 的块缩放。2D swizzled 128x4，形状 `(cum_m, k // 32)`，
  dtype `uint8`。
- `b_descale`（`torch.Tensor`）：B 的块缩放。3D swizzled 128x4，形状
  `(batch_size, n, k // 32)`，dtype `uint8`。
- `m_indptr`（`torch.Tensor`）：累计 token 计数，形状 `(batch_size + 1,)`，`int32`。
- `out`（`torch.Tensor`，可选）：预分配输出 `(m_out, n)`。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，`torch.bfloat16`（默认）或
  `torch.float16`、`torch.float32`。
- `backend`（`str`，默认 `"cudnn"`）：后端选择，目前仅支持 `"cudnn"`。
- `tactic`（`int`，默认 `-1`）：cuDNN 执行计划索引。

**返回**

- `torch.Tensor`：输出张量 `(m_out, n)`。

---

### `grouped_mm_fp4`

```python
grouped_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor
```

**功能**：FP4 数据类型的 MoE 分组矩阵乘（cuDNN MOE 后端）。镜像 `flashinfer.mm_fp4` 但针对
按专家划分的输入。

```text
out[start:end] = a[start:end] × b[e]ᵀ   （对每个专家 e）
```

其中 `start, end = m_indptr[e], m_indptr[e+1]`。

**参数**

- `a`（`torch.Tensor`）：token 激活，形状 `(cum_m, k)`，fp4 e2m1fn_x2 或 uint8。
- `b`（`torch.Tensor`）：专家权重，形状 `(batch_size, n, k)`，fp4 e2m1fn_x2 或 uint8。
- `a_descale`（`torch.Tensor`）：A 的块缩放。2D swizzled 128x4，形状
  `(cum_m, k // block_size)`，dtype `float8_e4m3fn` 或 `uint8`。
- `b_descale`（`torch.Tensor`）：B 的块缩放。3D swizzled 128x4，形状
  `(batch_size, n, k // block_size)`，dtype `float8_e4m3fn` 或 `uint8`。
- `m_indptr`（`torch.Tensor`）：累计 token 计数，形状 `(batch_size + 1,)`，`int32`。
- `alpha`（`torch.Tensor`，可选）：输出缩放，形状 `(1,)`。
- `out`（`torch.Tensor`，可选）：预分配输出 `(m_out, n)`。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，`torch.bfloat16`（默认）或
  `torch.float16`、`torch.float32`。
- `block_size`（`int`，默认 `16`）：FP4 缩放布局的块大小。`16` 选择 NVFP4（`float8_e4m3fn`
  缩放），`32` 选择 MXFP4（`uint8` 缩放）。
- `backend`（`str`，默认 `"cudnn"`）：后端选择，目前仅支持 `"cudnn"`。
- `tactic`（`int`，默认 `-1`）：cuDNN 执行计划索引。

**返回**

- `torch.Tensor`：输出张量 `(m_out, n)`。

---

### `moe_gemm_fp8_nt_groupwise`

```python
moe_gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal['MN'] = 'MN',
    backend: Literal['cute'] = 'cute',
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    is_gated: bool = False,
) -> torch.Tensor
```

**功能**：使用 groupwise float32 缩放的 FP8 输入、在零 padding 模式下执行分组 GEMM。当前
仅在 NVIDIA RTX PRO 6000 Blackwell（SM120）架构上支持。

零 padding 模式接受 token 打包输入 `a`（M 维无需按专家预 padding），缩放张量 `a_scale`
每专家 4 行 padding。组描述符为 CSR cumsum `m_indptr`。该模式针对每专家 M 较小（低至
`m_per_expert = 1`）的 decoding 优化，此时 DeepGEMM 风格的连续 padding 会浪费内存与算力。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(cum_m, k)`，dtype `torch.float8_e4m3fn`。跨专家
  token 打包；`cum_m` 为分段长度累计和。
- `b`（`torch.Tensor`）：列主序输入，形状 `(num_experts, n, k)`，dtype `torch.float8_e4m3fn`。
- `a_scale`（`torch.Tensor`）：`a` 的 float32 缩放，零 padding MN-major 布局：连续形状
  `(k_blocks, m_padded)`，其中 `k_blocks = ceil(k / 128)`、
  `m_padded = (cum_m + num_experts * 3) // 4 * 4`。专家 `i` 的缩放起始列由
  `(m_indptr[i] + 3 * i) // 4 * 4` 给出；padding 列被内核忽略。数据指针须 16 字节对齐。
- `b_scale`（`torch.Tensor`）：`b` 的 float32 缩放，连续形状
  `(num_experts, k_blocks, n_blocks)`，`n_blocks = ceil(n / 128)`（每个 `(128, 128)` 块一个缩放）。
- `m_indptr`（`torch.Tensor`）：分段长度 indptr，形状 `(num_experts + 1,)`，dtype `torch.int32`。
  `m_indptr[0] = 0`，`m_indptr[num_experts] = cum_m`。
- `scale_granularity_mnk`（`Tuple[int, int, int]`，默认 `(1, 128, 128)`）：缩放粒度。FP8
  float-scale 内核契约固定为 `(1, 128, 128)`（per-token A 缩放、(128, 128)-块 B 缩放），
  其它值抛 `ValueError`。
- `scale_major_mode`（`Literal["MN"]`，默认 `"MN"`）：缩放布局模式，目前仅支持 `"MN"`。
- `backend`（`Literal["cute"]`，默认 `"cute"`）：目前仅实现 `"cute"`。
- `out`（`torch.Tensor`，可选）：输出张量，形状 `(cum_m, n)`；未指定时创建。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，目前仅支持 `torch.bfloat16`。
- `is_gated`（`bool`，默认 `False`）：为 `True` 时运行融合 gate+up SwiGLU：`b` 沿 N 在列
  `[0, N)` 打包 up 投影、在 `[N, 2*N)` 打包 gate 投影（`b.shape[1] == 2 * out_n`），内核融合
  `SiLU(gate) * up`，输出 N 为 `b.shape[1] // 2`。

**返回**

- `torch.Tensor`：输出张量，形状 `(cum_m, n)`。

**注意事项**

- 与 `moe_gemm_mxfp8_nt_groupwise`（int32 打包 UE8M0 缩放，沿 M 与 N 逐 token）不同，FP8
  入口消费普通 float32 缩放：`a` 沿 M 逐 token，`b` 每个 `(128, 128)` 块一个缩放。
- 行主序 `(cum_m, k_blocks)` 的 per-token A 缩放必须转置并重打包为上文的 padded
  `(k_blocks, m_padded)` 布局（padding 列置零；将专家 `i` 的转置缩放复制到其 4 行对齐的
  起始列）。

---

### `moe_gemm_mxfp8_nt_groupwise`

```python
moe_gemm_mxfp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 1, 128),
    scale_major_mode: Literal['MN'] = 'MN',
    backend: Literal['cute'] = 'cute',
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    is_gated: bool = False,
) -> torch.Tensor
```

**功能**：使用 groupwise UE8M0 缩放的 MXFP8 输入、在零 padding 模式下执行分组 GEMM。当前
仅在 NVIDIA RTX PRO 6000 Blackwell（SM120）架构上支持。

零 padding 模式接受 token 打包输入 `a`（M 维无需按专家预 padding），缩放张量 `a_scale`
每专家 4 行 padding。组描述符为 CSR cumsum `m_indptr`。针对每专家 M 较小（低至 1）的
decoding 优化。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(cum_m, k)`，dtype `torch.float8_e4m3fn`。
- `b`（`torch.Tensor`）：列主序输入，形状 `(num_experts, n, k)`，dtype `torch.float8_e4m3fn`。
- `a_scale`（`torch.Tensor`）：`a` 的 int32 打包 UE8M0 缩放（每 int32 打包 4 个 UE8M0 缩放），
  形状 `(m_padded, k_align)`，其中 `m_padded = (cum_m + num_experts * 3) // 4 * 4`、
  `k_align = (k + 4 * k_granularity - 1) // (4 * k_granularity)`，dtype `torch.int32`。
- `b_scale`（`torch.Tensor`）：`b` 的 int32 打包 UE8M0 缩放，per-token 布局，形状
  `(num_experts, n, k_align)`，dtype `torch.int32`。
- `m_indptr`（`torch.Tensor`）：分段长度 indptr，形状 `(num_experts + 1,)`，dtype `torch.int32`。
  `m_indptr[0] = 0`，`m_indptr[num_experts] = cum_m`。
- `scale_granularity_mnk`（`Tuple[int, int, int]`，默认 `(1, 1, 128)`）：缩放粒度。接受
  `(1, 1, 128)`（DeepGEMM 风格生产，默认）或 `(1, 1, 32)`（OCP MXFP8）。`m_granularity` 与
  `n_granularity` 都须为 `1`；`k_granularity` 须为 `32` 或 `128`。其它值抛 `ValueError`。
- `scale_major_mode`（`Literal["MN"]`，默认 `"MN"`）：目前仅支持 `"MN"`（MN-major、per-token
  INT32 打包 UE8M0 缩放，TMA 对齐布局）。
- `backend`（`Literal["cute"]`，默认 `"cute"`）：目前仅实现 `"cute"`。
- `out`（`torch.Tensor`，可选）：输出张量，形状 `(cum_m, n)`；未指定时创建。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，目前仅支持 `torch.bfloat16`。
- `is_gated`（`bool`，默认 `False`）：为 `True` 时运行融合 gate+up SwiGLU（同 FP8 变体）。

**返回**

- `torch.Tensor`：输出张量，形状 `(cum_m, n)`。

**注意事项**

- MXFP8 对 K 轴块使用大小 32（OCP 规范）或 128（DeepGEMM 约定）的 UE8M0 缩放。`a_scale`
  与 `b_scale` 都须以 per-token 布局提供：沿 M（对 `a`）或 N（对 `b`）每行一个 UE8M0 缩放，
  沿 K 轴块每 int32 打包 4 个缩放。
- 若调用方从 2D `(k_granularity, k_granularity)` 块量化 `b_scale` 入手，须在调用前广播为
  per-token 形状 `(num_experts, n, k_align)`（每个 N 行一个缩放）。

## 相关链接 / 参考

- `flashinfer.gemm`：稠密 `mm_*` GEMM，本模块的分组变体与其镜像（见 `api-gemm.md`）。
- `flashinfer.experimental.cake_moe_grouped_gemm`：`grouped_mm_bf16(backend="cake")` 的
  实验性生成式后端。

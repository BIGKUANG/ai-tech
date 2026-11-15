# CuTe-DSL 内核（`flashinfer.cute_dsl`）

> 源码：`flashinfer/cute_dsl/__init__.py`、`availability.py`、`rmsnorm_fp4quant.py`、
> `add_rmsnorm_fp4quant.py`、`attention/wrappers/*.py`、`sparse/*.py`、
> `flashinfer/fused_moe/cute_dsl/*.py`
> 上游文档：`docs/api/cute_dsl.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.cute_dsl` 收集用 NVIDIA CuTe-DSL 实现的选定 FlashInfer 内核。这些符号仅在安装了
`nvidia-cutlass-dsl` 包且主机拥有受支持的 NVIDIA GPU 时可用；模块用 `is_cute_dsl_available()`
守卫其导入。

> **注意**：部分 GEMM 符号（`grouped_gemm_nt_masked`、
> `Sm100BlockScaledPersistentDenseGemmKernel`、`create_scale_factor_tensor`）过去位于
> `flashinfer.cute_dsl`，现在仍为向后兼容而再导出，但它们的规范位置是 `flashinfer.gemm`。
> 新代码应从 `flashinfer.gemm` 导入。这些符号未列于本 rst 的 autosummary，故不在此展开。

## 导入

```python
from flashinfer.cute_dsl import (
    is_cute_dsl_available,
    rmsnorm_fp4quant,
    add_rmsnorm_fp4quant,
    RMSNormFP4QuantKernel,
    AddRMSNormFP4QuantKernel,
)
from flashinfer.cute_dsl.attention.wrappers.batch_mla import BatchMLADecodeCuteDSLWrapper
from flashinfer.cute_dsl.attention.wrappers.batch_prefill import BatchPrefillCuteDSLWrapper
from flashinfer.cute_dsl.attention.wrappers.batch_decode import (
    BatchDecodeCuteDSLWrapper,
    BatchDecodePagedCuteDSLWrapper,
)
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
    bsa_attn_sm120_blk64_fwd,
    bsa_attn_sm120_blk64_sage_fwd,
)
from flashinfer.cute_dsl.attention.wrappers.batch_hca import cute_dsl_hca_decode
from flashinfer.fused_moe.cute_dsl import (
    cute_dsl_fused_moe_bf16,
    cute_dsl_fused_moe_nvfp4,
    cute_dsl_fused_moe_mxfp8_mxfp4,
    b12x_fused_moe,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `is_cute_dsl_available` | 函数 | 探测 CuTe DSL 栈是否可导入 |
| `rmsnorm_fp4quant` | 函数 | 融合 RMSNorm + FP4 量化 |
| `add_rmsnorm_fp4quant` | 函数 | 融合 Add + RMSNorm + FP4 量化 |
| `RMSNormFP4QuantKernel` | 类 | RMSNorm + FP4 量化内核 |
| `AddRMSNormFP4QuantKernel` | 类 | Add + RMSNorm + FP4 量化内核 |
| `BatchMLADecodeCuteDSLWrapper` | 类 | 模块化 MLA decode 的 PyTorch 封装 |
| `BatchPrefillCuteDSLWrapper` | 类 | ragged/paged KV batch prefill 封装 |
| `BatchDecodeCuteDSLWrapper` | 类 | ragged-KV GQA decode 封装 |
| `BatchDecodePagedCuteDSLWrapper` | 类 | paged-KV GQA decode 封装 |
| `bsa_attn_sm100_blk128_fwd` | 函数 | blk128 块稀疏注意力前向（SM100/103/107） |
| `bsa_attn_sm100_blk64_fwd` | 函数 | blk64 块稀疏注意力前向（SM100/103） |
| `bsa_attn_sm120_blk64_fwd` | 函数 | sm120 blk64 块稀疏注意力前向（SM120/121） |
| `bsa_attn_sm120_blk64_sage_fwd` | 函数 | 预量化 SM120 Sage 块稀疏注意力 |
| `cute_dsl_hca_decode` | 函数 | FP8 DeepSeek-V4 HCA decode（SM100/SM103） |
| `cute_dsl_fused_moe_bf16` | 函数 | SM90 CuTe-DSL 融合 MoE（BF16/FP16） |
| `cute_dsl_fused_moe_nvfp4` | 函数 | NVFP4 融合 MoE（已废弃） |
| `cute_dsl_fused_moe_mxfp8_mxfp4` | 函数 | MXFP8 激活 + MXFP4 权重融合 MoE（已废弃） |
| `b12x_fused_moe` | 函数 | SM120/SM121 b12x CuTe-DSL 融合 MoE |

## API 详情

### `is_cute_dsl_available`

```python
is_cute_dsl_available() -> bool
```

**功能**：当可选的 CuTe DSL 栈可导入时返回 `True`。通过 `importlib.util.find_spec` 探测
`cutlass` 与 `cutlass.cute`。高层封装据此决定是否分派到 CuTe-DSL 后端或回退到纯 CUDA 实现。

**参数**

- 无。

**返回**

- `bool`：当 `cutlass` 与 `cutlass.cute` 在当前 Python 环境中均可导入时为 `True`。

**示例**

```python
from flashinfer.cute_dsl import is_cute_dsl_available

if is_cute_dsl_available():
    ...
```

---

### `rmsnorm_fp4quant`

```python
rmsnorm_fp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    block_size: int = 16,
    scale_format: str | None = None,
    is_sf_swizzled_layout: bool = False,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用 CuTe-DSL 的融合 RMS 归一化 + FP4 量化。计算 `y = RMSNorm(input) * weight`，
可选应用全局缩放（`y = y / global_scale`），然后把 `y` 量化为 FP4。

**参数**

- `input`（`torch.Tensor`）：输入，形状 `(batch_size, hidden_size)` 或
  `(batch_size, seq_len, hidden_size)`，须为 `torch.float16` 或 `torch.bfloat16`。
- `weight`（`torch.Tensor`）：RMSNorm 权重，形状 `(hidden_size,)`，dtype 须与 input 相同。
- `y_fp4`（`torch.Tensor`，可选）：量化值输出，FP4_E2M1 格式，dtype 为
  `torch.float4_e2m1fn_x2`，形状 `(batch_size, hidden_size // 2)` 或匹配 3D 输入；为 `None`
  时自动申请。
- `block_scale`（`torch.Tensor`，可选）：逐块 scale factor 输出。`is_sf_swizzled_layout=False`
  （默认）时为 row-major，形状 `(batch_size, hidden_size // block_size)` 或匹配 3D 输入；
  为 `True` 时为面向张量核访问的 swizzled 布局，形状展平为
  `(batch_size * hidden_size // block_size,)`，采用 128x4 tile，排列为
  `[m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]`。dtype 应为
  `torch.float8_e4m3fn`（E4M3）或 `torch.uint8`（UE8M0）；为 `None` 时自动申请。
- `global_scale`（`torch.Tensor`，可选）：全局 scale，形状 `(1,)`，dtype `torch.float32`。
  提供时在量化前把 RMSNorm 输出除以该值：`y = rmsnorm(x, w) / global_scale`。用于 NVFP4 格式，
  其中预计算的全局 scale 把逐块 scale 提升到最优动态范围。为 `None` 时不做全局缩放
  （等价 `global_scale=1.0`）。
- `eps`（`float`，默认 `1e-6`）：RMSNorm 的数值稳定 epsilon。
- `block_size`（`int`，默认 `16`）：每个量化块的元素数。`16` 为 NVFP4（E4M3 scale factor）；
  `32` 为 MXFP4（UE8M0 scale factor）。
- `scale_format`（`str`，可选）：scale factor 格式，`"e4m3"` 或 `"ue8m0"`；为 `None` 时按
  `block_size` 自动选择（`16` → `"e4m3"`，`32` → `"ue8m0"`）。
- `is_sf_swizzled_layout`（`bool`，默认 `False`）：为 `True` 时以面向张量核 GEMM 的 swizzled
  布局输出 scale factor。swizzle 采用 128x4 tile，模式为
  `[m_tile_idx * k_tiles * 512 + k_tile_idx * 512 + outer_m * 16 + inner_m * 4 + inner_k]`。
- `enable_pdl`（`bool`，可选）：是否以 Programmatic Dependent Launch（PDL）启动。为 `None`
  （默认）或 `True` 时，仅当当前设备支持 PDL 才启用（经 `device_support_pdl` 探测）；传 `False`
  强制禁用。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(y_fp4, block_scale)`。`y_fp4` 为以 uint8 packed 的
  量化 FP4 值；`block_scale` 为逐块 scale factor。

**注意事项**

- 需要 SM100+（Blackwell）以使用 FP4 量化 PTX 内建。
- `hidden_size` 必须能被 `block_size` 整除且 `>= 64`。
- block_size=16（NVFP4）使用 E4M3 scale factor（最大值 448.0）；block_size=32（MXFP4）使用
  UE8M0 scale factor（2 的幂次 scale）。
- FP4 E2M1 格式的最大可表示值为 6.0。

**示例**

```python
import torch
from flashinfer.cute_dsl import rmsnorm_fp4quant

x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
y_fp4, block_scale = rmsnorm_fp4quant(x, w, block_size=16)
```

---

### `add_rmsnorm_fp4quant`

```python
add_rmsnorm_fp4quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    block_size: int = 16,
    scale_format: str | None = None,
    is_sf_swizzled_layout: bool = False,
    output_both_sf_layouts: bool = False,
    block_scale_unswizzled: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
    y_out: torch.Tensor | None = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
```

**功能**：使用 CuTe-DSL 的融合 Add + RMS 归一化 + FP4 量化。计算：

1. `residual = residual + input`（原地更新）；
2. `y = RMSNorm(residual) * weight`；
3. 可选应用全局缩放（`y = y / global_scale`）；
4. 把 `y` 量化为 FP4。

`residual` 张量被原地修改为融合后的值。

**参数**

- `input`（`torch.Tensor`）：输入，形状 `(batch_size, hidden_size)` 或
  `(batch_size, seq_len, hidden_size)`，须为 `torch.float16` 或 `torch.bfloat16`，只读。
- `residual`（`torch.Tensor`）：残差，形状与 dtype 须与 `input` 相同。**原地修改**为 `residual + input`。
- `weight`（`torch.Tensor`）：RMSNorm 权重，形状 `(hidden_size,)`，dtype 须与 input 相同。
- `y_fp4`（`torch.Tensor`，可选）：量化值输出，dtype `torch.float4_e2m1fn_x2`，形状
  `(batch_size, hidden_size // 2)` 或匹配 3D 输入；为 `None` 时自动申请。
- `block_scale`（`torch.Tensor`，可选）：逐块 scale factor 输出。当
  `is_sf_swizzled_layout=False` 且 `output_both_sf_layouts=False` 时为 row-major，形状
  `(batch_size, hidden_size // block_size)` 或匹配 3D 输入；当 `is_sf_swizzled_layout=True`
  或 `output_both_sf_layouts=True` 时为 swizzled，展平为
  `(batch_size * hidden_size // block_size,)`，采用 128x4 tile。dtype 应为
  `torch.float8_e4m3fn`（E4M3）或 `torch.uint8`（UE8M0）；为 `None` 时自动申请。
- `global_scale`（`torch.Tensor`，可选）：全局 scale，形状 `(1,)`，dtype `torch.float32`。提供时
  在量化前把 RMSNorm 输出除以该值：`y = rmsnorm(h, w) / global_scale`，其中 `h = input + residual`。
  用于 NVFP4。为 `None` 时等价 `global_scale=1.0`。
- `eps`（`float`，默认 `1e-6`）：RMSNorm 的数值稳定 epsilon。
- `block_size`（`int`，默认 `16`）：每量化块元素数，`16`（NVFP4/E4M3）或 `32`（MXFP4/UE8M0）。
- `scale_format`（`str`，可选）：`"e4m3"` 或 `"ue8m0"`；为 `None` 时按 `block_size` 自动选择。
- `is_sf_swizzled_layout`（`bool`，默认 `False`）：为 `True` 时使用 swizzled scale factor 布局，
  模式同 `rmsnorm_fp4quant`。`output_both_sf_layouts=True` 时该参数被忽略。
- `output_both_sf_layouts`（`bool`，默认 `False`）：为 `True` 时同时返回 swizzled 与 unswizzled
  scale factor：`block_scale` 为 swizzled、`block_scale_unswizzled` 为 row-major。覆盖
  `is_sf_swizzled_layout`。
- `block_scale_unswizzled`（`torch.Tensor`，可选）：unswizzled（row-major）逐块 scale factor 输出，
  仅在 `output_both_sf_layouts=True` 时使用，形状 `(batch_size, hidden_size // block_size)` 或
  匹配 3D 输入；为 `None` 且启用时自动申请。
- `enable_pdl`（`bool`，可选）：是否以 PDL 启动；语义同 `rmsnorm_fp4quant`。
- `y_out`（`torch.Tensor`，可选）：可选的、量化前 normed 值
  `y = RMSNorm(input + residual) * weight` 的输出（输入 dtype，fp16/bf16），形状
  `(batch_size, hidden_size)` 或匹配 3D 输入、连续。**原地写入**；返回值不变（元组长度与无
  `y_out` 时相同）。为 `None`（默认）时不物化 normed 值，生成的内核与基线一致。

**返回**

- `output_both_sf_layouts=False` 时：`(y_fp4, block_scale)`，其中 `y_fp4` 为 uint8 packed 的
  FP4 值，`block_scale` 为逐块 scale factor（依 `is_sf_swizzled_layout` 为 swizzled 或 row-major）。
- `output_both_sf_layouts=True` 时：`(y_fp4, block_scale, block_scale_unswizzled)`，其中
  `block_scale` 为 swizzled、`block_scale_unswizzled` 为 row-major。

**注意事项**

- 需要 SM100+（Blackwell）以使用 FP4 量化 PTX 内建。
- `hidden_size` 必须能被 `block_size` 整除且 `>= 64`。

**示例**

```python
import torch
from flashinfer.cute_dsl import add_rmsnorm_fp4quant

x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
res = torch.zeros_like(x)
w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
y_fp4, block_scale = add_rmsnorm_fp4quant(x, res, w, block_size=16)
```

---

### `RMSNormFP4QuantKernel`

```python
class RMSNormFP4QuantKernel:
    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,
    )
```

**功能**：融合 RMSNorm + FP4 量化内核。关键优化包括 Half2/BFloat2 SIMD 求 max-abs、无分支的
scale clamp（`fmin_f32`）、针对大 H 维度的 cluster 同步，以及直接的 128-bit 向量化全局加载。

**`__init__` 参数**

- `dtype`（`cutlass.Numeric`）：元素数据类型。
- `H`（`int`）：hidden 维度。
- `block_size`（`int`）：量化块大小，须为 `16` 或 `32`。
- `output_swizzled`（`bool`）：是否输出 swizzled 布局。
- `is_fp16`（`bool`）：输入是否为 fp16。
- `sm_version`（`int`，可选）：SM 版本；为 `None` 时调用 `get_sm_version()`。
- `scale_format`（`str`，可选）：`"e4m3"` 或 `"ue8m0"`；为 `None` 时按 `block_size`
  自动选择（`32` → `"ue8m0"`，否则 `"e4m3"`）。

**`__init__` 注意事项**

- `block_size` 须为 `16` 或 `32`，`scale_format` 须为 `"e4m3"` 或 `"ue8m0"`，否则 `assert` 失败。
- 依据 `H` 与设备 shared memory 动态选择 cluster size。

**关键方法**

- `__call__(mX, mW, mY, mS, mGlobalScale, M, eps, enable_pdl, stream)`：宿主启动函数，直接以
  TVM-FFI 传入张量。`mX` 形状 `(M, H)` row-major；`mW` 形状 `(H,)`；`mY` 形状 `(M, H // 2)`
  row-major（packed）；`mS` 形状取决于 swizzle 模式；`mGlobalScale` 形状 `(1,)`，float32。
- `tensor_api(x, w, y, s, global_scale, M, eps) -> None`：运行时 API，直接以 TVM-FFI 传入 torch
  张量；内部把 `y` 视图为 uint8，并在 swizzled 模式下展平 scale 张量。

**示例**

```python
import torch
from flashinfer.cute_dsl import RMSNormFP4QuantKernel

kernel = RMSNormFP4QuantKernel(cutlass.BFloat16, 1024, 16, False, False)
```

---

### `AddRMSNormFP4QuantKernel`

```python
class AddRMSNormFP4QuantKernel:
    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,
        output_both_sf_layouts: bool = False,
        output_norm: bool = False,
        launch_config: tuple[int, int, int] | None = None,
    )
```

**功能**：融合 Add + RMSNorm + FP4 量化内核。计算：`residual = input + residual`（原地更新）、
`y = RMSNorm(residual) * weight`、把 `y` 量化为 FP4。残差张量被原地修改。支持 NVFP4
（block_size=16）与 MXFP4（block_size=32）。

**`__init__` 参数**

- `dtype`（`cutlass.Numeric`）、`H`（`int`）、`block_size`（`int`）、`output_swizzled`（`bool`）、
  `is_fp16`（`bool`）：含义同 `RMSNormFP4QuantKernel`。
- `sm_version`（`int`，可选）：SM 版本；为 `None` 时调用 `get_sm_version()`。
- `scale_format`（`str`，可选）：`"e4m3"` 或 `"ue8m0"`；为 `None` 时按 `block_size` 自动选择。
- `output_both_sf_layouts`（`bool`，默认 `False`）：是否同时输出两种 scale factor 布局。
- `output_norm`（`bool`，默认 `False`）：是否输出 normed（量化前）值。
- `launch_config`（`tuple[int, int, int]`，可选）：`(cluster_n, threads_per_row, num_threads)`；
  为 `None` 时按 `H` 与设备自动推导。

**关键方法**

- `__call__(mX, mR, mW, mY, mS, mS_unswizzled, mYNorm, mGlobalScale, M, eps, enable_pdl, stream)`：
  宿主启动函数。`mR` 原地修改为 `input + residual`；`mS_unswizzled` 仅在
  `output_both_sf_layouts=True` 时使用；`mYNorm` 仅在 `output_norm=True` 时写入（否则可传任意
  dummy 张量）。
- `tensor_api(x, r, w, y, s, s_unswizzled, y_norm, global_scale, M, eps) -> None`：运行时 API，
  直接以 TVM-FFI 传入 torch 张量。

**示例**

```python
import torch
from flashinfer.cute_dsl import AddRMSNormFP4QuantKernel

kernel = AddRMSNormFP4QuantKernel(cutlass.BFloat16, 1024, 16, False, False)
```

---

### `BatchMLADecodeCuteDSLWrapper`

```python
class BatchMLADecodeCuteDSLWrapper:
    def __init__(self, workspace_buffer: torch.Tensor) -> None
```

**功能**：模块化 MLA decode 内核的 PyTorch 封装。典型用法：

```python
wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
wrapper.plan(
    kv_lora_rank=512, qk_rope_head_dim=64, num_heads=128,
    page_size=64, q_dtype=torch.bfloat16,
)
out = wrapper.run(query, kv_cache, block_tables, seq_lens, max_seq_len,
                  softmax_scale=0.125)
```

**`__init__` 参数**

- `workspace_buffer`（`torch.Tensor`）：目标 CUDA 设备上的预分配工作区，dtype 须为
  `torch.int8` 或 `torch.uint8`；其大小决定该封装在无需重新申请时可处理的最大 batch。

**关键方法**

- `plan(kv_lora_rank=512, qk_rope_head_dim=64, num_heads=128, page_size=1,
  q_dtype=torch.bfloat16, out_dtype=None, is_var_seq=True, enable_pdl=None, variant=None) -> None`：
  为给定配置编译（或取缓存）MLA decode 内核。
  - `kv_lora_rank`（`int`，默认 `512`）：latent 维度。
  - `qk_rope_head_dim`（`int`，默认 `64`）：RoPE 维度。
  - `num_heads`（`int`，默认 `128`）：注意力 head 数（DeepSeek-V3 通常为 128）。
  - `page_size`（`int`，默认 `1`）：KV cache page 大小。
  - `q_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：query/KV 数据类型。
  - `out_dtype`（`torch.dtype`，可选）：输出类型，默认为 `q_dtype`（fp8 输入时默认 bf16）。
  - `is_var_seq`（`bool`，默认 `True`）：batch 内序列长度是否变化。
  - `enable_pdl`（`bool`，可选）：是否启用 PDL；为 `None` 时自动探测。
  - `variant`（`AttentionVariant`，可选）：注意力变体（ALiBi、SoftCapping、AttentionWithSink
    等）；`None` 使用标准 softmax。MLA decode 不支持 `logits_transform`。
- `run(q, kv_cache, block_tables, seq_lens, max_seq_len, softmax_scale, output_scale=1.0,
  out=None) -> torch.Tensor`：运行 MLA decode。
  - `q`（`torch.Tensor`）：`[B, q_len, H, D_qk]`，`D_qk = kv_lora_rank + qk_rope_head_dim`。
  - `kv_cache`（`torch.Tensor`）：`[num_pages, page_size, D_total]`（3D）或
    `[num_pages, 1, page_size, D_total]`（4D）。
  - `block_tables`（`torch.Tensor`）：`[B, max_pages]` page table 索引。
  - `seq_lens`（`torch.Tensor`）：`[B]` 每个请求的 KV 序列长度。
  - `max_seq_len`（`int`）：batch 中的最大序列长度。
  - `softmax_scale`（`float`）：softmax 前对 QK^T 的缩放。
  - `output_scale`（`float`，默认 `1.0`）：施加于输出的缩放。
  - `out`（`torch.Tensor`，可选）：预分配输出 `[B, q_len, H, kv_lora_rank]`。
  - 返回：输出张量 `[B, q_len, H, kv_lora_rank]`。

**注意事项**

- 必须先调用 `plan()` 再调用 `run()`，否则抛 `RuntimeError`。

**示例**

```python
wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
wrapper.plan(kv_lora_rank=512, qk_rope_head_dim=64, num_heads=128, page_size=64)
out = wrapper.run(q, kv_cache, block_tables, seq_lens, max_seq_len, softmax_scale=0.125)
```

---

### `BatchPrefillCuteDSLWrapper`

```python
class BatchPrefillCuteDSLWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None
```

**功能**：CuTe-DSL ragged-KV batch prefill 内核的 PyTorch 封装。暴露与
`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` 兼容的 `plan` + `run` API，
但底层编译的是 CuTe-DSL 内核而非 C++ FA2/FA3 路径。

**`__init__` 参数**

- `float_workspace_buffer`（`torch.Tensor`）：目标 CUDA 设备上的预分配工作区。命名是为了与
  `BatchPrefillWithRaggedKVCacheWrapper` 保持 API 一致，调用方通常传 `torch.uint8`。
  CuTe-DSL 内核本身不消费该 buffer，但保留它以镜像父类 API。
- `use_cuda_graph`（`bool`，默认 `False`）：该封装是否会在 CUDA Graph 捕获中使用。

**关键方法**

- `plan(qo_indptr, kv_indptr=None, num_qo_heads=None, num_kv_heads=None, head_dim_qk=None,
  head_dim_vo=None, causal=True, sm_scale=None, q_data_type=torch.float16,
  kv_data_type=torch.float16, window_left=-1, variant=None, window_right=-1,
  page_size=None, paged_kv_indptr=None, paged_kv_indices=None,
  paged_kv_last_page_len=None, kv_layout="NHD") -> None`：为给定配置编译 FMHA prefill 内核。
  - `qo_indptr`：累计 query 序列长度，形状 `[batch_size + 1]`。
  - `kv_indptr`：ragged KV 的累计 KV 序列长度，形状 `[batch_size + 1]`；规划 paged KV 时必须为
    `None`（改传 `page_size` + paged 三元组）。
  - `num_qo_heads`/`num_kv_heads`：query/KV head 数（后者须整除前者）。
  - `head_dim_qk`：query/key 的 head 维度。
  - `head_dim_vo`：value/输出的 head 维度；非 `None` 时须等于 `head_dim_qk`。
  - `causal`（`bool`，默认 `True`）：是否应用 causal 掩码。
  - `sm_scale`：softmax scale；为 `None` 时默认 `1/sqrt(head_dim_qk)`。
  - `q_data_type`/`kv_data_type`：query / key-value 数据类型（float16、bfloat16 或 float8_e4m3fn）。
  - `window_left`（`int`，默认 `-1`）：最大回看距离；`-1` 无界，与 `causal` 可组合成 causal 滑动窗口。
  - `variant`：注意力变体；`None` 使用标准 softmax。
  - `window_right`（`int`，默认 `-1`）：非 causal 掩码的最大前瞻距离；`-1` 无界，与 `causal=True` 互斥。
  - `page_size`：每 KV page 的 token 数；`None`（默认）规划 ragged KV，int 规划 paged KV 并要求
    下面的 `paged_kv_*` 三元组，且须整除 128-token KV tile（16/32/64/128；接受 8）。
  - `paged_kv_indptr`（`[batch_size + 1]`，int32）：每个 batch 项的 page 计数前缀和。
  - `paged_kv_indices`（`[paged_kv_indptr[-1]]`，int32）：扁平物理 page id。
  - `paged_kv_last_page_len`（`[batch_size]`）：每项最后一个 page 的有效条目数。
  - `kv_layout`（`str`，默认 `"NHD"`）：`"NHD"` 或 `"HND"` page 布局；仅 paged 使用，ragged 路径保持 NHD。
- `run(q, k, v, out=None, return_lse=False, lse=None)`：运行 ragged prefill 注意力（需要 ragged
  `plan()`）。`q`/`k`/`v` 形状 `[total_*_len, num_heads, head_dim]`；`v` 可用不同于 q/k 的 dtype；
  `return_lse=True` 时额外返回 `[total_q_len, num_heads]`、float32、log2 域的逐行 LSE（仅标准
  注意力）。返回输出张量，或在 `return_lse=True` 时返回 `(output, lse)`。
- `run_paged(q, paged_kv_cache, out=None, return_lse=False, lse=None)`：运行 paged-KV prefill
  注意力（需要 paged `plan()`）。`paged_kv_cache` 可以是合并 cache `[num_pages, 2, page_size,
  num_kv_heads, head_dim]`（NHD）/ `[num_pages, 2, num_kv_heads, page_size, head_dim]`（HND），
  或 `(k_cache, v_cache)` 元组（各自 `[num_pages, page_size, num_kv_heads, head_dim]` (NHD) /
  `[num_pages, num_kv_heads, page_size, head_dim]` (HND)）。元组形式可携带 dtype 不同于 K 的 V
  cache（混合 dtype PV 路径）；K 必须始终匹配 plan。

**注意事项**

- 每个 batch 项的 KV 长度须 `>= 1`，零长度 KV 项不受支持，否则抛 `ValueError`。
- 每个被引用的 page 必须处处有限（含超过 `last_page_len` 的部分），内核会整块读取并依赖掩码。

**示例**

```python
wrapper = BatchPrefillCuteDSLWrapper(workspace_buffer)
wrapper.plan(qo_indptr, kv_indptr, num_qo_heads=32, num_kv_heads=8, head_dim_qk=128)
out = wrapper.run(q, k, v)
```

---

### `BatchDecodeCuteDSLWrapper`

```python
class BatchDecodeCuteDSLWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None
```

**功能**：ragged-KV CuTe DSL GQA decode 内核的 PyTorch 封装。假设连续的（非 paged）KV cache，
且所有 batch 具有相同的 KV 序列长度。对变长序列的 paged KV 请改用
`BatchDecodePagedCuteDSLWrapper`。

**`__init__` 参数**

- `float_workspace_buffer`（`torch.Tensor`）：预分配的 float32 工作区，供底层 CuTe DSL 内核做
  split-K 部分归约。封装不调整其大小，调用方需为最大预期 batch 分配合适大小。buffer 所在设备
  决定后续内核启动的设备。
- `use_cuda_graph`（`bool`，默认 `False`）：为 `True` 时把封装准备为 CUDA Graph 捕获安全
  （无 host sync、工作区指针稳定）。

**关键方法**

- `plan(batch_size, max_kv_len, num_qo_heads, num_kv_heads, head_dim, *deprecated_positional_args,
  **kwargs) -> None`：为规划配置编译 ragged-KV decode 内核。
  - `batch_size`（`int`）：用于自动调优 `kv_splits` 的代表性 batch；运行时 batch 可不同。
  - `max_kv_len`（`int`）：用于调优 `kv_splits` 的代表性 KV 长度。
  - `num_qo_heads, num_kv_heads, head_dim`（`int`）：GQA 配置；`num_qo_heads` 须为 `num_kv_heads`
    的倍数，`head_dim` 须为 64 的倍数。
  - `q_data_type, kv_data_type, o_data_type`（`torch.dtype`）：Q/K/V/O 类型；Q 与 KV 须匹配，
    `o_data_type` 默认 `q_data_type`（fp8 输入时为 float16）。
  - `sm_scale`（`float`，可选）：softmax scale，默认 `1 / sqrt(head_dim)`。
  - `kv_splits`（`int`，可选）：每序列的线程块数（flash decoding）；`None` 时按规划形状与设备
    SM 数自动调优。
  - `reduction`（`str`，默认 `"auto"`）：`"kernel"`、`"atomic"`、`"none"` 或 `"auto"`。
    `"none"` 完全跳过 flash-decoding（无归约内核、无 cluster atomic）并要求 `kv_splits == 1`；
    atomic 归约更快但要求 `kv_splits ∈ {1, 2, 4, 8, 16}` 且输出 dtype ∈ {float32, float16,
    bfloat16}；`"auto"` 在 `kv_splits == 1` 时取 `"none"`，兼容 dtype 与小 `kv_splits` 时取
    `"atomic"`，否则取 `"kernel"`。
  - `q_len_per_req`（`int`）：每请求预测 token 数（普通 decode 为 1，投机解码 > 1）。
  - `window_left`（`int`）：滑动窗口左界；`None` 关闭左界。
  - `window_right`（`int`）：滑动窗口右界；`None` 关闭右界。
  - `head_dim` 之后的可选参数可按位置传入以向后兼容，但该调用约定已废弃，将来移除；请改用关键字。
    `window_left` 与 `window_right` 为关键字限定；旧参数 `is_causal` 已废弃。
- `run(q, k, v, *deprecated_positional_args, **kwargs) -> torch.Tensor`：运行 ragged-KV GQA decode。
  - `q`（`torch.Tensor`）：形状 `[batch_size, q_len_per_req, num_qo_heads, head_dim]`。
    `q_len_per_req` 运行时从 `q.shape[1]` 读取，不必与 `plan` 时一致（后者只是编译期 tile 提示）。
  - `k, v`（`torch.Tensor`）：形状 `[batch_size, seq_len, num_kv_heads, head_dim]`，二者 seq_len 须一致。
  - `out`（`torch.Tensor`，可选）：预分配输出；atomic 归约时须在传入前零初始化。
  - `sm_scale`（`float`，可选）：每次调用覆盖 plan 时设置的 softmax scale。
  - `o_scale`（`float`，默认 `1.0`）：写出前施加于最终 O 的输出缩放（在归约 epilogue 中免费融合）。
  - `sinks`（`torch.Tensor`，可选）：query 设备上连续 float32 的逐 head attention sink logits，
    形状 `(num_qo_heads,)`；提供时 sink logit 计入 softmax 分母且不获得输出值贡献。
  - `lse`（`torch.Tensor`，可选）：预分配 float32 buffer，形状
    `(batch_size, q_len_per_req, num_qo_heads)`，接收 LSE（log2 base，与 flashinfer 约定一致）。
    为 `None`（默认）时内核完全跳过 LSE 写入；否则首次使用时惰性编译 log2 LSE 变体。
  - `enable_pdl`（`bool`，默认 `True`）：是否以 PDL 启动；设备不支持时设 `False`。

**示例**

```python
wrapper = BatchDecodeCuteDSLWrapper(workspace_buffer)
wrapper.plan(batch_size=32, max_kv_len=8192, num_qo_heads=32, num_kv_heads=8, head_dim=128)
out = wrapper.run(q, k, v)
```

---

### `BatchDecodePagedCuteDSLWrapper`

```python
class BatchDecodePagedCuteDSLWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None
```

**功能**：paged CuTe DSL GQA decode 内核的 PyTorch 封装。

**`__init__` 参数**

- `float_workspace_buffer`（`torch.Tensor`）：预分配 float32 工作区，供底层内核做 split-K
  部分归约。封装不调整其大小，调用方需为最大预期 batch 与 page table 分配。设备决定后续内核
  启动的设备。
- `use_cuda_graph`（`bool`，默认 `False`）：为 `True` 时准备为 CUDA Graph 捕获安全。

**关键方法**

- `plan(indptr, indices, seq_lens, num_qo_heads, num_kv_heads, head_dim, page_size,
  *deprecated_positional_args, **kwargs) -> None`：为给定问题规划 paged GQA decode。
  - `indptr`（`torch.Tensor`，int32，`[batch_size + 1]`）：指向 `indices` 的前缀和偏移。
  - `indices`（`torch.Tensor`，int32，`[num_pages_total]`）：扁平逐序列虚拟 page 索引。
  - `seq_lens`（`torch.Tensor`，int32，`[batch_size]`）：每序列 KV token 长度。
  - `num_qo_heads, num_kv_heads, head_dim, page_size`（`int`）：GQA + 分页配置；`page_size`
    须在 `{8, 16, 32, 64}`，`head_dim` 须为 64 的正倍数。
  - `q_data_type, kv_data_type, o_data_type`（`torch.dtype`）：Q/K/V/O 类型；`kv_data_type` 须等于
    `q_data_type`。
  - `sm_scale`（`float`，可选）：softmax scale，默认 `1 / sqrt(head_dim)`。
  - `kv_splits`（`int`，可选）：每序列线程块数；`None` 时自动调优。
  - `reduction`（`str`）：`"kernel"`（带工作区、确定性）、`"atomic"`（cluster 归约、更快但精度
    更低）、`"none"`（无 flash-decoding split-K，要求 `kv_splits == 1`）或 `"auto"`。
  - `q_len_per_req`（`int`）：每请求预测 token 数（普通 decode 为 1）。
  - `window_left`（`int`）：滑动窗口左界；`None` 关闭左界。
  - `window_right`（`int`）：滑动窗口右界；`None` 关闭右界，默认 `0`。
  - `max_kv_len`（`int`，可选）：batch 内最大 KV 序列长度，用于自动调优 `kv_splits`；显式传入
    可避免 GPU→CPU 同步。
  - `non_blocking`（`bool`）：规划期整数 buffer 的异步设备拷贝。
  - `precompile_skip_softmax_kernel`（`bool`）：为 `True` 时在 plan() 期同时编译内核的 BLASST
    skip-softmax 变体，使首次传入 `skip_softmax_threshold_scale_factor` 的 run 调用更快。
  - `page_size` 之后的可选参数可按位置传入以向后兼容，但已废弃；`window_left`/`window_right`
    为关键字限定，旧 `is_causal` 已废弃。
- `run(q, k_cache, v_cache, *deprecated_positional_args, **kwargs) -> torch.Tensor`：运行 paged
  GQA decode。
  - `q`（`torch.Tensor`）：`[batch_size * q_len_per_req, num_qo_heads, head_dim]` 或
    `[batch_size, q_len_per_req, num_qo_heads, head_dim]`；`q_len_per_req` 运行时读取。
  - `k_cache, v_cache`（`torch.Tensor`）：逻辑形状 `[num_pages, page_size, num_kv_heads, head_dim]`；
    NHD 连续布局与 HND 布局（以转置视图呈现）均可，只要 `head_dim` 最内层。
  - `out`（`torch.Tensor`，可选）：预分配输出；atomic 归约时须先零初始化。
  - `sm_scale`（`float`，可选）：每次调用覆盖 plan 的 softmax scale。
  - `o_scale`（`float`，默认 `1.0`）：写出前施加于最终 O 的输出缩放。
  - `sinks`（`torch.Tensor`，可选）：连续 float32 逐 head sink logits，形状 `(num_qo_heads,)`。
  - `skip_softmax_threshold_scale_factor`（`float`，可选）：BLASST skip-softmax 缩放因子，须
    `> 0`（设置时）；内核用它除以每个 batch 的 KV seqlen 得到逐请求有效阈值。`None`（默认）
    分派到标准内核。
  - `lse`（`torch.Tensor`，可选）：预分配 float32 buffer，形状 `(batch_size, q_len_per_req,
    num_qo_heads)` 或展平等价形式，接收 log2 域 LSE；为 `None` 时跳过 LSE 写入。
  - `enable_pdl`（`bool`，默认 `True`）：是否以 PDL 启动。

**示例**

```python
wrapper = BatchDecodePagedCuteDSLWrapper(workspace_buffer)
wrapper.plan(indptr, indices, seq_lens, num_qo_heads=32, num_kv_heads=8,
             head_dim=128, page_size=16)
out = wrapper.run(q, k_cache, v_cache)
```

---

### `bsa_attn_sm100_blk128_fwd`

```python
bsa_attn_sm100_blk128_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    allow_empty_block_nums: bool = True,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：blk128 CuTe-DSL 内核的 BSA 块稀疏注意力前向。支持 SM100 与 SM103（以及 SM107）。

**参数**

- `q`：query `(batch, seqlen_q, num_heads, head_dim)`。
- `k`：key `(batch, seqlen_k, num_heads_kv, head_dim)`。
- `v`：value `(batch, seqlen_k, num_heads_kv, head_dim_v)`。
- `q2k_block_index`：`(batch, num_heads, num_q_blocks, max_kv_blocks)`，int32。
- `block_sparse_num`（`int`）：每个 Q block 对应的 KV block 数（偶数且 `>= 2`）；提供
  `q2k_block_nums` 时忽略。
- `block_sizes`（可选，int32）：每个 KV block 的实际 token 数，形状 `(num_kv_blocks,)`；
  传 `None` 跳过 block-size 掩码（假设全满块）。
- `q2k_block_nums`（可选，int32）：每个 `(batch, head, q_block)` 的 KV block 数，形状
  `(batch, num_heads, num_q_blocks)`。
- `allow_empty_block_nums`（`bool`，默认 `True`）：允许 `q2k_block_nums` 含 0。
- `softmax_scale`（可选）：softmax scale，默认 `1/sqrt(head_dim)`。
- `pack_gqa`（可选）：是否 pack GQA heads。
- `return_lse`（`bool`，默认 `False`）：是否返回 log-sum-exp。
- `out`（可选）：预分配输出。
- `lse`（可选）：预分配 LSE。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：输出与 LSE。

**注意事项**

- `q`/`k`/`v` dtype 须为 float16 或 bfloat16 且相同；须在 CUDA 上。
- 设备架构须为 SM100/SM103/SM107，且 `num_head % num_head_kv == 0`。

**示例**

```python
out, lse = bsa_attn_sm100_blk128_fwd(q, k, v, q2k_block_index, block_sparse_num=4)
```

---

### `bsa_attn_sm100_blk64_fwd`

```python
bsa_attn_sm100_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    kv_splits: "int | str" = 1,
    use_clc: Optional[bool] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    backend: str = "cute",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：blk64 CuTe-DSL 内核的 BSA 块稀疏注意力前向。支持 SM100 与 SM103。仅 MHA
（`num_kv_heads` 须等于 `num_heads`），`head_dim` 须为 128。

**参数**

- `q`/`k`/`v`：`(batch, seqlen_q/seqlen_k, num_heads, head_dim)`。
- `q2k_block_index`：`(batch, num_heads, num_q_blocks, max_kv_blocks)`，int32。
- `block_sparse_num`（`int`）：每个 Q block 关注的 KV block 数（`>= 1`）；提供 `q2k_block_nums`
  时忽略。
- `block_sizes`（可选，int32）：每个 KV block 的实际 token 数，形状 `(num_kv_blocks,)`；
  传 `None` 跳过逐块 padding 掩码。
- `q2k_block_nums`（可选，int32）：`(batch, num_heads, num_q_blocks)`，每个 `(batch, head,
  q_block)` 关注的 KV block 数；为 `None` 时用固定 `block_sparse_num`。
- `softmax_scale`（可选）：默认 `1/sqrt(head_dim)`。
- `return_lse`（`bool`，默认 `False`）：是否返回 log-sum-exp。
- `out`/`lse`（可选）：预分配输出 / LSE。
- `kv_splits`（`int | str`，默认 `1`）：KV split 数，`"auto"` 或 `[1, 256]` 内的显式整数；
  `kv_splits=1`（或 `"auto"` 解析为 1）精确复现原始单内核行为。
- `use_clc`（`Optional[bool]`，默认 `None`）：调度器覆盖；`True` 强制 CLC persistent 调度器，
  `False` 强制静态调度器，`None` 用基于形状的启发式。
- `q_scale, k_scale, v_scale`（可选）：Sage FP8 量化 scale。三者须同时提供以启用 FP8 路径
  （此时 q/k/v 须为 float8_e4m3fn）；否则三者须均为 `None` 且 q/k/v 为 bfloat16。`backend="cute"`
  时 FP8 路径额外要求 `batch_size == 1`、`num_head ∈ (4, 8)`、`q2k_block_nums is None` 且
  `block_sizes is None`。
- `backend`（`str`，默认 `"cute"`）：`"cute"` 运行 CuTe-DSL 内核；`"cake"` 运行 Cake 生成的
  Sage-FP8 内核，要求 FP8 路径并解除其限制（任意 batch/head、GQA、非对齐 seqlen、允许
  `q2k_block_nums` 空行与 `block_sizes`）；`kv_splits` 与 `use_clc` 被 Cake 后端忽略。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor]]`：输出与（可选的）LSE。

**示例**

```python
out, lse = bsa_attn_sm100_blk64_fwd(q, k, v, q2k_block_index, block_sparse_num=4)
```

---

### `bsa_attn_sm120_blk64_fwd`

```python
bsa_attn_sm120_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：sm120_blk64 CuTe-DSL 内核的 BSA 块稀疏注意力前向（仅 SM120/SM121）。

**参数**

- `q`：query `(batch, seqlen_q, num_heads, head_dim)`，fp16/bf16。
- `k`/`v`：`(batch, seqlen_k, num_kv_heads, head_dim)`。
- `q2k_block_index`：`(batch, num_heads, num_q_blocks, max_kv_blocks)`，int32。
- `block_sparse_num`（`int`）：每个 Q block 的 KV block 数；提供 `q2k_block_nums` 时忽略。
- `block_sizes`（可选，int32）：每个 KV block 的实际 token 数，形状 `(num_kv_blocks,)`、
  `(batch, num_kv_blocks)` 或 `(batch, num_heads, num_kv_blocks)`；传 `None` 跳过逐块 padding 掩码。
- `q2k_block_nums`（可选，int32）：`(batch, num_heads, num_q_blocks)`。
- `softmax_scale`（可选）：默认 `1/sqrt(head_dim)`。
- `return_lse`（`bool`，默认 `False`）：是否返回 log-sum-exp。
- `out`（可选）：预分配 `(batch, seqlen_q, num_heads, head_dim)`。
- `lse`（可选）：预分配 `(batch, num_heads, seqlen_q)`，float32。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor]]`：输出与（可选的）LSE。

**注意事项**

- 仅支持 SM120/SM121；`head_dim` 与 `head_dim_v` 须为 128；`num_heads` 须能被 `num_kv_heads` 整除。
- `out` 须与 `q` 同 dtype（内核复用 Q shared memory 做 O epilogue）。

**示例**

```python
out, lse = bsa_attn_sm120_blk64_fwd(q, k, v, q2k_block_index, block_sparse_num=4)
```

---

### `bsa_attn_sm120_blk64_sage_fwd`

```python
bsa_attn_sm120_blk64_sage_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    *,
    out: Optional[torch.Tensor] = None,
    tma_descriptor_workspace: Optional[torch.Tensor] = None,
    uniform_block_count: bool = False,
    contiguous_block_indices: bool = False,
    backend: str = "cake",
) -> torch.Tensor
```

**功能**：运行预量化的 SM120 Sage 块稀疏注意力。支持两个后端：

- `"cake"`（默认）：tvm-ffi 生成的内核。所有 CUDA 存储由调用方持有：`out` 为必需的关键字参数。
  Tensor-map descriptor 以值（grid-constant 内核参数）传入，`tma_descriptor_workspace` 被接受
  但忽略。`uniform_block_count` 与 `contiguous_block_indices` 调优标志仅此后端遵循。
- `"cute_dsl"`：上游 Block-Sparse-Attention 的 CuTe-DSL 移植。全程使用 BHSD 张量布局，未提供
  `out` 时内部申请。仅支持 SM120（精确 compute capability 12.0）、仅 MHA、head 维度 128、非
  causal 前向、无 LSE。`uniform_block_count` 与 `contiguous_block_indices` 被静默忽略。

Q/K 使用连续 INT8 BHSD，V 使用连续 FP8 E4M3 HDS。

**参数**

- `q_int8`（`torch.Tensor`）：连续 INT8 query，形状 `[B, H, Sq, 128]`。
- `k_int8`（`torch.Tensor`）：连续 INT8 key，形状 `[B, H, Sk, 128]`。
- `v_fp8`（`torch.Tensor`）：连续 FP8 E4M3 value，HDS 布局，形状 `[B, H, 128, ceil(Sk / 64) * 64]`。
- `q_scale`（`torch.Tensor`）：连续 FP32 query scale，形状 `[B, H, ceil(Sq / 128) * 4]`。
- `k_scale`（`torch.Tensor`）：连续 FP32 key scale，形状 `[B, H, ceil(Sk / 64)]`。
- `v_scale`（`torch.Tensor`）：连续 FP32 value scale，形状 `[B, H, 128]`。
- `q2k_block_index`（`torch.Tensor`）：连续 INT32 选中的 KV block 索引，形状
  `[B, H, ceil(Sq / 64), capacity]`。
- `block_sparse_num`（`int`）：省略 `q2k_block_nums` 时每个 query block 的均匀选中 KV block 数；
  须在 0 到 `capacity` 之间。
- `block_sizes`（可选，int32）：KV block 的有效 token 数，支持 `[num_kv_blocks]`、
  `[B, num_kv_blocks]`、`[B, H, num_kv_blocks]`。
- `q2k_block_nums`（可选，int32）：选中的 block 数，形状 `[B, H, ceil(Sq / 64)]`。
- `softmax_scale`（可选）：正有限 softmax scale；`None` 取 `1 / sqrt(128)`。
- `out`（可选）：预分配连续 BF16 输出，形状 `[B, H, Sq, 128]`。`backend="cake"` 时必需；
  `backend="cute_dsl"` 未提供时内部申请。
- `tma_descriptor_workspace`（可选）：为兼容而接受，两个后端都忽略。
- `uniform_block_count`（`bool`，默认 `False`）：仅 `"cake"` 后端遵循的调优标志。
- `contiguous_block_indices`（`bool`，默认 `False`）：仅 `"cake"` 后端遵循的调优标志。
- `backend`（`str`，默认 `"cake"`）：`"cake"` 或 `"cute_dsl"`。

**返回**

- `torch.Tensor`：输出张量。

**示例**

```python
out = bsa_attn_sm120_blk64_sage_fwd(
    q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
    q2k_block_index, block_sparse_num=4, out=out,
)
```

---

### `cute_dsl_hca_decode`

```python
cute_dsl_hca_decode(
    query: torch.Tensor,
    window_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_block_tables: torch.Tensor,
    hca_seq_lens: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    window_valid_lens: torch.Tensor,
    softmax_scale: float,
    output_scale: float = 1.0,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    is_persistent: bool = False,
) -> torch.Tensor
```

**功能**：在 SM100/SM103 上运行 FP8 DeepSeek-V4 HCA decode 内核。所有张量使用 PyTorch 原生
布局。`query` 为 `[B, Q, H, 512]`；window pool 为扁平 `[pool_rows, 512]`，以绝对
`window_indices[B * Q, 128]` 选择；compressed pool 保持 `[num_pages, page_size, 512]` 并配
page table。当前 DSV4 集成仅支持 causal。`hca_seq_lens` 统计 HCA slot 数（128 window slot 加
compressed slot），而非原始未压缩 token。

**参数**

- `query`（`torch.Tensor`）：`[B, Q, H, 512]`，FP8 E4M3、连续、16 字节对齐。
- `window_kv_cache`（`torch.Tensor`）：2D `[pool_rows, 512]`。
- `compressed_kv_cache`（`torch.Tensor`）：3D `[num_pages, page_size, 512]`。
- `workspace_buffer`（`torch.Tensor`）：工作区。
- `window_indices`（`torch.Tensor`）：`[B * Q, 128]` INT32 绝对行索引。
- `compressed_block_tables`（`torch.Tensor`）：`[B * Q, max_pages]` INT32 block table。
- `hca_seq_lens`（`torch.Tensor`）：每个查询行的 HCA footprint 长度。
- `sparse_topk_lens`（`torch.Tensor`）：有效 top-k 长度。
- `window_valid_lens`（`torch.Tensor`）：窗口有效长度。
- `softmax_scale`（`float`）：softmax scale。
- `output_scale`（`float`，默认 `1.0`）：输出缩放。
- `sinks`（`torch.Tensor`，可选）：attention sink logits。
- `out`（`torch.Tensor`，可选）：预分配输出。
- `is_causal`（`bool`，默认 `True`）：当前必须为 `True`（DSV4 HCA 仅支持 causal decode）。
- `is_persistent`（`bool`，默认 `False`）：是否使用 persistent 网格（非 persistent 网格受
  `_check_nonpersistent_grid` 约束）。

**返回**

- `torch.Tensor`：输出张量。

**注意事项**

- 要求 FP8 E4M3 query、连续、16 字节对齐，compute capability 为 SM100/SM103，head 维度 512，
  `num_heads ∈ [1, 128]`。
- 推荐的公开入口是 `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` 配 `backend="cute-dsl"`；
  `cute_dsl_hca_decode` 是面向已使用显式 HCA metadata ABI 的调用方的更低层封装。
- masked window padding 仍须包含合法行索引，因为 gather4 在掩码前读取每个坐标。
  `hca_seq_lens` 描述 TMA 调度的底层 HCA footprint，而非有效 top-k；每个 compressed block-table
  行因此必须为该 footprint（向上取整到 128-slot tile）包含合法 page id，包括随后被更短
  `sparse_topk_lens` 掩掉的 slot。

**示例**

```python
out = cute_dsl_hca_decode(
    query, window_kv_cache, compressed_kv_cache, workspace_buffer,
    window_indices, compressed_block_tables, hca_seq_lens,
    sparse_topk_lens, window_valid_lens, softmax_scale=0.125,
)
```

---

### `cute_dsl_fused_moe_bf16`

```python
cute_dsl_fused_moe_bf16(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    *,
    intermediate_buffer: Optional[torch.Tensor] = None,
    tactic: Optional[Tuple[Any, ...]] = None,
) -> torch.Tensor
```

**功能**：SM90 CuTe-DSL 融合 MoE 前向（BF16/FP16，非量化）。对默认 SwiGLU，
`out[t] = sum_k scale[t,k] * ffn_expert(x[t]; e[t,k])`，其中
`ffn(x; e) = (silu(x @ w1_gate[e].T) * (x @ w1_up[e].T)) @ w2_weight[e].T`。

支持的配置：

- 架构：仅 SM90（Hopper）。
- dtype：bf16 或 fp16 激活与权重（须一致），fp32 累加，输出 dtype = 输入 dtype，无量化路径。
- 激活（融合进 GEMM1）：`ActivationType.Swiglu`（默认，OAI 变体经 `swiglu_alpha`/`swiglu_beta`/
  `swiglu_limit`，SiTU 经 `situ_beta`）、`ActivationType.GegluTanh` 或非门控的
  `ActivationType.Relu2`。
- 路由：仅预路由契约——调用方运行 router 并传入全局 expert id 与**归一化** scale；`top_k`
  是内核的编译期常量。
- 并行：按权重形状做 TP；经 `num_local_experts` + `local_expert_offset` 做 EP（路由完全在本地
  shard 之外的 token 贡献零）。
- 形状：`hidden % 8 == 0`；门控激活要求 `I % 32 == 0`（32 列 up/gate 交错），`Relu2` 要求
  `I % 8 == 0`；支持部分最后 K/N tile；支持 `num_tokens == 0`。
- 执行：可 CUDA Graph 捕获；默认开 PDL；融合 finalize（默认）是 atomic 的、非按位可复现——
  `use_fused_finalize=False` 选择确定性路径。

tile 选择经 FlashInfer AutoTuner：在 `autotune` 上下文内会 profile runner 为该 token bucket 提供的
每个 tactic 并缓存逐 bucket 赢家；之外则用缓存赢家或固定默认。显式 `tactic` 会绕过调优器。

**参数**

- `x`：`[num_tokens, hidden]`，bf16/fp16。
- `token_selected_experts`：`[num_tokens, top_k]`，int32。
- `token_final_scales`：`[num_tokens, top_k]`，float32，由调用方归一化。
- `w1_weight`：门控激活为 `[num_local_experts, 2I, hidden]`，up/gate 在 32 列交错；`Relu2` 为
  `[num_local_experts, I, hidden]`，无交错。
- `w2_weight`：`[num_local_experts, hidden, I]`。
- `num_experts`（`int`）：全局 expert 数。
- `top_k`（`int`）：每 token 的 expert 数。
- `num_local_experts`（`int`，可选）：本 rank 持有的 expert 数（EP shard）；默认 `num_experts`。
- `local_expert_offset`（`int`，默认 `0`）：本 shard 首个 expert 的全局 id。
- `use_fused_finalize`（`bool`，默认 `True`）：True 把 router-scaled scatter-reduce 融合进 GEMM2，
  top-k 组合以输出 dtype 累加，且非按位可复现；False 使用确定性两阶段路径（GEMM2 以展开
  `(token, slot)` 顺序 scatter 未缩放行，再由 `moe_unpermute` 以固定顺序在 float32 应用 scale 并
  组合），代价是额外一个内核与展开的中间结果。
- `moe_output`（可选）：预分配 `[num_tokens, hidden]` 输出（内容被覆盖；融合 finalize 时内部清零）。
- `enable_pdl`（`bool`，默认 `True`）：GEM 与确定性路径的 `moe_unpermute` 以 PDL 启动，使各内核
  prologue 与其前驱 tail 重叠；数值不受影响，是内核编译缓存键的一部分。
- `activation_type`（`int`，默认 `ActivationType.Swiglu.value`）：GEMM1 激活，连同下面的常量是
  内核编译键与 autotuner 缓存键的一部分。
- `swiglu_alpha`/`swiglu_beta`/`swiglu_limit`（`float`）：SwiGLU 参数（仅 `Swiglu`）。
- `situ_beta`（`float`，可选）：SiTU beta。
- `situ_linear_beta`（`float`，可选）：SiTU linear beta。
- `intermediate_buffer`（可选，关键字限定）：中间缓冲区。
- `tactic`（`Tuple`，可选，关键字限定）：显式 tactic，绕过 tuner。

**返回**

- `torch.Tensor`：`[num_tokens, hidden]` 输出。

**示例**

```python
from flashinfer import autotune
from flashinfer.fused_moe.cute_dsl import cute_dsl_fused_moe_bf16

with autotune(True):
    output = cute_dsl_fused_moe_bf16(
        x, token_selected_experts, token_final_scales,
        w1_weight, w2_weight, num_experts=256, top_k=8,
    )
```

---

### `cute_dsl_fused_moe_nvfp4`

```python
cute_dsl_fused_moe_nvfp4(
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor],
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    *,
    quant_mode: str = "w4a4",
    per_token_scale: Optional[torch.Tensor] = None,
    localized_weights: Optional[list] = None,
    localized_streams: Optional[list] = None,
    localized_sm_count: Optional[int] = None,
    localized_memset_stream: Optional[torch.cuda.Stream] = None,
    localized_allow_nonlocalized: bool = False,
    autotune_use_cuda_graph: bool = False,
) -> torch.Tensor
```

**功能**：使用 CuTe-DSL NVFP4 内核运行融合 MoE 前向。**该 API 将在未来废弃**，请改用
`cute_dsl_fused_moe` 且 `quant_mode="w4a4"`。完整参数文档见 `cute_dsl_fused_moe`；本函数原样转发
每个参数。调用时会发出 `DeprecationWarning`。

**返回**

- `torch.Tensor`：MoE 输出。

**示例**

```python
from flashinfer.fused_moe.cute_dsl import cute_dsl_fused_moe_nvfp4

output = cute_dsl_fused_moe_nvfp4(
    x, x_sf, token_selected_experts, token_final_scales,
    w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
    w2_weight, w2_weight_sf, w2_alpha, num_experts=256, top_k=8,
)
```

---

### `cute_dsl_fused_moe_mxfp8_mxfp4`

```python
cute_dsl_fused_moe_mxfp8_mxfp4(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    tactic: Optional[Tuple[Any, ...]] = None,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor
```

**功能**：使用 MXFP8 激活与 packed MXFP4 权重运行融合 MoE。**该 API 将在未来废弃**，请改用
`cute_dsl_fused_moe` 且 `quant_mode="w4a8"`。与 NVFP4 入口不同，本接口没有 `fc2_input_scale`，
转发为 `None`。完整参数文档见 `cute_dsl_fused_moe`。调用时发出 `DeprecationWarning`。

**注意事项**

- 支持的 compute capability：`[100, 103]`。

**返回**

- `torch.Tensor`：MoE 输出。

**示例**

```python
from flashinfer.fused_moe.cute_dsl import cute_dsl_fused_moe_mxfp8_mxfp4

output = cute_dsl_fused_moe_mxfp8_mxfp4(
    x, x_sf, token_selected_experts, token_final_scales,
    w1_weight, w1_weight_sf, w1_alpha,
    w2_weight, w2_weight_sf, w2_alpha, num_experts=256, top_k=8,
)
```

---

### `b12x_fused_moe`

```python
b12x_fused_moe(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor] = None,
    input_global_scale: Optional[torch.Tensor] = None,
    num_local_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: Optional[float] = None,
    activation_precision: str = "fp4",
    quant_mode: Optional[str] = None,
    source_format: str = "modelopt",
) -> torch.Tensor
```

**功能**：在 SM120/SM121 上使用 b12x CuTe-DSL 内核运行融合 MoE。内核接收 bf16 输入，并通过所选
后端执行 routing、FC1、activation、FC2、scatter。依据 routed 行数自动选择 micro（decode）、
static 或 dynamic 后端。

**参数**

- `x`（`torch.Tensor`）：输入激活，形状 `[num_tokens, hidden_size]`，`bfloat16`。
- `w1_weight`（`torch.Tensor`）：FC1 权重，FP4 packed。门控（SiLU）布局
  `[E, 2 * intermediate_size, hidden_size // 2]`；非门控（ReLU2）布局
  `[E, intermediate_size, hidden_size // 2]`。
- `w1_weight_sf`（`torch.Tensor`）：`w1_weight` 的 scale factor。
- `w2_weight`（`torch.Tensor`）：FC2 权重，形状 `[E, hidden_size, intermediate_size // 2]`，FP4。
- `w2_weight_sf`（`torch.Tensor`）：`w2_weight` 的 scale factor。
- `token_selected_experts`（`torch.Tensor`）：expert 分配，形状 `[num_tokens, top_k]`；负 id 标记
  未路由 slot（如 CUDA Graph padding），不贡献输出，全为负的 token 产生零输出行。
- `token_final_scales`（`torch.Tensor`）：路由权重，形状 `[num_tokens, top_k]`。
- `num_experts`（`int`）：总 expert 数。
- `top_k`（`int`）：每 token 路由到的 expert 数。
- `w1_alpha`（`torch.Tensor`，关键字限定）：FC1 的逐 expert 全局 scale。
- `w2_alpha`（`torch.Tensor`，关键字限定）：FC2 的逐 expert 全局 scale。
- `fc2_input_scale`（`torch.Tensor`，可选）：FC2 输入量化的全局 scale；`quant_mode="nvfp4"` 时必需，
  `"mxfp4"` 与 `"w4a16"` 时忽略。
- `input_global_scale`（`torch.Tensor`，可选）：FC1 输入量化的全局 scale，标量或 `[num_experts]`；
  使 `w1_alpha` 仅承载精确 fp32 权重 scale，并内部折入输出乘子。默认取 `w1_alpha`。
  `"mxfp4"` 与 `"w4a16"` 时忽略。
- `num_local_experts`（`int`，可选）：本 rank 持有的 expert 数（EP）；默认 `num_experts`。当不同于
  `num_experts` 时要求 `quant_mode="w4a16"` 与 `expert_map`。
- `expert_map`（`torch.Tensor`，可选）：EP 的 global-to-local expert 映射，int32，形状
  `[num_experts]`。条目 `g` 为全局 expert `g` 的 rank-local 权重索引，或 `-1` 表示本 rank 不持有。
  每次调用值被信任，须恰好覆盖 `[0, num_local_experts)` 一次。仅 `quant_mode="w4a16"` 支持。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `[num_tokens, hidden_size]`，`bfloat16`。
- `output_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出类型，当前仅支持 bfloat16。
- `activation`（`str`，默认 `"silu"`）：激活函数，`"silu"`（门控 SwiGLU）、`"gelu_tanh"`（门控
  tanh-approx GeGLU）、`"swigluoai_uninterleave"`（门控 SwiGLU-OAI）或 `"relu2"`。
- `swiglu_alpha, swiglu_beta, swiglu_limit`（`float`）：SwiGLU-OAI 参数，仅在
  `activation="swigluoai_uninterleave"` 时使用：`gate*sigmoid(alpha*gate)*(up+beta)`，可选 clamp
  到 `swiglu_limit`（`None` 禁用）。默认 1.702 / 1.0 / None，为近似 GELU 的标准参数。
- `activation_precision`（`str`，默认 `"fp4"`）：`quant_mode` 的向后兼容别名；`"fp4"` 选择
  `quant_mode="nvfp4"`，`"bf16"` 选择 `quant_mode="w4a16"`。
- `quant_mode`（`str`，可选）：量化模式，`"nvfp4"`/`"w4a4"`、`"mxfp4"` 或 `"w4a16"`；设置时选择
  后端与内部工作区族。
- `source_format`（`str`，默认 `"modelopt"`）：`quant_mode="w4a16"` 的源权重格式，`"modelopt"` 或
  `"compressed_tensors"`。

**返回**

- `torch.Tensor`：输出张量，形状 `[num_tokens, hidden_size]`。

**注意事项**

- 需要 CUDA 13 或更新，否则抛 `ValueError`。
- SM12x 内核为 `scatter_output` 硬编码 BF16，其他 dtype 会导致内核张量绑定失败；`output_dtype`
  非 bfloat16 时抛 `ValueError`。

**示例**

```python
from flashinfer.fused_moe.cute_dsl import b12x_fused_moe

output = b12x_fused_moe(
    x, w1_weight, w1_weight_sf, w2_weight, w2_weight_sf,
    token_selected_experts, token_final_scales,
    num_experts=256, top_k=8, w1_alpha=w1_alpha, w2_alpha=w2_alpha,
)
```

## 相关链接 / 参考

- CuTe-DSL 相关的可用性探测（`is_rubin_cute_dsl_available`、
  `is_cute_dsl_experimental_available`、`is_cute_dsl_arch_supported`、`require_cute_dsl_arch`）
  定义于 `flashinfer/cute_dsl/availability.py`，但本 rst 仅在 Availability 小节列出
  `is_cute_dsl_available`。
- `flashinfer/cute_dsl/__init__.py` 还再导出了若干 norm 与 GEMM 符号，均未列于本 rst 的
  autosummary。


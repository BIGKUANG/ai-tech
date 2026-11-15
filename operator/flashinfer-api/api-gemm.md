# GEMM 算子（`flashinfer.gemm`）

> 源码：`flashinfer/gemm/__init__.py`、`gemm_base.py`、`gemm_svdquant.py`、`gemm_bf16_fp4.py`、`routergemm.py`、`cake_*.py`、`kernels/`、`flashinfer/trtllm_low_latency_gemm.py`、`flashinfer/prims_ts/gemm/api.py`
> 上游文档：`docs/api/gemm.rst`
> 适用版本：FlashInfer 0.7.2

本模块提供一组 GEMM 算子：BF16/FP16、FP4（NVFP4 / MXFP4）、MXFP8、FP8（含块缩放与
分组变体）、SVDQuant NVFP4、BF16 x FP4（W4A16）、混合精度、Router GEMM、TGV Blackwell
GEMM、cuTile GEMM、分组 GEMM（CuTe-DSL / Ampere-Hopper），以及 PrimsTS 稠密 FP8/NVFP4
GEMM。多数算子通过 `backend=` 参数在多个后端间分发。

## 导入

```python
from flashinfer import mm_bf16, bmm_bf16, mm_fp4, mm_fp8, mm_mxfp8, SegmentGEMMWrapper
# 或
from flashinfer.gemm import mm_bf16, mm_fp4
```

## API 列表

### BF16 GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_bf16` | 函数 | BF16 矩阵乘（M x K · K x N）。 |
| `bmm_bf16` | 函数 | BF16 批矩阵乘。 |

### FP4 GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_fp4` | 函数 | FP4（NVFP4 / MXFP4）块缩放矩阵乘。 |

### SVDQuant NVFP4 GEMM（SM100）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_nvfp4_svdquant` | 函数 | SVDQuant NVFP4 GEMM，融合秩-r BF16 LoRA 修正。 |
| `nvfp4_quantize_smooth` | 函数 | 平滑 + NVFP4 量化融合。 |
| `svdquant_linear` | 函数 | SVDQuant 线性层完整算子。 |

### BF16 x FP4 GEMM（W4A16）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `prepare_bf16_fp4_weights` | 函数 | 为指定后端准备 FP4 权重。 |
| `mm_bf16_fp4` | 函数 | BF16 激活 x FP4 权重 GEMM。 |

### MXFP8 GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_mxfp8` | 函数 | MXFP8（block size 32）矩阵乘。 |
| `bmm_mxfp8` | 函数 | MXFP8 批矩阵乘。 |

### FP8 GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_fp8` | 函数 | 低延迟 FP8 矩阵乘。 |
| `bmm_fp8` | 函数 | FP8 批矩阵乘。 |
| `gemm_fp8_nt_blockscaled` | 函数 | FP8 块缩放 GEMM（128x128x128）。 |
| `gemm_fp8_nt_groupwise` | 函数 | FP8 groupwise 缩放 GEMM。 |
| `group_gemm_fp8_nt_groupwise` | 函数 | FP8 groupwise 分组 GEMM。 |
| `group_gemm_fp8_nt_groupwise_contiguous` | 函数 | 连续分组 FP8 GEMM（CuTe DSL）。 |
| `group_deepgemm_fp8_nt_groupwise` | 函数 | DeepGEMM 后端分组 FP8 GEMM。 |
| `batch_deepgemm_fp8_nt_groupwise` | 函数 | DeepGEMM 后端批 FP8 GEMM。 |
| `fp8_blockscale_gemm_sm90` | 函数 | SM90 FP8 块缩放 GEMM（自动 swapAB）。 |
| `prepare_low_latency_gemm_weights` | 函数 | 低延迟 TRT-LLM GEMM 权重准备。 |

### 混合精度 GEMM（fp8 x fp4）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `group_gemm_mxfp8_mxfp4_nt_groupwise` | 函数 | MXFP4 分组 GEMM（groupwise 缩放）。 |
| `group_gemm_nvfp4_nt_groupwise` | 函数 | NVFP4 分组 GEMM（groupwise 缩放）。 |

### Router GEMM（DeepSeek-V3 / Mistral / GLM / Kimi-K2 / Kimi-K3）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mm_M1_16_K7168_N128` | 函数 | Mistral Large 3 router GEMM（K=7168, N=128）。 |
| `mm_M1_16_K7168_N256` | 函数 | DeepSeek-V3 router GEMM（K=7168, N=256）。 |
| `mm_M1_16_K6144_N256` | 函数 | GLM-MoE-DSA router GEMM（K=6144, N=256）。 |
| `mm_M1_16_K7168_N256_bf16` | 函数 | DeepSeek-V3 router GEMM（bf16 输出）。 |
| `mm_M1_16_K7168_N384` | 函数 | Kimi-K2 router GEMM（K=7168, N=384）。 |
| `mm_M1_16_K7168_N384_bf16` | 函数 | Kimi-K2 router GEMM（bf16 输出）。 |
| `mm_M1_16_K7168_N896` | 函数 | Kimi-K3 router GEMM（K=7168, N=896）。 |
| `mm_M1_16_K7168_N896_bf16` | 函数 | Kimi-K3 router GEMM（bf16 输出）。 |
| `tinygemm_bf16` | 函数 | SM90+ 小 GEMM（`out = input @ weight.T + bias`）。 |

### Blackwell SM100 GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `tgv_gemm_sm100` | 函数 | TGV GEMM，自动 dtype 检测。 |

### cuTile GEMM

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `gemm_alpha_beta` | 函数 | `C = alpha * (A @ B) + beta * C`。 |
| `masked_bmm` | 函数 | 带逐 batch M 掩码的批矩阵乘。 |
| `masked_scaled_bmm` | 函数 | 掩码块缩放批矩阵乘（FP8/FP4）。 |
| `ragged_bmm` | 函数 | 非均匀 M 分段的 ragged 批矩阵乘。 |
| `ragged_block_scaled_bmm` | 函数 | ragged 块缩放批矩阵乘（FP8）。 |
| `ragged_scaled_bmm` | 函数 | ragged 块缩放批矩阵乘（FP8/FP4，`mma_scaled`）。 |

### 分组 GEMM（CuTe-DSL, Blackwell）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_gemm_nt_masked` | 函数 | Blackwell/Rubin 掩码批块缩放 GEMM。 |

### 分组 GEMM（Ampere/Hopper）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `SegmentGEMMWrapper` | 类 | segment GEMM 内核封装。 |

### PrimsTS 稠密 FP8 / NVFP4 GEMM（SM100, SM103, SM107）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `fp8_linear` | 函数 | FP8 线性层（per-token/per-channel 缩放）。 |
| `fp8_linear_swiglu` | 函数 | FP8 线性 + SwiGLU 融合。 |
| `fp8_qkv_qknorm_rope` | 函数 | FP8 QKV 投影 + QK RMSNorm + RoPE。 |
| `fp4_linear` | 函数 | 打包 NVFP4 线性层。 |
| `fp4_linear_swiglu` | 函数 | 打包 NVFP4 线性 + SwiGLU 融合。 |
| `fp4_qkv_qknorm_rope` | 函数 | 打包 NVFP4 QKV 投影 + QK RMSNorm + RoPE。 |

## API 详情

### `mm_bf16`

```python
mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal['cudnn', 'cutlass', 'tgv', 'cublaslt', 'tinygemm', 'cutile', 'cute-dsl', 'auto'] = 'cudnn',
) -> torch.Tensor
```

**功能**：BF16 矩阵乘 `out = a @ b`。

**参数**

- `a`（`torch.Tensor`）：形状 `(m, k)`，bf16，行主序。
- `b`（`torch.Tensor`）：权重，形状 `(k, n)`，bf16，列主序。
- `bias`（`torch.Tensor`，可选）：形状 `(n,)`。适用于 TGV、TinyGEMM 与 CuTeDSL 后端；
  cuBLASLt 在 BF16 输出下支持 bias。默认 `None`。
- `pdl`（`bool`，默认 `False`）：是否使用 PDL。适用于 TGV、TinyGEMM 与 CuTeDSL。
- `out`（`torch.Tensor`，可选）：形状 `(m, n)`，bf16/fp16/fp32。FP16/FP32 输出适用于
  CUTLASS 与 cuDNN；TinyGEMM 与 CuTeDSL 要求 bf16 输出。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，bf16/fp16/fp32。适用于
  CUTLASS、cuDNN 与 cuBLASLt；TinyGEMM 与 CuTeDSL 要求 `torch.bfloat16`。
- `backend`（`Literal["cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "cutile", "cute-dsl", "auto"]`，默认 `"cudnn"`）：
  `"cudnn"`（cuDNN）、`"cutlass"`（CUTLASS）、`"tgv"`（TGV）、`"cublaslt"`（cuBLASLt，
  可带融合 BF16 bias epilogue）、`"tinygemm"`（小 M BF16）、`"cutile"`（cuda.tile Python
  后端，纯 Python 持久调度 GEMM，逐形状穷举 autotune，忽略 `bias`/`pdl`，要求 SM >= 90）、
  `"cute-dsl"`（Blackwell 低 M 独立内核 + cuBLASLt 回退，绝不被自动选择，需服务框架显式
  指定）、`"auto"`（允许在启用 autotune 时从所有可用后端中选择最佳 tactic）。

**返回**

- `torch.Tensor`：形状 `(m, n)`，bf16/fp16/fp32，行主序。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> a = torch.randn([48, 64], device="cuda", dtype=torch.bfloat16)
>>> b = torch.randn([80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
>>> out = flashinfer.mm_bf16(a, b, backend="cudnn")
>>> out.shape
torch.Size([48, 80])
```

---

### `bmm_bf16`

```python
bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal['cudnn', 'cutlass', 'cutile', 'tgv', 'cake', 'auto'] = 'cudnn',
) -> torch.Tensor
```

**功能**：BF16 批矩阵乘 `out[i] = A[i] @ B[i]`。

**参数**

- `A`（`torch.Tensor`）：形状 `(b, m, k)`，bf16，行主序。
- `B`（`torch.Tensor`）：权重，形状 `(b, k, n)`，bf16，列主序。
- `out`（`torch.Tensor`，可选）：形状 `(b, m, n)`，bf16/fp16/fp32，默认 `None`。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，bf16/fp16/fp32。
- `backend`（`Literal["cudnn", "cutlass", "cutile", "tgv", "cake", "auto"]`，默认 `"cudnn"`）：
  `"cake"` 选择冻结的 SM100a/SM103a CAKE 生成调度器，要求 A/输出连续、B 精确转置列主序、
  16 字节对齐、N 可被 8 整除、K ∈ {64, 256, 1024}，且输出不与任一输入重叠；`"cake"`
  仅显式选择且不被 `"auto"` 考虑。

**返回**

- `torch.Tensor`：形状 `(b, m, n)`，bf16/fp16/fp32，行主序。

---

### `mm_fp4`

```python
mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: Literal['cudnn', 'trtllm', 'cutlass', 'cute-dsl', 'cutedsl_low_latency', 'b12x', 'cake', 'auto'] = 'auto',
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor
```

**功能**：FP4（NVFP4 / MXFP4）块缩放矩阵乘。

**参数**

- `a`（`torch.Tensor`）：形状 `(m, k)`，fp4 e2m1fn_x2 或 uint8。
- `b`（`torch.Tensor`）：形状 `(k, n)`，列主序，fp4 e2m1fn_x2 或 uint8。
- `a_descale`（`torch.Tensor`）：A 的块缩放，形状 `(m, k // block_size)`，float8_e4m3fn 或 uint8。
- `b_descale`（`torch.Tensor`）：B 的块缩放，形状 `(k, n // block_size)`，float8_e4m3fn 或 uint8。
- `alpha`（`torch.Tensor`，可选）：全局缩放（浮点标量），或形状 `(m,)`、dtype float32 的
  逐行反量化缩放（动态 per-token NVFP4 全局缩放）。per-token 形式由 `"cute-dsl"` 后端在
  SM100/SM103 实现（`backend="auto"` 会选择），也由实验性 `"cake"` 后端实现（显式选择）。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，bf16 或 fp16；
  `backend="trtllm"` 时仅支持 bf16。
- `out`（`torch.Tensor`，可选）：形状 `(m, n)`，bf16 或 fp16，默认 `None`。
- `block_size`（`int`，默认 `16`）：FP4 量化块大小，仅支持 16（NVFP4）与 32（MXFP4）。
- `use_8x4_sf_layout`（`bool`，默认 `False`）：使用 8x4 还是 128x4 缩放布局。
- `backend`（`Literal[...]`，默认 `"auto"`）：SM120 上 `"auto"` 优先 `"b12x"`（仅 NVFP4），
  其次 `"cutlass"`、`"cudnn"`；其它架构上 `"auto"` 依当前 CUDA 与 cuDNN 版本在
  `"cudnn"` 与 `"cutlass"` 间选择。`"trtllm"`、`"cute-dsl"` 因需不同权重准备而不被自动
  选择。`"cutedsl_low_latency"` 是 SM100/SM103 问题的最后启发式候选，要求 M <= 8、
  128x4 缩放、K 可被 64（NVFP4）或 128（MXFP4）整除。实验性 `"cake"`（SM100/SM103，不
  自动选择）仅服务 per-token alpha NVFP4 情形。
- `use_nvfp4`（`bool`，默认 `True`）：使用 nvfp4 还是 mxfp4 量化。与 `block_size` 约束相关。
- `enable_pdl`（`bool`，默认 `True`）：为 `cute_dsl` 与 `cutedsl_low_latency` 后端启用 PDL；
  其它后端忽略。

**返回**

- `torch.Tensor`：形状 `(m, n)`，bf16 或 fp16。

**注意事项**

- cudnn/cutlass 后端：a、b 都应以 `nvfp4_quantize` 用 128x4 缩放布局、`do_shuffle=False` 量化。
- trtllm 后端：b 必须以 128x4 布局、`do_shuffle=True` 量化；a 可用 128x4 或 8x4
  （`use_8x4_sf_layout` 控制）、`do_shuffle=False`。
- cute_dsl 后端：a、b 都应以 128x4 布局量化（NVFP4 用 `nvfp4_quantize(..., do_shuffle=False)`，
  MXFP4 用 `mxfp4_quantize(...)`）。cutedsl_low_latency 使用相同的 128x4 布局。

**示例**

```python
>>> import torch
>>> from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
>>> a = torch.randn([48, 128], device="cuda", dtype=torch.bfloat16)
>>> b = torch.randn([256, 128], device="cuda", dtype=torch.bfloat16)
>>> a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
>>> b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
>>> a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
>>> b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True)
>>> out = mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1.0/(a_global_sf * b_global_sf), torch.bfloat16, None, backend="trtllm")
```

---

### `mm_nvfp4_svdquant`

```python
mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal['cutlass', 'cake', 'cute-dsl', 'cute-dsl-unfused', 'auto'] = 'auto',
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：SVDQuant NVFP4 GEMM，`out = alpha * (a @ bᵀ + d @ l1ᵀ) [+ bias]`。SM100/SM103 上
CUTLASS 与显式 `"cake"` 后端将块缩放 NVFP4 残差 GEMM 与秩-r BF16 LoRA-up 修正及可选 bias
融合；SM120/SM121 上 `"cute-dsl"` 将修正与 bias 融合进 b12x CuTe DSL 内核的 FP32 累加器
epilogue，`"cute-dsl-unfused"` 保留组合式实现作为差分 oracle 与可选 autotune 候选。
LoRA 秩 `r` 由 `d`/`l1` 形状推断，必须是 32 的正倍数（校验 32-128）。`1/alpha` 必须由调用方
折进 `l1`（`l1 = svdquant_lora_b / alpha`）。

**参数**

- `a`（`torch.Tensor`）：量化激活，形状 `(m, k // 2)` uint8（打包 e2m1），行主序。用
  `nvfp4_quantize_smooth` 生成（会折叠 SVDQuant 的 `pre_quant_scale`）。
- `b`（`torch.Tensor`）：量化残差权重，形状 `(n, k // 2)` uint8，行主序（即计算 `a @ bᵀ`）。
- `a_sf`（`torch.Tensor`）：激活块缩放，uint8（ue4m3），128x4 swizzled 布局，
  `numel >= ceil(m / 128) * 128 * ceil(k / 16 / 4) * 4`。
- `b_sf`（`torch.Tensor`）：权重块缩放，布局同 `a_sf`，`n` 行。
- `alpha`（`torch.Tensor`）：逐张量残差反量化缩放，非空 float32 设备张量；仅消费第一个元素。
- `d`（`torch.Tensor`）：LoRA-down 输出 `x_hat @ L2ᵀ`，形状 `(m, r)` bf16，连续且 16 字节对齐。
- `l1`（`torch.Tensor`）：预除以 alpha 的 LoRA-up 权重，形状 `(n, r)` bf16。
- `bias`（`torch.Tensor`，可选）：逐列 bias，形状 `(n,)` bf16，被 CUTLASS 与 SM120/SM121
  CuTe DSL 内核融合。
- `out`（`torch.Tensor`，可选）：形状 `(m, n)` bf16，`None` 时分配。
- `backend`（`Literal[...]`，默认 `"auto"`）：`"cutlass"`（SM100/SM103 融合）、`"cake"`
  （生成式 SM100/SM103 融合，需 CUDA 13.0+）、`"cute-dsl"`（SM120/SM121 融合）、
  `"cute-dsl-unfused"`（组合参考路径）、`"auto"`（按 compute capability 选择）。
- `enable_pdl`（`bool`，可选）：是否以 PDL 启动。

**返回**

- `torch.Tensor`：形状 `(m, n)` bf16。

---

### `nvfp4_quantize_smooth`

```python
nvfp4_quantize_smooth(
    x: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: Optional[bool] = None,
    backend: Literal['cutlass', 'cute-dsl', 'auto'] = 'auto',
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：平滑 + NVFP4 量化，`(xq, sf) = nvfp4-quantize(x * pre_quant_scale)`。SM100/SM103
CUTLASS 后端一次完成逐输入通道平滑与 NVFP4 量化；SM120/SM121 CuTe DSL 后端也在 NVFP4 量化器
内部应用平滑，避免 BF16 中间结果。两者均使用 ue4m3 块缩放、128x4 swizzled 布局与 SF
向量大小 16。

**参数**

- `x`（`torch.Tensor`）：输入激活，形状 `(m, n)` bf16。
- `pre_quant_scale`（`torch.Tensor`）：逐输入通道平滑缩放，形状 `(n,)` bf16。
- `global_scale`（`torch.Tensor`）：全局缩放，float32 设备标量：
  `(448 * 6) / (x * pre_quant_scale).abs().max()`。
- `enable_pdl`（`bool`，可选）：是否以 PDL 启动。
- `backend`（`Literal["cutlass", "cute-dsl", "auto"]`，默认 `"auto"`）：按 compute capability 选择。

**返回**

- `xq`（`torch.Tensor`）：形状 `(m, n // 2)` uint8（打包 e2m1）。
- `sf`（`torch.Tensor`）：块缩放，uint8（ue4m3），128x4 swizzled 布局，1-D，
  大小 `ceil(m / 128) * 128 * ceil(n / 16 / 4) * 4`。

---

### `svdquant_linear`

```python
svdquant_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    l2t_smoothed: torch.Tensor,
    l1_scaled: torch.Tensor,
    global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    backend: Literal['cutlass', 'cute-dsl', 'cute-dsl-unfused', 'auto'] = 'auto',
) -> torch.Tensor
```

**功能**：SVDQuant 线性层完整算子，`y = x_hat @ (R + L1 @ L2)ᵀ [+ bias]`，其中
`x_hat = x * pre_quant_scale`，`R` 为 NVFP4 量化的残差权重。执行三步链：①
`nvfp4_quantize_smooth`；② `down = x @ l2t_smoothed`（BF16 张量核 GEMM）；③
`mm_nvfp4_svdquant`。

**参数**

- `x`（`torch.Tensor`）：输入激活，形状 `(m, k)` bf16。
- `weight_fp4`（`torch.Tensor`）：NVFP4 残差权重，形状 `(n, k // 2)` uint8。
- `weight_sf`（`torch.Tensor`）：权重块缩放，uint8（ue4m3），128x4 swizzled。
- `alpha`（`torch.Tensor`）：逐张量残差反量化缩放，float32 设备标量。
- `pre_quant_scale`（`torch.Tensor`）：逐输入通道平滑缩放，形状 `(k,)` bf16。
- `l2t_smoothed`（`torch.Tensor`）：`pre_quant_scale[:, None] * L2ᵀ`，形状 `(k, r)` bf16。
- `l1_scaled`（`torch.Tensor`）：`L1 / alpha`，形状 `(n, r)` bf16，`r` 为 32 的正倍数。
- `global_scale`（`torch.Tensor`）：激活全局缩放，float32 设备标量。
- `bias`（`torch.Tensor`，可选）：逐列 bias，形状 `(n,)` bf16。
- `enable_pdl`（`bool`，可选）：是否以 PDL 启动。
- `backend`（`Literal[...]`，默认 `"auto"`）：转发给平滑量化与 SVDQuant GEMM。

**返回**

- `torch.Tensor`：形状 `(m, n)` bf16。

---

### `prepare_bf16_fp4_weights`

```python
prepare_bf16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: _Bf16Fp4Backend,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
```

**功能**：为 bf16 x fp4 GEMM 的指定后端准备 FP4 权重。调用方应从 `flashinfer.nvfp4_quantize`
（`sfLayout=layout_128x4`）产出的规范格式开始：`b` 为 `(N, K // 2)` uint8（每字节两个 FP4
码，低半字节 = K=2i，高半字节 = K=2i+1）；`b_descale` 为 128x4-swizzled FP8-E4M3 逐块缩放
（1-D 字节缓冲区或 2-D 张量）。各后端将其转换为各自计算内核期望的布局。

**参数**

- `b`（`torch.Tensor`）：`(N, K // 2)` uint8 打包 FP4 权重。
- `b_descale`（`torch.Tensor`）：来自 `nvfp4_quantize` 的 128x4-swizzled FP8-E4M3 缩放。
- `alpha`（`torch.Tensor`，可选）：`(1,) float32` 全局标量；隐式 `alpha=1.0` 时传 `None`
  （默认）。原样返回。
- `backend`（关键字）：支持的后端标识：`"cudnn"`、`"cute-dsl"`、`"blackwell-native"` 或
  `"blackwell-tiled"`。`"cute-dsl-native"` 直接返回原始规范缓冲区而不做处理。
- `block_size`（关键字，`int`，默认 `16`）：SF 块大小，FP4 恒为 16。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`：`(b_prepared,
  b_descale_prepared, alpha_prepared)`——三者都应以**相同** `backend` 传给 `mm_bf16_fp4`。

**注意事项**

- 后端未知、或输入形状非法（`b` 非 2-D、`K` 非 `block_size` 的倍数、`alpha` 非 `(1,)`）
  时抛 `ValueError`；`b` 非 uint8 或 `alpha` 非 float32 时抛 `TypeError`。

---

### `mm_bf16_fp4`

```python
mm_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: _Bf16Fp4Backend,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
) -> torch.Tensor
```

**功能**：BF16 x FP4 GEMM，`out = (a @ dequant(b).T) * alpha`，支持 W4A16（4-bit 权重、
16-bit 激活）工作负载。权重须先用 `prepare_bf16_fp4_weights` 为所选后端准备。显式的
`cute-dsl-native` 后端无需准备：直接接受规范打包权重与 128x4-swizzled 缩放。

**参数**

- `a`（`torch.Tensor`）：`(M, K)` 激活矩阵，`torch.bfloat16`（当前唯一支持的激活 dtype）。
- `b`（`torch.Tensor`）：已准备权重张量（后端特定布局）。
- `b_descale`（`torch.Tensor`）：已准备缩放张量（后端特定布局）。
- `alpha`（`torch.Tensor`，可选）：`(1,) float32` 全局标量，透传
  `prepare_bf16_fp4_weights` 的返回值（后端将其折进 `b_descale` 时可能为 `None`）。
- `backend`（关键字）：与 `prepare_bf16_fp4_weights` 相同的标识；`cute-dsl-native` 在
  SM120/121 上直接接受规范打包权重与 128x4 缩放。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `a.dtype`（`bfloat16`）。
- `out`（`torch.Tensor`，可选）：预分配 `(M, N)` 输出。
- `block_size`（`int`，默认 `16`）：SF 块大小，FP4 恒为 16。
- `enable_pdl`（`bool`，默认 `True`）：启用 PDL。

**返回**

- `torch.Tensor`：`out_dtype` 的 `(M, N)` 张量。

**示例**

```python
# 1) 在模型加载时为目标后端准备权重（一次）。
b_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(b, b_descale, alpha, backend="cute-dsl")
# 2) 使用*相同* backend 标签运行 GEMM。
out = flashinfer.mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend="cute-dsl")
```

---

### `mm_mxfp8`

```python
mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = False,
    backend: Literal['cutlass', 'cute-dsl', 'cutedsl_low_latency', 'trtllm', 'cudnn', 'b12x', 'auto'] = 'auto',
) -> torch.Tensor
```

**功能**：MXFP8（block size 32）块缩放矩阵乘。

**参数**

- `a`（`torch.Tensor`）：A，形状 `(m, k)`，mxfp8 e4m3。
- `b`（`torch.Tensor`）：B，形状 `(k, n)`，列主序，mxfp8 e4m3。
- `a_descale`（`torch.Tensor`）：A 的块缩放，uint8（fp8 e8m0），1-D swizzled 布局，形状
  `(M_padded * K_padded,)`，`M_padded = round_up(m, 8 if 8x4 else 128)`、
  `K_padded = round_up(k // 32, 4)`，由 `mxfp8_quantize(..., is_sf_swizzled_layout=True)` 产生。
  8x4 布局（`use_8x4_sf_layout=True`）仅被 trtllm 后端消费。任何后端都不支持 2-D 线性缩放。
- `b_descale`（`torch.Tensor`）：B 的块缩放，uint8（fp8 e8m0），1-D swizzled 128x4 布局，
  形状 `(N_padded * K_padded,)`。trtllm 后端应以线性布局量化并用 `shuffle_matrix_sf_a` 洗牌。
- `out`（`torch.Tensor`，可选）：形状 `(m, n)`，bf16 或 fp16，`None` 时分配。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，bf16 或 fp16。
- `use_8x4_sf_layout`（`bool`，默认 `False`）：a 的缩放是否为 8x4 布局。
- `backend`（`Literal[...]`，默认 `"auto"`）：`"auto"` 在 CUTLASS 可用时选择 CUTLASS，否则
  回退 cuDNN；SM120/SM121 上当 b12x 满足要求时优先 b12x。`"b12x"`（SM120/SM121）要求
  CUDA 13+、nvidia-cutlass-dsl >= 4.6.0、1-D swizzled 128x4 缩放、K 可被 128 整除；
  `"cute-dsl"` 要求 swizzled 1-D 缩放；`"cutedsl_low_latency"` 要求 SM100/SM103、M <= 8、
  K % 128 == 0、128x4 swizzled 缩放；`"trtllm"` 要求 b 以 128x4 swizzle 布局量化并洗牌，
  a 可用 128x4 或 8x4；`"cutlass"` 仅支持 1-D swizzled 缩放（`SfLayout.layout_128x4`）；
  `"cudnn"` 消费 F8_128x4 swizzled 布局（`use_8x4_sf_layout=False`），支持 SM100/103/110/120/121。

**返回**

- `torch.Tensor`：形状 `(m, n)`，bf16 或 fp16。

---

### `bmm_mxfp8`

```python
bmm_mxfp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal['cudnn', 'cutlass', 'auto'] = 'auto',
) -> torch.Tensor
```

**功能**：MXFP8 批矩阵乘。

**参数**

- `A`（`torch.Tensor`）：形状 `(b, m, k)`，fp8 e4m3 或 e5m2。
- `B`（`torch.Tensor`）：形状 `(b, k, n)`，必须列主序，fp8 e4m3 或 e5m2。量化连续的
  `[b, n, k]` 权重（使 32 元素缩放块沿 k，归约维）并传入量化张量的转置，例如
  `B = mxfp8_quantize(weight)[0].transpose(-2, -1)`（不要对转置调用 `.contiguous()`）。
- `A_scale`（`torch.Tensor`）：A 的缩放，uint8（fp8 e8m0），F8_128x4 swizzled 布局，由
  `mxfp8_quantize(..., is_sf_swizzled_layout=True)` 产生。
- `B_scale`（`torch.Tensor`）：B 的缩放，uint8（fp8 e8m0），F8_128x4 swizzled 布局。
- `dtype`（`torch.dtype`）：输出 dtype，bf16 或 fp16。
- `out`（`torch.Tensor`，可选）：形状 `(b, m, n)`，bf16 或 fp16，默认 `None`。
- `backend`（`Literal["cudnn", "cutlass", "auto"]`，默认 `"auto"`）：SM120/121 上 `"auto"`
  选择 CUTLASS，缩放必须 1-D swizzled（`SfLayout.layout_128x4`）；两种后端都读取 F8_128x4
  swizzled 布局，不支持线性布局。

**返回**

- `torch.Tensor`：形状 `(b, m, n)`，bf16 或 fp16。

---

### `mm_fp8`

```python
mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    backend: Literal['trtllm_low_latency', 'cutedsl_low_latency'] = 'trtllm_low_latency',
)
```

**功能**：FP8 矩阵乘。

**参数**

- `a`（`torch.Tensor`）：输入，形状 `(m, k)`，fp8 e4m3；`"cutedsl_low_latency"` 后端下也
  可为 e5m2。
- `b`（`torch.Tensor`）：`"trtllm_low_latency"` 后端下为形状 `(k // block_size, n,
  block_size)` 的 fp8 e4m3 权重（block_size 为 128），需用
  `prepare_low_latency_gemm_weights` 预处理；`"cutedsl_low_latency"` 后端下为未处理的连续
  `(n, k)` fp8 e4m3/e5m2 权重。
- `alpha`（`torch.Tensor`，可选）：输出缩放；`None` 时默认 1.0（不缩放）。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype。
- `out`（`torch.Tensor`，可选）：输出，形状 `(m, n)`；`None` 时分配。
- `backend`（`Literal["trtllm_low_latency", "cutedsl_low_latency"]`，默认 `"trtllm_low_latency"`）：
  `"trtllm_low_latency"` 针对小 M 优化；`"cutedsl_low_latency"` 要求 SM100/SM103、m <= 8、
  k 可被 128 整除。

**返回**

- `torch.Tensor`：形状 `(m, n)`，dtype `out_dtype`。

---

### `bmm_fp8`

```python
bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal['cudnn', 'cublas', 'cutlass', 'cute-dsl', 'auto'] = 'cublas',
) -> torch.Tensor
```

**功能**：FP8 批矩阵乘。

**参数**

- `A`（`torch.Tensor`）：形状 `(b, m, k)`，fp8 e4m3 或 e5m2。
- `B`（`torch.Tensor`）：形状 `(b, k, n)`，列主序，fp8 e4m3 或 e5m2。
- `A_scale`、`B_scale`（`torch.Tensor`）：A、B 的浮点缩放。
- `dtype`（`torch.dtype`）：输出 dtype，bf16 或 fp16。
- `out`（`torch.Tensor`，可选）：形状 `(b, m, n)`，bf16 或 fp16，默认 `None`。
- `backend`（`Literal[...]`，默认 `"cublas"`）：`"auto"` 在启用 autotune 时可从所有可用后端
  选择最佳 tactic；`"cute-dsl"` 使用为 SM100+（Blackwell/Rubin）优化的 CuTe-DSL 内核。

**返回**

- `torch.Tensor`：形状 `(b, m, n)`，bf16 或 fp16。

---

### `gemm_fp8_nt_blockscaled`

```python
gemm_fp8_nt_blockscaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal['MN', 'K']] = 'MN',
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

**功能**：使用块缩放缩放的 FP8 矩阵乘。块缩放是 groupwise 缩放的特例，粒度固定为
`(128, 128, 128)`。各参数语义见 `gemm_fp8_nt_groupwise`。

**参数**

- `a`（`torch.Tensor`）：FP8 输入，形状 `(M, K)`。
- `b`（`torch.Tensor`）：FP8 输入（转置权重），形状 `(N, K)`。
- `a_scale`、`b_scale`（`torch.Tensor`）：a、b 的 FP32 块缩放，布局由 `scale_major_mode` 决定。
- `scale_major_mode`（`Optional[Literal["MN", "K"]]`，默认 `"MN"`）：`"MN"` 将非收缩维放在
  major 方向；`"K"` 将收缩维放在 major 方向。
- `mma_sm`（`int`，默认 `1`）：每个 MMA 融合的 SM 数（`1` 或 `2`）。
- `out`（`torch.Tensor`，可选）：预分配输出 `(M, N)`；`None` 时分配。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `torch.bfloat16`。

**返回**

- `torch.Tensor`：形状 `(M, N)`，dtype `out_dtype`。

---

### `gemm_fp8_nt_groupwise`

```python
gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal['MN', 'K']] = None,
    mma_sm: int = 1,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal['cutlass', 'trtllm', 'cutile'] = 'cutlass',
) -> torch.Tensor
```

**功能**：使用 groupwise 缩放的 FP8 矩阵乘，可细粒度控制各维度的缩放粒度。当前仅在
NVIDIA Blackwell 架构上支持。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(m, k)`，fp8 e4m3 或 e5m2。
- `b`（`torch.Tensor`）：列主序输入，形状 `(n, k)`，fp8 e4m3 或 e5m2。
- `a_scale`（`torch.Tensor`）：cutlass 后端下为列主序缩放，形状 `(m, k // block_size)`
  （`scale_major_mode="K"`）或 `(k // block_size, m)`（`"MN"`）；trtllm 后端下
  `scale_major_mode` 应为 `None`，缩放形状 `(m, k // block_size)` 且第一维连续。
- `b_scale`（`torch.Tensor`）：cutlass 后端下为行主序缩放，形状 `(n // block_size,
  k // block_size)`（`"K"`）或 `(k // block_size, n // block_size)`（`"MN"`）；trtllm 后端下
  形状 `(k // block_size, n // block_size)` 且第一维连续。
- `scale_granularity_mnk`（`Tuple[int, int, int]`，默认 `(1, 128, 128)`）：缩放粒度
  `(m, n, k)`。
- `scale_major_mode`（`Literal["MN", "K"]`，默认 `None`）：缩放布局模式。
- `mma_sm`（`int`，默认 `1`）：MMA 使用的 SM 数（1 或 2）。每组 M 较大（>= 256）时 2 更快。
- `out`（`torch.Tensor`，可选）：输出 `(m, n)`；未给定时显式创建。
- `out_dtype`（`torch.dtype`，可选）：未给 `out` 时创建输出的 dtype，默认 `torch.bfloat16`。
- `backend`（`Literal["cutlass", "trtllm", "cutile"]`，默认 `"cutlass"`）：`"cutile"`
  （sm_100/103/110/120/121）为纯 `cuda.tile` Python 内核，v1 限制：`scale_major_mode` 必须为
  `"K"`（或 `None`），`scale_granularity_mnk` 必须为 `(1, 128, 128)`，`out_dtype` 仅
  bfloat16/float16。

**返回**

- `torch.Tensor`：形状 `(m, n)`。

**注意事项**

- `m` 应在调用前 padding 到 4 的倍数，以满足内核要求。

---

### `group_gemm_fp8_nt_groupwise`

```python
group_gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal['MN', 'K'] = 'MN',
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal['trtllm', 'cutile'] = 'trtllm',
    segment_alignment: int = 1,
) -> torch.Tensor
```

**功能**：使用 groupwise 缩放的 FP8 分组 GEMM。当前仅支持 NVIDIA Blackwell 架构。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(cum_m, k)`，dtype `torch.float8_e4m3fn` 或
  `torch.float8_e5m2`；`cum_m` 为分段长度累计和。
- `b`（`torch.Tensor`）：列主序输入，形状 `(batch_size, n, k)`。
- `a_scale`（`torch.Tensor`）：a 的列主序缩放，形状 `(cum_m, k // block_size)`
  （`"K"`）或 `(k // block_size, cum_m)`（`"MN"`），dtype `torch.float32`。
- `b_scale`（`torch.Tensor`）：b 的行主序缩放，形状 `(batch_size, n // block_size,
  k // block_size)`（`"K"`）或 `(batch_size, k // block_size, n // block_size)`（`"MN"`）。
- `m_indptr`（`torch.Tensor`）：分段长度 indptr，形状 `(batch_size + 1,)`，dtype `torch.int32`；
  每个元素必须是 4 的倍数。
- `scale_granularity_mnk`（默认 `(1, 128, 128)`）：缩放粒度。
- `scale_major_mode`（默认 `"MN"`）：缩放布局模式。
- `mma_sm`（`int`，默认 `1`）：MMA 使用的 SM 数（1 或 2）。
- `out`（`torch.Tensor`，可选）：输出 `(cum_m, n)`。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，`torch.bfloat16` 或 `torch.float16`。
- `backend`（`Literal["trtllm", "cutile"]`，默认 `"trtllm"`）：`"trtllm"` 使用
  TensorRT-LLM 分组 GEMM 内核；`"cutile"` 使用 cuTile Python 内核。
- `segment_alignment`（`int`，默认 `1`）：调用方**保证**的每个 `m_indptr` 分段偏移的行对齐。
  仅 cutile 后端。默认 `1`（任意）使用 gather 融合内核；传 128 的倍数（常见 MoE 情形）
  选择更快的对齐分段 TMA 内核。这是调用方契约：运行时无法校验（否则会破坏 CUDA-graph
  捕获），错误值会静默损坏输出。trtllm 忽略。

**返回**

- `torch.Tensor`：形状 `(cum_m, n)`。

**注意事项**

- `m_indptr` 中每个值应在调用前 padding 到 4 的倍数。

---

### `group_gemm_fp8_nt_groupwise_contiguous`

```python
group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    validate_indices: bool = False,
) -> torch.Tensor
```

**功能**：使用 CuTe DSL 在 SM100/SM103 上计算连续分组 FP8 GEMM。`a` 的每行乘以由
`m_indices` 选择的专家矩阵的转置。A 使用逐行、128 元素 K 块缩放；B 使用 128x128 块缩放。
输出为 bfloat16。

**参数**

- `a`（`torch.Tensor`）：连续 FP8 E4M3 输入，形状 `(M, K)`。
- `b`（`torch.Tensor`）：连续 FP8 E4M3 专家权重，形状 `(G, N, K)`。
- `a_scale`（`torch.Tensor`）：连续 float32 缩放，形状 `(M, K // 128)`。
- `b_scale`（`torch.Tensor`）：连续 float32 缩放，形状 `(G, N // 128, K // 128)`。
- `m_indices`（`torch.Tensor`）：连续 int32 专家索引，形状 `(M,)`，按非降序排序。内部专家
  边界必须对齐到 128 行；最后一个专家可以以部分 tile 结尾。所有值须满足 `0 <= index < G`；
  不支持 `-1` padding。
- `scale_granularity_mnk`（默认 `(1, 128, 128)`）：仅支持 `(1, 128, 128)`。
- `out`（`torch.Tensor`，可选）：连续 bfloat16 输出，形状 `(M, N)`；省略时分配。
- `out_dtype`（`torch.dtype`，可选）：分配时的输出 dtype，仅支持 `torch.bfloat16`；提供
  `out` 时忽略。
- `validate_indices`（`bool`，默认 `False`）：校验专家索引值、有序性与边界对齐。默认
  `False` 以避免每次调用的 GPU-CPU 同步。

**返回**

- `torch.Tensor`：提供或分配的 bfloat16 `(M, N)` 输出。

**注意事项**

- 需要 `nvidia-cutlass-dsl`。N、K 必须为正的 128 的倍数，G 为正。M 可为零（此时不启动
  内核）。所有张量须在同一 CUDA 设备且至少 16 字节对齐。索引在 `validate_indices=False`
  时不做检查，违反前置条件会导致未定义行为。

---

### `group_deepgemm_fp8_nt_groupwise`

```python
group_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
)
```

**功能**：使用 DeepGEMM 后端的 FP8 分组矩阵乘。对每个组 i，`b` 中对应组与 `a` 中由
`m_indices` 指定的行相乘（MoE 场景）。当前仅支持 NVIDIA Blackwell（SM100）。

**参数**

- `a`（`torch.Tensor`）：形状 `(m, k)`，FP8（`torch.float8_e4m3fn`）。
- `b`（`torch.Tensor`）：形状 `(batch_size, n, k)`，FP8。
- `a_scale`（`torch.Tensor`）：a 的缩放，形状 `(m, k // block_size)`，`torch.float32`。
- `b_scale`（`torch.Tensor`）：b 的缩放，形状 `(batch_size, n // block_size, k // block_size)`。
- `m_indices`（`torch.Tensor`）：组分配，形状 `(m,)`，`torch.int32`；`m_indices[i] = j`
  表示 a 的第 i 行与 b 的第 j 组相乘。
- `scale_granularity_mnk`（默认 `(1, 128, 128)`）：缩放粒度（a 为 per-token，b 为 128x128
  块缩放）。
- `out`（`torch.Tensor`，可选）：预分配 `(m, n)` 输出；未提供时新建。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype；提供 `out` 时忽略，默认 `torch.bfloat16`。

**返回**

- `torch.Tensor`：形状 `(m, n)`。

**注意事项**

- 需要 NVIDIA Blackwell（SM100）架构；所有输入张量须在同一 CUDA 设备。

---

### `batch_deepgemm_fp8_nt_groupwise`

```python
batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal['deepgemm', 'cake'] = 'deepgemm',
)
```

**功能**：使用 DeepGEMM 后端的 FP8 批矩阵乘。每个组 i：`output[i] = a[i][:masked_m[i]] @
b[i][:masked_m[i]].T`。当前仅支持 NVIDIA Blackwell（SM100）。

**参数**

- `a`（`torch.Tensor`）：形状 `(batch_size, m, k)`，FP8。
- `b`（`torch.Tensor`）：形状 `(batch_size, n, k)`，FP8。
- `a_scale`（`torch.Tensor`）：形状 `(batch_size, m, k // block_size)`，`torch.float32`。
- `b_scale`（`torch.Tensor`）：形状 `(batch_size, n // block_size, k // block_size)`。
- `masked_m`（`torch.Tensor`）：形状 `(batch_size,)`，`torch.int32`；`masked_m[i] = j` 表示
  a[i] 的前 j 行与 b 的第 i 组相乘。
- `expected_m`（`int`）：每组 M 期望值的 CPU 提示，正确设置可提升性能。
- `scale_granularity_mnk`（默认 `(1, 128, 128)`）：缩放粒度。
- `out`（`torch.Tensor`，可选）：预分配 `(batch_size, m, n)` 输出。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `torch.bfloat16`；提供 `out` 时忽略。
- `backend`（`{"deepgemm", "cake"}`，默认 `"deepgemm"`）：`"deepgemm"` 运行内置 DeepGEMM
  masked 内核；`"cake"` 运行生成的 Cake 程序（SM100a/SM103a，`(n, k)` 属于特定集合；
  bfloat16 输出；FP32 累加），对其它问题抛错。`"cake"` 下缩放也可以是服务路由的原生
  MN-major 打包 UE8M0 `torch.int32` 张量，无需转换。

**返回**

- `torch.Tensor`：形状 `(batch_size, m, n)`。

---

### `fp8_blockscale_gemm_sm90`

```python
fp8_blockscale_gemm_sm90(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

**功能**：带自动 swapAB 优化的 FP8 块缩放 GEMM。根据 M 维自动在普通内核与 swapAB 内核间
选择：M 小（< 32）时用 swapAB 内核。

支持的 dtype 组合：**BF16 + BF16 → BF16**（内部量化，无需缩放）；**BF16 + FP8 → BF16**；
**FP8 + FP8 → BF16**（W8A8，需缩放）。

**参数**

- `input`（`torch.Tensor`）：形状 `(M, K)` 的激活，BF16（内部量化）。
- `weight`（`torch.Tensor`）：形状 `(N, K)` 的权重，FP8（需 `weight_scale`）或 BF16
  （内部量化）。
- `input_scale`、`weight_scale`（`torch.Tensor`，可选）：weight 为 FP8 时 `weight_scale` 必需。
- `out`（`torch.Tensor`，可选）：形状 `(M, N)` 输出，`None` 时分配。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `torch.bfloat16`。

**返回**

- `torch.Tensor`：形状 `(M, N)`，dtype `out_dtype`。

**注意事项**

- 需要 NVIDIA Hopper（SM90）架构与 CUDA 12.8+。
- FP8 输入必须提供缩放。weight 缩放支持两种粒度：per-token（1x128 块，`(N, K//128)`）与
  per-block（128x128 块，`(N//128, K//128)`）。input 缩放仅支持 per-token 格式
  `(M, K//128)`。

---

### `prepare_low_latency_gemm_weights`

```python
prepare_low_latency_gemm_weights(
    w: torch.Tensor,
    permutation_indices_cache: Dict[torch.Size, torch.Tensor],
) -> torch.Tensor
```

**功能**：为低延迟 TRT-LLM GEMM 准备输入权重张量，包括洗牌与转换为块布局。

**参数**

- `w`（`torch.Tensor`）：待洗牌的权重，形状 `(n, k)`，fp8 e4m3。
- `permutation_indices_cache`（`dict`）：用于缓存置换索引的位置（计算开销大）。

**返回**

- `torch.Tensor`：洗牌并块布局后的权重，形状 `(k // 128, n, 128)`，fp8 e4m3。

---

### `group_gemm_mxfp8_mxfp4_nt_groupwise`

```python
group_gemm_mxfp8_mxfp4_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    mma_sm: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    swap_ab: bool = True,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

**功能**：使用 groupwise 缩放的 MXFP4 分组 GEMM。当前仅在 NVIDIA Blackwell、Blackwell
GeForce 与 DGX Spark 架构上支持。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(cum_m, k)`，`torch.float8_e4m3fn` 或 `e5m2`。
- `b`（`torch.Tensor`）：列主序输入，形状 `(batch_size, n, k // 2)`，`torch.uint8`。
- `a_scale`（`torch.Tensor`）：a 的列主序缩放，形状 `(cum_m_padded, k // 32)`，`torch.uint8`。
- `b_scale`（`torch.Tensor`）：b 的行主序缩放，形状 `(batch_size, n_padded, k // 32)`，`torch.uint8`。
- `m_indptr`（`torch.Tensor`）：形状 `(batch_size + 1,)`，`torch.int32`；每个元素须为 4 的倍数。
- `mma_sm`（`int`，默认 `1`）：MMA 使用的 SM 数（1 或 2，SM120/121 不支持 2）。
- `tile_m`（`int`，默认 `128`）：M 维 tile，必须为 128。
- `tile_n`（`int`，默认 `128`）：N 维 tile，必须为 32/64/128/192/256（SM120/121 仅 32/64/128；
  SM100/103/110 仅 64/128/192/256）。
- `tile_k`（`int`，默认 `128`）：K 维 tile，必须为 128 或 256（SM120/121 仅 128）。
- `swap_ab`（`bool`，默认 `True`）：是否交换 A、B。
- `out`（`torch.Tensor`，可选）：输出 `(cum_m, n)`。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，`torch.bfloat16` 或 `torch.float16`。

**返回**

- `torch.Tensor`：形状 `(cum_m, n)`。

---

### `group_gemm_nvfp4_nt_groupwise`

```python
group_gemm_nvfp4_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    swap_ab: bool = True,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor
```

**功能**：使用 groupwise 缩放的 NVFP4 分组 GEMM。当前仅在 NVIDIA Blackwell GeForce 与
DGX Spark 架构上实现。

**参数**

- `a`（`torch.Tensor`）：行主序输入，形状 `(cum_m, k // 2)`，`torch.uint8`（打包 NVFP4）。
- `b`（`torch.Tensor`）：列主序输入，形状 `(batch_size, n, k // 2)`，`torch.uint8`。
- `a_scale`（`torch.Tensor`）：a 的列主序缩放，形状 `(cum_m_padded, k // 16)`，`torch.uint8`。
- `b_scale`（`torch.Tensor`）：b 的行主序缩放，形状 `(batch_size, n_padded, k // 16)`，`torch.uint8`。
- `m_indptr`（`torch.Tensor`）：形状 `(batch_size + 1,)`，`torch.int32`；每个元素须为 4 的倍数。
- `alpha`（`torch.Tensor`，可选）：形状 `(batch_size, )`，`torch.float32`。
- `tile_m`（`int`，默认 `128`）：必须为 128。
- `tile_n`（`int`，默认 `128`）：必须为 32/64/128。
- `tile_k`（`int`，默认 `128`）：必须为 128 或 256。
- `swap_ab`（`bool`，默认 `True`）：是否计算 `Output^T = Weight^T Activation^T` 而非
  `Output = Activation Weight`。
- `out`（`torch.Tensor`，可选）：输出 `(cum_m, n)`。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，`torch.bfloat16` 或 `torch.float16`。

**返回**

- `torch.Tensor`：形状 `(cum_m, n)`。

## Router GEMM

以下函数的公共签名统一为：

```python
mm_M1_16_K<...>(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None
```

它们为特定模型的 MoE router 操作提供高度优化的 `out = mat_a @ mat_b`（`mat_a` 为 1-16
行 token 嵌入，`mat_b` 为专家路由权重矩阵），就地写入 `out`。

**公共参数说明**

- `mat_a`（`torch.Tensor`）：形状 `(M, K)`，`M` 为 token 数（1-16），`K` 为隐藏维。
  bfloat16、行主序（连续）。
- `mat_b`（`torch.Tensor`）：形状 `(K, N)`，`N` 为专家数。bfloat16、列主序（转置布局）。
- `out`（`torch.Tensor`）：预分配输出 `(M, N)`，行主序（连续），就地修改。dtype 因函数而异
  （见下表）。
- `launch_with_pdl`（`bool`，默认 `True`）：是否以 PDL 启动内核。

| 函数 | 模型 / 专精 | K | N | `out` dtype |
| --- | --- | --- | --- | --- |
| `mm_M1_16_K7168_N128` | Mistral Large 3 | 7168 | 128 | bfloat16 |
| `mm_M1_16_K7168_N256` | DeepSeek-V3 | 7168 | 256 | float32 |
| `mm_M1_16_K6144_N256` | GLM-MoE-DSA | 6144 | 256 | float32 |
| `mm_M1_16_K7168_N256_bf16` | DeepSeek-V3 | 7168 | 256 | bfloat16 |
| `mm_M1_16_K7168_N384` | Kimi-K2 | 7168 | 384 | float32 |
| `mm_M1_16_K7168_N384_bf16` | Kimi-K2 | 7168 | 384 | bfloat16 |
| `mm_M1_16_K7168_N896` | Kimi-K3 | 7168 | 896 | float32 |
| `mm_M1_16_K7168_N896_bf16` | Kimi-K3 | 7168 | 896 | bfloat16 |

**注意事项**

- `mm_M1_16_K7168_N128`、`mm_M1_16_K7168_N256`、`mm_M1_16_K6144_N256` 要求 Blackwell
  SM100/SM103 架构（经 `get_router_gemm_module(..., backend="cake")` 分发）。
- 其余 `*_bf16` 与 Kimi-K2/K3 float32 变体要求 SM90（Hopper）或更新架构。维度、stride 或
  dtype 不匹配时抛 `ValueError`。

---

### `tinygemm_bf16`

```python
tinygemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> None
```

**功能**：SM90+ 优化的微小 GEMM，`out = input @ weight.T + bias`（等价 `F.linear`）。
面向极小 batch（理想 1-8 行，单个 `TILE_N=8` tile 覆盖整个 batch 维），使用 Ampere 风格
HMMA 指令、TMA 异步批量加载，以及 `mma.sync.aligned.m16n8k16` 张量核指令，BF16 输入/权重/
bias/输出与 FP32 内部累加。改编自 TensorRT-LLM 的 `tinygemm2` 内核。

**参数**

- `input`（`torch.Tensor`）：形状 `(batch_size, input_features)`，bfloat16、连续，
  `input_features` 须为 64 的倍数。
- `weight`（`torch.Tensor`）：形状 `(output_features, input_features)`，bfloat16、连续
  （行主序），`output_features` 须为 16 的倍数。
- `out`（`torch.Tensor`）：预分配输出 `(batch_size, output_features)`，bfloat16、连续，就地修改。
- `bias`（`torch.Tensor`，可选）：形状 `(output_features,)`，bfloat16、连续；`None` 时用零 bias。
- `use_pdl`（`bool`，默认 `False`）：启用 PDL（流串行化）。为 `True` 时内核用
  `cudaGridDependencySynchronize()` 将 DMA 与前序内核计算重叠。仅当**所有**前序流操作也都
  使用 PDL 时才启用，否则内核会挂起。

**返回**

- `None`：就地写入 `out`。

**注意事项**

- 要求 SM90+（Hopper 或更新）。维度、dtype 或对齐约束被违反时抛 `ValueError`。
- SM100/SM103（B200/B300 类）设备上 bias 路径会分派到 `tinygemm2_sm100`（输出逐位相同、
  延迟更低）。设置 `FLASHINFER_DISABLE_TINYGEMM2_SM100=1` 可在所有设备强制参考实现。

---

### `tgv_gemm_sm100`

```python
tgv_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    a_descale: Optional[torch.Tensor] = None,
    b_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：在 SM100 架构上执行带自动 dtype 检测的 TGV GEMM，计算 `out = a @ b + bias`。
稠密输入须为同一浮点 dtype（`torch.bfloat16` 或 `torch.float16`）。提供缩放因子时也支持
块缩放 FP4 与 FP8 输入。

**参数**

- `a`（`torch.Tensor`）：形状 `(M, K)`，行主序。
- `b`（`torch.Tensor`）：形状 `(K, N)`，列主序。
- `bias`（`torch.Tensor`）：形状 `(N,)`，加到 `a @ b` 的每一行。
- `pdl`（`bool`，默认 `False`）：是否使用 PDL。
- `out`（`torch.Tensor`，可选）：预分配 `(M, N)`；`None` 时新建。
- `a_descale`、`b_descale`（`torch.Tensor`，可选）：块缩放 FP4/FP8 输入所需的 a、b 缩放。

**返回**

- `torch.Tensor`：形状 `(M, N)`，行主序。

**注意事项**

- 要求 SM100、SM103 或 SM107 架构。
- 稠密输入须同 dtype 且不使用缩放。`b` 期望列主序。
- 块缩放输入要求 `M <= 8` 且扁平 128x4 缩放布局。NVFP4 要求两个 FP4 操作数与 FP8 E4M3
  缩放，`K` 可被 64 整除。MX 缩放接受 FP4 或 FP8 操作数（含混合），使用 FP8 E8M0 缩放，
  `K` 可被 128 整除。FP4 每字节打包两个值，其物理 K 维为 `K // 2`。
- 支持的操作数 dtype：`torch.bfloat16`、`torch.float16`、`torch.float4_e2m1fn_x2`、
  `torch.float8_e4m3fn`、`torch.float8_e5m2`。

## cuTile GEMM

### `gemm_alpha_beta`

```python
gemm_alpha_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    num_sms: Optional[int] = None,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：带 alpha/beta 缩放的 GEMM，`C = alpha * (A @ B) + beta * C`。

**参数**

- `a`（`torch.Tensor`）：形状 `(M, K)`（`trans_a` 时为 `(K, M)`）。
- `b`（`torch.Tensor`）：形状 `(K, N)`（`trans_b` 时为 `(N, K)`，默认 `trans_b=True`）。
- `c`（`torch.Tensor`）：累加 / 输出张量，形状 `(M, N)`。`beta != 0` 时读取，就地写入。
- `trans_a`、`trans_b`（`bool`）：`a` / `b` 是否转置存储。
- `alpha`、`beta`（`float`）：缩放因子。
- `num_sms`（`int`，可选）：grid 使用 SM 数的可选覆盖。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：输出张量 `C`。

---

### `masked_bmm`

```python
masked_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    masked_m: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[torch.Tensor] = None,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：带逐 batch M 掩码的批矩阵乘 `C = A @ B`。每个 batch `q` 只产生输出的前
`masked_m[q]` 行；超出掩码的行未定义（调用方通常置零）。这是 MoE 风格专家路由使用的
grouped/masked GEMM。

**参数**

- `a`（`torch.Tensor`）：形状 `(Q, M, K)`（`transpose_a` 时为 `(Q, K, M)`），float16 或
  bfloat16，连续。
- `b`（`torch.Tensor`）：形状 `(Q, K, N)`（`transpose_b` 时为 `(Q, N, K)`），与 `a` 同 dtype，连续。
- `masked_m`（`torch.Tensor`）：逐 batch 行数，形状 `(Q,)`，int32。
- `transpose_a`、`transpose_b`（`bool`）：`a` / `b` 是否转置存储。
- `out`（`torch.Tensor`，可选）：输出 `(Q, M, N)`，省略时分配。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：输出 `C`，形状 `(Q, M, N)`。

---

### `masked_scaled_bmm`

```python
masked_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    block_scale_type: str,
    max_m_device: Optional[torch.Tensor] = None,
    transpose_a: bool = False,
    transpose_b: bool = True,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：掩码块缩放批矩阵乘（FP8/FP4 块缩放）。类似 `masked_bmm`，但用逐块缩放张量作用于
`a` 与 `b`。每个 batch `q` 只产生前 `masked_m[q]` 行。仅 Blackwell（使用 `mma_scaled`）。

**参数**

- `a`（`torch.Tensor`）：块缩放批输入（FP8/FP4），形状 `(Q, M, K_A)`。
- `b`（`torch.Tensor`）：块缩放批权重（FP8/FP4），形状 `(Q, N, K_B)`。
- `a_scale`、`b_scale`（`torch.Tensor`）：MX-swizzled 逐块缩放。
- `masked_m`（`torch.Tensor`）：逐 batch 行数，形状 `(Q,)`，int32。
- `block_scale_type`（`str`）：`"nvfp4"`、`"mxfp4"`、`"mxfp8"` 或 `"mixed"`。
- `max_m_device`（`torch.Tensor`，可选）：设备侧标量 `max(masked_m)`；省略时在设备上计算
  （避免主机同步）。
- `transpose_a`、`transpose_b`（`bool`）：是否转置存储（仅支持 NT）。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `torch.bfloat16`。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：输出 `C`，形状 `(Q, M, N)`。

---

### `ragged_bmm`

```python
ragged_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m: int,
    max_m_device: Optional[torch.Tensor] = None,
    transpose_a: bool = False,
    transpose_b: bool = True,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：非均匀 M 分段的 ragged 批矩阵乘。矩阵 `A` 沿 M 维展平，`m_indptr` 定义逐组分段
边界（分组 / 变长 GEMM）。

**参数**

- `a`（`torch.Tensor`）：展平批输入，M 维由 `m_indptr` 分段。
- `b`（`torch.Tensor`）：批权重，形状 `(Q, K, N)`（或按 `transpose_b` 转置）。
- `m_indptr`（`torch.Tensor`）：分段偏移，形状 `(Q + 1,)`，int32。
- `max_m`（`int`）：最大分段长度（主机 int，用于 grid 大小）。
- `max_m_device`（`torch.Tensor`，可选）：`max_m` 的设备侧副本。
- `transpose_a`、`transpose_b`（`bool`）：是否转置存储。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype，默认 `a.dtype`。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：ragged 输出张量。

---

### `ragged_block_scaled_bmm`

```python
ragged_block_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m: int,
    max_m_device: Optional[torch.Tensor] = None,
    transpose_a: bool = False,
    transpose_b: bool = True,
    out_dtype: Optional[torch.dtype] = None,
    segment_alignment: int = 128,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：ragged 块缩放批矩阵乘（FP8 块缩放）。类似 `ragged_bmm`，但用逐块缩放张量作用于
`a` 与 `b`（块缩放 FP8 输入反量化到 `out_dtype`）。

**参数**

- `a`、`b`（`torch.Tensor`）：块缩放（FP8）批输入；`a` 的 M 维由 `m_indptr` 分段。
- `a_scale`、`b_scale`（`torch.Tensor`）：a、b 的逐块缩放。
- `m_indptr`（`torch.Tensor`）：分段偏移，形状 `(Q + 1,)`，int32。
- `max_m`（`int`）：最大分段长度（主机 int，用于 grid 大小）。
- `max_m_device`（`torch.Tensor`，可选）：`max_m` 的设备侧副本。
- `transpose_a`、`transpose_b`（`bool`）：是否转置存储。
- `out_dtype`（`torch.dtype`，可选）：输出 dtype（如 `torch.bfloat16`）。
- `segment_alignment`（`int`，默认 `128`）：调用方保证的每个 `m_indptr` 分段偏移的行对齐。
  界定最大内部 tile（`BLOCK_M` 必须整除它）；传 256（且分段 256 对齐）可启用大 M 快速路径。
  这是调用方契约——运行时无法检查（否则破坏 CUDA-graph 捕获）。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：ragged 块缩放输出张量。

---

### `ragged_scaled_bmm`

```python
ragged_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m: int,
    block_scale_type: str,
    transpose_a: bool = False,
    transpose_b: bool = True,
    static_persistent: bool = True,
    swizzled_layout_a: bool = True,
    a_global_scale: Optional[torch.Tensor] = None,
    b_global_scale: Optional[torch.Tensor] = None,
    backend: Literal['cutile'] = 'cutile',
) -> torch.Tensor
```

**功能**：ragged 块缩放批矩阵乘（FP8/FP4 块缩放）。类似 `ragged_block_scaled_bmm`，但使用
`mma_scaled`（仅 Blackwell），支持 NVFP4/MXFP4/MXFP8/mixed 及可选全局缩放。矩阵 `A` 为
按 `m_indptr` 划分的 ragged 栈 `(total_m, K_A)`；`B` 为批 `(Q, N, K_B)`。输出 `(total_m, N)`
（float32）。

**参数**

- `a`（`torch.Tensor`）：ragged 块缩放输入（FP8/FP4），形状 `(total_m, K_A)`，由 `m_indptr` 分段。
- `b`（`torch.Tensor`）：批块缩放权重（FP8/FP4），形状 `(Q, N, K_B)`。
- `a_scale`、`b_scale`（`torch.Tensor`）：a、b 的 MX-swizzled 逐块缩放。
- `m_indptr`（`torch.Tensor`）：分段偏移，形状 `(Q + 1,)`；每项须为 128 的倍数。
- `max_m`（`int`）：任意单分段长度的上界（主机 int，用于 grid 大小）。
- `block_scale_type`（`str`）：`"nvfp4"`、`"mxfp4"`、`"mxfp8"` 或 `"mixed"`。
- `transpose_a`、`transpose_b`（`bool`）：是否转置存储（仅支持 NT）。
- `static_persistent`（`bool`，默认 `True`）：为与 ocean 签名兼容而保留。
- `swizzled_layout_a`（`bool`，默认 `True`）：`a_scale` 是否使用 swizzled 布局（仅 `True` 支持）。
- `a_global_scale`（`torch.Tensor`，可选）：a 的可选标量全局缩放。
- `b_global_scale`（`torch.Tensor`，可选）：b 的可选逐 batch `(Q,)` 全局缩放。
- `backend`（`str`，默认 `"cutile"`）：目前仅支持 `"cutile"`。

**返回**

- `torch.Tensor`：输出 `C`，形状 `(total_m, N)`。

## 分组 GEMM（CuTe-DSL, Blackwell）

### `grouped_gemm_nt_masked`

```python
grouped_gemm_nt_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    topk_weights: Optional[torch.Tensor] = None,
    idx_src_info: Optional[torch.Tensor] = None,
    rank_src_info: Optional[torch.Tensor] = None,
    out_ptrs: Optional[torch.Tensor] = None,
    num_ranks: int = 0,
    dst_signals: Optional[torch.Tensor] = None,
    sm_count: Optional[int] = None,
    barrier_flag_local: Optional[torch.Tensor] = None,
    barrier_flag_multicast: Optional[torch.Tensor] = None,
    is_combine_fusion: bool = False,
    is_swap_ab: bool = False,
    **kwargs,
)
```

**功能**：Blackwell（SM100/SM103）与 Rubin（SM107）上的掩码、批、块缩放 GEMM。按设备
compute capability 路由到架构特定内核（SM100/SM103 用 Blackwell 内核，SM107 用 Rubin）。
执行带缩放因子与可选逐 batch alpha 缩放的掩码批矩阵乘。

**参数**

- `lhs`（`Tuple[torch.Tensor, torch.Tensor]`）：`(A, SFA)`——左操作数及其缩放。`A` 逻辑
  形状 `(m, k, l)`（物理 `(l, m, k)`）；FP4 8-bit 存储时物理形状为 `(m, k/2, l)`。`SFA`
  逻辑形状 `(m32, m4, rm, k4, rk, l)`（物理 `(l, rm, rk, m32, m4, k4)`）。
- `rhs`（`Tuple[torch.Tensor, torch.Tensor]`）：`(B, SFB)`——右操作数及其缩放。`B` 逻辑
  形状 `(n, k, l)`；`SFB` 逻辑形状 `(n32, n4, rn, k4, rk, l)`。
- `out`（`torch.Tensor`）：形状 `(l, m, n)`，就地修改。
- `masked_m`（`torch.Tensor`）：1-D `int32`，形状 `(l,)`，给出每 batch 有效行数；超过
  `masked_m[batch]` 的行被忽略。
- `ab_dtype`（`str`）：`A`/`B` dtype，`"float4_e2m1fn"`、`"float8_e4m3fn"` 或 `"float8_e5m2"`。
- `sf_dtype`（`str`）：缩放 dtype，`"float8_e8m0fnu"` 或 `"float8_e4m3fn"`。
- `c_dtype`（`str`）：输出 `C` dtype，`"float16"`、`"bfloat16"`、`"float32"`、
  `"float8_e4m3fn"` 或 `"float8_e5m2"`。
- `sf_vec_size`（`int`）：缩放向量大小（通常 16 或 32）。
- `topk_weights`（`torch.Tensor`，可选）：2-D `float32`，形状 `(l, m)`，top-k 路由权重。
- `idx_src_info`、`rank_src_info`（`torch.Tensor`，可选）：2-D `int32`，形状 `(l, m)`，
  combine 融合路径的源索引 / rank 元数据。
- `out_ptrs`（`torch.Tensor`，可选）：1-D `int64`，形状 `(num_ranks,)`，多 rank combine 的
  远端输出指针。
- `num_ranks`（`int`，默认 `0`）：参与 combine 路径的 rank 数。
- `dst_signals`（`torch.Tensor`，可选）：combine 融合路径用于通知消费者的一维信号。
- `sm_count`（`int`，可选）：使用的 SM 数；`None` 时运行时在 CTA 配置下取最大可用值。
- `barrier_flag_local`、`barrier_flag_multicast`（`torch.Tensor`，可选）：1-D `int32`，
  形状 `(sm_count,)`，本地 / 多播屏障标志。
- `is_combine_fusion`（`bool`，默认 `False`）：是否启用融合的 GEMM + combine 模式。
- `is_swap_ab`（`bool`，默认 `False`）：是否交换 `lhs`/`rhs`。
- `**kwargs`：附加关键字参数，目前识别 `mma_tiler_mn`（`Tuple[int, int]`，默认 `(128, 128)`；
  `[0] == 256` 启用 2-CTA MMA 路径，`is_combine_fusion=True` 时必须为 `(128, 128)`）、
  `cluster_shape_mn`（默认 `(1, 1)`）、`alpha`（可选 1-D `(l,)` 逐 batch 缩放，`out =
  alpha * (A @ B)`）、`alpha_dtype`（提供 `alpha` 时必需）。

**注意事项**

- 布局约定：`l` 为 batch，`m`/`n` 为行 / 列数，`k` 为收缩维；`m/n32 = 32`、`m/n4 = 4`、
  `k4 = 4`；`m32 * m4 * rm = M`（M 为 m 上取整到 128 的倍数），`n32 * n4 * rn = N`，
  `k4 * rk = K`（K 为 `k / sf_vec_size` 上取整到 4 的倍数）。
- 掩码按 batch 通过 `masked_m` 应用；提供 `alpha` 时每 batch 输出乘以其对应 alpha。

## 分组 GEMM（Ampere/Hopper）：`SegmentGEMMWrapper`

Wrapper for segment GEMM kernels（segment GEMM 内核封装）。

**示例**

```python
>>> import torch
>>> from flashinfer import SegmentGEMMWrapper
>>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
>>> segment_gemm = SegmentGEMMWrapper(workspace_buffer)
>>> seq_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
>>> x = torch.randn(10, 128, device="cuda", dtype=torch.float16)
>>> weights = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
>>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens)
>>> y.shape
torch.Size([10, 256])
```

### `SegmentGEMMWrapper.__init__`

```python
__init__(self, float_workspace_buffer: torch.Tensor, backend: str = 'auto') -> None
```

**功能**：初始化封装。

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：内核工作区缓冲，用于在 cutlass segment GEMM
  内核中存放中间结果。建议大小 128 MiB。
- `backend`（`str`，默认 `"auto"`）：后端选择。`"auto"` 让运行时选择最佳可用实现（若可用
  则 CUTLASS Hopper，否则 Ampere 兼容内核）。显式值为 `"sm80"`（SM80/Ampere CUTLASS 路径）
  与 `"sm90"`（SM90 Hopper 特化）。

### `SegmentGEMMWrapper.reset_workspace_buffer`

```python
reset_workspace_buffer(
    self,
    float_workspace_buffer: torch.Tensor,
    int_workspace_buffer: torch.Tensor,
) -> None
```

**功能**：重置工作区缓冲。

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：新的 float 工作区缓冲。
- `int_workspace_buffer`（`torch.Tensor`）：新的 int 工作区缓冲。

### `SegmentGEMMWrapper.run`

```python
run(
    self,
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    out: Optional[torch.Tensor] = None,
    seg_lens: Optional[torch.Tensor] = None,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：运行 segment GEMM 内核。计算一个批输入张量（行数可变、列数固定）与一批权重
张量（行、列固定）之间的矩阵乘：`y[i] = x[i] × W[i]`；提供 `weight_indices` 时按索引选择
权重：`y[i] = x[i] × W[weight_indices[i]]`。输入 `x` 与输出 `y` 使用 ragged 张量表示，每个
`x[i]` 是拼接张量的一个分段，用 `seg_len` 或 `seg_indptr` 指示分段起止。

**参数**

- `x`（`torch.Tensor`）：输入，形状 `(sum(seg_lens), d_in)`。
- `weights`（`torch.Tensor`）：3D 权重，`weight_column_major=False` 时形状
  `(num_weights, d_in, d_out)`，否则 `(num_weights, d_out, d_in)`。
- `batch_size`（`int`）：分段数。
- `weight_column_major`（`bool`）：权重是否列主序。
- `out`（`torch.Tensor`，可选）：输出，形状 `(sum(seg_lens), d_out)`；未提供时内部新建。
- `seg_lens`（`torch.Tensor`，可选）：每个分段的长度，形状 `(batch_size,)`，1D `torch.int64`。
- `seg_indptr`（`torch.Tensor`，可选）：分段 indptr，形状 `(batch_size + 1,)`，1D
  `torch.int64`。提供时忽略 `seg_lens`，否则内部由 `seg_lens` 计算。
- `weight_indices`（`torch.Tensor`，可选）：每个分段选择的权重索引，形状 `(batch_size,)`，
  1D `torch.int64`。

**返回**

- `torch.Tensor`：形状 `(sum(seg_lens), d_out)`。

## PrimsTS 稠密 FP8 / NVFP4 GEMM（SM100, SM103, SM107）

以下算子均为关键字参数形式，签名中 `a`、`weight` 等省略类型注解；返回类型见各条。

### `fp8_linear`

```python
fp8_linear(
    a,
    weight,
    a_scale,
    weight_scale,
    bias = None,
    *,
    qkv_scale = None,
    config = None,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：FP8 `[M, K]` 乘 `[N, K]`，带 per-token 与 per-channel 缩放。在 SM100、SM103、
SM107 上支持。没有全局缩放。

**参数**

- `a`：`[M, K]` `torch.float8_e4m3fn` 激活，`K` 须为 128 的倍数。
- `weight`：`[N, K]` `torch.float8_e4m3fn` 权重。
- `a_scale`：`[M]` float32 per-token 缩放。
- `weight_scale`：`[N]` float32 per-channel 缩放。
- `bias`：可选 `[N]` bfloat16 bias。
- `qkv_scale`：可选 `[3]` float32 缩放，当 `N` 均分为三组时分别作用于 Q、K、V 组。
- `config`：可选 `PrimsTsGemmConfig`，匹配本次调用；`None` 选择 autotune tactic。
- `out_dtype`：`torch.bfloat16` 或 `torch.float8_e4m3fn`，默认 bfloat16。
- `output_quant_scale`：可选 `[1]` float32，FP8 输出编码缩放。提供时每次调用都读取。省略时
  内核使用 `1`，返回的缩放是独立张量；写入它不影响后续调用。
- `out`：可选预分配输出，BF16 / FP8 形状均为 `[M, N]`。

**返回**

- 输出张量。当 `out_dtype` 为 FP8 且未提供 `output_quant_scale` 时，返回 `(output, scale)`。

---

### `fp8_linear_swiglu`

```python
fp8_linear_swiglu(
    a,
    weight,
    a_scale,
    weight_scale,
    bias = None,
    *,
    config = None,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：FP8 融合线性与 SwiGLU，带 per-token 与 per-channel 缩放。权重、缩放与 bias 行是
相邻的 gate 与 activation 对。逻辑输出宽度为 `N / 2`。在 SM100、SM103、SM107 上支持。

**参数**

- `a`：`[M, K]` `torch.float8_e4m3fn` 激活，`K` 须为 128 的倍数。
- `weight`：`[N, K]` `torch.float8_e4m3fn` 权重，`N` 须为偶数。
- `a_scale`：`[M]` float32 per-token 缩放。
- `weight_scale`：`[N]` float32 per-channel 缩放。
- `bias`：可选 `[N]` bfloat16 bias。
- `config`：可选 `PrimsTsGemmConfig`。
- `out_dtype`：`torch.bfloat16` 或 `torch.float8_e4m3fn`，默认 bfloat16。
- `output_quant_scale`：可选 `[1]` float32，同 `fp8_linear` 语义。
- `out`：可选预分配输出，形状 `[M, N / 2]`。

**返回**

- 输出张量。`out_dtype` 为 FP8 且未提供 `output_quant_scale` 时返回 `(output, scale)`。

---

### `fp8_qkv_qknorm_rope`

```python
fp8_qkv_qknorm_rope(
    a,
    qkv_weight,
    a_scale,
    qkv_weight_scale,
    q_norm_weight,
    k_norm_weight,
    cos_sin,
    positions,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_neox = False,
    qkv_scale = None,
    config = None,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：FP8 QKV 投影，带 QK RMSNorm 与 RoPE。`positions[i]` 选择 token `i` 使用的 `cos_sin`
行；超出 `[0, M)` 的值使用第 0 行。在 SM100、SM103、SM107 上支持。`num_q_heads` 必须等于
`num_kv_heads`，`head_dim` 必须为 128，`is_neox` 必须为 false。

**参数**

- `a`：`[M, K]` `torch.float8_e4m3fn` 激活，`K` 须为 128 的倍数。
- `qkv_weight`：`[N, K]` `torch.float8_e4m3fn` 权重，打包为三个等分的 Q、K、V 组；`N` 为
  `3 * num_q_heads * head_dim`。
- `a_scale`：`[M]` float32 per-token 缩放。
- `qkv_weight_scale`：`[N]` float32 per-channel 缩放。
- `q_norm_weight`、`k_norm_weight`：`[head_dim]` bfloat16 Q / K 的 RMSNorm 权重。
- `cos_sin`：`[M, head_dim]` float32 表，每行前半为余弦、后半为正弦。
- `positions`：`[M]` int64，`cos_sin` 的索引。
- `num_q_heads`、`num_kv_heads`：查询 / 键值头数，两者须相等。
- `head_dim`：头大小，仅支持 128。
- `is_neox`：RoPE 布局，仅支持 false（交错对布局）。
- `qkv_scale`：可选 `[3]` float32，分别作用于 Q、K、V 组。
- `config`：可选 `PrimsTsGemmConfig`。
- `out_dtype`：`torch.bfloat16` 或 `torch.float8_e4m3fn`，默认 bfloat16。
- `output_quant_scale`：可选 `[1]` float32，同 `fp8_linear` 语义。
- `out`：可选预分配输出 `[M, N]`。

**返回**

- 输出张量。`out_dtype` 为 FP8 且未提供 `output_quant_scale` 时返回 `(output, scale)`。

---

### `fp4_linear`

```python
fp4_linear(
    a_packed,
    a_block_scale,
    a_global_scale,
    weight_packed,
    weight_block_scale,
    weight_global_scale,
    bias = None,
    *,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：打包 NVFP4 `[M, K]` 乘 `[N, K]`。块缩放为连续的 1D 128x4 FP8-E4M3 缓冲区；全局
缩放为主机 float。在 SM100、SM103、SM107 上支持。逻辑 `K` 须为 256 的倍数。

**参数**

- `a_packed`：`[M, K / 2]` uint8 激活，每字节两个 FP4 值。
- `a_block_scale`：`a_packed` 的 128x4 块缩放。
- `a_global_scale`：主机 float，乘入激活缩放。
- `weight_packed`：`[N, K / 2]` uint8 权重。
- `weight_block_scale`：`weight_packed` 的 128x4 块缩放。
- `weight_global_scale`：主机 float，乘入权重缩放。
- `bias`：可选 `[N]` bfloat16 bias。
- `out_dtype`：`torch.bfloat16` 或 `torch.float8_e4m3fn`，默认 bfloat16。NVFP4 输出仅由
  `fp4_linear_swiglu` 提供。
- `output_quant_scale`：可选 `[1]` float32，FP8 输出编码缩放。
- `out`：可选预分配输出 `[M, N]`。

**返回**

- 输出张量。`out_dtype` 为 FP8 且未提供 `output_quant_scale` 时返回 `(output, scale)`。

---

### `fp4_linear_swiglu`

```python
fp4_linear_swiglu(
    a_packed,
    a_block_scale,
    a_global_scale,
    weight_packed,
    weight_block_scale,
    weight_global_scale,
    bias = None,
    *,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：打包 NVFP4 融合线性与 SwiGLU。块缩放为连续的 1D 128x4 FP8-E4M3 缓冲区；权重行是
相邻的 gate 与 activation 对，逻辑输出宽度为 `N / 2`。在 SM100、SM103、SM107 上支持。

**参数**

- `a_packed`：`[M, K / 2]` uint8 激活，每字节两个 FP4 值。
- `a_block_scale`：`a_packed` 的 128x4 块缩放。
- `a_global_scale`：主机 float，乘入激活缩放。
- `weight_packed`：`[N, K / 2]` uint8 权重，`N` 须为偶数。
- `weight_block_scale`：`weight_packed` 的 128x4 块缩放。
- `weight_global_scale`：主机 float，乘入权重缩放。
- `bias`：可选 `[N]` bfloat16 bias。
- `out_dtype`：`torch.bfloat16`、`torch.float8_e4m3fn` 或 `torch.uint8`（打包 NVFP4 输出），
  默认 bfloat16。NVFP4 输出要求逻辑 `N / 2` 是 16 的倍数。
- `output_quant_scale`：可选 `[1]` float32，FP8 或 NVFP4 输出的编码缩放。
- `out`：可选预分配输出。BF16 与 FP8 形状为 `[M, N / 2]`；打包 NVFP4 形状为 `[M, N / 4]`。

**返回**

- 输出张量。FP8 输出且无 `output_quant_scale` 时返回 `(output, scale)`。NVFP4 输出返回
  `(output, block_scale)`，当未提供 `output_quant_scale` 时返回 `(output, block_scale, scale)`。

---

### `fp4_qkv_qknorm_rope`

```python
fp4_qkv_qknorm_rope(
    a_packed,
    a_block_scale,
    a_global_scale,
    qkv_weight_packed,
    qkv_weight_block_scale,
    qkv_weight_global_scale,
    q_norm_weight,
    k_norm_weight,
    cos_sin,
    positions,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_neox = False,
    qkv_scale = None,
    out_dtype = torch.bfloat16,
    output_quant_scale = None,
    out = None,
)
```

**功能**：打包 NVFP4 QKV 投影，带 QK RMSNorm 与 RoPE。`positions[i]` 选择 token `i` 使用的
`cos_sin` 行；超出 `[0, M)` 的值使用第 0 行。在 SM100、SM103、SM107 上支持。`num_q_heads`
须等于 `num_kv_heads`，`head_dim` 须为 128，`is_neox` 须为 false。输出为 bfloat16。

**参数**

- `a_packed`：`[M, K / 2]` uint8 激活，每字节两个 FP4 值。
- `a_block_scale`：`a_packed` 的 128x4 块缩放。
- `a_global_scale`：主机 float，乘入激活缩放。
- `qkv_weight_packed`：`[N, K / 2]` uint8 权重，打包为三个等分的 Q、K、V 组。
- `qkv_weight_block_scale`：`qkv_weight_packed` 的 128x4 块缩放。
- `qkv_weight_global_scale`：主机 float，乘入权重缩放。
- `q_norm_weight`、`k_norm_weight`：`[head_dim]` bfloat16 Q / K 的 RMSNorm 权重。
- `cos_sin`：`[M, head_dim]` float32 表，每行前半为余弦、后半为正弦。
- `positions`：`[M]` int64，`cos_sin` 的索引。
- `num_q_heads`、`num_kv_heads`：须相等。
- `head_dim`：仅支持 128。
- `is_neox`：仅支持 false（交错对布局）。
- `qkv_scale`：可选 `[3]` float32，分别作用于 Q、K、V 组。
- `out_dtype`：输出 dtype，仅支持 `torch.bfloat16`。
- `output_quant_scale`：不使用；该 epilogue 不支持量化输出。
- `out`：可选预分配 `[M, N]` bfloat16 输出。

**返回**

- bfloat16 输出张量。

## 相关链接 / 参考

- `flashinfer.quantization`：为上述 FP4/MXFP8 GEMM 生成量化张量与缩放布局。
- `flashinfer.grouped_mm`：面向 MoE 的分组 GEMM 门面（见 `api-grouped-mm.md`）。

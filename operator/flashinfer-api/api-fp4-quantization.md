# FP4 量化（`flashinfer.fp4_quantization`，向后兼容垫片）

> 源码：`flashinfer/fp4_quantization.py`
> 上游文档：`docs/api/fp4_quantization.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.fp4_quantization` 现在是一个**向后兼容垫片**：它重新导出
`flashinfer.quantization.fp4_quantization`（及其在 `flashinfer.quantization` 上的规范
重新导出）中的同一批符号，使 `from flashinfer.fp4_quantization import fp4_quantize`
之类的既有代码继续可用。

本页**有意不重新记录**这些 FP4 符号：每个符号与
[`api-quantization.md`](./api-quantization.md) 上渲染的是同一 Python 对象；在此重复
autosummary 条目会让 Sphinx 在 `-W` 下发出 “duplicate object description” 警告。

## 导入

```python
# 既有代码（兼容，继续可用）
from flashinfer.fp4_quantization import fp4_quantize, nvfp4_quantize, SfLayout

# 新代码应使用规范位置
from flashinfer.quantization import fp4_quantize, nvfp4_quantize, SfLayout
from flashinfer.quantization.fp4_quantization import fp4_quantize
```

## API 列表

下表列出 `flashinfer.fp4_quantization` 曾经承载、现由本垫片重新导出的符号。完整签名、
参数、返回值与注意事项见 [`api-quantization.md`](./api-quantization.md)。

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `SfLayout` | 枚举（IntEnum） | 缩放因子内存布局（`layout_128x4` / `layout_8x4` / `layout_linear`）。 |
| `fp4_quantize` | 函数 | 通用 FP4 量化（NVFP4 / MXFP4）。 |
| `nvfp4_quantize` | 函数 | NVFP4 量化（含可选 per-token 激活缩放）。 |
| `nvfp4_batched_quantize` | 函数 | 批处理 NVFP4 量化。 |
| `block_scale_interleave` | 函数 | 将块缩放张量 swizzle 为 FP4 布局。 |
| `nvfp4_block_scale_interleave` | 函数 | `block_scale_interleave` 的别名（同一对象）。 |
| `e2m1_and_ufp8sf_scale_to_float` | 函数 | 将 E2M1 + UFP8 缩放反量化为 float32。 |
| `scaled_fp4_grouped_quantize` | 函数 | 带逐行掩码的批处理 NVFP4 量化。 |
| `silu_and_mul_nvfp4_quantize` | 函数 | 融合 SwiGLU 与 NVFP4 量化。 |
| `shuffle_matrix_a` | 函数 | 矩阵 A 行洗牌（TRT-LLM-gen 等价实现）。 |
| `shuffle_matrix_sf_a` | 函数 | 线性布局缩放因子行洗牌并重排到 128x4。 |
| `nvfp4_kv_quantize` | 函数 | KV cache NVFP4 量化（线性块缩放布局，SM100+）。 |
| `nvfp4_kv_dequantize` | 函数 | KV cache NVFP4 反量化（SM80+）。 |
| `nvfp4_quantize_paged_kv_cache` | 函数 | 分页 KV cache NVFP4 量化（trtllm-gen MHA 布局）。 |

另有以下**非公共**符号也会被重新导出，供部分测试使用，不属于稳定 API：
`get_fp4_quantization_module`、`gen_fp4_quantization_module`、
`gen_fp4_quantization_sm90_module`、`gen_fp4_quantization_sm100_module`、
`gen_fp4_quantization_sm103_module`、`gen_fp4_quantization_sm107_module`、
`gen_fp4_quantization_sm110_module`、`gen_fp4_quantization_sm120_module`、
`gen_fp4_quantization_sm120f_module`、`gen_fp4_quantization_sm121_module`、
`nvfp4_kv_dequantize_paged`、`_pad_scale_factors`、`_compute_swizzled_layout_sf_size`。

## API 详情（签名速查）

以下签名与规范源码 `flashinfer/quantization/fp4_quantization.py` 完全一致；垫片中它们是
同一对象。

### `fp4_quantize`

```python
fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    is_global_scale_inversed: bool = False,
    enable_pdl: Optional[bool] = None,
    backend: str = "cuda",
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：将输入张量量化为 FP4 格式，支持 NVFP4 与 MXFP4 配方。详见
[`api-quantization.md#fp4_quantize`](./api-quantization.md)。

---

### `nvfp4_quantize`

```python
nvfp4_quantize(
    a,
    a_global_sf,
    sfLayout=SfLayout.layout_128x4,
    do_shuffle=False,
    sf_vec_size=16,
    enable_pdl=None,
    backend: str = "cuda",
    per_token_activation: bool = False,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：将输入张量量化为 NVFP4 格式。详见
[`api-quantization.md#nvfp4_quantize`](./api-quantization.md)。

---

### `nvfp4_batched_quantize`

```python
nvfp4_batched_quantize(
    a,
    a_global_sf,
    sf_vec_size=16,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：将批处理输入张量量化为 NVFP4 格式。

---

### `block_scale_interleave`

```python
block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor
```

**功能**：将 FP4 块缩放张量 swizzle，输出在 `m` 维 padding 到 128 的倍数。
`nvfp4_block_scale_interleave` 是其别名（同一 Python 对象）。

---

### `e2m1_and_ufp8sf_scale_to_float`

```python
e2m1_and_ufp8sf_scale_to_float(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
) -> torch.Tensor
```

**功能**：将 E2M1 张量结合 UFP8 缩放反量化为 float32。低于 SM90 时走 CPU 回退。

---

### `scaled_fp4_grouped_quantize`

```python
scaled_fp4_grouped_quantize(
    a,
    mask,
    a_global_sf,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：使用逐行掩码将批处理输入张量量化为 NVFP4。

---

### `silu_and_mul_nvfp4_quantize`

```python
silu_and_mul_nvfp4_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    enable_pdl: Optional[bool] = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：单个 CuTe-DSL 内核中融合 SwiGLU 与 NVFP4 量化，要求 SM100+。

---

### `shuffle_matrix_a`

```python
shuffle_matrix_a(input_tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor
```

**功能**：TRT-LLM-gen `shuffleMatrixA` 的 PyTorch 等价实现。

---

### `shuffle_matrix_sf_a`

```python
shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
)
```

**功能**：面向线性布局缩放因子的 `shuffleMatrixSfA` 实现，重排进入 128x4 布局。

---

### `nvfp4_kv_quantize`

```python
nvfp4_kv_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用线性块缩放布局将数据量化为 NVFP4 KV-cache 格式，要求 SM100+。

---

### `nvfp4_kv_dequantize`

```python
nvfp4_kv_dequantize(
    fp4_data: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor
```

**功能**：使用线性块缩放布局对 NVFP4 KV cache 做 GPU 反量化，要求 SM80+。

---

### `nvfp4_quantize_paged_kv_cache`

```python
nvfp4_quantize_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_layout: str = "HND",
    k_global_sf: Optional[torch.Tensor] = None,
    v_global_sf: Optional[torch.Tensor] = None,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    float,
    float,
]
```

**功能**：将分页 KV cache 量化为 NVFP4，供 trtllm-gen MHA 内核使用。

---

### `SfLayout`

```python
class SfLayout(IntEnum):
    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2
```

**功能**：缩放因子内存布局枚举。定义于 `flashinfer/tllm_enums.py`。

## 相关链接 / 参考

- [量化（`flashinfer.quantization`）](./api-quantization.md) —— 上述所有符号的规范
  FP4 / FP8 / packbits API 参考。
- `flashinfer.gemm.mm_fp4`：消费本垫片导出函数产出的量化张量。

# 旋转位置编码（`flashinfer.rope`）

> 源码：`flashinfer/rope.py`
> 上游文档：`docs/api/rope.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.rope` 提供应用旋转位置编码（Rotary Position Embedding，RoPE）的内核。算子分为两组：

- **即时计算 cos/sin 的 ragged 变体**：以 RaggedTensor 布局（`q`/`k` 形状 `(nnz, num_heads, head_dim)`，
  配合 `indptr` 或 `pos_ids`）输入，cos/sin 在内核内即时计算。支持普通 RoPE、Llama 3.1（YARN）风格 RoPE，
  以及原地 / 非原地两种形式。
- **使用预计算 cos/sin cache 的变体**：与 SGL/vLLM 实现兼容，`cos_sin_cache` 形状 `(max_seq_len, rotary_dim)`。
- **量化变体**：`rope_quantize_fp8` 系列在应用 RoPE 的同时将 Q/K 量化为 FP8，并可将 K/V 写入分页 KV cache。

RaggedTensor 布局说明：`indptr` 表示每个 segment 的起始指针，第 i 个 segment 的 query 为
`q[indptr[i]:indptr[i+1]]`，`indptr[0]` 恒为 0，`indptr[-1]` 为 batch 中 query/key 总数。

## 导入

```python
import flashinfer

from flashinfer.rope import (
    apply_rope_inplace,
    apply_llama31_rope_inplace,
    apply_rope,
    apply_llama31_rope,
    apply_rope_pos_ids,
    apply_rope_pos_ids_inplace,
    apply_llama31_rope_pos_ids,
    apply_llama31_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
    rope_quantize_fp8,
    rope_quantize_fp8_append_paged_kv_cache,
    mla_rope_quantize_fp8,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `apply_rope_inplace` | 函数 | 对 ragged Q/K 原地应用 RoPE（按 `indptr`/`offsets`）。 |
| `apply_llama31_rope_inplace` | 函数 | 原地应用 Llama 3.1 风格 RoPE。 |
| `apply_rope` | 函数 | 对 ragged Q/K 应用 RoPE（非原地）。 |
| `apply_llama31_rope` | 函数 | 应用 Llama 3.1 风格 RoPE（非原地）。 |
| `apply_rope_pos_ids` | 函数 | 按显式 `pos_ids` 应用 RoPE（非原地）。 |
| `apply_rope_pos_ids_inplace` | 函数 | 按显式 `pos_ids` 原地应用 RoPE。 |
| `apply_llama31_rope_pos_ids` | 函数 | 按显式 `pos_ids` 应用 Llama 3.1 风格 RoPE（非原地）。 |
| `apply_llama31_rope_pos_ids_inplace` | 函数 | 按显式 `pos_ids` 原地应用 Llama 3.1 风格 RoPE。 |
| `apply_rope_with_cos_sin_cache` | 函数 | 使用预计算 cos/sin cache 应用 RoPE（非原地）。 |
| `apply_rope_with_cos_sin_cache_inplace` | 函数 | 使用预计算 cos/sin cache 原地应用 RoPE。 |
| `rope_quantize_fp8` | 函数 | 应用 RoPE 并量化为 FP8。 |
| `rope_quantize_fp8_append_paged_kv_cache` | 函数 | 应用 RoPE、量化为 FP8 并追加 K/V 到分页 cache。 |
| `mla_rope_quantize_fp8` | 函数 | MLA 布局下的 `rope_quantize_fp8` 薄封装。 |

## API 详情

### 通用参数约定（ragged 变体）

下列 ragged 变体的公共参数含义一致：

- `q`（`torch.Tensor`）：Query ragged 张量，形状 `(nnz, num_q_heads, head_dim)`，`nnz` 为 `indptr` 最后一个元素。
- `k`（`torch.Tensor`）：Key ragged 张量，形状 `(nnz, num_k_heads, head_dim)`。
- `indptr`（`torch.Tensor`）：形状 `(batch_size + 1)` 的指针张量。
- `offsets`（`torch.Tensor`）：batch 中每个 query 的相对位置偏移，形状 `(batch_size)`。
- `pos_ids`（`torch.Tensor`）：位置索引，形状 `(nnz)`。
- `rotary_dim`（`Optional[int]`，默认 `None`）：应用 RoPE 的维度数；`None` 时对整个 head 维度应用 RoPE，
  否则仅对前 `rotary_dim` 维应用。
- `interleave`（`bool`，默认 `False`）：是否使用最后一维交错布局。
  - `True`：最后一维交错，旋转偶数维（`[..., ::2]`）与奇数维（`[..., 1::2]`）。
  - `False`：不交错，旋转前半维（`[..., :head_dim//2]`）与后半维（`[..., head_dim//2:]`）。
- `rope_scale`（`float`）：RoPE 缩放因子。
- `rope_theta`（`float`）：RoPE theta 值。
- `low_freq_factor` / `high_freq_factor` / `old_context_len`（Llama 3.1 变体）：Llama 3.1 RoPE 的低/高频
  因子与旧上下文长度，默认分别为 `1`、`4`、`8192`。

### `apply_rope_inplace`

```python
apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None
```

**功能**：对一批 query/key（RaggedTensor 存储）**原地**应用 RoPE，cos/sin 在内核内即时计算。
默认参数为 `rope_scale=1`、`rope_theta=1e4`。

**返回**

- `None`：`q`、`k` 被原地修改。

**示例**

```python
import torch
import flashinfer

batch_size = 128
qkv_len = 1024
num_qo_heads, num_kv_heads, head_dim = 32, 32, 128
nnz = batch_size * qkv_len
qkv_packed = torch.randn(
    nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim,
    dtype=torch.float16, device="cuda:0",
)
q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
k = qkv_packed[
    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
].reshape(nnz, num_kv_heads, head_dim)
indptr = torch.tensor(
    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
)
offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
flashinfer.apply_rope_inplace(q, k, indptr, offsets)
```

---

### `apply_rope`

```python
apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：对一批 query/key（RaggedTensor 存储）应用 RoPE，**非原地**，返回新的旋转结果。

**返回**

- `q_rope`（`torch.Tensor`）：旋转后的 query，形状 `(nnz, num_q_heads, head_dim)`。
- `k_rope`（`torch.Tensor`）：旋转后的 key，形状 `(nnz, num_k_heads, head_dim)`。

---

### `apply_rope_pos_ids`

```python
apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：按显式 `pos_ids`（形状 `(nnz)`）对一批 query/key 应用 RoPE，非原地。

**返回**

- `(q_rope, k_rope)`：旋转后的 query/key。

---

### `apply_rope_pos_ids_inplace`

```python
apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None
```

**功能**：按显式 `pos_ids` 对一批 query/key **原地**应用 RoPE。

**返回**

- `None`：`q`、`k` 被原地修改。

---

### `apply_llama31_rope_inplace`

```python
apply_llama31_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None
```

**功能**：对一批 query/key **原地**应用 Llama 3.1 风格 RoPE，cos/sin 内核内即时计算。默认参数为
`rope_scale=8`、`rope_theta=5e5`、`low_freq_factor=1`、`high_freq_factor=4`、`old_context_len=8192`。

**返回**

- `None`：`q`、`k` 被原地修改。

---

### `apply_llama31_rope`

```python
apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：对一批 query/key 应用 Llama 3.1 风格 RoPE，非原地。

**返回**

- `(q_rope, k_rope)`：旋转后的 query/key，形状分别为 `(nnz, num_q_heads, head_dim)`、
  `(nnz, num_k_heads, head_dim)`。

---

### `apply_llama31_rope_pos_ids`

```python
apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：按显式 `pos_ids`（形状 `(nnz)`）应用 Llama 3.1 风格 RoPE，非原地。

**返回**

- `(q_rope, k_rope)`：旋转后的 query/key。

---

### `apply_llama31_rope_pos_ids_inplace`

```python
apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None
```

**功能**：按显式 `pos_ids` 应用 Llama 3.1 风格 RoPE，原地。

**返回**

- `None`：`q`、`k` 被原地修改。

---

### `apply_rope_with_cos_sin_cache`

```python
apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用预计算 cos/sin 值对 key 与 query 应用 RoPE。设计上与 SGL/vLLM 实现兼容。

**参数**

- `positions`（`torch.Tensor`）：位置索引，形状 `(nnz)`。
- `query`（`torch.Tensor`）：Query 张量，形状 `(nnz, num_q_heads * head_size)`。
- `key`（`torch.Tensor`）：Key 张量，形状 `(nnz, num_k_heads * head_size)`。
- `head_size`（`int`）：每头特征维。`query` 与 `key` 会在应用 RoPE 前被 reshape 为
  `(nnz, num_q_heads, head_size)` 与 `(nnz, num_k_heads, head_size)`；必须整除 query/key 的尾维。
- `cos_sin_cache`（`torch.Tensor`）：cos/sin cache，形状 `(max_seq_len, rotary_dim)`。前一半为 cosine，
  后一半为 sine（沿 rotary_dim 方向）。dtype 必须为 float32。
- `is_neox`（`bool`，默认 `True`）：是否使用 NeoX 风格 RoPE。
  - `True`：最后一维不交错，旋转前半维与后半维。
  - `False`：最后一维交错，旋转偶数维与奇数维。

**返回**

- `query_out`（`torch.Tensor`）：旋转后的 query，形状 `(nnz, num_q_heads * head_size)`。
- `key_out`（`torch.Tensor`）：旋转后的 key，形状 `(nnz, num_k_heads * head_size)`。

**注意事项**

- 旋转维度由 cos cache 与 sine cache 决定。
- `cos_sin_cache` 必须为 float32，否则抛出 `ValueError`。

---

### `apply_rope_with_cos_sin_cache_inplace`

```python
apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None
```

**功能**：使用预计算 cos/sin 值对 key 与 query **原地**应用 RoPE；结果直接写入输入张量。

**参数**：同 `apply_rope_with_cos_sin_cache`。

**返回**

- `None`：`query`、`key` 被原地修改。

**注意事项**

- 旋转维度由 cos cache 与 sine cache 决定；`cos_sin_cache` 必须为 float32。

---

### `rope_quantize_fp8`

```python
rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
    backend: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：应用 RoPE 并量化为 FP8。接收已切分的 query/key 张量（旋转与非旋转部分分离），对旋转部分应用
RoPE，并将旋转与非旋转张量都量化为 FP8。支持 MLA、GQA、MHA 架构。

**参数**

- `q_rope`（`torch.Tensor`）：Query 旋转部分，形状 `(nnz, num_qo_heads, rope_dim)`，float16 或 bfloat16。
- `k_rope`（`torch.Tensor`）：Key 旋转部分。GQA/MHA 为 `(nnz, num_kv_heads, rope_dim)`；MLA 为
  `(nnz, rope_dim)`。float16 或 bfloat16。
- `q_nope`（`Optional[torch.Tensor]`）：Query 非旋转部分，形状 `(nnz, num_qo_heads, no_rope_dim)`；
  为 `None` 时按零维处理（内部创建 size-0 张量）。
- `k_nope`（`Optional[torch.Tensor]`）：Key 非旋转部分。GQA/MHA 为 `(nnz, num_kv_heads, no_rope_dim)`；
  MLA 为 `(nnz, no_rope_dim)`；为 `None` 时按零维处理。
- `cos_sin_cache`（`torch.Tensor`）：预计算 cos/sin 值，形状 `(max_seq_len, rope_dim)`，float32。
- `pos_ids`（`torch.Tensor`）：每个 token 的位置索引，形状 `(nnz,)`。
- `is_neox`（`bool`，默认 `True`）：`True` 为非交错（前/后半维），`False` 为交错（偶/奇维）。
- `quantize_dtype`（`Optional[torch.dtype]`，默认 `None`）：目标量化 dtype；为 `None` 时从输出张量推断，
  否则默认 `torch.float8_e4m3fn`。必须为 `torch.float8_e4m3fn` 或 `torch.float8_e5m2`。
- `quant_scale_q`（`float`，默认 `1.0`）：query 张量的量化缩放因子。
- `quant_scale_kv`（`float`，默认 `1.0`）：key 张量的量化缩放因子。
- `q_rope_out` / `k_rope_out` / `q_nope_out` / `k_nope_out`（`Optional[torch.Tensor]`）：预分配输出张量；
  为 `None` 时自动分配。
- `enable_pdl`（`bool`，默认 `False`）：是否启用 PDL。
- `backend`（`str`，默认 `"cuda"`）：实现后端。`"cuda"` 使用融合 CUDA 内核；`"cutile"` 使用 cuda.tile
  Python 内核并要求 SM89 或更新。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`：`(q_rope_out, k_rope_out, q_nope_out, k_nope_out)`。

**注意事项**

- `backend="cutile"` 仅支持 `is_neox=False`（交错 / GPT-J rotary）与 MLA 风格二维 key（单一共享 K 头）；
  要求 SM89+ 与 float32 `cos_sin_cache`，否则抛出 `NotImplementedError`。

---

### `rope_quantize_fp8_append_paged_kv_cache`

```python
rope_quantize_fp8_append_paged_kv_cache(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
    kv_layout: str = "NHD",
    q_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：应用 RoPE、量化为 FP8，并将 K/V 追加到分页 KV cache。该融合函数对 query/key 的旋转部分应用
RoPE，将所有 Q/K 张量（以及 GQA/MHA 的 V）量化为 FP8，并把量化后的 K/V 直接追加到分页 KV cache，
返回供注意力计算使用的量化 Q 张量。基于输入张量形状自动检测 MLA / GQA / MHA。

**参数**

- `q_rope`（`torch.Tensor`）：Query 旋转部分，形状 `(nnz, num_qo_heads, rope_dim)`，fp16/bf16。
- `k_rope`（`torch.Tensor`）：Key 旋转部分。GQA/MHA 为 `(nnz, num_kv_heads, rope_dim)`；MLA 为
  `(nnz, rope_dim)`。
- `q_nope`（`Optional[torch.Tensor]`）：Query 非旋转部分，形状 `(nnz, num_qo_heads, no_rope_dim)`。
- `k_nope`（`Optional[torch.Tensor]`）：Key 非旋转部分，形状视架构而定（见 `k_rope`）。
- `v`（`Optional[torch.Tensor]`）：GQA/MHA 的 value 张量 `(nnz, num_kv_heads, head_dim)`；MLA 传 `None`
  （MLA 不使用独立的 V，K-non-RoPE 充当压缩 KV）。
- `cos_sin_cache`（`torch.Tensor`）：预计算 cos/sin，形状 `(max_seq_len, rope_dim)`，float32。
- `pos_ids`（`torch.Tensor`）：每个 token 的位置索引，形状 `(nnz,)`。
- `paged_kv_cache`（`Tuple[torch.Tensor, torch.Tensor]`）：
  - MLA：`(ckv_cache, kpe_cache)`，其中 `ckv_cache` 形状 `(max_pages, page_size, no_rope_dim)`，
    `kpe_cache` 形状 `(max_pages, page_size, rope_dim)`，均为 FP8。
  - GQA/MHA：`(k_cache, v_cache)`，形状为 `(max_pages, page_size, num_kv_heads, head_dim)` 或
    `(max_pages, num_kv_heads, page_size, head_dim)`（取决于布局），均为 FP8。
- `kv_indices`（`torch.Tensor`）：页索引映射，形状 `(total_pages,)`，通常为 `torch.arange(total_pages)`。
- `kv_indptr`（`torch.Tensor`）：每个请求的页指针数组，形状 `(batch_size + 1,)`。
- `batch_indices`（`torch.Tensor`）：每个 token 的 batch 索引，形状 `(nnz,)`。
- `positions`（`torch.Tensor`）：每个 token 在其请求序列内的位置，形状 `(nnz,)`。
- `is_neox`（`bool`，默认 `True`）：`True` 为非交错，`False` 为交错。
- `quantize_dtype`（`Optional[torch.dtype]`，默认 `None`）：目标量化 dtype；为 `None` 时从输出张量推断或默认
  `torch.float8_e4m3fn`。
- `quant_scale_q`（`float`，默认 `1.0`）：query 量化缩放因子。
- `quant_scale_kv`（`float`，默认 `1.0`）：key/value 量化缩放因子。
- `page_size`（`int`，默认 `16`）：分页 cache 每页条目数。
- `kv_layout`（`str`，默认 `"NHD"`）：GQA/MHA 的 cache 内存布局，可选 `"NHD"`（page, seq, head, dim）或
  `"HND"`（page, head, seq, dim）；MLA 忽略。
- `q_rope_out` / `q_nope_out`（`Optional[torch.Tensor]`）：预分配的量化 Q 输出（旋转 / 非旋转部分）。
- `enable_pdl`（`bool`，默认 `False`）：是否启用 PDL。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：量化 Q 张量 `(q_rope_out, q_nope_out)`。K/V 直接写入分页 cache，
  不返回。

**注意事项**

- 架构检测：自动区分 MLA（二维 K 张量）与 GQA/MHA（三维 K 张量）。
- MLA 将 K-RoPE 写入 `kpe_cache`，K-noRoPE 写入 `ckv_cache`；不使用 V。
- GQA/MHA 将完整 K（RoPE + noRoPE）写入 `k_cache`，V 写入 `v_cache`。
- `batch_indices` 与 `positions` 通常来自 `flashinfer.get_batch_indices_positions()`。
- cache 张量必须已按目标 FP8 dtype 分配，否则抛出 `ValueError`。

---

### `mla_rope_quantize_fp8`

```python
mla_rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：为 MLA 注意力应用 RoPE 并量化为 FP8。这是 `rope_quantize_fp8` 的薄封装，采用 MLA 张量布局：
`k_rope` / `k_nope` 为二维（无 KV-head 轴）。其余参数与行为均与 `rope_quantize_fp8` 相同。

**参数**

- `q_rope`（`torch.Tensor`）：Query 旋转部分，`(nnz, num_qo_heads, rope_dim)`。
- `k_rope`（`torch.Tensor`）：MLA 布局的 Key 旋转部分，`(nnz, rope_dim)`。
- `q_nope`（`torch.Tensor`）：Query 非旋转部分，`(nnz, num_qo_heads, no_rope_dim)`。
- `k_nope`（`torch.Tensor`）：MLA 布局的 Key 非旋转部分，`(nnz, no_rope_dim)`。
- 其余参数同 `rope_quantize_fp8`。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`：`(q_rope_out, k_rope_out, q_nope_out, k_nope_out)`。

## 相关链接 / 参考

- 上游文档：`docs/api/rope.rst`
- 相关：`flashinfer.deep_gemm`、`flashinfer.get_batch_indices_positions`
- RaggedTensor 布局说明：`:ref:`kv-layout``

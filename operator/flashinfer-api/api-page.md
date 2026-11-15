# 分页 KV-Cache 操作（`flashinfer.page`）

> 源码：`flashinfer/page.py`
> 上游文档：`docs/api/page.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.page` 提供操作分页 KV-Cache 的内核：把新追加的 K/V（或 MLA 的 ckv/kpe）张量写入分页
缓存、把 NVFP4 量化与写入融合进同一趟、以及把 ragged 追加元数据转换为 COO 形式的 batch 索引与
position，或者反过来从分页 indptr 推出序列长度。适用于任意支持 FlashInfer 注意力的后端（FA2、
FA3、trtllm-gen 等），并支持 `NHD`/`HND` 两种 KV layout。

## 导入

```python
from flashinfer import (
    append_paged_kv_cache,
    append_paged_mla_kv_cache,
    nvfp4_quantize_append_paged_kv_cache,
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping,
    nvfp4_quantize_append_paged_mla_kv_cache,
    get_batch_indices_positions,
)
# 或
from flashinfer.page import append_paged_kv_cache
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `append_paged_kv_cache` | 函数 | 把一批 K/V 追加到分页 KV-Cache。 |
| `append_paged_mla_kv_cache` | 函数 | 把一批 MLA 的 ckv/kpe 追加到分页 MLA KV-Cache。 |
| `nvfp4_quantize_append_paged_kv_cache` | 函数 | 量化并把 K/V 行写入 NVFP4 分页 KV-Cache。 |
| `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` | 函数 | 按 slot mapping 量化并写入 NVFP4 分页 KV-Cache。 |
| `nvfp4_quantize_append_paged_mla_kv_cache` | 函数 | 量化并把 MLA (ckv, kpe) 写入 NVFP4 MLA 分页 KV-Cache。 |
| `get_batch_indices_positions` | 函数 | 把 append indptr 与序列长度转换为 batch 索引与 position。 |

此外还有 `get_seq_lens`（未在 rst 中列出，但常与本模块配合使用）：把 `kv_indptr` 与
`kv_last_page_len` 转换为序列长度。

## API 详情

### `append_paged_kv_cache`

```python
append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
) -> None
```

**功能**：把一批 K/V 对追加到分页 KV-Cache。函数假定追加 K/V 的空间已分配好，即 `kv_indices`、
`kv_indptr`、`kv_last_page_len` 已经包含了追加的 K/V。

**参数**

- `append_key`（`torch.Tensor`）：要追加的 key，ragged 格式，形状
  `[append_indptr[-1], num_kv_heads, head_dim]`。
- `append_value`（`torch.Tensor`）：要追加的 value，ragged 格式，形状
  `[append_indptr[-1], num_kv_heads, head_dim]`。
- `batch_indices`（`torch.Tensor`）：追加的每个 entry 所属 batch 索引，形状 `[append_indptr[-1]]`。
- `positions`（`torch.Tensor`）：追加的每个 entry 的位置，形状 `[append_indptr[-1]]`。
- `paged_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：分页 KV-Cache，
  可以是张量元组或单个张量：
  - 元组 `(k_cache, v_cache)`，各为 4-D 张量，`kv_layout="NHD"` 时形状
    `[max_num_pages, page_size, num_kv_heads, head_dim]`，`"HND"` 时形状
    `[max_num_pages, num_kv_heads, page_size, head_dim]`；
  - 单个 5-D 张量，`"NHD"` 时形状 `[max_num_pages, 2, page_size, num_kv_heads, head_dim]`，
    `"HND"` 时形状 `[max_num_pages, 2, num_kv_heads, page_size, head_dim]`；其中
    `paged_kv_cache[:, 0]` 为 key-cache，`paged_kv_cache[:, 1]` 为 value-cache。
- `kv_indices`（`torch.Tensor`）：分页 KV-Cache 的页索引，形状 `[kv_indptr[-1]]`。
- `kv_indptr`（`torch.Tensor`）：分页 KV-Cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_last_page_len`（`torch.Tensor`）：每个请求分页 KV cache 最后一页的 entry 数，形状
  `[batch_size]`。
- `kv_layout`（`str`，默认 `"NHD"`）：分页 KV-Cache 的 layout，`"NHD"` 或 `"HND"`。

**返回**

- `None`：就地在 `paged_kv_cache` 中更新。

**示例**

```python
import torch
import flashinfer

nnz_kv, num_kv_heads, head_dim = 100, 32, 128
k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
kv_append_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
).int()
num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
kv_page_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
).int()
kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
page_size = 16
kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
batch_indices, positions = flashinfer.get_batch_indices_positions(
    kv_append_indptr,
    flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
    nnz_kv,
)
flashinfer.append_paged_kv_cache(
    k_append, v_append, batch_indices, positions,
    paged_kv_cache, kv_page_indices, kv_page_indptr, kv_last_page_len,
)
```

**注意事项**

- 空间必须已分配：`kv_indices`/`kv_indptr`/`kv_last_page_len` 已反映追加内容。
- 分页 KV-Cache 的 layout 由 `kv_layout` 决定，追加的 K/V 为独立 shape 时不受 layout 影响。

---

### `append_paged_mla_kv_cache`

```python
append_paged_mla_kv_cache(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None
```

**功能**：把一批 key-value 对追加到分页 KV-Cache（MLA 用途）。**当前仅支持 `ckv=512` 与
`kpe=64`。**

**参数**

- `append_ckv`（`torch.Tensor`）：要追加的 compressed kv，ragged 格式，形状
  `[append_indptr[-1], ckv_dim]`。
- `append_kpe`（`torch.Tensor`）：要追加的 value，ragged 格式，形状 `[append_indptr[-1], kpe_dim]`。
- `batch_indices`（`torch.Tensor`）：追加的每个 entry 的 batch 索引，形状 `[append_indptr[-1]]`。
- `positions`（`torch.Tensor`）：追加的每个 entry 的位置，形状 `[append_indptr[-1]]`。
- `ckv_cache`（`torch.Tensor`）：compressed kv 缓存，形状 `[page_num, page_size, ckv_dim]`。
- `kpe_cache`（`torch.Tensor`）：key position embedding 缓存，形状 `[page_num, page_size, kpe_dim]`。
- `kv_indices`（`torch.Tensor`）：分页 KV-Cache 的页索引，形状 `[kv_indptr[-1]]`。
- `kv_indptr`（`torch.Tensor`）：分页 KV-Cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。

**返回**

- `None`：就地在 `ckv_cache`、`kpe_cache` 中更新。

---

### `nvfp4_quantize_append_paged_kv_cache`

```python
nvfp4_quantize_append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_sf: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    k_scale: float,
    v_scale: float,
    kv_layout: str = "NHD",
) -> None
```

**功能**：量化并把 K/V 行写入 NVFP4 分页 KV-Cache。`append_key`/`append_value` 必须是
fp16/bf16、形状 `[nnz, num_kv_heads, head_dim]`。函数把打包 E2M1 数据写入 last dim 为
`head_dim // 2` 的 uint8 分页 K/V cache，把 FP8 E4M3 块 scale 写入 last dim 为 `head_dim // 16`
的 `kv_cache_sf`。

`k_scale`/`v_scale` 是 NVFP4 注意力内核消费的全局 decode scale，即反量化重建为
`e2m1_value * block_scale * global_scale`。

**参数**

- `append_key`（`torch.Tensor`）：要量化并追加的 key，形状 `[nnz, num_kv_heads, head_dim]`，
  dtype `torch.float16` 或 `torch.bfloat16`。
- `append_value`（`torch.Tensor`）：与 `append_key` 同形状同 dtype 的 value。
- `batch_indices`（`torch.Tensor`）：每个追加行的 batch 索引，形状 `[nnz]`。
- `positions`（`torch.Tensor`）：每个追加行的逻辑 token 位置，形状 `[nnz]`。
- `paged_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：调用方持有的打包
  NVFP4 K/V cache。元组输入时每个张量形状 `kv_layout="NHD"` 为
  `[max_num_pages, page_size, num_kv_heads, head_dim // 2]`，`"HND"` 为
  `[max_num_pages, num_kv_heads, page_size, head_dim // 2]`。也接受 K/V 在第二维的 5-D 堆叠 cache。
- `kv_cache_sf`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：调用方持有的 FP8 E4M3
  scale cache，tuple 或堆叠格式同 `paged_kv_cache`，把 `head_dim // 2` 换成 `head_dim // 16`。
- `kv_indices`（`torch.Tensor`）：分页 KV-Cache 的页索引，形状 `[kv_indptr[-1]]`。
- `kv_indptr`（`torch.Tensor`）：分页 KV-Cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。
- `k_scale`（`float`）：K 的正有限全局 decode scale。
- `v_scale`（`float`）：V 的正有限全局 decode scale。
- `kv_layout`（`str`，默认 `"NHD"`）：`"NHD"` 或 `"HND"`。

**返回**

- `None`：就地在 `paged_kv_cache` 和 `kv_cache_sf` 中更新。

**注意事项**

- 空间必须已分配并由 `kv_indices`/`kv_indptr`/`kv_last_page_len` 描述。
- `append_key` dtype 必须为 fp16/bf16，`append_value` 同 dtype；NVFP4 paged K/V cache 必须为
  `uint8`，scale cache 必须为 `torch.float8_e4m3fn`；`k_scale`/`v_scale` 必须为正有限。

---

### `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`

```python
nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_sf: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    k_scale: Union[float, torch.Tensor],
    v_scale: Union[float, torch.Tensor],
    kv_layout: str = "NHD",
) -> None
```

**功能**：按 slot mapping 量化并写入 K/V 行到 NVFP4 分页 KV-Cache。适用于已经把每个 token 分配到
扁平 cache 槽位的运行时。`slot_mapping[i]` 解释为 `page_id * page_size + entry_idx`。负槽位是填充，
被忽略。`append_key`/`append_value` 可包含额外的填充行；只考虑 `slot_mapping.shape[0]` 行。

**参数**

- `append_key`（`torch.Tensor`）：要量化并写入的 key，形状 `[num_rows, num_kv_heads, head_dim]`，
  dtype fp16/bf16。`num_rows` 至少为 `slot_mapping.shape[0]`。
- `append_value`（`torch.Tensor`）：与 `append_key` 同形状同 dtype 的 value。
- `slot_mapping`（`torch.Tensor`）：每行要写入的扁平 cache 槽位，形状 `[nnz]`，dtype `int32` 或
  `int64`。负 entry 忽略。
- `paged_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：同
  `nvfp4_quantize_append_paged_kv_cache`。
- `kv_cache_sf`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：同
  `nvfp4_quantize_append_paged_kv_cache`。
- `k_scale`（`Union[float, torch.Tensor]`）：K 的正有限全局 decode scale。CUDA Graph 捕获期间必须
  是与 `append_key` 同设备、连续的标量 `torch.float32` CUDA 张量。
- `v_scale`（`Union[float, torch.Tensor]`）：V 的正有限全局 decode scale，捕获期间要求同上。
- `kv_layout`（`str`，默认 `"NHD"`）：`"NHD"` 或 `"HND"`。

**返回**

- `None`：就地在 `paged_kv_cache` 和 `kv_cache_sf` 中更新。

**注意事项**

- `slot_mapping` 必须为 int32 或 int64；`append_key`/`append_value` 至少要有
  `slot_mapping.shape[0]` 行。
- CUDA Graph 捕获期间 `k_scale`/`v_scale` 必须是同设备连续的 float32 CUDA 标量张量。

---

### `nvfp4_quantize_append_paged_mla_kv_cache`

```python
nvfp4_quantize_append_paged_mla_kv_cache(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    ckv_scale: float,
    kpe_scale: float,
) -> None
```

**功能**：量化并把 MLA (ckv, kpe) 行写入 NVFP4 MLA 分页 KV-Cache。compressed-kv 部分量化为 NVFP4
（打包 E2M1，每 16 通道一个 FP8 E4M3 scale）；rope 部分保留 FP8 E4M3（DeepSeek 系列 latent 中
rope 通道比 ckv 通道热得多）。反量化重建 compressed-kv 部分为 `e2m1_value * block_scale * ckv_scale`，
rope 部分为 `fp8_value * kpe_scale`。输入直接从 fp16/bf16 量化，无中间 FP8 重量化。

**当前仅支持 `ckv_dim=512` 与 `kpe_dim=64`。**

**参数**

- `append_ckv`（`torch.Tensor`）：要追加的 compressed kv，ragged 格式，形状
  `[append_indptr[-1], 512]`，dtype fp16 或 bf16。
- `append_kpe`（`torch.Tensor`）：要追加的 rope 部分，ragged 格式，形状 `[append_indptr[-1], 64]`，
  与 `append_ckv` 同 dtype。rope 须由调用方已施加（NoPE 风格模型传入未旋转值）。
- `batch_indices`（`torch.Tensor`）：每个追加行的 batch 索引，形状 `[append_indptr[-1]]`。
- `positions`（`torch.Tensor`）：每个追加行的位置，形状 `[append_indptr[-1]]`。
- `ckv_cache`（`torch.Tensor`）：打包 NVFP4 compressed-kv cache，dtype `uint8`，形状
  `[num_pages, page_size, 256]`（每字节两个 E2M1 值）。
- `ckv_sf_cache`（`torch.Tensor`）：compressed-kv 块 scale cache，dtype `torch.float8_e4m3fn`，形状
  `[num_pages, page_size, 32]`（每 16 通道一个 scale）。
- `kpe_cache`（`torch.Tensor`）：FP8 rope cache，dtype `torch.float8_e4m3fn`，形状
  `[num_pages, page_size, 64]`。
- `kv_indices`（`torch.Tensor`）：分页 KV-Cache 的页索引，形状 `[kv_indptr[-1]]`。
- `kv_indptr`（`torch.Tensor`）：分页 KV-Cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。
- `ckv_scale`（`float`）：compressed-kv 部分的全局 decode scale；必须在正规格 float32 范围内
  （内核以 float32 计算）。
- `kpe_scale`（`float`）：rope 部分的全局 decode scale，范围要求同上。

**返回**

- `None`：就地在 `ckv_cache`、`ckv_sf_cache`、`kpe_cache` 中更新。

**注意事项**

- `append_ckv.dtype` 必须为 fp16/bf16，`append_kpe` 必须与其同 dtype；`ckv_cache` 必须为
  `uint8`，`ckv_sf_cache` 与 `kpe_cache` 必须为 `torch.float8_e4m3fn`。
- `ckv_scale`/`kpe_scale` 必须为正、处于规格 float32 范围（`2**-126` 到 `3.402823466e38`）。

---

### `get_batch_indices_positions`

```python
get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int,
    batch_indices: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：把 append indptr 与序列长度转换为 batch 索引与 position。类似 cuSPARSE 的 CSR2COO
转换，区别在于：这里从 ragged 张量（不需要 column indices 数组）转换到 COO 格式。

**参数**

- `append_indptr`（`torch.Tensor`）：ragged 张量的 indptr，形状 `[batch_size + 1]`。
- `seq_lens`（`torch.Tensor`）：KV-Cache 中每个请求的序列长度，形状 `[batch_size]`。
- `nnz`（`int`）：ragged 张量中的 entry 数。
- `batch_indices`（`Optional[torch.Tensor]`，默认 `None`）：batch_indices 的预分配输出张量。为
  `None` 时自动分配。
- `positions`（`Optional[torch.Tensor]`，默认 `None`）：positions 的预分配输出张量。为 `None` 时
  自动分配。

**返回**

- `batch_indices`（`torch.Tensor`）：ragged 张量中每个 entry 的 batch 索引，形状 `[nnz]`。
- `positions`（`torch.Tensor`）：ragged 张量中每个 entry 的位置，形状 `[nnz]`。

**注意事项**

- 提供预分配张量时会检查其 shape/dtype/device（须为 int32、形状 `[nnz]`、与输入同设备）。

**示例**

```python
import torch
import flashinfer

nnz_kv = 10
append_indptr = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device="cuda:0")
seq_lens = torch.tensor([5, 5, 5, 5])
batch_indices, positions = flashinfer.get_batch_indices_positions(append_indptr, seq_lens, nnz_kv)
# batch_indices: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
# positions:     [4, 3, 4, 2, 3, 4, 1, 2, 3, 4]
```

---

### `get_seq_lens`（辅助，未在 rst 中列出）

```python
get_seq_lens(
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor
```

**功能**：把 KV indptr 与最后一页长度转换为序列长度：
`clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size + kv_last_page_len`。

**参数**

- `kv_indptr`（`torch.Tensor`）：分页 kv-cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。
- `page_size`（`int`）：分页 kv-cache 的页大小。

**返回**

- `torch.Tensor`：每个请求的序列长度，形状 `[batch_size]`。

## 相关链接 / 参考

- 上游 `docs/api/page.rst`
- 源码 `flashinfer/page.py`
- `flashinfer.get_seq_lens`、`flashinfer.quantization.packbits`

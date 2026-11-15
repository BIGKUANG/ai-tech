# 级联注意力（`flashinfer.cascade`）

> 源码：`flashinfer/cascade.py`
> 上游文档：`docs/api/cascade.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.cascade` 实现级联推断（cascade inference）：把多个共享前缀段与各请求独有后缀段的
注意力结果合并，避免为每个请求重复计算公共前缀。它既提供合并注意力状态的基础算子
（`merge_state` / `merge_state_in_place` / `merge_states`），也提供管理多级/共享前缀生命周期的
Wrapper 类。典型使用场景是带共享系统提示（system prompt）或多个层级公共前缀的批量服务。

## 导入

```python
from flashinfer import merge_state, merge_state_in_place, merge_states
from flashinfer import MultiLevelCascadeAttentionWrapper
from flashinfer import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
# 或
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `merge_state` | 函数 | 合并两个 KV 段的注意力输出 `V` 与 logsumexp `S`。 |
| `merge_state_in_place` | 函数 | 就地合并自注意力状态 `(v, s)` 与另一状态 `(v_other, s_other)`。 |
| `merge_states` | 函数 | 合并多个注意力状态 `(v, s)`。 |
| `MultiLevelCascadeAttentionWrapper` | 类 | 多级级联推断 Wrapper，假定所有级 KV-Cache 存于统一分页表。 |
| `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` | 类 | 共享前缀 + 分页后缀的批量 decode（将废弃）。 |
| `BatchPrefillWithSharedPrefixPagedKVCacheWrapper` | 类 | 共享前缀 + 分页后缀的批量 prefill/append（将废弃）。 |

## API 详情

### `merge_state`

```python
merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：合并两个 KV 段的注意力输出 `V` 与 logsumexp 值 `S`。数学细节见递归注意力教程。

**参数**

- `v_a`（`torch.Tensor`）：KV 段 `A` 的注意力输出，形状 `[seq_len, num_heads, head_dim]`。
- `s_a`（`torch.Tensor`）：KV 段 `A` 的 logsumexp 值，float32，形状 `[seq_len, num_heads]`。
- `v_b`（`torch.Tensor`）：KV 段 `B` 的注意力输出，形状 `[seq_len, num_heads, head_dim]`。
- `s_b`（`torch.Tensor`）：KV 段 `B` 的 logsumexp 值，float32，形状 `[seq_len, num_heads]`。

**返回**

- `V`（`torch.Tensor`）：合并后的注意力输出（等价于对合并 KV 段 `[A: B]` 的注意力），形状
  `[seq_len, num_heads, head_dim]`。
- `S`（`torch.Tensor`）：合并 KV 段的 logsumexp 值，形状 `[seq_len, num_heads]`。

**示例**

```python
import torch
import flashinfer

seq_len, num_heads, head_dim = 2048, 32, 128
va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
v_merged, s_merged = flashinfer.merge_state(va, sa, vb, sb)
```

---

### `merge_state_in_place`

```python
merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None
```

**功能**：就地合并自注意力状态 `(v, s)` 与另一状态 `(v_other, s_other)`。内部把 `s`/`s_other`
转换为 float32 后调用内核，并修改 `v`、`s`。

**参数**

- `v`（`torch.Tensor`）：要就地更新的部分注意力输出，形状 `(seq_len, num_heads, head_dim)`。
- `s`（`torch.Tensor`）：要就地更新的部分 logsumexp，float32，形状 `(seq_len, num_heads)`。
- `v_other`（`torch.Tensor`）：要合并的另一注意力输出，形状 `(seq_len, num_heads, head_dim)`。
- `s_other`（`torch.Tensor`）：要合并的另一 logsumexp，float32，形状 `(seq_len, num_heads)`。
- `mask`（`Optional[torch.Tensor]`，默认 `None`）：布尔掩码，指示对应序列是否合并状态，用于 CUDA
  Graphs。不指定时对所有序列合并。形状 `[seq_len]`。

**返回**

- `None`：就地在 `v`、`s` 中更新。

**示例**

```python
flashinfer.merge_state_in_place(v, s, v_other, s_other)
```

---

### `merge_states`

```python
merge_states(v: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：合并多个注意力状态 `(v, s)`。

**参数**

- `v`（`torch.Tensor`）：各 KV 段的注意力输出，形状 `[seq_len, num_states, num_heads, head_dim]`。
- `s`（`torch.Tensor`）：各 KV 段的 logsumexp 值，float32，形状 `[seq_len, num_states, num_heads]`。

**返回**

- `V`（`torch.Tensor`）：合并后的注意力输出，形状 `[seq_len, num_heads, head_dim]`。
- `S`（`torch.Tensor`）：合并 KV 段的 logsumexp，形状 `[seq_len, num_heads]`。

**示例**

```python
v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
v_merged, s_merged = flashinfer.merge_states(v, s)
```

---

### `MultiLevelCascadeAttentionWrapper`

```python
MultiLevelCascadeAttentionWrapper(
    num_levels,
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    qo_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
    paged_kv_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
    paged_kv_indices_buf_arr: Optional[List[torch.Tensor]] = None,
    paged_kv_last_page_len_buf_arr: Optional[List[torch.Tensor]] = None,
) -> None
```

**功能**：内存高效的多级级联推断注意力 Wrapper，假定所有级的 KV-Cache 存于统一分页表。级联推断
思想见 FlashInfer 博客。注意：由于合并注意力结果的开销，增加级数并不总是有利。

**生命周期**：构造 `__init__` → `plan`（构造辅助数据结构，可在同一 decode step 的多次 forward 中
复用）→ 可选 `prewarm_paged_kv_stride_variant`（CUDA Graph 捕获前）→ `run`。`begin_forward` 是
`plan` 的别名，`forward` 是 `run` 的别名。

#### `__init__`

**参数**

- `num_levels`（`int`）：级联注意力的级数。
- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲，用于存放 split-k
  中间结果。推荐 128MB，设备须与输入张量一致。
- `kv_layout`（`str`，默认 `"NHD"`）：输入 K/V 的 layout，`"NHD"` 或 `"HND"`。
- `use_cuda_graph`（`bool`，默认 `False`）：是否用 CUDA Graph 捕获内核；启用时辅助数据结构存于
  提供的缓冲中。
- `qo_indptr_buf_arr`（`Optional[List[torch.Tensor]]`，默认 `None`）：每级一个 qo indptr 缓冲，
  数组长度等于级数。每个张量的最后一个元素应为 query/output 总数。
- `paged_kv_indptr_buf_arr`（`Optional[List[torch.Tensor]]`，默认 `None`）：每级一个分页 kv-cache
  indptr 缓冲，长度等于级数。
- `paged_kv_indices_buf_arr`（`Optional[List[torch.Tensor]]`，默认 `None`）：每级一个分页 kv-cache
  索引缓冲，长度等于级数。
- `paged_kv_last_page_len_buf_arr`（`Optional[List[torch.Tensor]]`，默认 `None`）：每级一个分页
  kv-cache 最后一页长度缓冲，长度等于级数。

**属性**

- `is_cuda_graph_enabled`（`bool`）：返回构造时的 `use_cuda_graph`。

#### `plan`（= `begin_forward`）

```python
plan(
    qo_indptr_arr: List[torch.Tensor],
    paged_kv_indptr_arr: List[torch.Tensor],
    paged_kv_indices_arr: List[torch.Tensor],
    paged_kv_last_page_len: List[torch.Tensor],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    causal: bool = False,
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    q_data_type: str = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
)
```

**功能**：为同一 decode step 内的多次 forward 调用创建多级级联注意力的辅助数据结构。对每级分别
调用底层分页 prefill wrapper 的 `plan`；只有最底层（最后一层）施加 `causal`，其余层级恒为非因果。

**参数**

- `qo_indptr_arr`（`List[torch.Tensor]`）：每级一个 qo indptr，数组长度等于级数，各张量最后一元素
  为 query/output 总数。
- `paged_kv_indptr_arr`（`List[torch.Tensor]`）：每级一个分页 kv-cache indptr。
- `paged_kv_indices_arr`（`List[torch.Tensor]`）：每级一个分页 kv-cache 索引。
- `paged_kv_last_page_len`（`List[torch.Tensor]`）：每级一个分页 kv-cache 最后一页长度。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：头维度。
- `page_size`（`int`）：分页 kv-cache 的页大小。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码（仅在未提供 `custom_mask` 时生效；本包装器
  只对最底层生效）。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）：`NONE`/`ROPE_LLAMA`/`ALIBI`。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否用 f16 做 qk reduction。
- `window_left`（`int`，默认 `-1`）：注意力窗口左边界（含）；`-1` 表示全序列长度。
- `logits_soft_cap`（`Optional[float]`，默认 `None`）：logits soft cap 值；不提供时为 0。
- `sm_scale`（`Optional[float]`，默认 `None`）：softmax scale；默认 `1.0 / sqrt(head_dim)`。
- `rope_scale`（`Optional[float]`，默认 `None`）：RoPE 插值 scale；默认 `1.0`。
- `rope_theta`（`Optional[float]`，默认 `None`）：RoPE theta；默认 `1e4`。
- `q_data_type`（`str`，默认 `"float16"`）：query 数据类型。
- `kv_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：key/value 数据类型；默认取
  `q_data_type`。

#### `prewarm_paged_kv_stride_variant`

```python
prewarm_paged_kv_stride_variant(variant: str = "independent") -> None
```

**功能**：在 `plan` 之后预热每级的惰性 paged-KV-stride 变体。在 CUDA Graph 捕获前调用。
`variant` 仅支持 `"independent"`（默认），用于 K 与 V 数据 stride 不同的情形。

#### `run`（= `forward`）

```python
run(q: torch.Tensor, paged_kv_cache: torch.Tensor)
```

**功能**：计算多级级联注意力。先对最底层 wrapper 运行并取回 LSE，然后依次对更早层级运行，并用
`merge_state_in_place` 合并，返回合并后的输出。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[batch_size, num_qo_heads, head_dim]`。
- `paged_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：分页 KV-Cache，
  可为 tuple `(k_cache, v_cache)`（4-D，`NHD` 为
  `[max_num_pages, page_size, num_kv_heads, head_dim]`，`HND` 为
  `[max_num_pages, num_kv_heads, page_size, head_dim]`），或单个 5-D 张量（`NHD` 为
  `[max_num_pages, 2, page_size, num_kv_heads, head_dim]`，`HND` 为
  `[max_num_pages, 2, num_kv_heads, page_size, head_dim]`）。

**返回**

- `torch.Tensor`：注意力输出，形状 `[batch_size, num_qo_heads, head_dim]`。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(2, workspace_buffer, "NHD")
wrapper.plan(qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
             paged_kv_last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size)
o = wrapper.run(q, kv_cache_at_layer)
```

---

### `BatchDecodeWithSharedPrefixPagedKVCacheWrapper`

```python
BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
) -> None
```

**功能**：共享前缀分页 KV-Cache 的批量 decode 注意力。共享前缀 KV-Cache 存于独立张量，每个请求
独有的 KV-Cache 存于分页数据结构中。

> **警告**：该 API 未来将被废弃，请改用 `MultiLevelCascadeAttentionWrapper`。

**生命周期**：构造 `__init__` → `begin_forward`（构造辅助数据结构）→ `forward`（可跨多个
Transformer 层复用辅助结构）。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户提供的 float workspace 缓冲（如 128 MiB），与
  输入同一 CUDA 设备，并与底层 `BatchDecodeWithPagedKVCacheWrapper` 共享。
- `kv_layout`（`str`，默认 `"NHD"`）：KV-cache 张量的 layout，`"NHD"` 或 `"HND"`。

#### `begin_forward`

```python
begin_forward(
    unique_kv_indptr: torch.Tensor,
    unique_kv_indices: torch.Tensor,
    unique_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    data_type: str = "float16",
) -> None
```

**功能**：为给定问题规格规划共享前缀批量 decode。必须在任何 `forward` 之前调用。辅助数据结构在
此创建并缓存供多次 forward 复用。`num_qo_heads` 必须是 `num_kv_heads` 的倍数；不等时使用 GQA。

**参数**

- `unique_kv_indptr`（`torch.Tensor`）：指向每个请求**非共享**后缀的 CSR 页偏移，形状
  `[batch_size + 1]`，int32。
- `unique_kv_indices`（`torch.Tensor`）：非共享后缀分页 KV cache 的页索引，形状
  `[unique_kv_indptr[-1]]`，int32。
- `unique_kv_last_page_len`（`torch.Tensor`）：每个请求 unique 后缀最后一页的有效 entry 数，形状
  `[batch_size]`，int32。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：头维度。
- `page_size`（`int`）：分页 kv cache 的页大小。
- `data_type`（`str`，默认 `"float16"`）：分页 kv cache 的数据类型。

#### `forward`

```python
forward(
    q: torch.Tensor,
    k_shared: torch.Tensor,
    v_shared: torch.Tensor,
    unique_kv_cache: torch.Tensor,
) -> torch.Tensor
```

**功能**：计算 query 与共享前缀分页 kv-cache 之间的批量 decode 注意力。内部对共享前缀调用
`single_prefill_with_kv_cache`（非因果、返回 LSE），对 unique 后缀调用底层 decode wrapper 的
`forward_return_lse`，再用 `merge_state_in_place` 合并。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[batch_size, num_qo_heads, head_dim]`。
- `k_shared`（`torch.Tensor`）：共享前缀 key，`NHD` 时形状
  `[shared_prefix_len, num_kv_heads, head_dim]`，`HND` 时
  `[num_kv_heads, shared_prefix_len, head_dim]`。
- `v_shared`（`torch.Tensor`）：共享前缀 value，shape 规则同 `k_shared`。
- `unique_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：请求无关的后缀分页
  KV-Cache，tuple 或单张量，形状规则同 `MultiLevelCascadeAttentionWrapper.run`。

**返回**

- `torch.Tensor`：注意力输出，形状 `[batch_size, num_heads, head_dim]`。

---

### `BatchPrefillWithSharedPrefixPagedKVCacheWrapper`

```python
BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
) -> None
```

**功能**：共享前缀分页 KV-Cache 的批量 prefill/append 注意力。

> **警告**：该 API 未来将被废弃，请改用 `MultiLevelCascadeAttentionWrapper`。

**生命周期**：构造 `__init__` → `begin_forward` → 可选 `prewarm_paged_kv_stride_variant` →
`forward`。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲，用于 split-k 中间
  结果。推荐 128MB，设备须与输入张量一致。
- `kv_layout`（`str`，默认 `"NHD"`）：输入 K/V 的 layout，`"NHD"` 或 `"HND"`。

#### `begin_forward`

```python
begin_forward(
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> None
```

**功能**：为同一 prefill/append step 内的多次 forward 调用创建共享前缀批量 prefill/append 的辅助
数据结构。必须在任何 `forward` 之前调用。`num_qo_heads` 必须是 `num_kv_heads` 的倍数；不等时使用
GQA。

**参数**

- `qo_indptr`（`torch.Tensor`）：query/output 张量的 indptr，形状 `[batch_size + 1]`。
- `paged_kv_indptr`（`torch.Tensor`）：分页 kv-cache 的 indptr，形状 `[batch_size + 1]`。
- `paged_kv_indices`（`torch.Tensor`）：分页 kv-cache 的页索引，形状 `[qo_indptr[-1]]`。
- `paged_kv_last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：头维度。
- `page_size`（`int`）：分页 kv-cache 的页大小。

#### `prewarm_paged_kv_stride_variant`

```python
prewarm_paged_kv_stride_variant(variant: str = "independent") -> None
```

**功能**：预热底层 prefill wrapper 的 paged-KV-stride 变体。在 `begin_forward` 之后、CUDA Graph
捕获之前调用。`variant` 仅支持 `"independent"`。

#### `forward`

```python
forward(
    q: torch.Tensor,
    k_shared: torch.Tensor,
    v_shared: torch.Tensor,
    unique_kv_cache: torch.Tensor,
    causal: bool = False,
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
) -> torch.Tensor
```

**功能**：计算 query 与共享前缀分页 kv-cache 之间的批量 prefill/append 注意力。内部对共享前缀调用
`single_prefill_with_kv_cache`（非因果、返回 LSE），对 unique 后缀调用底层 prefill wrapper 的
`forward_return_lse`，再用 `merge_state_in_place` 合并。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[qo_indptr[-1], num_qo_heads, head_dim]`。
- `k_shared`（`torch.Tensor`）：共享前缀 key，`NHD` 时形状
  `[shared_prefix_len, num_kv_heads, head_dim]`，`HND` 时
  `[num_kv_heads, shared_prefix_len, head_dim]`。
- `v_shared`（`torch.Tensor`）：共享前缀 value，shape 规则同 `k_shared`。
- `unique_kv_cache`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：请求无关的后缀分页
  KV-Cache。
- `causal`（`bool`，默认 `False`）：是否对注意力矩阵施加因果掩码。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否用 f16 做 qk reduction。
- `sm_scale`（`Optional[float]`，默认 `None`）：softmax scale；默认 `1 / sqrt(head_dim)`。
- `rope_scale`（`Optional[float]`，默认 `None`）：RoPE 插值 scale；默认 `1.0`。
- `rope_theta`（`Optional[float]`，默认 `None`）：RoPE theta；默认 `1e4`。

**返回**

- `torch.Tensor`：注意力输出，形状 `[qo_indptr[-1], num_heads, head_dim]`。

## 相关链接 / 参考

- 上游 `docs/api/cascade.rst`
- 源码 `flashinfer/cascade.py`
- 递归注意力教程（recursive attention）
- FlashInfer 级联推断博客：https://flashinfer.ai/2024/02/02/cascade-inference.html

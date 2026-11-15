# V3.2 Indexer MQA Logits API

- 源码：`csrc/apis/attention.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM90 / SM100

这组算子是 DeepSeek V3.2 lightning indexer 的打分内核：对每个 query token 计算加权 ReLU 的 MQA logits。分为**非分页**（prefilling）与**分页**（decoding）两族，以及 SM100 上的**稀疏**变体。

核心公式：对 query 行 `i`，遍历其有效 KV 区间 `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])` 中的每个 `j`：

```python
kv_j = kv_data[j, :] * kv_sf[j]          # [head_dim]
out_ij = q[i, :, :] @ kv_j               # [num_heads]
out_ij = out_ij.relu() * weights[i, :]   # [num_heads]
logit_i = out_ij.sum()                   # 标量
```

输出被压缩为 `[seq_len, max_seqlen_k]`：第 `i` 行从其有效 KV 区间的开头起，连续存放有效列的 logit。区间外的列不保证被清理，调用方需自行掩码。

## 函数总览

| 函数 | 说明 | 架构 |
| --- | --- | --- |
| `fp8_fp4_mqa_logits` | 非分页 MQA logits | SM90/SM100 |
| `get_mqa_logits_metadata` | 非分页调度元数据 | SM100 |
| `fp8_fp4_paged_mqa_logits` | 分页 MQA logits | SM90/SM100 |
| `get_paged_mqa_logits_metadata` | 分页调度元数据 | SM90/SM100 |
| `fp8_fp4_sparse_mqa_logits` | 稀疏（连续 KV）MQA logits | SM100 |
| `get_sparse_mqa_logits_metadata` | 稀疏元数据 | SM100 |
| `fp8_fp4_paged_sparse_mqa_logits` | 稀疏（分页 KV）MQA logits | SM100 |
| `get_paged_sparse_mqa_logits_metadata` | 分页稀疏元数据 | SM100 |

### 缩放因子模式（重要）

`q_sf` 与 `kv_sf` 的 dtype 是**耦合**的：是否传 `q_sf` 决定 “MX 模式”，并翻转对 `kv_sf` 的要求。不存在混合模式。

| 模式 | `q_sf` | `kv_sf` | Q/KV 数据 dtype | 架构 |
| --- | --- | --- | --- | --- |
| MX（`q_sf` 非空） | `int32` 打包 UE8M0，连续 | `int32` 打包 UE8M0，连续 | FP8（MXFP8）或打包 FP4（MXFP4） | 仅 SM100 |
| 非 MX（`q_sf=None`） | — | `float32`（每 token 一个标量），连续 | 仅 FP8 | 仅 SM90 |

- FP4 的 Q/KV 数据**必须**使用 MX 模式（`q_sf` 非空）。
- MX 粒度为沿 `head_dim` 每 32 元素一个块；`head_dim <= 128` 时一个 token/head 的最多 4 个 UE8M0 指数正好装入一个 `int32`，因此 `q_sf` 没有末尾 K 维。
- 权重与 logits：SM100 为 BF16，SM90 为 FP32（输出 dtype 由架构决定）。

---

## 非分页 MQA logits

### get_mqa_logits_metadata

```python
get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads) -> torch.Tensor
```

构造非分页调度元数据（仅 SM100）。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `cu_seq_len_k_start` | `[num_q_tokens]`，`torch.int`，CUDA，连续 | 每行 KV 区间起点 |
| `cu_seq_len_k_end` | `[num_q_tokens]`，`torch.int`，CUDA，连续 | 每行 KV 区间终点 |
| `num_kv_tokens` | int | KV token 总数（> 0） |
| `num_heads` | int | query head 数 |

**返回**：一维 `int32` 调度元数据张量。

**示例**

```python
schedule_meta = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
logits = deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, ks, ke, max_seqlen_k,
                                      schedule_meta=schedule_meta)
```

### fp8_fp4_mqa_logits

```python
fp8_fp4_mqa_logits(
    q, kv, weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    max_seqlen_k,
    schedule_meta=None,
) -> torch.Tensor
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(q_fp, q_sf)` | `q_fp` 形状 `[seq_len, num_heads, head_dim]`；`q_sf` 可选，`[seq_len, num_heads]`，`int32`（MX 模式） |
| `kv` | `(kv_fp, kv_sf)` | `kv_fp` 形状 `[seq_len_kv, head_dim]`；`kv_sf` 形状 `[seq_len_kv]`，MX 模式为 `int32`、非 MX 为 `float32` |
| `weights` | `[seq_len, num_heads]` | SM100 = BF16，SM90 = FP32；`stride(1) == 1` 且行 stride 为 16 字节对齐（需将 `num_heads` 上调补齐） |
| `cu_seq_len_k_start` | `[seq_len]`，`torch.int`，连续 | KV 区间起点 |
| `cu_seq_len_k_end` | `[seq_len]`，`torch.int`，连续 | KV 区间终点 |
| `max_seqlen_k` | int | 任意 query 行的最大有效 KV 跨度（> 0） |
| `schedule_meta` | `torch.Tensor`，可选 | `get_mqa_logits_metadata` 的输出；SM90 不接受 |

**约束**

- `head_dim`：FP8 时为 32/64/128，FP4 时为 64/128（SM100 下 FP8 为 32/64/128）。
- 输出 dtype 由架构决定（SM100 BF16 / SM90 FP32）。
- 返回张量形状 `[seq_len, max_seqlen_k]`，其行 stride 按 1024 字节对齐；区间外条目未清理。

**示例（MX 模式，SM100）**

```python
from deep_gemm.utils import per_token_cast_to_fp8

q_fp8_2d, q_sf = per_token_cast_to_fp8(q.view(-1, head_dim), use_ue8m0=True,
                                       gran_k=32, use_packed_ue8m0=True)
kv_fp8, kv_sf = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
logits = deep_gemm.fp8_fp4_mqa_logits(
    q=(q_fp8_2d.view(seq_len, num_heads, head_dim), q_sf.view(seq_len, num_heads)),
    kv=(kv_fp8, kv_sf.view(seq_len_kv)),
    weights=weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
    max_seqlen_k=int((ke - ks).max().item()))
```

---

## 分页 MQA logits

### get_paged_mqa_logits_metadata

```python
get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None) -> torch.Tensor
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `context_lens` | `[batch_size, next_n]`，`torch.int`，连续 | 每行上下文长度（目前仅支持二维） |
| `block_kv` | int | KV 页大小。SM100 需为 32/64/128；SM90 必须为 64 |
| `num_sms` | int | 使用的 SM 数（> 0） |
| `indices` | `[batch_size]`，`torch.int`，连续，可选 | SM100 要求 varlen 请求（每行一个 token，由 `indices` 给出请求归属）；SM90 不接受 |

**返回**：`int32` 矩阵，形状 `[num_sms + 1, 2]`。

**说明**：SM100 上 `indices` 的数据指针必须 8 字节对齐（元数据内核每次扫描两个 token）。

### fp8_fp4_paged_mqa_logits

```python
fp8_fp4_paged_mqa_logits(
    q, kv_cache, weights,
    context_lens, block_table, schedule_meta,
    max_context_len,
    indices=None,
) -> torch.Tensor
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(q_fp, q_sf)` | `q_fp` 形状 `[batch_size, next_n, num_heads, head_dim]`；`q_sf` 见 MX 模式 |
| `kv_cache` | `torch.uint8` | 分页 KV cache，形状 `[num_kv_blocks, block_kv, 1, kv_head_dim + 4]`：每 token 值字节（FP8 为 `head_dim`，FP4 为 `head_dim/2`）后跟 4 字节 SF（MX 为 `int32`，非 MX 为 `float32`）。要求 `stride(1) == kv_head_dim + 4`、`stride(3) == 1` |
| `weights` | `[batch_size * next_n, num_heads]` | 同上 |
| `context_lens` | `[batch_size, next_n]`，`torch.int`，连续 | 上下文长度 |
| `block_table` | `[batch_size, max_block_len]`，`torch.int` | `stride(1) == 1` |
| `schedule_meta` | `[num_sms + 1, 2]`，`torch.int`，连续 | `get_paged_mqa_logits_metadata` 输出 |
| `max_context_len` | int | 最大上下文长度 |
| `indices` | `[batch_size]`，`torch.int`，连续，可选 | SM100 必填（varlen），且要求 `next_n == 1` |

**返回**：`[batch_size * next_n, max_context_len]`，dtype 由架构决定。

**示例**

```python
metadata = deep_gemm.get_paged_mqa_logits_metadata(
    context_lens, block_kv, deep_gemm.get_num_sms(), indices)
logits = deep_gemm.fp8_fp4_paged_mqa_logits(
    q=q_in, kv_cache=kv_in, weights=weights,
    context_lens=context_lens, block_table=block_table,
    schedule_meta=metadata, max_context_len=max_model_len, indices=indices)
```

---

## 稀疏 MQA logits（SM100）

稀疏变体跳过 KV 中未被选中的块：`sparse_kv_block_indices` 的每一行给出该 query 要参与计算的块索引（**必须唯一且严格递增**，处于该行的 KV 范围内）。块大小 `sparse_block_kv ∈ {8, 16}`。

### get_sparse_mqa_logits_metadata

```python
get_sparse_mqa_logits_metadata(
    cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens,
    sparse_kv_block_indices, qk_dtype, sparse_block_kv,
    use_unaligned_ks=False) -> torch.Tensor
```

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `cu_seq_len_k_start` / `cu_seq_len_k_end` | `[num_q_tokens]`，`torch.int`，连续 | KV 区间 |
| `num_kv_tokens` | int | KV 总数 |
| `sparse_kv_block_indices` | `[num_q_tokens, num_max_sparse_blocks]`，`torch.int`，连续 | 每行的稀疏块索引（每行前缀是有效的递增块） |
| `qk_dtype` | `torch.dtype` | `q[0].dtype`（FP8 或打包 FP4） |
| `sparse_block_kv` | int | 8 或 16 |
| `use_unaligned_ks` | bool | KV 起点未按 `sparse_block_kv` 对齐时置 `True`（块 `i` 从 `i * sparse_block_kv + ks % sparse_block_kv` 开始） |

**返回**：一维 `uint8` 元数据张量。

### fp8_fp4_sparse_mqa_logits

```python
fp8_fp4_sparse_mqa_logits(
    q, kv, weights, metadata,
    num_max_sparse_blocks, sparse_block_kv,
    use_unaligned_ks=False) -> torch.Tensor
```

| 名称 | 说明 |
| --- | --- |
| `q` | `(q_fp, q_sf)`：`q_fp` 形状 `[num_q_tokens, num_heads, head_dim]`，`head_dim` 必须为 128，`num_heads % 4 == 0` 且 `<= kNumMaxHeads` |
| `kv` | `(kv_fp, kv_sf)`：`kv_fp` 形状 `[num_kv_tokens, head_dim]`，`kv_sf` 形状 `[num_kv_tokens]`，`int32`，连续 |
| `weights` | `[num_q_tokens, num_heads]`，BF16 |
| `metadata` | 上一步元数据 |
| `num_max_sparse_blocks` | int，`> 0`，`% 4 == 0`，`<= 4096` |
| `sparse_block_kv` | 8 或 16 |

**返回**：BF16 张量，形状 `[num_q_tokens, num_max_sparse_blocks * sparse_block_kv]`（有效列之外未清理）。

### get_paged_sparse_mqa_logits_metadata

```python
get_paged_sparse_mqa_logits_metadata(
    context_lens, block_table, indices, page_kv,
    sparse_kv_block_indices, qk_dtype, sparse_block_kv) -> torch.Tensor
```

| 名称 | 说明 |
| --- | --- |
| `context_lens` | `[num_q_tokens]`，`torch.int`，连续 |
| `block_table` | `[num_q_tokens, max_pages]`，`torch.int`，`stride(1) == 1`；成对 query 必须拥有相同的 block-table 行 |
| `indices` | `[num_q_tokens]`，`torch.int`，连续；同一请求的 query 必须连续 |
| `page_kv` | int，`> 0` 且 `% sparse_block_kv == 0` |
| `sparse_kv_block_indices` | `[num_q_tokens, num_max_sparse_blocks]`，`torch.int`，连续；块索引为逻辑块、唯一递增 |
| `qk_dtype` | `torch.dtype` |
| `sparse_block_kv` | 8 或 16 |

### fp8_fp4_paged_sparse_mqa_logits

```python
fp8_fp4_paged_sparse_mqa_logits(
    q, kv_cache, weights, metadata,
    num_max_sparse_blocks, sparse_block_kv) -> torch.Tensor
```

| 名称 | 说明 |
| --- | --- |
| `q` | `(q_fp, q_sf)`：`q_fp` 形状 `[num_q_tokens, 1, num_heads, head_dim]`，`head_dim == 128` |
| `kv_cache` | `torch.uint8`，形状 `[num_kv_pages, page_kv, 1, kv_head_dim + 4]`；`stride(0)` 是 512 的倍数且 `% 512 == 0`，数据指针 16 字节对齐（内核用 16 字节 `cp.async` 拷贝） |
| `weights` | `[num_q_tokens, num_heads]`，BF16 |
| `metadata` | 上一步元数据 |

**返回**：BF16，形状 `[num_q_tokens, num_max_sparse_blocks * sparse_block_kv]`。

## 相关

- 缩放因子（含 MQA SF 硬编码要求）：[scaling-factor-format.md](./scaling-factor-format.md) 第 6.1 节
- 量化和 cast 工具：[api-utils.md](./api-utils.md)
- 测试参考：`tests/test_attention.py`

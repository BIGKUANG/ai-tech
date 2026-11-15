# FlashAttention-3 (Hopper) 注意力 API

- 源码：`hopper/flash_attn_interface.py`
- 模块名：`flash_attn_3`（编译产物 `flash_attn_3_cuda` / `flash_attn_3_gpu`）
- 目标架构：Hopper (SM90)

FA3 是面向 Hopper 的第三代实现，相比 FA2 增加了：

- **FP8 支持**：通过 `q_descale` / `k_descale` / `v_descale` 传入反量化缩放。
- **`attention_chunk`**：chunked (block-diagonal) 注意力，用于如块内注意力等。
- **`num_splits` / `pack_gqa` / `sm_margin`**：性能调优参数。
- **MLA (`qv`)**：DeepSeek 风格 MLA 吸收形式，`qv` 为额外的 "value of Q"。
- **`flash_attn_combine`**：合并 split-KV 的部分结果。

> FA3 各接口**不支持 dropout**（无 `dropout_p` 参数），支持 `torch.autograd`。

## 函数总览

| 函数 | 说明 |
| --- | --- |
| `flash_attn_func` | 标准注意力 |
| `flash_attn_varlen_func` | 变长注意力 |
| `flash_attn_qkvpacked_func` | 打包 QKV |
| `flash_attn_combine` | 合并 split-KV 部分结果 |
| `flash_attn_with_kvcache` | 推理 KV Cache / paged KV |

公共新增参数：

- `q_descale, k_descale, v_descale` (Tensor, optional)：FP8 反量化缩放张量。
- `attention_chunk` (int)：`> 0` 时启用 chunked 注意力（每个 query 只关注同一 chunk 内的 key）。
- `num_splits` (int)：split-KV 的切分数；`0` 表示启发式自动确定。
- `pack_gqa` (bool, optional)：是否将多个 Q head 打包以提升 GQA 效率；`None` 表示自动。
- `sm_margin` (int)：为通信预留的 SM 数（当部分 SM 用于通信时调优）。
- `window_size` (int, int)：滑动窗口，`-1` 表示无限。
- `softcap` (float)：softcapping。

---

## flash_attn_func

```python
flash_attn_func(
    q, k, v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(batch_size, seqlen, nheads, headdim)` | 查询 |
| `k` | `(batch_size, seqlen, nheads_k, headdim)` | 键 |
| `v` | `(batch_size, seqlen, nheads_k, headdim_v)` | 值 |
| `softmax_scale` | float | 缩放，默认 `1/sqrt(headdim)` |
| `causal` | bool | 因果掩码（右下角对齐） |
| `qv` | `(batch, seqlen, nheads, headdim_v)`，可选 | MLA：`O = softmax(scale*(QK^T + Qv V^T)) V` |
| `q_descale/k_descale/v_descale` | Tensor，可选 | FP8 反量化缩放 |
| `window_size` | (int, int) | 滑动窗口 |
| `attention_chunk` | int | chunked 注意力 |
| `softcap` | float | softcapping |
| `num_splits` | int | split-KV 切分数 |
| `pack_gqa` | bool, optional | GQA 打包 |
| `deterministic` | bool | 确定性反向 |
| `sm_margin` | int | 预留 SM 数 |
| `return_attn_probs` | bool | 返回注意力概率（测试用） |

**返回**

- `out`：`(batch_size, seqlen, nheads, headdim_v)`
- `softmax_lse`（`return_attn_probs=True`）：`(batch_size, nheads, seqlen)`

**示例**

```python
from flash_attn_3 import flash_attn_func

q = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
```

---

## flash_attn_varlen_func

```python
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    seqused_q=None, seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
)
```

**参数**

- `q`：`(total_q, nheads, headdim)`
- `k`：`(total_k, nheads_k, headdim)`
- `v`：`(total_k, nheads_k, headdim_v)`
- `cu_seqlens_q` / `cu_seqlens_k`：`(batch_size + 1,)`，`int32`
- `max_seqlen_q` / `max_seqlen_k`：int
- `seqused_q` / `seqused_k`：`(batch_size,)`，可选，实际使用的序列长度（允许 `cu_seqlens` 描述容量、`seqused` 描述有效长度）
- `qv`：`(total_q, nheads, headdim_v)`，可选（MLA）

其余同 `flash_attn_func`。

**返回**

- `out`：`(total_q, nheads, headdim_v)`
- `softmax_lse`（可选）：`(nheads, total_q)`

---

## flash_attn_qkvpacked_func

```python
flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
)
```

**参数**

- `qkv`：`(batch_size, seqlen, 3, nheads, headdim)`
- `num_heads_q`：int, optional，用于 GQA（QKV 打包但 Q 头数与 K/V 不同）的情况

其余同 `flash_attn_func`。

**返回**

- `out`：`(batch_size, seqlen, nheads, headdim)`
- `softmax_lse`（可选）：`(batch_size, nheads, seqlen)`
- `S_dmask`（可选）：`(batch_size, nheads, seqlen, seqlen)`

---

## flash_attn_combine

```python
flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None)
```

合并 split-KV 产生的部分输出。

**参数**

- `out_partial`：`(num_splits, total_q, nheads, headdim)` 或 `(num_splits, batch, seqlen, nheads, headdim)`
- `lse_partial`：`(num_splits, ...)`，各 split 的 logsumexp
- `out`：可选，预分配输出
- `out_dtype`：可选，输出 dtype

**返回**：合并后的 `out`。

---

## flash_attn_with_kvcache

```python
flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=None, v=None, qv=None,
    rotary_cos=None, rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None, k_descale=None, v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
)
```

推理解码注意力，支持 paged KV、变长 query、MLA，可融合 RoPE。

> **注意：不支持反向传播。**

**关键参数**

- `k_cache`：无 `page_table` 时 `(batch_size_cache, seqlen_cache, nheads_k, headdim)`；有 `page_table` 时 `(num_blocks, page_block_size, nheads_k, headdim)`。FA3 中 `page_block_size` **可为任意值**（如 1, 2, 3, 64）。
- `v_cache`：同上，值维度为 `headdim_v`。
- `k` / `v`：可选，新 K/V，原地写入 cache。
- `page_table`：paged KV 的页表（FA3 命名，对应 FA2 的 `block_table`）。
- `cu_seqlens_q` / `cu_seqlens_k_new` / `max_seqlen_q`：变长 query / 新增 KV 的累积长度。
- `rotary_seqlens`：变长 RoPE 的位置索引。
- `scheduler_metadata`：由调度器元数据预计算得到的张量（性能优化）。
- `num_splits` / `pack_gqa` / `sm_margin`：性能调优。

**返回**

- `out`：`(batch_size, seqlen, nheads, headdim_v)`
- `softmax_lse`（`return_softmax_lse=True`）

---

## FA3 vs FA2 差异小结

| 特性 | FA2 | FA3 |
| --- | --- | --- |
| Dropout | 支持 | 不支持 |
| FP8 | 不支持 | 支持（descale 参数） |
| Chunked attention | 不支持 | 支持（`attention_chunk`） |
| MLA (`qv`) | 不支持 | 支持 |
| Split-KV 调优 | `num_splits`（kvcache） | `num_splits` + `scheduler_metadata` |
| Paged block 大小 | 必须 256 的倍数 | 任意 |
| `pack_gqa` / `sm_margin` | 无 | 支持 |

# FlashAttention-2 注意力 API

- 源码：`flash_attn/flash_attn_interface.py`
- 导入：`from flash_attn import <func>`
- 目标架构：Ampere (SM80) 及以上

本文档介绍 FA2 标准注意力算子。所有函数都通过 `torch.autograd.Function` 实现，前向/反向均可微。反向默认使用非确定性（atomic）实现；设置 `deterministic=True` 可获得可复现的反向结果（更慢、占用内存更多），前向始终是确定性的。

## 函数总览

| 函数 | 输入布局 | 典型用途 |
| --- | --- | --- |
| `flash_attn_func(q, k, v, ...)` | 分离的 `q, k, v` | 通用单次注意力 |
| `flash_attn_qkvpacked_func(qkv, ...)` | 打包的 `qkv` | Q/K/V 已拼接时更快 |
| `flash_attn_kvpacked_func(q, kv, ...)` | `q` + 打包 `kv` | 交叉注意力 / KV 已拼接 |
| `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` | 变长 `q, k, v` | 变长序列训练 |
| `flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, ...)` | 变长打包 `qkv` | 变长自注意力 |
| `flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, ...)` | 变长 `q` + 打包 `kv` | 变长交叉注意力 |
| `flash_attn_with_kvcache(...)` | KV Cache / paged KV | 自回归推理解码 |

公共参数（各函数共有）：

- `dropout_p` (float)：Dropout 概率，推理/评估时必须为 `0.0`。
- `softmax_scale` (float, optional)：`QK^T` 在 softmax 前的缩放，默认 `1 / sqrt(head_dim)`。
- `causal` (bool)：是否应用因果掩码（自回归建模）。
- `window_size` (tuple[int, int])：`(left, right)`，非 `(-1, -1)` 时启用滑动窗口局部注意力，`-1` 表示无限。
- `softcap` (float)：`> 0` 时启用 softcapping 注意力（对 logits 做 `tanh` 软截断）。
- `alibi_slopes` ((nheads,) 或 (batch_size, nheads), fp32)：ALiBi 位置偏置斜率。
- `deterministic` (bool)：反向是否使用确定性实现。
- `return_attn_probs` (bool)：是否返回注意力概率，仅用于测试，返回值不保证缩放正确。

GQA/MQA：`q` 的头数必须能被 `k`/`v` 的头数整除。例如 Q 有 6 个头、KV 有 2 个头时，Q 的头 0/1/2 关注 KV 头 0，头 3/4/5 关注 KV 头 1。

因果掩码对齐规则（右下角对齐）：若 `seqlen_q = 2, seqlen_k = 5`，掩码为

```
1 1 1 1 0
1 1 1 1 1
```

若 `seqlen_q = 5, seqlen_k = 2`：

```
0 0
0 0
0 0
1 0
1 1
```

若某行掩码全为 0，则该行输出为 0。

---

## flash_attn_func

```python
flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

标准注意力：`out = softmax(scale * Q K^T) V`。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(batch_size, seqlen, nheads, headdim)` | 查询 |
| `k` | `(batch_size, seqlen, nheads_k, headdim)` | 键 |
| `v` | `(batch_size, seqlen, nheads_k, headdim)` | 值 |
| `dropout_p` | float | Dropout 概率 |
| `softmax_scale` | float | 缩放因子，默认 `1/sqrt(headdim)` |
| `causal` | bool | 因果掩码 |
| `window_size` | (int, int) | 滑动窗口 `(left, right)` |
| `softcap` | float | `> 0` 启用 softcapping |
| `alibi_slopes` | `(nheads,)` 或 `(batch, nheads)`，fp32 | ALiBi 偏置 `(-slope * |i + seqlen_k - seqlen_q - j|)` |
| `deterministic` | bool | 确定性反向 |
| `return_attn_probs` | bool | 返回注意力概率（测试用） |

**返回**

- `out`：`(batch_size, seqlen, nheads, headdim)`
- `softmax_lse`（`return_attn_probs=True`）：`(batch_size, nheads, seqlen)`，`QK^T * scaling` 每行的 logsumexp
- `S_dmask`（`return_attn_probs=True`）：`(batch_size, nheads, seqlen, seqlen)`，softmax 输出（负值表示被 dropout，非负表示保留）

**示例**

```python
from flash_attn import flash_attn_func

q = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
out.sum().backward()
```

---

## flash_attn_qkvpacked_func

```python
flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

当 Q、K、V 已打包为一个张量时，比分别传入 `flash_attn_func` 更快，因为反向避免了显式拼接 Q/K/V 的梯度。

**参数**

- `qkv`：`(batch_size, seqlen, 3, nheads, headdim)`

其余参数与返回值同 `flash_attn_func`（`nheads` 为 Q/K/V 共用头数，不支持 GQA；GQA 请用 `flash_attn_kvpacked_func` 或 `flash_attn_func`）。

---

## flash_attn_kvpacked_func

```python
flash_attn_kvpacked_func(
    q, kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

当 K、V 已打包为一个张量时使用。支持 MQA/GQA（`q` 头数多于 KV 头数且可整除）。

**参数**

- `q`：`(batch_size, seqlen, nheads, headdim)`
- `kv`：`(batch_size, seqlen, 2, nheads_k, headdim)`

ALiBi 偏置为 `(-alibi_slope * |i + seqlen_k - seqlen_q - j|)`。返回值同 `flash_attn_func`。

---

## flash_attn_varlen_func

```python
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
)
```

变长（unpadded）注意力，用于一个 batch 内序列长度不同的情况，避免 padding 带来的浪费。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(total_q, nheads, headdim)` | `total_q` 为 batch 中所有 query token 总数 |
| `k` | `(total_k, nheads_k, headdim)` | `total_k` 为所有 key token 总数 |
| `v` | `(total_k, nheads_k, headdim)` | 值 |
| `cu_seqlens_q` | `(batch_size + 1,)`，`int32` | query 的累积序列长度 |
| `cu_seqlens_k` | `(batch_size + 1,)`，`int32` | key 的累积序列长度 |
| `max_seqlen_q` | int | batch 内最大 query 序列长度 |
| `max_seqlen_k` | int | batch 内最大 key 序列长度 |
| `block_table` | `(batch_size, max_num_blocks_per_seq)`，`int32`，可选 | paged KV 块表 |

其余参数同 `flash_attn_func`。

**返回**

- `out`：`(total, nheads, headdim)`
- `softmax_lse`（`return_attn_probs=True`）：`(nheads, total_q_seqlen)`
- `S_dmask`（`return_attn_probs=True`）：`(batch_size, nheads, seqlen, seqlen)`

**示例**

```python
import torch
from flash_attn import flash_attn_varlen_func

cu_seqlens = torch.tensor([0, 128, 256, 384], dtype=torch.int32, device="cuda")
q = torch.randn(384, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn(384, 8, 64, dtype=torch.float16, device="cuda")
v = torch.randn(384, 8, 64, dtype=torch.float16, device="cuda")
out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 128, 128)
```

---

## flash_attn_varlen_qkvpacked_func

```python
flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

变长 + 打包。当 Q、K、V 已打包时比 `flash_attn_varlen_func` 更快。

**参数**

- `qkv`：`(total, 3, nheads, headdim)`，`total` 为 batch 中 token 总数
- `cu_seqlens`：`(batch_size + 1,)`，`int32`
- `max_seqlen`：int，batch 内最大序列长度

**返回**

- `out`：`(total, nheads, headdim)`
- `softmax_lse`（可选）：`(nheads, total_q_seqlen)`
- `S_dmask`（可选）：`(batch_size, nheads, seqlen, seqlen)`

---

## flash_attn_varlen_kvpacked_func

```python
flash_attn_varlen_kvpacked_func(
    q, kv,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

变长 + KV 打包，支持 GQA/MQA。

**参数**

- `q`：`(total_q, nheads, headdim)`
- `kv`：`(total_k, 2, nheads_k, headdim)`
- 其余同 `flash_attn_varlen_func`

**返回**：同 `flash_attn_varlen_func`。

---

## flash_attn_with_kvcache

```python
flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=None, v=None,
    rotary_cos=None, rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
)
```

自回归推理的解码注意力，可直接读写 KV Cache，并可选融合 RoPE。

> **注意：不支持反向传播**（推理专用）。

**行为说明**

- 若传入 `k`、`v`，则 `k_cache`、`v_cache` 会被**原地更新**：从 `cache_seqlens` 指定的位置写入新的 K/V，然后基于更新后的 cache 做注意力。cache 必须预留足够空间。
- 若传入 `rotary_cos`/`rotary_sin` 且 `k`、`v` 非空，则对 `k`、`q` 应用旋转位置编码。`causal` 或局部注意力时，`q` 在 `cache_seqlens, cache_seqlens+1, ...` 这些位置旋转；非 causal 非局部时，所有 `q` token 视作位于位置 `cache_seqlens`。
- 支持 GQA/MQA。
- 因果掩码对齐与 `flash_attn_func` 相同。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(batch_size, seqlen, nheads, headdim)` | 查询 |
| `k_cache` | 无 `block_table`：`(batch_size_cache, seqlen_cache, nheads_k, headdim)`；有 `block_table`：`(num_blocks, page_block_size, nheads_k, headdim)` | KV Cache（paged 时 `page_block_size` 必须是 256 的倍数） |
| `v_cache` | 同上 | 值 Cache |
| `k` | `(batch_size, seqlen_new, nheads_k, headdim)`，可选 | 新 key，写入 cache |
| `v` | 同上，可选 | 新 value，写入 cache |
| `rotary_cos` | `(seqlen_ro, rotary_dim / 2)`，可选 | 旋转余弦，`rotary_dim` 必须能被 16 整除 |
| `rotary_sin` | 同上，可选 | 旋转正弦 |
| `cache_seqlens` | int 或 `(batch_size,)`，`int32` | 每个序列的 KV Cache 长度 |
| `cache_batch_idx` | `(batch_size,)`，`int32`，可选 | 索引到 KV Cache 的 batch 下标；`None` 表示 `[0..batch-1]`。若索引重复且传入了 `k`/`v`，写入值可能来自任一重复下标 |
| `cache_leftpad` | `(batch_size,)`，`int32`，可选 | KV Cache 起始位置；`None` 表示 0 |
| `block_table` | `(batch_size, max_num_blocks_per_seq)`，`int32`，可选 | paged KV 块表 |
| `softmax_scale` | float | 缩放因子 |
| `causal` | bool | 因果掩码 |
| `window_size` | (int, int) | 滑动窗口 |
| `softcap` | float | `> 0` 启用 softcapping |
| `rotary_interleaved` | bool | `True` 时组合维度 0&1、2&3…（GPT-J 风格）；`False` 时组合 0&rotary_dim/2（GPT-NeoX 风格） |
| `alibi_slopes` | `(nheads,)` 或 `(batch, nheads)`，fp32 | ALiBi |
| `num_splits` | int | `>1` 沿序列切分 KV；`1` 不切分；`0` 启发式自动确定。除非了解其含义，否则不要改动 |
| `return_softmax_lse` | bool | 是否返回注意力分数的 logsumexp |

**返回**

- `out`：`(batch_size, seqlen, nheads, headdim)`
- `softmax_lse`（`return_softmax_lse=True`）：`(batch_size, nheads, seqlen)`

**示例**

```python
import torch
from flash_attn import flash_attn_with_kvcache

batch, seqlen, nheads, headdim = 2, 1, 8, 64
max_seqlen = 1024
q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
k_cache = torch.randn(batch, max_seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
v_cache = torch.randn(batch, max_seqlen, nheads, headdim, dtype=torch.float16, device="cuda")

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=128,
    causal=True,
)
```

---

## 相关：底层算子与自动微分

`flash_attn_interface.py` 同时封装了 `torch.autograd.Function`（`FlashAttnFunc`、`FlashAttnQKVPackedFunc`、`FlashAttnKVPackedFunc`、`FlashAttnVarlenFunc` 等），它们通过 `torch.library.custom_op` 注册了 `_flash_attn_forward`、`_flash_attn_backward`、`_flash_attn_varlen_forward`、`_flash_attn_varlen_backward` 等 CUDA 算子（含 fake tensor 实现以支持 `torch.compile`）。一般用户直接调用上面的公开函数即可。

另有一个纯 Triton 的参考实现 `flash_attn/flash_attn_triton.py`（`flash_attn_func = FlashAttnFunc.apply`），用于不支持 CUDA 扩展时的回退与教学，性能低于 CUDA 内核。

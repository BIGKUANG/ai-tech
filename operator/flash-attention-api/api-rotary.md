# 旋转位置编码 (RoPE) API

- 源码：`flash_attn/layers/rotary.py`
- 底层 Triton 内核：`flash_attn/ops/triton/rotary.py`（`apply_rotary`）

实现 RoFormer（Su et al., [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)）的旋转位置编码。当 `scale_base` 非空时实现 XPos。

## 风格说明

- `interleaved=False`：GPT-NeoX 风格，旋转前半维与后半维（维度 `i` 与 `i + rotary_dim/2`）。
- `interleaved=True`：GPT-J 风格，旋转相邻的偶数维与奇数维（维度 `0 & 1`、`2 & 3`…）。
- `rotary_dim` 必须 `<= headdim`，只对 `x` 的前 `rotary_dim` 维应用；且（底层约束）`rotary_dim` 应能被 16 整除。

## 函数总览

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `rotate_half(x, interleaved=False)` | 函数 | 旋转辅助函数 |
| `apply_rotary_emb_torch(x, cos, sin, interleaved=False)` | 函数 | 纯 PyTorch 参考实现 |
| `apply_rotary_emb(x, cos, sin, ...)` | 函数 | 通用 RoPE 应用（可微） |
| `apply_rotary_emb_qkv_(qkv, cos, sin, ...)` | 函数 | 对打包 QKV 原地应用 |
| `apply_rotary_emb_kv_(kv, cos, sin, ...)` | 函数 | 对打包 KV 原地应用 K |
| `RotaryEmbedding` | `nn.Module` | 缓存 cos/sin 的模块封装 |

---

## apply_rotary_emb

```python
apply_rotary_emb(
    x, cos, sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `x` | `(batch, seqlen, nheads, headdim)`；若给出 `cu_seqlens` 则为 `(total_seqlen, nheads, headdim)` | 输入 |
| `cos`, `sin` | `(seqlen_rotary, rotary_dim / 2)` | 旋转用的余弦/正弦 |
| `interleaved` | bool | 旋转风格 |
| `inplace` | bool | 是否原地修改 |
| `seqlen_offsets` | `(batch,)` 或 int | 每个序列的位置偏移，常用于带 KV Cache 的推理 |
| `cu_seqlens` | `(batch + 1,)` 或 `None` | 变长累积长度 |
| `max_seqlen` | int | 最大序列长度 |

**返回**：与 `x` 形状相同的输出。

> 别名：`apply_rotary_emb_func = apply_rotary_emb`（向后兼容）。

---

## apply_rotary_emb_qkv_

```python
apply_rotary_emb_qkv_(
    qkv, cos, sin,
    cos_k=None, sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
)
```

对打包的 QKV 张量**原地**应用 RoPE（只旋转 Q 和 K）。

**参数**

- `qkv`：`(batch, seqlen, 3, nheads_kv?, headdim)` 或 `(batch, seqlen, num_heads_q + 2*num_heads_k, headdim)`
  - 若为 `(batch, seqlen, num_heads_q + 2*num_heads_k, headdim)`（如 MQA/GQA），必须提供 `num_heads_q`
- `cos`, `sin`：`(seqlen, rotary_dim / 2)`
- `cos_k`, `sin_k`：`(seqlen, rotary_dim / 2)`，可选（XPos 时为 Q 与 K 使用不同的缩放）
- `seqlen_offsets`：`(batch,)` 或 int

**返回**：与输入同形状的 `qkv`。

---

## apply_rotary_emb_kv_

```python
apply_rotary_emb_kv_(
    kv, cos, sin,
    interleaved=False,
    seqlen_offsets: Union[int, Tensor] = 0,
)
```

对打包 KV 张量**原地**应用 RoPE（只旋转 K，不动 V）。

**参数**

- `kv`：`(batch, seqlen, 2, nheads, headdim)`
- `cos`, `sin`：`(seqlen, rotary_dim / 2)`

**返回**：`(batch, seqlen, 2, nheads, headdim)`

---

## RotaryEmbedding

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None, device=None):
        ...
    def forward(self, qkv, kv=None, seqlen_offset=0, max_seqlen=None, num_heads_q=None):
        ...
```

缓存 `cos`/`sin` 表的模块，序列长度变化或设备/dtype 变化时自动重建缓存。

**构造参数**

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `dim` | int | rotary 维度（偶数） |
| `base` | float | RoPE 基频，默认 10000.0 |
| `interleaved` | bool | 旋转风格 |
| `scale_base` | float, optional | 非空时实现 XPos；推荐值 512 |
| `device` | optional | 设备 |

**forward 参数**

- `qkv`：`(batch, seqlen, 3, nheads, headdim)`，或 `(batch, seqlen, num_heads_q + 2*num_heads_k, headdim)`；若 `kv` 非空，则 `qkv` 实际是 Q，形状 `(batch, seqlen, nheads, headdim)`。
- `kv`：`(batch, seqlen, 2, nheads, headdim)`，可选。
- `seqlen_offset`：`(batch,)` 或 int，序列位置偏移。
- `max_seqlen`：int，可选；当 `seqlen_offset` 为张量时，用它更新 cos/sin 缓存。
- `num_heads_q`：GQA/MQA 打包 QKV 时必需。

**返回**

- `kv is None`：原地旋转后的 `qkv`
- 否则：`(q, kv)` 元组

**示例**

```python
from flash_attn.layers.rotary import RotaryEmbedding

rotary = RotaryEmbedding(dim=64, base=10000.0, interleaved=False).cuda()
q = torch.randn(2, 1024, 8, 64, device="cuda")
kv = torch.randn(2, 1024, 2, 8, 64, device="cuda")
q, kv = rotary(q, kv=kv)
```

---

## rotate_half

```python
rotate_half(x, interleaved=False)
```

RoPE 旋转的一半操作：

- `interleaved=False`：`x -> cat(-x2, x1)`，其中 `x1, x2 = x.chunk(2, dim=-1)`
- `interleaved=True`：对偶数/奇数维配对后 `cat(-x2, x1)` 并展平

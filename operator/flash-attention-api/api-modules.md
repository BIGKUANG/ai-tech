# 模块 API（MHA / Block / MLP / Embedding）

- 源码：`flash_attn/modules/`
  - `mha.py`：多头注意力层
  - `block.py`：Transformer Block
  - `mlp.py`：MLP / Gated MLP
  - `embedding.py`：词嵌入

这些是构建 Transformer 的**参考实现**（源自 GPT-NeoX / Megatron 风格），展示如何在模型中集成 FlashAttention。

---

## MHA（`modules/mha.py`）

```python
class MHA(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_heads_kv=None,
        cross_attn=False, qkv_proj_bias=True, out_proj_bias=True,
        dropout=0.0, softmax_scale=None, causal=False, layer_idx=None,
        dwconv=False, rotary_emb_dim=0, rotary_emb_base=10000.0,
        rotary_emb_scale_base=None, rotary_emb_interleaved=False,
        use_alibi=False, window_size=(-1, -1),
        fused_bias_fc=False, use_flash_attn=False,
        return_residual=False, checkpointing=False,
        device=None, dtype=None,
    )
```

**要点**

- `num_heads_kv`：用于 MQA/GQA，`None` 时等于 `num_heads`；必须满足 `num_heads % num_heads_kv == 0`。
- `use_flash_attn=True` 时走 FlashAttention 内核路径；否则用朴素实现（支持 `key_padding_mask`）。
- `use_alibi` 与 `window_size != (-1, -1)` 都**要求** `use_flash_attn=True`。
- `rotary_emb_dim > 0` 时内置 `RotaryEmbedding`（不支持 cross-attention）。

**forward**

```python
def forward(self, x, x_kv=None, key_padding_mask=None,
            cu_seqlens=None, max_seqlen=None,
            mixer_subset=None, inference_params=None, **kwargs):
```

- `x`：`(batch, seqlen, hidden_dim)`；若提供 `cu_seqlens`/`max_seqlen` 则为 `(total, hidden_dim)`
- `x_kv`：交叉注意力的 KV，`(batch, seqlen, hidden_dim)`
- `key_padding_mask`：仅在非 FlashAttention 路径使用
- `cu_seqlens` / `max_seqlen`：仅在 FlashAttention 路径使用
- `inference_params`：生成（增量解码）时使用

`allocate_inference_cache(batch_size, max_seqlen, dtype=None)` 预分配 KV Cache，形状 `(batch, max_seqlen, 2, num_heads_kv, head_dim)`。

### 内部注意力子模块

| 类 | 说明 |
| --- | --- |
| `FlashSelfAttention` | 使用 `flash_attn_qkvpacked_func` / `flash_attn_varlen_qkvpacked_func` 的自注意力 |
| `FlashCrossAttention` | 使用 `flash_attn_kvpacked_func` 系列的交叉注意力 |
| `SelfAttention` / `CrossAttention` | 朴素 PyTorch 实现（支持 padding mask） |
| `ParallelMHA` | 张量并行的 MHA |

`get_alibi_slopes(nheads)` 生成 ALiBi 斜率。

---

## Block（`modules/block.py`）

```python
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout, prenorm=True,
        resid_dropout1=0.0, resid_dropout2=0.0,
        drop_path1=0.0, drop_path2=0.0,
        fused_dropout_add_ln=False, return_residual=False,
        residual_in_fp32=False, sequence_parallel=False,
        mark_shared_params=False,
    )
    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None)
```

结构说明（源码注释）：

- **prenorm=True**：采用「Dropout → Add → LN → MHA → Dropout → Add → LN → MLP」的顺序，返回 `(hidden_states, residual)` 两者。这种改写是为了把 dropout + add + LayerNorm 融合；`residual` 需由上层传入（第一个 block 除外）。
- **prenorm=False**：标准 postnorm：「MHA → Dropout → Add → LN → MLP → Dropout → Add → LN」。
- `fused_dropout_add_ln=True`：使用 Triton 融合的 dropout+add+LayerNorm。

`ParallelBlock` 为张量并行版本。`allocate_inference_cache(...)` 透传到 mixer。

---

## MLP（`modules/mlp.py`）

| 类 | 说明 |
| --- | --- |
| `Mlp(in_features, hidden_features=None, out_features=None, activation=F.gelu, bias1=True, bias2=True, return_residual=False, ...)` | 标准两层 MLP，`hidden_features` 默认 `4 * in_features` |
| `ParallelMLP(...)` | 张量并行 MLP（`ColumnParallelLinear` + `RowParallelLinear`） |
| `GatedMlp(in_features, hidden_features=None, out_features=None, activation=F.sigmoid, bias1=True, bias2=True, multiple_of=128, return_residual=False, ...)` | 门控 MLP（GLU/SwiGLU），`fc1` 输出 `2 * hidden_features` 后对半做门控 |
| `ParallelGatedMlp(...)` | 张量并行门控 MLP |

---

## Embedding（`modules/embedding.py`）

| 类 | 说明 |
| --- | --- |
| `GPT2Embeddings` | GPT-2 风格 token + 可学习位置嵌入 |
| `BertEmbeddings` | BERT 风格 token + position + token_type 嵌入 |
| `VocabParallelEmbedding` | 词表并行嵌入（Megatron 风格） |
| `ColumnParallelEmbedding` | 嵌入维列并行 |
| `ParallelGPT2Embeddings` | 并行 GPT-2 嵌入 |

---

## 典型用法

```python
from functools import partial
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp

block = Block(
    dim=768,
    mixer_cls=partial(MHA, num_heads=12, use_flash_attn=True, causal=True),
    mlp_cls=partial(Mlp, hidden_features=3072),
    fused_dropout_add_ln=True,
    prenorm=True,
).cuda()

x = torch.randn(2, 128, 768, device="cuda", dtype=torch.float16)
y, residual = block(x)
```

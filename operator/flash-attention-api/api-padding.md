# BERT Padding 工具 API

- 源码：`flash_attn/bert_padding.py`
- 用途：在「padded 布局」与「varlen（打包）布局」之间转换，配合 `flash_attn_varlen_func` 使用，避免 padding 带来的算力/显存浪费。

## 布局约定

- padded：`(batch, seqlen, ...)`
- packed/unpadded：`(total_nnz, ...)`，`total_nnz` 为有效 token 总数
- `cu_seqlens`：`(batch + 1,)` int32
- `indices`：`(total_nnz,)`，标记每个有效 token 在 flattened 输入中的位置，用于还原

## 函数总览

| 名称 | 说明 |
| --- | --- |
| `unpad_input(hidden_states, attention_mask, unused_mask=None)` | padded → packed |
| `unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length)` | 拼接短样本场景的 unpad |
| `pad_input(hidden_states, indices, batch, seqlen)` | packed → padded |
| `index_first_axis` / `index_put_first_axis` / `index_first_axis_residual` | 带自定义反向的索引算子 |

---

## unpad_input

```python
unpad_input(hidden_states, attention_mask, unused_mask=None)
```

**参数**

- `hidden_states`：`(batch, seqlen, ...)`
- `attention_mask`：`(batch, seqlen)`，bool/int，`1` 表示有效
- `unused_mask`：`(batch, seqlen)`，bool/int，`1` 表示已分配但未使用（如 FlashAttention 的 padding 边界）

**返回**

- `hidden_states`：`(total_nnz, ...)`，`total_nnz = attention_mask + unused_mask` 的 token 数
- `indices`：`(total_nnz,)`，被选中 token 在 flattened 输入中的下标
- `cu_seqlens`：`(batch + 1,)` int32
- `max_seqlen_in_batch`：int
- `seqused`：`(batch,)`，每条序列选中的 token 数

**示例**：先 unpad 再调用 `flash_attn_varlen_func`

```python
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_func

q_unpad, indices_q, cu_q, max_q, _ = unpad_input(q, attention_mask)
k_unpad, _, cu_k, max_k, _ = unpad_input(k, attention_mask)
v_unpad, _, _, _, _ = unpad_input(v, attention_mask)

out_unpad = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_q, cu_k, max_q, max_k, causal=True)
out = pad_input(out_unpad, indices_q, batch, seqlen)
```

---

## unpad_input_for_concatenated_sequences

```python
unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length)
```

用于把多条短样本拼接进一条序列的场景（如 SFT）。`attention_mask_in_length[b, i]` 表示第 `b` 条序列中第 `i` 段拼接样本的长度：

```
[
  [2, 3, 0, 0, 0, 0],
  [3, 2, 0, 0, 0, 0],
  [6, 0, 0, 0, 0, 0]
]
```

**参数**

- `hidden_states`：`(batch, seqlen, ...)`
- `attention_mask_in_length`：`(batch, seqlen)`，int，非零值表示该段拼接样本长度

**返回**

- `hidden_states`：`(total_nnz, ...)`
- `indices`：`(total_nnz,)`
- `cu_seqlens`：`(batch + 1,)`
- `max_seqlen_in_batch`：int

---

## pad_input

```python
pad_input(hidden_states, indices, batch, seqlen)
```

packed → padded 的逆操作。

**参数**

- `hidden_states`：`(total_nnz, ...)`
- `indices`：`(total_nnz,)`，来自 `unpad_input`
- `batch`：int
- `seqlen`：int

**返回**：`(batch, seqlen, ...)`

---

## 索引算子

| 名称 | 说明 |
| --- | --- |
| `index_first_axis(input, indices)` | 沿第 0 维按 `indices` gather（带反向 scatter） |
| `index_put_first_axis(values, indices, first_axis_dim)` | 沿第 0 维 scatter（带反向 gather） |
| `index_first_axis_residual(input, indices)` | 同时返回 gather 结果与 `input.detach()` 作为残差，供自定义反向使用 |

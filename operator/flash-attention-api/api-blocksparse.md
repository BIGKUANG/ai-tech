# 块稀疏注意力 API

- 源码：`flash_attn/flash_blocksparse_attn_interface.py`
- 底层：`flash_attn_cuda` 的块稀疏算子
- 适用：已知注意力矩阵中只有部分 `BLOCK_M × BLOCK_N` 块非零的场景（如 Longformer/稀疏 BERT）

## convert_blockmask

```python
convert_blockmask(blockmask, causal)
```

把 0-1 形式的块掩码转换成 CUDA 内核所需的压缩格式。

- `blockmask`：`(row, col)` 的 0-1 张量（bool/uint8），`0` 表示该块被跳过。
- `causal`：当前**必须为 False**（源码断言 `assert not causal`）。
- 返回 `(col, row)`、`dtype=torch.int32`：对每一列，给出非零块的行索引，用 `-1` 补齐到长度 `row`。索引乘以 4，最低位编码「是否为该行第一个非零块」，次低位编码「是否为该行最后一个非零块」。

## flash_blocksparse_attn_func

```python
flash_blocksparse_attn_func(
    qkv,
    cu_seqlens,
    blockmask,
    dropout_p,
    max_s,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    convert_mask=True,
)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `qkv` | `(total, 3, nheads, headdim)` | 打包 QKV（变长布局） |
| `cu_seqlens` | `(batch_size + 1,)` int32 | 累积序列长度 |
| `blockmask` | `(row, col)` | 块掩码；`convert_mask=True` 时内部调用 `convert_blockmask` 转换 |
| `dropout_p` | float | 评估时必须为 0.0 |
| `max_s` | int | 最大序列长度 |
| `softmax_scale` | float | 缩放，默认 `1/sqrt(headdim)` |
| `causal` | bool | 因果掩码（注意：`convert_mask=True` 且 `causal=True` 时转换函数会断言失败） |
| `return_attn_probs` | bool | 是否返回注意力概率 |

**返回**：`out`（`(total, nheads, headdim)`），或 `(out, S)`（`return_attn_probs=True`）。

**示例**

```python
import torch
from flash_attn.flash_blocksparse_attn_interface import flash_blocksparse_attn_func

total, nheads, headdim = 512, 8, 64
qkv = torch.randn(total, 3, nheads, headdim, dtype=torch.float16, device="cuda")
cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32, device="cuda")
blockmask = torch.ones(4, 4, dtype=torch.bool, device="cuda")  # 4x4 个块全部保留
out = flash_blocksparse_attn_func(qkv, cu_seqlens, blockmask, 0.0, 256)
```

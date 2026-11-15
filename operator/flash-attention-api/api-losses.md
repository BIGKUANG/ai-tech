# CrossEntropy 损失 API

- 源码：`flash_attn/losses/cross_entropy.py`（封装）
- 底层 Triton 内核：`flash_attn/ops/triton/cross_entropy.py`（`cross_entropy_loss`）

融合的交叉熵损失，支持 label smoothing、logit scaling、z-loss、张量并行与原地反向。

## CrossEntropyLoss

```python
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        inplace_backward=False,
        process_group=None,
        return_z_loss=False,
    ):
```

**构造参数**

| 名称 | 说明 |
| --- | --- |
| `ignore_index` | int，`target == ignore_index` 的样本 loss 置 0 |
| `reduction` | `"mean"` / `"sum"` / `"none"` |
| `label_smoothing` | float，标签平滑系数 |
| `logit_scale` | float，对 logits 的缩放（如 `1/temperature`） |
| `lse_square_scale` | float，`> 0` 时向 loss 加入 `lse_square_scale * lse(logits)^2`（z-loss） |
| `inplace_backward` | bool，为 True 时反向原地修改 logits 以省显存 |
| `process_group` | 张量并行进程组；非空时按 vocab 切分并跨进程聚合 loss |
| `return_z_loss` | bool，为 True 时额外返回 z-loss 分量（仅用于日志，不支持反传） |

**forward**

```python
def forward(self, input, target, precomputed_lse=None):
```

- `input`：`(batch, vocab_size)`
- `target`：`(batch,)`
- `precomputed_lse`：可选，若外部已算好 logits 的 logsumexp 可传入复用

**返回**

- `reduction="none"`：`losses` 为 `(batch,)`，否则为 `(1,)`，dtype float
- `return_z_loss=True` 时返回 `(losses, z_loss)`

**示例**

```python
import torch
from flash_attn.losses.cross_entropy import CrossEntropyLoss

criterion = CrossEntropyLoss(ignore_index=-100, reduction="mean").cuda()
logits = torch.randn(32, 32000, device="cuda", dtype=torch.float16, requires_grad=True)
targets = torch.randint(0, 32000, (32,), device="cuda")
loss = criterion(logits, target)
loss.backward()
```

## 底层函数 `cross_entropy_loss`

```python
cross_entropy_loss(x, target, ..., label_smoothing=0.0, logit_scale=1.0,
                   lse_square_scale=0.0, ignore_index=-100,
                   inplace_backward=False, process_group=None, ...)
```

定义于 `flash_attn/ops/triton/cross_entropy.py`，由 `CrossEntropyLoss` 调用；也可直接使用。

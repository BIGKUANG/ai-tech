# 采样（`flashinfer.sampling`）

> 源码：`flashinfer/sampling.py`
> 上游文档：`docs/api/sampling.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.sampling` 提供用于 LLM 解码的采样内核，包括：从 logits 或概率分布采样、在线安全 Softmax、
top-p（nucleus）采样、top-k 采样、min-p 采样、top-k + top-p 联合采样、概率重归一化 / logits 掩码，
以及链式投机采样（speculative sampling）。多数算子采用 GPU 上的拒绝采样（rejection sampling）实现，
无需显式排序，将多轮拒绝采样融合进单个 CUDA 内核，比朴素的多内核实现更高效。

## 关于 batch-length `seed` / `offset` 的重要说明

> **重要**
>
> Batch-length 的 `seed` 与 `offset` 张量现在按 **每个输出行一个值** 生效。此前即使传入 batch-length
> 张量，所有行也统一使用第零个元素。因此，当这些张量中含有不同值时，修复后可能产生与之前不同的采样结果。
> 标量值、长度为 1 的张量，以及使用共享 `torch.Generator` 的调用保持原有行为不变。
>
> 输出行索引仍会参与 Philox 子序列（subsequence）的计算：即使 seed 与 offset 相同，逐行 seed 也**不会**使
> 采样结果对 batch 位置保持不变。将相同的 seed 与 offset 移到不同的 batch 槽位会改变 RNG 流。

> **另见**
>
> 若需要高效的 Top-K 选择（而非采样），参见 `docs/api/topk.rst`，其提供
> `flashinfer.top_k`、`flashinfer.top_k_page_table_transform` 与 `flashinfer.top_k_ragged_transform`。

## 导入

```python
import flashinfer

from flashinfer.sampling import (
    sampling_from_probs,
    sampling_from_logits,
    softmax,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    min_p_sampling_from_probs,
    top_k_top_p_sampling_from_logits,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_probs,
    top_k_renorm_probs,
    top_k_mask_logits,
    chain_speculative_sampling,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `softmax` | 函数 | 带温度缩放的在线安全 Softmax。 |
| `sampling_from_logits` | 函数 | 从 logits 采样（等价于先 softmax 再采样）。 |
| `sampling_from_probs` | 函数 | 从概率分布采样。 |
| `top_p_sampling_from_probs` | 函数 | 从概率做 top-p（nucleus）采样。 |
| `top_k_sampling_from_probs` | 函数 | 从概率做 top-k 采样。 |
| `min_p_sampling_from_probs` | 函数 | 从概率做 min-p 采样。 |
| `top_k_top_p_sampling_from_logits` | 函数 | 从 logits 做 top-k 与 top-p 联合采样。 |
| `top_k_top_p_sampling_from_probs` | 函数 | 从概率做 top-k 与 top-p 联合采样。 |
| `top_p_renorm_probs` | 函数 | 按 top-p 阈值重归一化概率。 |
| `top_k_renorm_probs` | 函数 | 按 top-k 阈值重归一化概率。 |
| `top_k_mask_logits` | 函数 | 按 top-k 阈值掩码 logits（其余置 `-inf`）。 |
| `chain_speculative_sampling` | 函数 | 序列生成的链式投机采样。 |

## API 详情

### 采样类算子的公共参数约定

下列采样函数共享以下参数含义：

- `probs` / `logits`（`torch.Tensor`）：待采样的概率 / 预 softmax logits。未提供 `indices` 时形状应为
  `(batch_size, num_classes)`，第 i 个输出从第 i 行采样；提供 `indices` 时形状应为
  `(unique_batch_size, num_classes)`。
- `indices`（`Optional[torch.Tensor]`，默认 `None`）：形状 `(batch_size,)`、dtype `torch.int32` 或
  `torch.int64` 的映射张量，将每个输出映射到 `probs`/`logits` 中的某一行。输出张量 dtype 与 `indices` 相同。
  若 `indices[i] = j`，则第 i 个输出从第 j 行采样。未提供 `indices` 时输出 dtype 默认为 `torch.int32`。
- `deterministic`（`bool`，默认 `True`）：是否使用确定性内核实现。
- `generator`（`Optional[torch.Generator]`）：随机数生成器。
- `check_nan`（`bool`，默认 `False`）：是否检查 `probs`/`logits` 中的 NaN。
- `seed`（`Optional[Union[int, torch.Tensor]]`）：采样用的随机种子，可为整数或 `torch.Tensor`。作为张量时
  必须为 int64/uint64、一维。长度为 1 的张量会广播到所有输出行；否则长度必须等于输出行数，按**输出行**
  而非输入分布行取值。使用 `torch.Tensor` 是 CUDA graph 兼容所必需的。

  输出行索引也参与 Philox 子序列：将相同 seed 和 offset 移到不同 batch 槽位会改变 RNG 流，故逐行 seed
  不会使采样对 batch 位置保持不变。

  ⚠️ 若显式提供 `seed` 与 `offset`，调用方需负责在多次调用间更新其取值以保证随机样本不同（例如按消耗的
  随机值数量递增 offset，或按调用次数更新 seed）。
- `offset`（`Optional[Union[int, torch.Tensor]]`）：采样用的随机 offset，规则同 `seed`。同样需由调用方在
  调用间更新。
- `return_valid`（`bool`，默认 `False`）：为 `True` 时额外返回一个布尔 mask，标记哪些行具有有效
  （非退化）分布。

**返回（含 `return_valid` 的算子）**

- `return_valid=False`：一维 `samples` 张量，形状 `(batch_size,)`。
- `return_valid=True`：`(samples, valid)`，其中 `valid` 为形状 `(batch_size,)` 的布尔张量。

### `softmax`

```python
softmax(
    logits: torch.Tensor,
    temperature: Optional[Union[torch.Tensor, float]] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**功能**：带温度缩放的融合 GPU `在线安全 Softmax <https://arxiv.org/abs/1805.02867>`_。

**参数**

- `logits`（`torch.Tensor`）：二维 CUDA logits 张量。
- `temperature`（`Optional[Union[torch.Tensor, float]]`，默认 `None`）：标量或形状 `(batch_size,)` 的张量。
  为标量时所有请求使用同一温度；为张量时每个请求各有温度。张量须与 `logits` 在同一 CUDA 设备，dtype 为
  float16、bfloat16、float32 或 float64。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL。为 `None` 时在计算能力 >= 9.0 的设备上自动启用。

**返回**

- `torch.Tensor`：与输入同形状的 softmax 概率张量。

**注意事项**

- 自定义 Blackwell softmax 路由目前在 SM103 上应用 PDL；SM100 快速路径保持非 PDL，回退路由保持既有 PDL
  行为。在 SM100/SM103 上，32 字节对齐、词表大小为 8 的倍数且不超过 262144 的行采用行缓存 cluster 路由，
  对每行恰好读写一次。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> logits = torch.rand(batch_size, vocab_size).to(0)
>>> probs = flashinfer.sampling.softmax(logits, temperature=1.0)
>>> probs
tensor([[0.2309, 0.2385, 0.1401, 0.2493, 0.1412],
        [0.2019, 0.1431, 0.2448, 0.2837, 0.1265],
        [0.2401, 0.1707, 0.2249, 0.1664, 0.1979],
        [0.1724, 0.2719, 0.1991, 0.1465, 0.2101]], device='cuda:0')
```

---

### `sampling_from_logits`

```python
sampling_from_logits(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor
```

**功能**：从 logits 做类别采样的融合 GPU 内核，等价于对 `logits` 先应用 softmax 再采样。由于不依赖 CUB
的 `BlockScan`，该采样是确定性的；`deterministic` 参数为与其他采样函数保持兼容而保留。

**参数**：见「采样类算子的公共参数约定」。

**返回**

- `samples`（`torch.Tensor`）：采样得到的类别，形状 `(batch_size,)`。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> logits = torch.rand(batch_size, vocab_size).to(0)
>>> samples = flashinfer.sampling.sampling_from_logits(logits)
>>> samples
tensor([0, 1, 1, 1], device='cuda:0', dtype=torch.int32)
```

---

### `sampling_from_probs`

```python
sampling_from_probs(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：从概率分布做类别采样的融合 GPU 内核。

**参数**：见「采样类算子的公共参数约定」。

**返回**

- 见公共约定。`return_valid=True` 时 `valid` 标记哪些行具有有效分布。

**注意事项**

- 该函数期望 float32 输入，输出为 int32。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> samples = flashinfer.sampling.sampling_from_probs(norm_prob)
>>> samples
tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)
```

---

### `top_p_sampling_from_probs`

```python
top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：从概率做 top-p（nucleus）采样的融合 GPU 内核，采用 GPU 上的拒绝采样，无需显式排序。

**参数**

- `top_p`（`Union[torch.Tensor, float]`）：top-p 阈值，float 或形状 `(batch_size,)` 的张量；为 float 时所有
  请求共用同一阈值。
- 其余见「采样类算子的公共参数约定」。

**返回**

- 见公共约定；`valid` 标记重归一化后哪些行具有有效分布。

**注意事项**

- 期望 float32 输入，输出为 int32。

**另见**：`top_k_top_p_sampling_from_probs`、`top_k_sampling_from_probs`、`top_p_renorm_probs`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_p = 0.5
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> samples = flashinfer.sampling.top_p_sampling_from_probs(norm_prob, top_p)
>>> samples
tensor([1, 2, 0, 4], device='cuda:0', dtype=torch.int32)
```

---

### `top_k_sampling_from_probs`

```python
top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：从概率做 top-k 采样的融合 GPU 内核，采用 GPU 上的拒绝采样，无需显式排序。

**参数**

- `top_k`（`Union[torch.Tensor, int]`）：top-k 阈值，标量或形状 `(batch_size,)` 的张量；为标量时所有请求共用。
- 其余见「采样类算子的公共参数约定」。

**返回**

- 见公共约定；`valid` 标记重归一化后哪些行具有有效分布。

**注意事项**

- 期望 float32 输入，输出为 int32。

**另见**：`top_k_top_p_sampling_from_probs`、`top_p_sampling_from_probs`、`top_k_renorm_probs`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_k = 1
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> samples = flashinfer.sampling.top_k_sampling_from_probs(norm_prob, top_k)
>>> samples
tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)
```

---

### `min_p_sampling_from_probs`

```python
min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：从概率做 `min-p 采样 <https://arxiv.org/abs/2407.01082>`_ 的融合 GPU 内核，采用 GPU 上的拒绝
采样，无需显式排序。

**参数**

- `min_p`（`Union[torch.Tensor, float]`）：min-p 阈值，标量或形状 `(batch_size,)` 的张量。
- 其余见「采样类算子的公共参数约定」。

**返回**

- 见公共约定；`valid` 标记重归一化后哪些行具有有效分布。

**注意事项**

- 期望 float32 输入，输出为 int32。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> min_p = torch.full((batch_size,), 0.05).to(0)
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> samples = flashinfer.sampling.min_p_sampling_from_probs(norm_prob, min_p)
>>> samples
tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)
```

---

### `top_k_top_p_sampling_from_logits`

```python
top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor
```

**功能**：从预 softmax logits 做 top-k 与 top-p 采样的融合 GPU 内核，采用 GPU 上的拒绝采样，无需显式排序。

**参数**

- `top_k`（`Union[torch.Tensor, int]`）：top-k 阈值，标量或形状 `(batch_size,)` 的张量。
- `top_p`（`Union[torch.Tensor, float]`）：top-p 阈值，标量或形状 `(batch_size,)` 的张量。
- `filter_apply_order`（`str`，默认 `"top_k_first"`）：top-k 与 top-p 的应用顺序，为 `"top_k_first"` 或
  `"joint"`。`"top_k_first"` 先做 top-k 过滤，再在 top-k 结果上做 top-p；`"joint"` 每轮同时应用 top-k 与
  top-p。
- 其余见「采样类算子的公共参数约定」。

**返回**

- `samples`（`torch.Tensor`）：采样得到的类别，形状 `(batch_size,)`。

**注意事项**

- 期望 float32 输入，输出为 int32。
- 当 `filter_apply_order="top_k_first"` 且满足快速路径条件（无 `indices`、标量 `top_k` 且
  `0 < top_k <= 256`、词表 `>= 65536`）时，会使用 "top_k_first" 快速路径：先用并行 radix/cluster top-k 选出
  top-k，再仅在这 k 个元素上做 top-p 采样。

**另见**：`top_k_top_p_sampling_from_probs`、`top_k_mask_logits`、`top_p_sampling_from_probs`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_p, top_k = 0.5, 3
>>> logits = torch.rand(batch_size, vocab_size).to(0)
>>> samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)
>>> samples
tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32)
```

---

### `top_k_top_p_sampling_from_probs`

```python
top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：从概率做 top-k 与 top-p 采样的融合 GPU 内核，采用 GPU 上的拒绝采样，无需显式排序。

**参数**

- `top_k`（`Union[torch.Tensor, int]`）：top-k 阈值。
- `top_p`（`Union[torch.Tensor, float]`）：top-p 阈值。
- `filter_apply_order`（`str`，默认 `"top_k_first"`）：含义同 `top_k_top_p_sampling_from_logits`。
- 其余见「采样类算子的公共参数约定」。

**返回**

- 见公共约定；`valid` 标记重归一化后哪些行具有有效分布。

**注意事项**

- 期望 float32 输入，输出为 int32。

**另见**：`top_k_sampling_from_probs`、`top_p_sampling_from_probs`、`top_k_renorm_probs`、
`top_p_renorm_probs`、`top_k_mask_logits`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_p = torch.full((batch_size,), 0.2).to(0)
>>> top_k = torch.full((batch_size,), 2).to(0)
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(norm_prob, top_k, top_p)
>>> samples
tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)
```

---

### `top_p_renorm_probs`

```python
top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    is_deterministic: bool = False,
) -> torch.Tensor
```

**功能**：按 top-p 阈值重归一化概率的融合 GPU 内核，使用 AIR Top-P 算法（基于 radix）高效寻找阈值。

**参数**

- `probs`（`torch.Tensor`）：概率，形状 `(batch_size, num_classes)`。
- `top_p`（`Union[torch.Tensor, float]`）：top-p 阈值，应在 `(0, 1)` 内；标量或形状 `(batch_size,)` 的张量。
  掩掉小于阈值的概率（使 `probs[probs >= threshold]` 的累积和为 `top_p`）并重归一化。
- `is_deterministic`（`bool`，默认 `False`）：为 `True` 时使用确定性整数累加以获得可复现结果，会影响性能。

**返回**

- `renorm_probs`（`torch.Tensor`）：重归一化后的概率，形状 `(batch_size, num_classes)`。

**注意事项**

- `top_p_renorm_probs` 与 `sampling_from_probs` 的组合应等价于 `top_p_sampling_from_probs`。

**另见**：`top_p_sampling_from_probs`、`sampling_from_probs`、`top_k_renorm_probs`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_p = 0.3
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> renormed_probs = flashinfer.sampling.top_p_renorm_probs(prob, top_p)
>>> renormed_probs
tensor([[0.0000, 0.4882, 0.0000, 0.5118, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.5181, 0.0000, 0.4819, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')
```

---

### `top_k_renorm_probs`

```python
top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    is_deterministic: bool = False,
) -> torch.Tensor
```

**功能**：按 top-k 阈值重归一化概率的融合 GPU 内核。

**参数**

- `probs`（`torch.Tensor`）：概率，形状 `(batch_size, num_classes)`。支持 dtype：`float32`、`float16`、
  `bfloat16`。
- `top_k`（`Union[torch.Tensor, int]`）：top-k 阈值，应在 `(0, num_classes)` 内；标量或形状 `(batch_size,)`
  的张量。保留 top-k 概率，其余置零并重归一化。
- `is_deterministic`（`bool`，默认 `False`）：为 `True` 时使用与顺序无关的定点整数原子操作累加重归一化
  求和，使结果在多次调用间位级可复现（默认的浮点累加依赖 CTA 到达顺序），适合张量并行各 rank 需从相同输入
  得到相同输出的场景。会轻微影响性能。

**返回**

- `renorm_probs`（`torch.Tensor`）：重归一化后的概率，形状 `(batch_size, num_classes)`，dtype 与输入相同。

**注意事项**

- `top_k_renorm_probs` 与 `sampling_from_probs` 的组合应等价于 `top_k_sampling_from_probs`。

**另见**：`top_k_sampling_from_probs`、`sampling_from_probs`、`top_p_renorm_probs`、`top_k`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_k = 3
>>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
>>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
>>> renormed_probs = flashinfer.sampling.top_k_renorm_probs(prob, top_k)
>>> renormed_probs
tensor([[0.3201, 0.3319, 0.0000, 0.3480, 0.0000],
        [0.2573, 0.0000, 0.3398, 0.4028, 0.0000],
        [0.3672, 0.0000, 0.3416, 0.0000, 0.2912],
        [0.0000, 0.4243, 0.2750, 0.0000, 0.3007]], device='cuda:0')
```

---

### `top_k_mask_logits`

```python
top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor
```

**功能**：按 top-k 阈值掩码 logits 的融合 GPU 内核。

**参数**

- `logits`（`torch.Tensor`）：softmax 前的 logits，形状 `(batch_size, num_classes)`。支持 dtype：`float32`、
  `float16`、`bfloat16`。
- `top_k`（`Union[torch.Tensor, int]`）：top-k 阈值，应在 `(0, num_classes)` 内；标量或形状 `(batch_size,)`
  的张量。保留 top-k logits，其余置为负无穷。

**返回**

- `masked_logits`（`torch.Tensor`）：掩码后的 logits，形状 `(batch_size, num_classes)`，dtype 与输入相同。

**注意事项**

- `top_k_mask_logits` 与 `softmax` 的组合应等价于 `top_k_renorm_probs`。

**另见**：`top_k_renorm_probs`、`top_k`

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 5
>>> top_k = 3
>>> logits = torch.randn(batch_size, vocab_size).to(0)
>>> masked_logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
>>> masked_logits
tensor([[ 1.9269,  1.4873,  0.9007,    -inf,    -inf],
        [ 1.0783,  0.8008,  1.6806,    -inf,    -inf],
        [   -inf,  0.2415, -0.2316,  0.0418,    -inf],
        [ 0.8599, -0.3097,    -inf,  0.8034,    -inf]], device='cuda:0')
```

---

### `chain_speculative_sampling`

```python
chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    target_probs,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor
```

**功能**：用于序列生成的融合 GPU 投机采样内核（论文
`Accelerating Large Language Model Decoding with Speculative Sampling
<https://arxiv.org/pdf/2302.01318>`_），draft 模型为每个请求生成一个 token 序列（chain）。

**参数**

- `draft_probs`（`torch.Tensor`）：draft 模型生成的词表概率。形状 `(batch_size, num_speculate_tokens, vocab_size)`。
- `draft_token_ids`（`torch.Tensor`）：draft 模型生成的 token 索引。形状
  `(batch_size, num_speculate_tokens)`。
- `target_probs`（`torch.Tensor`）：target 模型生成的词表概率。相比 `draft_probs` 多一个槽位，因为 target
  模型会比 draft 模型多生成一个 token。形状 `(batch_size, num_speculate_tokens + 1, vocab_size)`。
- `maybe_output_accepted_token_num`（`Optional[torch.Tensor]`，默认 `None`）：若每个 token 独立考虑时每个请求
  可接受的 token 数（不考虑遇到首个不满足 `r < p/q` 即停止）。形状 `(batch_size)`。指定时接受数会被原地
  累加到该张量。
- `maybe_output_emitted_draft_token_num`（`Optional[torch.Tensor]`，默认 `None`）：每个请求最终发出的 draft
  token 数，不含 bonus token（故总采样 token 数为 `output_emitted_draft_token_num + 1`）。形状 `(batch_size)`。
  指定时发出数会被原地累加。
- `deterministic`（`bool`，默认 `True`）：是否使用确定性内核实现。
- `generator`、`seed`、`offset`：随机数来源，含义同公共约定。

**返回**

- `output_token_ids`（`torch.Tensor`）：经 target 模型验证的输出 token 索引，被拒绝的样本以 `-1` 填充。相比
  `draft_token_ids` 末尾多一个 token 索引；若之前所有 token 均被接受，则会从 target 模型概率中额外采样一个
  "bonus" token。形状 `(batch_size, num_speculate_tokens + 1)`。
- `output_accepted_token_num`（`torch.Tensor`）：独立考虑时可接受的 token 数，形状 `(batch_size)`。
- `output_emitted_draft_token_num`（`torch.Tensor`）：最终发出的 draft token 数（不含 bonus），形状 `(batch_size)`。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> draft_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1]]]).to(0)
>>> draft_token_ids = torch.tensor([[2, 1]], dtype=torch.int32).to(0)
>>> target_probs = torch.tensor(
...     [[[0.0, 0.1, 0.6, 0.3], [1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1]]]
... ).to(0)
>>> out_ids, acc_num, emitted_num = flashinfer.sampling.chain_speculative_sampling(
...     draft_probs, draft_token_ids, target_probs)
>>> out_ids
tensor([[ 2,  0, -1]], device='cuda:0', dtype=torch.int32)
```

## 相关链接 / 参考

- 上游文档：`docs/api/sampling.rst`
- 相关：`flashinfer.top_k`、`flashinfer.top_k_page_table_transform`、`flashinfer.top_k_ragged_transform`
- Cake 采样实现：`docs/api/cake_sampling.rst`

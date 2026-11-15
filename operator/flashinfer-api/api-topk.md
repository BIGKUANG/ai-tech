# Top-K 选择（`flashinfer.topk`）

> 源码：`flashinfer/topk.py`、`flashinfer/topk_varlen/topk_varlen.py`
> 上游文档：`docs/api/topk.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.topk` 提供高效的 Top-K 选择内核。`top_k` 是一般用途的逐行 Top-K 选择，可作为
`torch.topk` 的替代（在大词表上更快）；`top_k_page_table_transform` 与 `top_k_ragged_transform`
是面向稀疏注意力第二阶段、将 Top-K 选择与索引变换融合的算子；`top_k_varlen` 面向批量解码步的变长
logits 选择。

这些算子会在多个后端之间自动调度：原生 radix 内核、clusters 内核（SM100）以及 CUB 的
`DeviceBatchedTopK`。可通过环境变量 `FLASHINFER_TOPK_ALGO`（`"default"`、`"clusters"`、`"cub"`）
强制指定后端。

> **另见**
>
> 若需要基于 Top-K 的采样，参见 `docs/api/sampling.rst`，其提供
> `flashinfer.sampling.top_k_sampling_from_probs`、`flashinfer.sampling.top_k_top_p_sampling_from_probs`、
> `flashinfer.sampling.top_k_renorm_probs` 与 `flashinfer.sampling.top_k_mask_logits`。

## 导入

```python
import flashinfer
from flashinfer import top_k, top_k_varlen, top_k_page_table_transform, top_k_ragged_transform

# 后端能力查询（位于 flashinfer.topk 子模块）
from flashinfer.topk import can_implement_filtered_topk, TopKTieBreak
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `top_k` | 函数 | 一般用途的逐行 Top-K 选择。 |
| `top_k_varlen` | 函数 | 变长（per-request KV 长度）批量解码 Top-K 选择。 |
| `top_k_page_table_transform` | 函数 | 融合 Top-K 选择 + Page Table 索引变换。 |
| `top_k_ragged_transform` | 函数 | 融合 Top-K 选择 + Ragged 索引变换。 |
| `can_implement_filtered_topk` | 函数 | 查询设备是否支持 FilteredTopK。 |
| `TopKTieBreak` | 枚举（`IntEnum`） | Top-K 并列（tie）处理模式。 |

## API 详情

### `TopKTieBreak`

```python
class TopKTieBreak(IntEnum):
    NONE = 0   # 遗留行为，无显式索引 tie-break
    SMALL = 1  # 偏好更小的索引
    LARGE = 2  # 偏好更大的索引
```

**功能**：Top-K 的并列（tie）处理模式。它以类枚举风格提供 API，同时保留与 FFI 调度兼容的整数值。

**成员**

- `NONE = 0`：无显式索引 tie-break（遗留行为）。
- `SMALL = 1`：偏好更小的索引。
- `LARGE = 2`：偏好更大的索引。

**注意事项**

- tie-break 控制选择哪些边界元素，并不蕴含确定性的输出顺序；若还需要可复现的输出顺序，请设置
  `deterministic=True`。

---

### `top_k`

```python
top_k(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：从输入张量的每一行中选择最大的 k 个元素。它会根据形状、dtype 与所请求的模式，在原生 radix
内核、clusters 内核（SM100）与 CUB 的 `DeviceBatchedTopK` 之间自动调度。设置 `FLASHINFER_TOPK_ALGO`
（`"default"`、`"clusters"`、`"cub"`）可强制指定后端。设计为 `torch.topk` 的替代，在大张量
（`vocab_size > 10000`）上性能更好。

**参数**

- `input`（`torch.Tensor`）：输入张量，形状 `(batch_size, d)`。支持 dtype：`float32`、`float16`、
  `bfloat16`。
- `k`（`int`）：每行选择的 top 元素个数。
- `sorted`（`bool`，默认 `False`）：为 `True` 时返回的 top-k 元素按降序排序；默认 `False`（不排序，更快）。
- `deterministic`（`bool`，默认 `False`）：为 `True` 时使用确定性模式。确定性模式保证在固定输入与系统上，
  所选 top-k 集合的 FlashInfer 输出顺序可复现。
- `tie_break`（`int`，默认 `TopKTieBreak.NONE`）：选择边界处相等值的 tie-break 模式，支持
  `0`（无显式索引 tie-break）、`1`（偏好更小索引）、`2`（偏好更大索引），或使用 `TopKTieBreak` 枚举值。
- `dsa_graph_safe`（`bool`，默认 `False`）：为 `True` 时要求 CUDA-graph 安全的执行路径。原生 radix 后端
  通过强制 FilteredTopK 并使用 graph-safe 向量化（`VEC_SIZE=1`）来满足；CUB 后端本身即 graph-safe，因此
  仍可能服务此类调用。

**返回**

- `values`（`torch.Tensor`）：top-k 值，形状 `(batch_size, k)`，dtype 与输入相同。
- `indices`（`torch.Tensor`）：top-k 元素的索引，形状 `(batch_size, k)`，dtype 为 int64。

**注意事项**

- 与 `torch.topk` 不同，默认返回 **未排序** 结果以获得更好性能；需要排序输出时设置 `sorted=True`。
- 基于 radix 的算法在词表大小上为 O(n)，相比基于堆方法的 O(n log k)，在大词表上更快。
- 对小词表（< 1000），`torch.topk` 可能更快。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> torch.manual_seed(42)
>>> batch_size = 4
>>> vocab_size = 32000
>>> k = 256
>>> logits = torch.randn(batch_size, vocab_size, device="cuda")
>>> values, indices = flashinfer.top_k(logits, k)
>>> values.shape, indices.shape
(torch.Size([4, 256]), torch.Size([4, 256]))
>>> # 排序输出（兼容 torch.topk）
>>> values_sorted, indices_sorted = flashinfer.top_k(logits, k, sorted=True)
>>> # 确定性模式（位级可复现输出）
>>> values, indices = flashinfer.top_k(logits, k, deterministic=True)
```

**另见**：`torch.topk`、`flashinfer.sampling.top_k_mask_logits`、`flashinfer.sampling.top_k_renorm_probs`

---

### `top_k_page_table_transform`

```python
top_k_page_table_transform(
    input: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_to_batch: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
    row_starts: Optional[torch.Tensor] = None,
    page_table_row_starts: Optional[torch.Tensor] = None,
    *,
    page_size: int = 1,
    out: Optional[torch.Tensor] = None,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：融合 Top-K 选择与 Page Table 变换，用于稀疏注意力。该函数对输入分数做 top-k 选择，并在单个融合
内核中将所选索引通过 page table 转换。它根据形状、dtype 与请求模式在原生 radix 内核、clusters 内核
（SM100）与 CUB 的 `DeviceBatchedTopK` 之间自动调度。每个 page-table 条目代表 `page_size` 个连续的分数位置。
对第 `i` 行中每个被选中的局部索引 `idx`：

```python
physical_page = src_page_table[batch_idx, page_table_row_start[i] + idx // page_size]
output[i, j] = physical_page * page_size + idx % page_size
```

其中 `batch_idx` 由 `row_to_batch[i]` 决定（若提供），否则为 `i`。`topk_indices` 相对 `row_starts[i]`。

**参数**

- `input`（`torch.Tensor`）：输入分数张量，形状 `(num_rows, max_len)`。支持 dtype：`float32`、`float16`、
  `bfloat16`。
- `src_page_table`（`torch.Tensor`）：源 page table，形状 `(batch_size, max_page_table_length)`，dtype `int32`。
  被选中索引用到的条目必须非负，且每个 `physical_page * page_size + offset` 必须符合有符号 `int32`。这些取值
  约束不在运行时检查。
- `lengths`（`torch.Tensor`）：每行的实际 KV 长度，形状 `(num_rows,)`，dtype `int32`。
- `k`（`int`）：每行选择的 top 元素个数。
- `row_to_batch`（`Optional[torch.Tensor]`，默认 `None`）：行索引到 batch 索引的映射，形状 `(num_rows,)`，
  dtype `int32`；为 `None` 时按 1:1 映射（`row_idx == batch_idx`）。
- `deterministic`（`bool`，默认 `False`）：为 `True` 时使用确定性模式。
- `tie_break`（`int`，默认 `TopKTieBreak.NONE`）：含义同 `top_k`。
- `dsa_graph_safe`（`bool`，默认 `False`）：含义同 `top_k`。
- `row_starts`（`Optional[torch.Tensor]`，默认 `None`）：每行起始索引，形状 `(num_rows,)`，dtype `int32`。
  对第 `i` 行在 `[row_starts[i], row_starts[i] + lengths[i])` 上计算 top-k；为 `None` 时等价于全零。
- `page_table_row_starts`（`Optional[torch.Tensor]`，默认 `None`）：每行 page-table 起始索引（以 page-table
  条目为单位），形状 `(num_rows,)`，dtype `int32`；为 `None` 时默认等于 `row_starts`，使分数窗口与
  page-table 窗口共享同一起点。当 `page_size > 1` 且提供了 `row_starts` 时必须提供本参数，因为两者单位不同。
- `page_size`（`int`，默认 `1`）：每个 page-table 条目代表的分数位置数。必须是 `<= 2**30` 的正的 2 的幂。
  设为 `1` 保持一条目一分数行为。
- `out`（`Optional[torch.Tensor]`，可选）：可选的连续 `int32` 输出缓冲，形状 `(num_rows, k)`。提供可避免分配并
  对 CUDA graph 友好。
- `out_raw_indices`（`Optional[torch.Tensor]`，可选）：可选的连续 `int32` 输出缓冲，形状 `(num_rows, k)`，
  接收 page-table 变换前、相对于每个分数窗口的被选索引。padding 位置置 `-1` 并与 `out` 保持位置对齐。不得与
  `out` 重叠。

**返回**

- `output`（`torch.Tensor`）：物理索引，形状 `(num_rows, k)`，dtype `int32`。当提供了 `out` 时返回同一张量。
  超出实际长度的位置置 `-1`。

**注意事项**

- 专为稀疏注意力的第二阶段设计。
- `input` 行间可有 padding，但其最后一维必须连续。
- 若 `lengths[i] <= k`，则选中所有有效索引 `0..lengths[i]-1`（包括 `k > max_len` 时）。输出保持形状
  `(num_rows, k)`；未用位置在映射输出与可选 raw 索引中均为 `-1`。padding 不经过 page table 变换。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> num_rows, max_len, k = 8, 4096, 256
>>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
>>> src_page_table = torch.randint(0, 1000, (num_rows, max_len), device="cuda", dtype=torch.int32)
>>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
>>> output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)
>>> output.shape
torch.Size([8, 256])
```

---

### `top_k_ragged_transform`

```python
top_k_ragged_transform(
    input: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：融合 Top-K 选择与 Ragged 索引变换，用于稀疏注意力。该函数对输入分数做 top-k 选择，并在单个融合
内核中通过加上一个偏移来变换所选索引。用于稀疏注意力第二阶段、ragged/变长 KV cache 布局。它根据形状、dtype
与请求模式在原生 radix 内核、clusters 内核（SM100）与 CUB 的 `DeviceBatchedTopK` 之间自动调度。

对第 i 行：

```python
output_indices[i, j] = topk_indices[j] + offsets[i]
```

**参数**

- `input`（`torch.Tensor`）：输入分数张量，形状 `(num_rows, max_len)`。支持 dtype：`float32`、`float16`、
  `bfloat16`。
- `offsets`（`torch.Tensor`）：每行要加的偏移，形状 `(num_rows,)`，dtype `int32`。
- `lengths`（`torch.Tensor`）：每行的实际 KV 长度，形状 `(num_rows,)`，dtype `int32`。
- `k`（`int`）：每行选择的 top 元素个数。
- `deterministic`（`bool`，默认 `False`）：为 `True` 时使用确定性模式。
- `tie_break`（`int`，默认 `TopKTieBreak.NONE`）：含义同 `top_k`。
- `dsa_graph_safe`（`bool`，默认 `False`）：含义同 `top_k`。
- `row_starts`（`Optional[torch.Tensor]`，默认 `None`）：每行起始索引，形状 `(num_rows,)`，dtype `int32`。
  对第 `i` 行在 `[row_starts[i], row_starts[i] + lengths[i])` 上计算 top-k。输出索引仍为
  `local_topk + offsets[i]`，其中 `local_topk` 相对 `row_starts[i]`。默认为 `None`（等价于全零）。

**返回**

- `output_indices`（`torch.Tensor`）：输出索引，形状 `(num_rows, k)`，dtype `int32`，包含 top-k 索引加偏移。
  超出实际长度的位置置 `-1`。

**注意事项**

- 专为稀疏注意力第二阶段、ragged KV cache 布局设计。
- 若 `lengths[i] <= k`，所有有效索引都加上 `offsets[i]`（包括 `k > max_len` 时）。输出保持形状
  `(num_rows, k)`；未用位置保持 `-1`，不加偏移。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> num_rows, max_len, k = 8, 4096, 256
>>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
>>> offsets = torch.arange(0, num_rows * max_len, max_len, device="cuda", dtype=torch.int32)
>>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
>>> output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)
>>> output.shape
torch.Size([8, 256])
```

---

### `top_k_varlen`

```python
top_k_varlen(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    pre_idx: Optional[torch.Tensor] = None,
    compress_ratio: int = 1,
    next_n: int = 1,
    return_values: bool = False,
    out_indices: Optional[torch.Tensor] = None,
    out_values: Optional[torch.Tensor] = None,
    backend: Literal[
        "radix", "gvr", "gvr_2", "radix_cutlass", "radix_filter", "auto"
    ] = "auto",
    load_balance: bool = True,
    workspace: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：在批量解码步 logits 上做 Top-K 选择。从 `logits` 的每一行选择 top-`top_k` 个元素，并遵守 `seq_lens`
给出的每请求 KV-cache 长度。

**后端选择**：`backend="auto"`（默认）按形状与 dtype 对可运行的后端排名（见 `_top_k_varlen_heuristic`）：
datacentre Blackwell 级 GPU 上的 fp32 用 `gvr_2`（带或不带 hint，缺 hint 在多数形状上约慢 10%），
在实测的大 N 区域用 `radix_filter`，大 hint 半精度批次的 mid-length 行用 `gvr`，否则用 Blackwell 上的 CuTe
DSL `radix`，并在其 fp32 大角落与其他所有 GPU 上用 `radix_cutlass` 掩码回退。可用
`backend="radix"`、`"gvr"`、`"gvr_2"`、`"radix_filter"` 或 `"radix_cutlass"` 强制指定。

**参数**

- `logits`（`torch.Tensor`）：二维 float 张量，形状 `(num_rows, max_seq_len)`。支持 dtype：`float32`、
  `bfloat16`、`float16`。对 `"gvr"` 后端，行宽 `max_seq_len` 必须是 `16 // itemsize` 的倍数（fp16/bf16 为 8，
  fp32 为 4），使每行 16 字节对齐以支持 GVR 的 128-bit 向量化加载，否则抛 `ValueError`。`"radix_cutlass"`
  无此约束。
- `seq_lens`（`torch.Tensor`）：一维 `int32` 张量，形状 `(num_rows // next_n,)`，存每个请求的有效 KV-cache
  长度，**以 KV-token 为单位**（无论 `compress_ratio` 如何，都是原始序列长度，而非已除以压缩因子的长度）。
  请求 `i` 的第 `t` 行（`t` 属于 `[0, next_n)`）对其前
  `max(0, (seq_lens[i] - next_n + t + 1) // compress_ratio)` 个 logit 列排名（padding 或 evicted 槽位可能使
  分子为负，这类行无有效列并被输出为全 `-1`）；等于或超过该计数的 logit 被排除在搜索之外。行长度超过 logits
  宽度时（动态长度超出静态缓冲，例如 CUDA-graph 重放下）会被所有后端钳制到宽度。
- `top_k`（`int`）：每行 top 元素个数。GVR 后端支持 `{512, 1024, 2048}`；radix 后端无限制。
- `pre_idx`（`torch.Tensor`，可选）：`int32[num_rows // next_n, top_k]`，由 **同一层** 在上一个 token 的解码步
  选出的 top-K KV-cache 索引。GVR 利用层在 step `t` 与 `t+1` 的注意力模式强相关；内核内部施加 `+1` 偏移
  （DSv3.2）使上一步索引正确落在当前步增长的 KV-cache 空间。`pre_idx[:, 0]` 必须是 argmax 索引。`"gvr"`
  后端必需；`"gvr_2"` 可选（仅作采样锚点，无 hint 亦精确但稍慢）；radix 后端忽略。必须是连续、16 字节对齐的
  int32 CUDA 张量，形状 `[seq_lens.shape[0], top_k]`。违反该契约的 hint 会以 `RuntimeWarning` **被丢弃**，
  调用以无 hint 方式运行（`gvr_2` 用其无 hint 引擎；显式 `"gvr"` 请求被拒绝）。
- `compress_ratio`（`int`，默认 `1`）：KV 索引压缩因子（DSv3.2 为 `1`，DSv4 为 `4`）。每个 logit 列代表
  `compress_ratio` 个连续 KV token。`seq_lens` 保持 KV-token 单位；内核按
  `max(0, (seq_len - next_n + t + 1) // compress_ratio)` 推导每行的有效列数。
- `next_n`（`int`，默认 `1`）：投机解码的时间步长。
- `return_values`（`bool`，默认 `False`）：为 `True` 时同时返回被选中的 logit 值。
- `out_indices`（`torch.Tensor`，可选）：预分配的 `int32[num_rows, top_k]` 输出缓冲：连续、16 字节对齐、位于
  `logits.device`，形状为该形状，或为恰好 `num_rows * top_k` 个元素的一维缓冲（原地视为 `[num_rows, top_k]`
  返回，返回的是同一存储的视图而非调用方对象）；更宽、更高或其他形状会抛 `ValueError`。
- `out_values`（`torch.Tensor`，可选）：预分配的值缓冲（dtype 同 `logits`，布局规则同 `out_indices`）。仅在
  `return_values=True` 时使用。
- `backend`（`{"radix", "gvr", "gvr_2", "radix_cutlass", "radix_filter", "auto"}`，默认 `"auto"`）：后端。
  - `"radix"`：CuTe DSL 单趟多 CTA radix top-K（Blackwell sm_100+ 含 Rubin sm_107；原生 varlen，无
    `pre_idx`，无 logit 掩码）。
  - `"gvr"`：GVR 内核（仅 Blackwell sm_100+，需要 `pre_idx`）。`load_balance` 选择 LB 与单 CTA 路径。
  - `"gvr_2"`：自采样 GVR V2（TRT-LLM PR #17821 移植）：采样校准的阈值阶梯、精确的可交换并列 top-K、每
    batch 一次 launch。仅 datacentre Blackwell 级（sm_100/103 或 Rubin sm_107）；fp32 logits；`pre_idx` 可选；
    `top_k` 属于 `{512, 1024, 2048}`。忽略 `load_balance`。
  - `"radix_cutlass"`：掩码 CUTLASS radix top-K（所有 GPU，无需 `pre_idx`）。
  - `"radix_filter"`：DKG filtered-radix top-K（coarse histogram → filter → on-chip refine）：与 `"radix"`
    一样无 hint，fp32/fp16/bf16，`top_k` 属于 `[1, 16384]`，仅 `compress_ratio == 1`，仅行相对索引
    （无融合 page-table 变换，并列不确定）。datacentre Blackwell 级（sm_100/103/107），需要
    nvidia-cutlass-dsl >= 4.8。接受并忽略 `pre_idx`。
  - `"auto"`：形状/dtype 感知的选择，追踪各配置的实测胜者。
- `load_balance`（`bool`，默认 `True`）：选择 GVR 内核路径（radix 后端忽略）。
  - `True`（默认）：双内核 LB 路径（`GvrTopKLBPrepareKernel` + `GvrTopKLBKernel`），先由 prepare 内核将请求
    分为长/短桶，再由主内核将长行跨 CTA cluster 拆分并打包短行。适合 GVR 面向的 ragged 解码批次。
  - `False`：单内核路径（`GvrTopKKernel`），每行一个 CTA，无 prepare 步骤。批次无长度方差时更快。
  两者都是 CUDA-graph 安全的（不基于设备数据做 host 分支）。
- `workspace`（`dict`，可选）：GVR `load_balance=True` 路径的可复用 workspace 缓冲。
  - 必需键（均为 `int32`，与 `logits` 同设备）：`"gvr_order_row"`，形状 `(M,)`，其中 `M` 为 `[64, 1024]` 中
    最小且 `>= seq_lens.shape[0]` 的 2 的幂；`"gvr_counters"`，形状 `(2,)`。
  - `"gvr_2"` 后端可选键 `"gvr2_workspace"`（CUDA 张量，至少
    `flashinfer.topk_varlen.kernels.gvr2_topk_host.workspace_bytes()` = 20,973,568 字节，首次使用前零初始化，
    16 字节对齐）可覆盖缓存默认 slab。
  - ⚠️ **警告**：不要在并发 CUDA 流之间共享同一 workspace dict——每个流须有自己的 workspace 以避免设备张量
    竞争。`"gvr"` 后端下 `workspace=None`（默认）按次分配，任何并发下都安全；`"gvr_2"` 下 `workspace=None`
    解析为 **每个设备与 CUDA 流一个 slab**。

**返回**

- `(indices, values)`：始终为 2 元组。`indices` 为 `int32[num_rows, top_k]`；`values` 在 `return_values=True`
  时保存被选中的 logits（dtype 同 `logits`），否则为 `None`。

**异常**

- `BackendSupportedError`：请求的后端在当前设备不受支持，或显式 `"gvr"` 请求没有可用的 `pre_idx`。
- `ValueError`：`logits` / `seq_lens` / `next_n` 违反形状、dtype 或分组契约，`out_indices` / `out_values`
  缓冲不符合契约，或显式后端的检查器拒绝该问题（报 "Problem size is not supported"）。
- `RuntimeWarning`（警告）：`pre_idx` 格式错误已被丢弃；调用以无 hint 方式运行。

**示例**

```python
>>> import torch, flashinfer
>>> torch.manual_seed(42)
>>> B, N_max, top_k = 32, 8192, 1024
>>> logits = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
>>> seq_lens_t = torch.randint(top_k + 1, N_max, (B,), dtype=torch.int32, device="cuda")
>>> indices_t, _ = flashinfer.top_k_varlen(logits, seq_lens_t, top_k, backend="radix_cutlass")
>>> # step t+1：追加一个 token，seq_lens 增加 1，把 indices_t 作为 pre_idx 传给 GVR
>>> logits_t1 = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
>>> seq_lens_t1 = seq_lens_t + 1
>>> indices_t1, _ = flashinfer.top_k_varlen(logits_t1, seq_lens_t1, top_k, pre_idx=indices_t)
```

**另见**：`flashinfer.top_k`（一般用途 radix/clusters top-K，长度一致）

---

### `can_implement_filtered_topk`

```python
can_implement_filtered_topk() -> bool
```

**功能**：检查 GPU 是否有足够共享内存支持 FilteredTopK 算法。FilteredTopK 需要 128KB 动态共享内存；本函数
检查当前 GPU 每 SM 的最大共享内存是否足够。

**参数**

- 无。

**返回**

- `bool`：若 GPU 支持 FilteredTopK 返回 `True`，否则返回 `False`。

## 相关链接 / 参考

- 上游文档：`docs/api/topk.rst`
- 相关采样 API：`flashinfer.sampling.top_k_sampling_from_probs`、
  `flashinfer.sampling.top_k_top_p_sampling_from_probs`、`flashinfer.sampling.top_k_renorm_probs`、
  `flashinfer.sampling.top_k_mask_logits`
- varlen 资源释放：`flashinfer.topk_varlen.release_gvr2_resources`

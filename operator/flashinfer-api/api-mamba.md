# Mamba / Mamba-2 状态空间模型 kernel（`flashinfer.mamba`）

> 源码：`flashinfer/mamba/__init__.py` 及其子模块
> 上游文档：`docs/api/mamba.rst`
> 适用版本：FlashInfer 0.7.2

本模块封装 Mamba / Mamba-2 状态空间模型（SSM）在 SSM block 中使用的 selective scan 与状态更新
原语。包含 generation 阶段的 `selective_state_update`，带 MTP replay 的 `checkpointing_ssu`，经
Cake 后端的 `ssd_combined_fwd`（完整 SSD 前向），以及从 ReplaySSM ring buffer 物化 SSM 状态的
`replayssm_materialize`。

## 导入

```python
from flashinfer.mamba import (
    selective_state_update,
    checkpointing_ssu,
    ssd_combined_fwd,
    replayssm_materialize,
)
# 亦可通过包访问：
from flashinfer import mamba
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `selective_state_update` | 函数 | Mamba 层的 selective state update（generation 阶段）。 |
| `checkpointing_ssu` | 函数 | 带 MTP replay 的 checkpointing SSU，基于 matmul 的并行 token 处理。 |
| `ssd_combined_fwd` | 函数 | 运行 source-built Cake SSDCombined 后端（完整 SSD 前向）。 |
| `replayssm_materialize` | 函数 | 从较旧状态与 ReplaySSM ring buffer 物化某一 token 处的 SSM 状态。 |

## API 详情

### `selective_state_update`

```python
selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor
```

**功能**：Mamba 层的 selective state update 操作（generation 阶段）。支持单 token、多 token、变长
（varlen）多 token 以及投机解码（MTP）等模式，支持状态缓存索引与量化状态。

**参数**

- `state`（`torch.Tensor`）：状态张量，形状 `(state_cache_size, dim, dstate)` 或
  `(state_cache_size, nheads, dim, dstate)`。
- `x`（`torch.Tensor`）：输入张量，形状：单 token 为 `(batch, dim)` 或 `(batch, nheads, dim)`；
  多 token 为 `(batch, T, nheads, dim)`；varlen 多 token（带 `cu_seqlens`）为
  `(total_tokens, nheads, dim)`。
- `dt`（`torch.Tensor`）：delta time 张量，布局与 `x` 相同。
- `A`（`torch.Tensor`）：A 矩阵，形状 `(dim, dstate)` 或 `(nheads, dim, dstate)`。
- `B`（`torch.Tensor`）：B 矩阵，形状：单 token 为 `(batch, dstate)` 或 `(batch, ngroups, dstate)`；
  多 token 为 `(batch, T, ngroups, dstate)`；varlen 多 token 为 `(total_tokens, ngroups, dstate)`。
- `C`（`torch.Tensor`）：C 矩阵，布局同 `B`。
- `D`（`torch.Tensor`）：D 向量，形状 `(dim,)` 或 `(nheads, dim)`。
- `z`（`torch.Tensor`，可选）：可选 z 张量，布局同 `x`。
- `dt_bias`（`torch.Tensor`，可选）：可选 dt bias，形状 `(dim,)` 或 `(nheads, dim)`。
- `dt_softplus`（`bool`，默认 `False`）：是否对 dt 应用 softplus。
- `state_batch_indices`（`torch.Tensor`，可选）：状态缓存读取的 batch 索引，形状 `(batch,)` 或
  `(N, max_seqlen)`。对带 `num_accepted_tokens` 的投机解码必须为 2D。
- `pad_slot_id`（`int`，默认 `-1`）：`state_batch_indices` 中 padded 条目的哨兵值。
- `state_scale`（`torch.Tensor`，可选）：可选 float32 scale 张量，形状 `(state_cache_size, nheads,
  dim)`，用于 int16 状态的分块量化。它列在自定义 op 的 `mutates_args` 中：当 `state` 量化（int16）
  时，kernel 把新的 per-block scale 原地写入此处——调用方在回读时依据该张量对 `state` 反量化
  （与下文 `intermediate_state_scales` 契约一致）。
- `out`（`torch.Tensor`，可选）：可选输出张量（形状同 `x`）。
- `disable_state_update`（`bool`，默认 `False`）：为 `True` 时跳过更新 state 张量（用于投机解码
  验证）。
- `intermediate_states_buffer`（`torch.Tensor`，可选）：可选缓冲，用于投机解码期间缓存中间状态，
  形状 `(batch, cache_steps, nheads, dim, dstate)`。同样列在 `mutates_args` 中——kernel 原地把中间
  状态写入其中。与 `dst_state_batch_indices` 互斥。
- `intermediate_state_indices`（`torch.Tensor`，可选）：可选索引，把 batch 元素映射到中间状态缓冲
  位置，形状 `(batch,)`。
- `intermediate_state_scales`（`torch.Tensor`，可选）：可选 per-block float32 scale 张量，与
  `intermediate_states_buffer` 匹配。当与 int16 `intermediate_states_buffer` 一起提供时，kernel 把
  计算出的 scale 写入该张量（它列在自定义 op 的 `mutates_args` 中），调用方负责在回读时用这些 scale
  对中间状态*反量化*。其布局镜像 `state_scale`，但针对投机解码的中间缓冲。
- `rand_seed`（`torch.Tensor`，可选）：可选单元素 int64 CUDA 张量，用作 stochastic rounding 种子。
  提供时 `philox_rounds` 须 `> 0`，且不得与 `state_scale` 同时提供。
- `philox_rounds`（`int`，默认 `10`）：stochastic rounding 的 Philox-4x32 PRNG 轮数。`rand_seed` 为
  `None` 时强制为 `0`（禁用随机舍入）。
- `cache_steps`（`int`，默认 `0`）：投机解码需缓存的步数/token 数。varlen 模式（提供 `cu_seqlens`）
  下它指定 `max_seqlen`。
- `algorithm`（`str`，默认 `"auto"`）：算法选择：`"auto"`、`"simple"`、`"vertical"`、`"horizontal"`。
  另接受 `"async_horizontal"`，已向后兼容地并入 `"simple"`。未知值抛 `ValueError`。
- `dst_state_batch_indices`（`torch.Tensor`，可选）：状态缓存写入目的地索引，形状 `(batch,)` 或
  `(N, max_seqlen)`。提供时 state 从 `state_batch_indices` 读取、写到 `dst_state_batch_indices`
  （支持读写不同状态 slot）。与 `intermediate_states_buffer` 互斥。
- `cu_seqlens`（`torch.Tensor`，可选）：累积序列长度，形状 `(N + 1,)`，整数 dtype（JIT 对实际
  dtype 特化，int32 或 int64 均可；省略时默认 int32）。提供时输入为 packed-token varlen 格式：
  `x` / `dt` 为 3D `(total_tokens, nheads, dim)`，`B` / `C` 为 3D `(total_tokens, ngroups, dstate)`。
- `num_accepted_tokens`（`torch.Tensor`，可选）：每序列已接受 token 数，形状 `(N,)`。决定每个序列
  读取哪个状态作为初始状态。
- `backend`（`str`，默认 `"auto"`）：后端选择：`"auto"`、`"flashinfer"` 或 `"cake"`。`"auto"` 与
  `"flashinfer"` 都使用 FlashInfer。Cake 是 SM100/SM103 上的 opt-in source-built 后端；
  `backend="cake"` 在 Cake promoted rows 之外回退到 FlashInfer。未知值抛 `ValueError`。

**返回**

- `torch.Tensor`：输出张量，形状同 `x`。

**注意事项**

- `rand_seed` 必须是 CUDA int64 单元素张量；违反时抛 `TypeError` / `ValueError`。
- `intermediate_states_buffer` 与 `dst_state_batch_indices` 互斥。
- 标量维度的输入（`state` 3D、`A`/`D` 2D/1D、`dt_bias` 1D）会被自动 unsqueeze 到期望的批次布局。

**示例**

```python
import torch
import flashinfer

batch, nheads, dim, dstate = 8, 16, 64, 128
state = torch.zeros(batch, nheads, dim, dstate, device="cuda", dtype=torch.float32)
x = torch.randn(batch, nheads, dim, device="cuda", dtype=torch.float16)
dt = torch.rand(batch, nheads, dim, device="cuda", dtype=torch.float16)
A = torch.randn(nheads, dim, dstate, device="cuda", dtype=torch.float32)
B = torch.randn(batch, 1, dstate, device="cuda", dtype=torch.float16)
C = torch.randn(batch, 1, dstate, device="cuda", dtype=torch.float16)
D = torch.randn(nheads, dim, device="cuda", dtype=torch.float16)

out = flashinfer.mamba.selective_state_update(state, x, dt, A, B, C, D, dt_softplus=True)
```

---

### `checkpointing_ssu`

```python
checkpointing_ssu(
    state: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    d_split: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    enable_pdl: bool = False,
    cb_scaled: Optional[torch.Tensor] = None,
    cumAdt_vec: Optional[torch.Tensor] = None,
    cb_old: Optional[torch.Tensor] = None,
    precompute_heads_per_cta: int = 0,
    algorithm: str = "auto",
) -> torch.Tensor
```

**功能**：带 MTP replay 的 checkpointing SSU，基于 matmul 的并行 token 处理。把新 token 的更新写入
可量化状态，并维护用于 ReplaySSM 的 x / B / dt ring buffer。

**参数**

- `state`（`torch.Tensor`）：SSM 状态，形状 `(state_cache_size, nheads, dim, dstate)`，原地更新。
- `x_cache`（`torch.Tensor`）：缓存 x 的 ring，形状 `(state_cache_size, nheads, RING_BUFFER_LEN,
  dim)`。`RING_BUFFER_LEN` 隐含（= `size(2)`）；逻辑 replay 窗口为
  `max_window = RING_BUFFER_LEN - T`（flush 规则 `pnat + 2T > RING_BUFFER_LEN`）。
- `B_cache`（`torch.Tensor`）：缓存 B 的 ring，形状 `(state_cache_size, ngroups, RING_BUFFER_LEN,
  dstate)`。
- `dt_cache`（`torch.Tensor`）：缓存处理后 dt 的 ring，形状 `(state_cache_size, nheads,
  RING_BUFFER_LEN)`，f32。replay decay 由它重新计算（不缓存 `cumAdt`——前缀和对 ring 移位不
  不变）。
- `ring_start`（`torch.Tensor`）：每 slot 的 ring 头（最旧活跃行），形状 `(state_cache_size,)`，
  int32。簿记由**主机**负责：flush 时按 replayed 数量推进。
- `prev_num_accepted_tokens`（`torch.Tensor`）：需 replay 的旧 token 数，形状
  `(state_cache_size,)`，int32。
- `x`（`torch.Tensor`）：新 token 输入，形状 `(batch, T, nheads, dim)`。
- `dt`（`torch.Tensor`）：delta time，形状 `(batch, T, nheads, dim)`，带 tie_hdim（`stride[-1]=0`）。
  接受原生 dtype（如 bf16），内部转 f32。
- `A`（`torch.Tensor`）：decay 率，形状 `(nheads, dim, dstate)`，带 tie_hdim。
- `B`（`torch.Tensor`）：输入投影，形状 `(batch, T, ngroups, dstate)`。
- `C`（`torch.Tensor`）：输出投影，形状 `(batch, T, ngroups, dstate)`。
- `out`（`torch.Tensor`）：预分配输出，形状 `(batch, T, nheads, dim)`。
- `D`（`torch.Tensor`，可选）：skip connection，形状 `(nheads, dim)`。
- `z`（`torch.Tensor`，可选）：gate，形状 `(batch, T, nheads, dim)`。
- `dt_bias`（`torch.Tensor`，可选）：加到 dt 的 bias，形状 `(nheads, dim)`，带 tie_hdim。
- `dt_softplus`（`bool`，默认 `False`）：是否对 dt 应用 softplus。
- `state_batch_indices`（`torch.Tensor`，可选）：batch 索引到 cache slot 的映射，形状 `(batch,)`，
  int32 | int64。
- `pad_slot_id`（`int`，默认 `-1`）：padded 条目的哨兵值。
- `state_scale`（`torch.Tensor`，可选）：量化状态的 block-scale 解码因子，形状 `(state_cache_size,
  nheads, dim)`，f32。量化（int8/fp8_e4m3fn）状态必需，且此时 `d_split` 必须为 1 或 `None`；非
  量化状态必须为 `None`。
- `rand_seed`（`torch.Tensor`，可选）：单元素 int64 CUDA 张量，用于 stochastic rounding 种子。
- `philox_rounds`（`int`，默认 `10`）：stochastic rounding 的 Philox PRNG 轮数。
- `d_split`（`Optional[int]`，默认 `None`）：per-head DIM 拆分因子。**仅用于 benchmarking**，不要
  使用，会导致变慢。
- `cu_seqlens`（`torch.Tensor`，可选）：packed 变长输入的累积序列边界，一维 int32 CUDA 张量，形状
  `(batch + 1,)`。须从 0 开始、单调非递减、以 `total_tokens` 结束。提供时 `x` 形状须为
  `(1, total_tokens, nheads, dim)`，且 `max_seqlen` 必需。
- `max_seqlen`（`Optional[int]`，默认 `None`）：每个 packed 序列长度的上界。变长模式下这是 JIT
  特化的 predicted-token 计数，用于从 ring buffer 大小推导逻辑 replay 窗口。未提供 `cu_seqlens`
  时必须为 `None`。
- `enable_pdl`（`bool`，默认 `False`）：为 `True` 时用
  `cudaLaunchAttributeProgrammaticStreamSerialization` 启动 kernel，启用 kernel 内
  `griddepcontrol.{wait,launch_dependents}` PTX 以上游（如 conv1d）为门、并向下游 kernel 发信号。
  调用方须保证上下游 kernel 也 PDL-paired，否则 wait/signal 无效果。
- `cb_scaled`（`torch.Tensor`，可选）：预分配、dtype 同 `x` 的 scratch，用于预计算新 token CB 矩阵，
  fragment-native 布局 `(batch, nheads, WARP_SIZE, MMA_FRAG_SIZE)`——每个 `(batch, head)` 的 CB 是
  一个 m16n8k16 MMA A-fragment，存为 `[warp lane, register]`。提供它（连同 `cumAdt_vec` /
  `cb_old`）使 **two-kernel**（precompute + main）路径可用——`algorithm` 决定是否运行；四个都为
  `None` 时总是运行 monolithic kernel。由调用方分配，故该路径 CUDA-graph-safe。
- `cumAdt_vec`（`torch.Tensor`，可选）：预分配 fp32 scratch，用于 per-head 原始 cumAdt 向量，形状
  `(batch, nheads, T_pad)`；main kernel 即时对其取指数得到 decay/β 因子。当且仅当提供 `cb_scaled`
  时提供。
- `cb_old`（`torch.Tensor`，可选）：预分配、dtype 同 `x` 的 scratch，用于预计算旧 token CB 矩阵，
  fragment-native 布局 `(batch, nheads, WARP_SIZE, K_old // 2)`，其中 `K_old =
  next_multiple_of_8(max_window)`——no-write（replay）路径消费的 m16n8k{K_old} MMA A-fragment，存为
  `[warp lane, register]`。当且仅当提供 `cb_scaled` 时提供。
- `precompute_heads_per_cta`（`int`，默认 `0`）：two-kernel PRECOMPUTE head-tiling：每个 precompute
  CTA 的 head 数。`0` 用 launcher 的 co-residency 启发式；`>0` 覆盖它且必须位于
  `HEADS_PER_GROUP >> k` 的对半链上。调优旋钮——仅 two-kernel 路径。
- `algorithm`（`str`，默认 `"auto"`）：kernel 选择：`"auto"`、`"monolith"` 或 `"two-kernel"`。有
  scratch trio 且无显式调优旋钮时，`"auto"` 使用 FlashInfer 缓存的 autotuner tactic（可能
  monolithic 或 two-kernel）。在 `autotune(True)` 上下文中它会 profile monolithic `d_split={1,2}` 与
  每种受支持的 two-kernel 组合。量化 int8/fp8 状态只 profile 其受支持的 monolithic `d_split=1`。
  无缓存 tactic 时保留 production 回退：当 `batch * nheads >= sm_count` 时用 split，否则用
  monolithic。`"two-kernel"` 强制 split（需 scratch trio），`"monolith"` 强制 monolithic（忽略
  scratch）。

**返回**

- `torch.Tensor`：输出张量，形状 `(batch, T, nheads, dim)`。

**注意事项**

- 量化状态与 `state_scale` 必须成对：int8 / fp8_e4m3fn 需要形状 `(cache, nheads, dim)` 的 float32
  `state_scale`；非量化 dtype 必须不传该张量。
- 8-bit 状态（int8/fp8）要求 `d_split == 1`，因为 per-warp M-shard 布局需要
  `D_PER_CTA / 4 >= 16`。
- source state 与 replay ring 在 ReplaySSM 物化路径中是只读的；本函数维护 ring 写入。

**示例**

```python
import torch
import flashinfer

# 详见 tests / benchmarks；此处给出形状示意
state_cache_size, nheads, dim, dstate = 256, 16, 64, 128
batch, T = 4, 4
RING = 16
state = torch.zeros(state_cache_size, nheads, dim, dstate, device="cuda", dtype=torch.float32)
x_cache = torch.zeros(state_cache_size, nheads, RING, dim, device="cuda", dtype=torch.float16)
B_cache = torch.zeros(state_cache_size, 1, RING, dstate, device="cuda", dtype=torch.float16)
dt_cache = torch.zeros(state_cache_size, nheads, RING, device="cuda", dtype=torch.float32)
ring_start = torch.zeros(state_cache_size, device="cuda", dtype=torch.int32)
pnat = torch.zeros(state_cache_size, device="cuda", dtype=torch.int32)
x = torch.randn(batch, T, nheads, dim, device="cuda", dtype=torch.float16)
dt = torch.rand(batch, T, nheads, dim, device="cuda", dtype=torch.float16)
A = torch.randn(nheads, dim, dstate, device="cuda", dtype=torch.float32)
B = torch.randn(batch, T, 1, dstate, device="cuda", dtype=torch.float16)
C = torch.randn(batch, T, 1, dstate, device="cuda", dtype=torch.float16)
out = torch.empty(batch, T, nheads, dim, device="cuda", dtype=torch.float16)

out = flashinfer.mamba.checkpointing_ssu(
    state, x_cache, B_cache, dt_cache, ring_start, pnat,
    x, dt, A, B, C, out,
)
```

---

### `ssd_combined_fwd`

```python
ssd_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    seq_chunk_cumsum: Optional[torch.Tensor] = None,
    update_seq_chunk_cumsum: bool = False,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
    num_seqs: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：运行 source-built Cake `SSDCombined` 后端的完整 SSD 前向。支持 batched 与
packed-varlen 两种模式，可选 checkpoint 捕获。

**参数**

- `x`（`torch.Tensor`）：BF16 输入，形状 `[batch, seqlen, nheads, headdim]`。接受任意正 `seqlen`
  （尾部不完整的 128-token chunk 在 kernel 内处理；短于一个 chunk 的调用直接绑定调用方张量，TMA
  对超过 `seqlen` 的行零填充并裁剪 store）。
- `dt`（`torch.Tensor`）：每 token 步长，形状 `[batch, seqlen, nheads]`。
- `A`（`torch.Tensor`）：float32 状态转移系数，形状 `[nheads]`。
- `B`（`torch.Tensor`）：BF16 输入投影，形状 `[batch, seqlen, ngroups, dstate]`。
- `C`（`torch.Tensor`）：BF16 输出投影，形状同 `B`。
- `D`（`torch.Tensor`，可选）：BF16 skip 系数，形状 `[nheads]` 或 `[nheads, headdim]`。
- `z`（`torch.Tensor`，可选）：BF16 gating 张量，形状同 `x`。
- `dt_bias`（`torch.Tensor`，可选）：步长偏置，形状 `[nheads]`。
- `dt_softplus`（`bool`，默认 `False`）：是否对 `dt + dt_bias` 应用 softplus。
- `dt_limit`（`Tuple[float, float]`，默认 `(0.0, inf)`）：处理后步长的闭区间上/下界。
- `initial_states`（`torch.Tensor`，可选）：BF16 / FP16 / FP32 初始状态，packed-varlen 模式下形状
  `[num_seqs, nheads, headdim, dstate]`，batched 模式下形状 `[batch, nheads, headdim, dstate]`，其
  dtype 决定 state dtype。packed-varlen 模式下省略时 scan 从零状态开始，序列数来自
  `seq_chunk_cumsum` 或 `num_seqs`。
- `seq_idx`（`torch.Tensor`，可选）：int32 或 int64 packed-sequence ID，形状 `[batch, seqlen]`。
- `chunk_indices`（`torch.Tensor`，可选）：每个逻辑 packed-varlen 段的 int32 物理 chunk 索引。
- `chunk_offsets`（`torch.Tensor`，可选）：每个逻辑 packed-varlen 段的 int32 in-chunk 起始 offset。
- `seq_chunk_cumsum`（`torch.Tensor`，可选）：调用方拥有的 int32 张量，形状 `[num_seqs + 1]`。除非
  `update_seq_chunk_cumsum` 为真，否则视为预计算；省略时 runner 计算内部缓冲。
- `update_seq_chunk_cumsum`（`bool`，默认 `False`）：是否把 `seq_chunk_cumsum` 重算到提供的张量中。
- `checkpoint_token_indices`（`torch.Tensor`，可选）：连续 int32 向量，每序列一个 exclusive
  checkpoint token 边界。batched 边界是序列相对，packed-varlen 边界是绝对；负值禁用该序列捕获。
- `checkpoint_state_slots`（`torch.Tensor`，可选）：连续 int32 向量，把每序列映射到
  `checkpoint_states` 的一行。负值禁用捕获。
- `checkpoint_states`（`torch.Tensor`，可选）：调用方拥有的连续 checkpoint 输出，形状
  `[num_checkpoints, nheads, headdim, dstate]`，dtype 为选定 state dtype。三个 checkpoint 参数必须
  一起提供。
- `out`（`torch.Tensor`，可选）：调用方拥有的连续 token-major 输出存储，形状
  `[batch, seqlen, nheads, headdim]`，由 kernel 直接写入。省略时新分配。
- `return_final_states`（`bool`，默认 `True`）：是否为每个 batch 元素或 packed 序列返回最终状态。
- `num_seqs`（`int`，可选）：varlen 模式下无 `initial_states` 且无 `seq_chunk_cumsum` 时的
  packed 序列数。
- `cu_seqlens`（`torch.Tensor`，可选）：设备上的 int32 累积 token offset `[num_seqs + 1]`，即 kernel
  自行推导段元数据的 packed-varlen 形式（不需要 `chunk_indices` / `chunk_offsets`；`seq_idx` 可选）。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor]]`：token-major 输出 `out`，形状
  `[batch, seqlen, nheads, headdim]`；以及 `return_final_states` 为真时形状
  `[num_seqs, nheads, headdim, dstate]` 的最终状态，否则为 `None`。

**注意事项**

- 本函数经 Cake 后端执行（`chunk_size=128`，`backend="cake"`）；无单独 backend 参数。
- state dtype 由 `initial_states` 决定，其次 `checkpoint_states`，否则默认 bfloat16。

**示例**

```python
import torch
import flashinfer

batch, seqlen, nheads, headdim, dstate = 2, 1024, 16, 64, 128
x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
dt = torch.rand(batch, seqlen, nheads, device="cuda", dtype=torch.bfloat16)
A = torch.randn(nheads, device="cuda", dtype=torch.float32)
B = torch.randn(batch, seqlen, 1, dstate, device="cuda", dtype=torch.bfloat16)
C = torch.randn(batch, seqlen, 1, dstate, device="cuda", dtype=torch.bfloat16)

out, final_states = flashinfer.mamba.ssd_combined_fwd(x, dt, A, B, C)
```

---

### `replayssm_materialize`

```python
replayssm_materialize(
    state_ptrs: torch.Tensor,
    state_slot_strides: torch.Tensor,
    x_cache_ptrs: torch.Tensor,
    x_cache_slot_strides: torch.Tensor,
    B_cache_ptrs: torch.Tensor,
    B_cache_slot_strides: torch.Tensor,
    dt_cache_ptrs: torch.Tensor,
    dt_cache_slot_strides: torch.Tensor,
    A_ptrs: torch.Tensor,
    state_scale_ptrs: torch.Tensor,
    state_scale_slot_strides: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: torch.Tensor,
    *,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    dim: int,
    dstate: int,
    num_heads: int,
    heads_per_group: int,
    max_window: int,
    ring_buffer_len: int,
    pad_slot_id: int = -1,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 0,
    dependency_inputs: Optional[list[torch.Tensor]] = None,
    dependency_outputs: Optional[list[torch.Tensor]] = None,
) -> None
```

**功能**：从较旧状态与 ReplaySSM ring buffer 物化某一选定 token 处的 SSM 状态。一个 source 状态
加上其 replay ring 表示从该 source 状态到由 accepted-token 前缀推进的状态之间的每个状态（见
`checkpointing_ssu`）。本操作从 `ring_start` 开始，对该 source 状态应用该 ring 的调用方选定前缀，
并把结果状态写到单独的 destination slot。它不消费也不以其它方式修改 replay-ring 数据。

**参数**

- `state_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表，形状 `(L,)`。每项是该层状态存储的基址。
- `state_slot_strides`（`torch.Tensor`）：一维 CUDA int64 表 `(L,)`，每层的状态存储外层 slot
  stride（以元素计）。
- `x_cache_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表 `(L,)`，每层的 per-head replay `x` 缓存。
- `x_cache_slot_strides`（`torch.Tensor`）：一维 CUDA int64 表 `(L,)`，`x` 缓存外层 slot stride。
- `B_cache_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表 `(L,)`，每层的 per-group replay `B` 缓存。
- `B_cache_slot_strides`（`torch.Tensor`）：一维 CUDA int64 表 `(L,)`，`B` 缓存外层 slot stride。
- `dt_cache_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表 `(L,)`，每层的 replay `dt` 缓存。
- `dt_cache_slot_strides`（`torch.Tensor`）：一维 CUDA int64 表 `(L,)`，`dt` 缓存外层 slot stride。
- `A_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表 `(L,)`，每层的 `A` 向量。
- `state_scale_ptrs`（`torch.Tensor`）：一维 CUDA int64 指针表 `(L,)`，量化状态 scale。非量化状态
  用零项。
- `state_scale_slot_strides`（`torch.Tensor`）：一维 CUDA int64 表 `(L,)`，scale 外层 slot stride。
- `src_slots, dst_slots`（`torch.Tensor`）：CUDA int32 张量 `(L, B)`，为每层与每个物理 batch 请求
  选择源与目的 slot。当任一 slot 等于 `pad_slot_id` 时跳过该请求/层。
- `ring_start, replay_prefix_len`（`torch.Tensor`）：CUDA int32 张量 `(B,)`。`replay_prefix_len[b]`
  是从 `ring_start[b]` 开始、写到 `dst_slots[:, b]` 之前要应用到 `src_slots[:, b]` 的 ring 条目数。
  零表示精确的状态/scale 拷贝。对共享的 source-state/ring 快照，它不得超过 checkpointing SSU 该
  请求的 `prev_num_accepted_tokens`。对 `active_request_indices` 选中的每个请求，`ring_start` 须在
  `[0, ring_buffer_len)`，`replay_prefix_len` 须在 `[0, max_window]`。本操作不接收也不校验
  checkpointing tracker。
- `active_request_indices`（`torch.Tensor`）：CUDA int32 张量 `(B,)`。其前缀包含选中的物理 batch
  索引，各恰好一次、顺序任意；其余项为 `-1`。kernel 在遇到首个 `-1` 时停止。每个选中索引须在
  `[0, B)`，且引用满足上述 `ring_start` 与 `replay_prefix_len` 要求的元数据。这些元数据张量的内容
  驻留 GPU 且不被检查；违反其值要求属于非法输入、行为未定义。
- `state_dtype`（`torch.dtype`）：JIT 状态存储 dtype。一字节状态要求 `dim=64` 且 `dstate=128`。
- `input_dtype`（`torch.dtype`）：`x` 与 `B` 缓存条目的 JIT dtype。必须为 `torch.bfloat16`，匹配
  checkpointing SSU 的 BF16 replay MMA 操作数。指针表无法校验被指向张量的 dtype，故调用方须确保
  两个缓存族都用 BF16 存储。
- `matrixA_dtype`（`torch.dtype`）：per-head `A` 值的 JIT dtype。
- `dim`（`int`）：JIT 状态 head 维度。
- `dstate`（`int`）：JIT SSM 状态维度。
- `num_heads`（`int`）：每层的运行 head 数。
- `heads_per_group`（`int`）：共享每个 `B` 缓存组的 JIT head 数，必须整除 `num_heads`。
- `max_window`（`int`）：JIT 最大 replay 前缀长度（ring 条目数），在 `[1, 16]`。
- `ring_buffer_len`（`int`）：每个 replay ring 的运行行数。
- `pad_slot_id`（`int`，默认 `-1`）：要跳过的 slot 哨兵，匹配 `checkpointing_ssu`。默认为 `-1`；
  除非显式传入，零是合法 slot。
- `rand_seed`（`torch.Tensor`，可选）：`philox_rounds > 0` 时必需的单元素 CUDA int64 种子。
- `philox_rounds`（`int`，默认 `0`）：Philox stochastic rounding 轮数。零禁用随机舍入。
- `dependency_inputs`（`list[torch.Tensor]`，可选）：通过指针表间接读取的后备 CUDA 张量：通常是
  由 `src_slots` 经 `state_ptrs` 与 `state_scale_ptrs` 选中的 source 状态与 scale 存储，以及
  `x_cache_ptrs`、`B_cache_ptrs`、`dt_cache_ptrs`、`A_ptrs` 代表的存储。这些张量是 `torch.compile`
  的依赖锚点，不传给 JIT kernel。当某状态或 scale 后备张量同时是输出锚点时，不要在此重复。
- `dependency_outputs`（`list[torch.Tensor]`，可选）：通过 `dst_slots` 经 `state_ptrs`（量化状态还有
  `state_scale_ptrs`）写入目的 slot 的后备 CUDA 张量。这些张量是 `torch.compile` 的可变依赖锚点，
  不传给 JIT kernel。使用 `torch.compile` 的调用方必须包含其内容需要 PyTorch 围绕本次调用排序的每个
  后备分配。

**返回**

- `None`。

**注意事项**

- source state 与 replay ring 只读。调用方拥有的 slot 表与元数据选择要应用的 token；本操作没有
  acceptance、alignment 或 cursor-update 语义。
- `input_dtype` 必须为 `torch.bfloat16`，否则抛 `ValueError`。
- 所有指针/stride 表必须是一维 CUDA int64 张量；`src_slots` / `dst_slots` 必须为 CUDA int32；
  `ring_start` / `replay_prefix_len` / `active_request_indices` 必须为 CUDA int32。
- 8-bit 状态物化当前要求 `dim=64` 且 `dstate=128`。

**示例**

若 `src_slots` 选中的是 token 123 之后的状态，且其后已有 7 个 token 被接受到 ring 中，则一个每 128
token 物化状态的 prefix cache 传 `replay_prefix_len=5` 即可写出 token 128 之后的状态。

```python
import torch
import flashinfer

# 指针/stride 表均为 1D CUDA int64；slots 与元数据为 CUDA int32
flashinfer.mamba.replayssm_materialize(
    state_ptrs, state_slot_strides,
    x_cache_ptrs, x_cache_slot_strides,
    B_cache_ptrs, B_cache_slot_strides,
    dt_cache_ptrs, dt_cache_slot_strides,
    A_ptrs, state_scale_ptrs, state_scale_slot_strides,
    src_slots, dst_slots, ring_start, replay_prefix_len, active_request_indices,
    state_dtype=torch.float32, input_dtype=torch.bfloat16, matrixA_dtype=torch.float32,
    dim=64, dstate=128, num_heads=16, heads_per_group=1,
    max_window=16, ring_buffer_len=16,
)
```

## 相关链接 / 参考

- `docs/api/gdn_decode.rst`：GDN 的 ReplaySSM verify/commit（`gated_delta_rule_mtp` /
  `gated_delta_rule_replayssm_commit`），与本模块的 ReplaySSM 物化互补。

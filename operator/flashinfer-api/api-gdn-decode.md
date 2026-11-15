# Gated Delta-Rule 解码（`flashinfer.gdn_decode`）

> 源码：`flashinfer/gdn_decode.py`
> 上游文档：`docs/api/gdn_decode.rst`
> 适用版本：FlashInfer 0.7.2

本模块提供 Mamba-2 / GDN 风格序列模型在**解码阶段**使用的 Gated Delta-Rule（门控 delta 规则）
kernel。这些函数消费预先构建好的 KV / 状态缓存，对当前解码步执行一次循环门控 delta-rule 更新。
除普通单 token 解码外，还包含多 token 处理（MTP，多用于投机解码验证）与 ReplaySSM 投机验证
及其提交（commit）接口。主要要求 SM90+（Hopper / Blackwell），其中 ReplaySSM 相关路径要求
SM100 / SM103。

## 导入

```python
from flashinfer.gdn_decode import (
    gated_delta_rule_decode,
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_mtp,
    gated_delta_rule_replayssm_commit,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `gated_delta_rule_decode` | 函数 | K-major（`[B,HV,K,V]`）状态布局的单 token 解码，无需转置。 |
| `gated_delta_rule_decode_pretranspose` | 函数 | V-major / K-last（`[B,HV,V,K]`）状态布局的单 token 及多 token 解码，支持状态池与索引。 |
| `gated_delta_rule_mtp` | 函数 | 多 token 处理（`T>1`）解码，用于投机解码验证；支持中间状态缓存与 ReplaySSM 原始窗口缓存。 |
| `gated_delta_rule_replayssm_commit` | 函数 | 将已接受的 ReplaySSM 前缀一次性重放到所有层的 checkpoint 中。 |

## API 详情

### `gated_delta_rule_decode`

```python
gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    backend: Literal["auto", "flashinfer", "cake_gdn"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：实现门控 delta-rule 线性注意力的解码阶段，每次处理一个 token 并更新循环状态。本变体
使用 **K-major** 状态布局 `[B, HV, K, V]`，无需转置。

**参数**

- `q`（`torch.Tensor`）：当前 query，形状 `[B, 1, H, K]`，dtype 为 float16 / bfloat16。
- `k`（`torch.Tensor`）：当前 key，形状 `[B, 1, H, K]`，dtype 为 float16 / bfloat16。
- `v`（`torch.Tensor`）：当前 value，形状 `[B, 1, HV, V]`，dtype 为 float16 / bfloat16。
- `state`（`torch.Tensor`）：当前状态，形状 `[B, HV, K, V]`（k-major），必须为 float32，**原地更新**。
- `A_log`（`torch.Tensor`）：对数衰减参数，形状 `[HV]`，必须为 float32。
- `a`（`torch.Tensor`）：输入相关衰减，形状 `[B, 1, HV]`，dtype 为 float16 / bfloat16。
- `dt_bias`（`torch.Tensor`）：衰减偏置，形状 `[HV]`，dtype 为 bfloat16 或 float32。
- `b`（`torch.Tensor`）：更新门（beta）输入，形状 `[B, 1, HV]`，dtype 为 float16 / bfloat16。
- `scale`（`float`，可选）：query 缩放因子。为 `None` 时默认为 `1 / sqrt(K)`。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `[B, 1, HV, V]`。为 `None` 时自动分配。
- `use_qk_l2norm`（`bool`，默认 `True`）：是否对 q 和 k 做 L2 归一化。
- `backend`（`{"auto", "flashinfer", "cake_gdn"}`，默认 `"auto"`）：`auto` 仅在命中精确的 frozen
  manifest 行时选择 GDN non-CP，否则使用现有 FlashInfer 实现；显式 `cake_gdn` 在契约不支持时
  fail-closed。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(output, state)`，其中 `output` 形状为 `[B, 1, HV, V]`，
  `state` 形状为 `[B, HV, K, V]`（原地更新）。

**注意事项**

- 要求 SM90（Hopper）架构。
- 状态原地更新。
- `K` 与 `V` 均须 `>= 128`。`V` 必须是 32（`TILE_V_NT`）的倍数：launcher 采用更严格的大 batch
  tile 约束以同时覆盖两条代码路径（尽管小 batch kernel 理论上可接受 `V % 16 == 0`）。

**示例**

```python
import torch
import flashinfer

B, H, K, V = 4, 8, 128, 128
q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
state = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
a = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)
dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)
b = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)

out, new_state = gated_delta_rule_decode(
    q, k, v, state, A_log, a, dt_bias, b
)
```

---

### `gated_delta_rule_decode_pretranspose`

```python
gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    backend: Literal["auto", "flashinfer", "cake_gdn"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：门控 delta-rule 解码 kernel（单 token 生成），采用 **V-major / K-last** 状态布局，即状态为
`[B, HV, V, K]`，无需在调用前转置。相比 `gated_delta_rule_decode`，本入口额外支持状态池 + 索引
（`initial_state` / `initial_state_indices`）、输出状态索引、中间状态缓存缓冲，以及 BF16 状态的
`K=V=128` 快速路径（`T=1` 或 MTP 形式）。

**参数**

- `q`（`torch.Tensor`）：当前 query，形状 `[B, 1, H, K]`，float16 / bfloat16。
- `k`（`torch.Tensor`）：当前 key，形状 `[B, 1, H, K]`，float16 / bfloat16。
- `v`（`torch.Tensor`）：当前 value，形状 `[B, 1, HV, V]`，float16 / bfloat16。
- `state`（`torch.Tensor`，可选）：当前状态，形状 `[B, HV, V, K]`（v-major / K-last）。
  float32 走 legacy kernel（仅 `T=1`）；bfloat16 走 BF16 状态后端（`K=V=128` 时 `T=1` 或 `T>1`
  的 MTP）。原地更新。使用 `initial_state` / `initial_state_indices` 时应传 `None`。
- `A_log`（`torch.Tensor`）：对数衰减参数，形状 `[HV]`，必须是 float32。
- `a`（`torch.Tensor`）：输入相关衰减，形状 `[B, 1, HV]`，float16 / bfloat16。
- `dt_bias`（`torch.Tensor`）：衰减偏置，形状 `[HV]`，bfloat16 或 float32。
- `b`（`torch.Tensor`）：更新门（beta）输入，形状 `[B, 1, HV]`，float16 / bfloat16。
- `scale`（`float`，可选）：query 缩放因子。为 `None` 时默认为 `1 / sqrt(K)`。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `[B, 1, HV, V]`，为 `None` 时自动分配。
- `use_qk_l2norm`（`bool`，默认 `True`）：是否对 q 和 k 做 L2 归一化。
- `initial_state`（`torch.Tensor`，可选）：状态池，形状 `[pool_size, HV, V, K]`（K-last / K 连续，
  与 per-batch `state` 相同布局）。提供后 kernel 直接按 `initial_state_indices` 从池中 gather，
  并原地写回，省去调用侧的 gather/scatter 开销。要求 bfloat16 状态且 `K=V=128`（bf16 快速路径）。
- `initial_state_indices`（`torch.Tensor`，可选）：每 batch 索引，形状 `[B]`（int32 或 int64），把
  每个 batch 项映射到 `initial_state` 中的 slot。提供 `initial_state` 时必须提供。
- `output_state_indices`（`torch.Tensor`，可选）：每 batch 索引，形状 `[B]`（int32 或 int64），指定
  每个 batch 项更新后状态的写入位置。要求同时提供 `initial_state`。为 `None` 时写回读取的同一
  slot（即 `initial_state_indices`）。
- `intermediate_states_buffer`（`torch.Tensor`，可选）：调用方拥有的 checkpoint 缓冲，形状
  `[B, >=T, HV, V, K]`。仅支持 `T > 1`；dtype 必须与 state dtype 一致。
- `disable_state_update`（`bool`，默认 `False`）：verify 调用时跳过最终状态写回。仅支持 `T > 1`。
- `backend`（`{"auto", "flashinfer", "cake_gdn"}`，默认 `"auto"`）：`auto` 仅在命中精确 frozen
  manifest 行时选择 GDN non-CP，否则使用现有 FlashInfer 实现；显式 `cake_gdn` 在契约不支持时
  fail-closed。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(output, state_or_initial_state)`，`output` 形状
  `[B, 1, HV, V]`，第二项为更新后的状态（原地修改）。

**注意事项**

- 要求 SM90+（Hopper、Blackwell 等）。
- 状态始终原地更新；池路径直接把更新写入 `initial_state` 内存（无需额外 scatter）。
- 状态布局为 v-major（K-last）：`[B, HV, V, K]`。当状态为 bfloat16 且 `K = V = 128` 时使用 BF16
  状态 kernel（`T=1` 或 `T>1` 的 MTP）；池 + 索引路径走 MTP kernel。
- 池 + 索引（`initial_state` / `initial_state_indices`）在 bf16 快速路径（`K=V=128`）与 float32
  legacy 路径（`T=1`）上都支持。两条路径都支持 `-1` padding 索引。
- **Padding / inactive 序列**：把应视为 padding 的 batch 项索引设为 `-1`。两个后端处理方式不同：
  - **bf16 快速路径**（bfloat16 状态，`K=V=128`）：`-1` 重定向到 `initial_state[0]`，该 slot 充当
    *牺牲性 null buffer*。kernel 读写 slot 0，该 batch 项的输出被计算但**未定义**（调用方不应
    使用）。调用方须为池额外分配一个前导 slot（`pool_size = num_real_slots + 1`），真实 slot
    保持在索引 `1..pool_size-1`。
  - **float32 legacy 路径**（`T=1`）：`-1` 项被完全跳过；状态池与输出都不被触碰，输出 slot 写为
    **零**。
- legacy 路径（float32 状态，`T=1`）：`K` 与 `V` 均须 `>= 128`，且 `V` 必须是 8（pretranspose 的
  tile 大小 `TILE_V`）的倍数。

**示例**

```python
import torch
import flashinfer

B, H, K, V = 4, 8, 128, 128
q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
state = torch.zeros(B, H, V, K, device="cuda", dtype=torch.float32)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
a = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)
dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)
b = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)

out, new_state = gated_delta_rule_decode_pretranspose(
    q, k, v, state, A_log, a, dt_bias, b
)

# 状态池 + 索引
pool = torch.zeros(B + 1, H, V, K, device="cuda", dtype=torch.bfloat16)
idx = torch.arange(1, B + 1, device="cuda", dtype=torch.int32)
out, pool = gated_delta_rule_decode_pretranspose(
    q, k, v, None, A_log, a, dt_bias, b,
    initial_state=pool, initial_state_indices=idx,
)
```

---

### `gated_delta_rule_mtp`

```python
gated_delta_rule_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
    cache_replayssm: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：门控 delta-rule MTP（Multiple Token Processing）kernel，每次调用处理多个 token
（`T > 1`），典型用于投机解码验证，并支持中间状态缓存以便回滚。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[B, T, H, K]`。
- `k`（`torch.Tensor`）：key，形状 `[B, T, H, K]`。
- `v`（`torch.Tensor`）：value，形状 `[B, T, HV, V]`。
- `initial_state`（`torch.Tensor`）：初始状态池，形状 `[pool_size, HV, V, K]`（K-last 布局），
  **必须为 float32**：本独立 MTP 入口不支持 BF16 快速路径；若需 BF16 `K=V=128` 状态池，请改调
  `gated_delta_rule_decode_pretranspose`（其在 `T > 1` 时分派到 BF16 MTP kernel）。连续时 kernel
  通过免拷贝的 4D→3D reshape 视图原地读写池；非连续池走原生 4D `use_pool_indexing=True` 路径，
  kernel 原地写 strided 池而不稠密化。
- `initial_state_indices`（`torch.Tensor`）：读索引，形状 `[B]`，把每个 batch 映射到
  `initial_state` 的 slot。负值视为 padding——kernel 跳过该 batch 的读与写回，输出 slot 保持
  调用方分配的值（`output` 为 `None` 时为零）。
- `A_log`（`torch.Tensor`）：对数衰减参数，形状 `[HV]`。
- `a`（`torch.Tensor`）：输入相关衰减，形状 `[B, T, HV]`。
- `dt_bias`（`torch.Tensor`）：衰减偏置，形状 `[HV]`。
- `b`（`torch.Tensor`）：更新门输入，形状 `[B, T, HV]`。
- `scale`（`float`，可选）：query 缩放因子。为 `None` 时使用 `1 / sqrt(K)`。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `[B, T, HV, V]`。
- `intermediate_states_buffer`（`torch.Tensor`，可选）：缓存中间状态的缓冲，形状
  `[B, T, HV, V, K]`（第一个维度按 batch 而非 pool slot 索引；缓冲至少要有 `B` 行且连续；提供时
  必须为 float32）。为 `None` 时不缓存中间状态。与 `ssm_state_indices` 互斥。
- `ssm_state_indices`（`torch.Tensor`，可选）：per-token 池 scatter 索引，形状 `[B, T]`，dtype
  `torch.int32`。提供后 kernel 把每个中间隐状态 `h_{t+1}` 直接写到
  `initial_state[ssm_state_indices[i, t]]`，而不是累积到稠密 `intermediate_states_buffer`。适用于
  每个 draft token 需要自己池 slot 的 FLA 风格投机解码流程。约束：`T >= 2`、
  `disable_state_update=False`、与 `intermediate_states_buffer` 互斥。默认 `None`。
- `disable_state_update`（`bool`，可选）：为 `True` 时不更新初始状态。目前默认 `True`；请显式传入
  该参数以消除 deprecation 警告——默认值将在 FlashInfer 0.8.0 变为 `False`。
  **已弃用**：隐式默认 `True` 已弃用，0.8.0 将改为 `False`。
- `use_qk_l2norm`（`bool`，默认 `True`）：是否对 q 和 k 做 L2 归一化。
- `output_state_indices`（`torch.Tensor`，可选）：写索引，形状 `[B]`（int32 或 int64），指定每个
  batch 更新后状态的目标池 slot。默认等于 `initial_state_indices`（读、写同一 slot）。负值跳过
  该 batch 的写回（读仍执行）。
- `cache_replayssm`（`bool`，默认 `False`）：仅 SM100/SM103、FP32 checkpoint、`K=V=128`、`T>=3`。
  只持久化 fold-every-commit 的 ReplaySSM 原始输入窗口，而非完整中间隐状态快照。要求 frozen-state
  verify 模式（`disable_state_update=True`）、无 `intermediate_states_buffer` 或
  `ssm_state_indices`、且四个 `replayssm_*` 缓冲齐全。活跃 `initial_state_indices` 必须唯一且在
  范围内。负索引保持输出与 cache 行不变。窗口按池 slot 索引，每次 verify 覆盖，并在下一次 verify
  前被 `gated_delta_rule_replayssm_commit` 消费一次。FP16 输入在缓存前转换为 BF16。所有张量基址
  须 16 字节对齐以支持向量化加载与 TMA。
  tcgen05 TF32 特化要求 `T=4..8`、`B=1..256`、连续 checkpoint、BF16 输出、int32 索引、Q/K 归一化
  以及偶数 `HV/H`。其它受支持的 SM100 情况使用 FP32 MTP 回退。TF32 改变归约精度；输出不必与回退
  逐位相等。
- `replayssm_rawv`（`torch.Tensor`，可选）：BF16 原始 V 窗口，形状 `[pool_size, HV, T, V]`。
- `replayssm_rawk`（`torch.Tensor`，可选）：BF16 归一化前 K 窗口，形状 `[pool_size, H, T, K]`。
- `replayssm_g`（`torch.Tensor`，可选）：FP32 log-decay 窗口，形状 `[pool_size, HV, T]`。
- `replayssm_beta`（`torch.Tensor`，可选）：FP32 sigmoid-beta 窗口，形状 `[pool_size, HV, T]`。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(output, initial_state)`，`output` 形状 `[B, T, HV, V]`，
  `initial_state` 为更新后的状态（`disable_state_update=True` 时不变）。

**注意事项**

- 标准 MTP 要求 SM90 或更新架构；ReplaySSM 要求 SM100/SM103。
- 支持 `T > 1`（多 token 处理）。
- 状态布局为 K-last：`[pool_size, HV, V, K]`。
- 针对投机解码验证场景做了优化。

**ReplaySSM 投机验证（SM100/SM103）**：使用
`gated_delta_rule_mtp(cache_replayssm=True, disable_state_update=True)` 并配合 FP32 128x128
checkpoint。对每一层，向全层 BF16 原始 K/V 窗口以及 FP32 log-decay / beta 窗口传入连续视图；
窗口与 checkpoint 使用相同的池 slot 索引。verify 在不改变 checkpoint、也不为每个 token 存储完整
状态的前提下产生输出。确知接受长度后，跨所有层调用一次 `gated_delta_rule_replayssm_commit`；它
只重放已接受前缀并原地更新每个活跃 checkpoint。零接受使 checkpoint 不变。下一次 verify 覆写原始
窗口；轮次之间不保留循环缓存或延迟状态更新。verify 与 commit 必须使用一致的 Q/K 归一化设置。

**示例（MTP 验证）**

```python
import torch
import flashinfer

B, T, H, K, V = 4, 4, 8, 128, 128
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
pool = torch.zeros(B + 1, H, V, K, device="cuda", dtype=torch.float32)
idx = torch.arange(1, B + 1, device="cuda", dtype=torch.int32)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
a = torch.zeros(B, T, H, device="cuda", dtype=torch.bfloat16)
dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)
b = torch.zeros(B, T, H, device="cuda", dtype=torch.bfloat16)
buf = torch.zeros(B, T, H, V, K, device="cuda", dtype=torch.float32)

out, pool = gated_delta_rule_mtp(
    q, k, v, pool, idx, A_log, a, dt_bias, b,
    intermediate_states_buffer=buf, disable_state_update=False,
)
```

---

### `gated_delta_rule_replayssm_commit`

```python
gated_delta_rule_replayssm_commit(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    track_state_indices: Optional[torch.Tensor] = None,
    track_steps: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    null_block_id: int = -1,
    backend: Literal["auto", "simt", "tcgen05"] = "auto",
) -> None
```

**功能**：将已接受的 ReplaySSM 前缀提交到所有层的 checkpoint 中。先对每一层调用
`gated_delta_rule_mtp(cache_replayssm=True)`。verify 使 checkpoint 保持冻结。确知接受情况后调用
本函数一次，然后在下一次 verify 时覆写窗口。缓存不会被移位或清空；未接受的 token 被丢弃。

**参数**

- `checkpoint_state`（`torch.Tensor`）：连续 FP32 `[layers, slots, HV, 128, 128]` 状态，K-last，
  原地更新。所有张量必须在同一 SM100/SM103 设备上。
- `rawv_cache`（`torch.Tensor`）：连续 BF16 `[layers, slots, HV, T, 128]` 原始 V 窗口。
- `rawk_cache`（`torch.Tensor`）：连续 BF16 `[layers, slots, H, T, 128]` 归一化前 K。
- `g_cache`（`torch.Tensor`）：连续 FP32 `[layers, slots, HV, T]` log-decay 窗口。
- `beta_cache`（`torch.Tensor`）：连续 FP32 `[layers, slots, HV, T]` sigmoid-beta 窗口。
- `state_indices`（`torch.Tensor`）：连续 int32 `[B]` 池 slot。活跃 slot 必须唯一且在范围内。
  小于等于 `null_block_id` 的 slot 被忽略。
- `accept_lens`（`torch.Tensor`）：连续 int32 `[B]` 已接受 token 数，范围 `[0, T]`。零为 no-op，
  包括可选的 track 写入。
- `track_state_indices`（`torch.Tensor`，可选）：连续 int32 `[B]` 额外 checkpoint 的目的地。活跃
  track slot 必须在范围内、彼此不同，且与所有活跃 `state_indices` slot 不相交。与 `track_steps`
  配对使用。
- `track_steps`（`torch.Tensor`，可选）：连续 int32 `[B]` 需保存的零基 token 索引。接受前缀之外
  的 step 使目的地不变。
- `use_qk_l2norm`（`bool`，默认 `True`）：以 epsilon `1e-6` 归一化缓存 K。必须与 verify 一致。
- `null_block_id`（`int`，默认 `-1`）：最大保留 / 无效 slot；须 `>= -1`。
- `backend`（`{"auto", "simt", "tcgen05"}`，默认 `"auto"`）：无 tracking 且归一化 K 时，`auto` 对
  `T=8` 选择 TF32 tcgen05，或在 `B>1` 且 `layers * B * HV >= 256` 时对 `T=6/7` 选择 tcgen05，否则
  使用 FP32 SIMT。较短接受前缀可能更适合显式 `simt`；`auto` 不会把接受长度读回主机。显式
  `tcgen05` 支持无 tracking 的归一化 `T=4..8`。使用 `simt` 保留 FP32 收缩。

**返回**

- `None`。

**注意事项**

- 索引值、接受长度与非别名性是调用方契约；不会读回主机。
- 所有张量基址须 16 字节对齐以支持向量化加载与 TMA。连续的 storage-offset 视图也须满足该对齐。
- CUDA Graph capture 前需先 warm up。

**示例**

```python
import torch
import flashinfer

# 假设已完成各层 gated_delta_rule_mtp(cache_replayssm=True) verify
gated_delta_rule_replayssm_commit(
    checkpoint_state, rawv_cache, rawk_cache, g_cache, beta_cache,
    state_indices, accept_lens,
)
```

## 相关链接 / 参考

- `docs/api/gdn_prefill.rst`：prefill 侧的 chunked GDN scan。
- `docs/api/gdn_fused_decode.rst`：把整条 per-layer 服务链融合为单步的 `gdn_fused_decode_step`。

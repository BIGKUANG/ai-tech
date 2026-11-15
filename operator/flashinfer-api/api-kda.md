# Kimi Delta Attention（KDA）

> 源码：`flashinfer/kda.py`、`flashinfer/kda_prefill.py`、`flashinfer/kda_decode.py`、`flashinfer/kda_kernels/`
> 上游文档：`docs/api/kda.rst`
> 适用版本：FlashInfer 0.7.2

Recurrent Kimi Delta Attention（KDA），是 GDN 的 per-K-dimension gating 变体。`flashinfer.recurrent_kda`
是每个阶段的规范入口：它分类每次调用并据此服务 prefill 或 decode。`flashinfer.kda_prefill` 与
`flashinfer.kda_decode` 两个模块持有它分派进入的阶段专属接口——它们记录在本页而非各自单独页面，
以便一次调用的整条路由集中呈现。KDA 覆盖 SM90+ 解码、SM100a（B200/GB200）/SM103a（B300/GB300）
的多种 prefill 后端、SM120a prefill，以及可选的 PTX / TIRx 后端。

## 导入

```python
from flashinfer import (
    recurrent_kda,                    # flashinfer.kda.recurrent_kda
    fused_kda_decode,                 # flashinfer.kda_decode.fused_kda_decode
    packed_kda_decode,                # flashinfer.kda_decode.packed_kda_decode
    packed_fused_kda_decode,          # flashinfer.kda_decode.packed_fused_kda_decode
    RecurrentKDAPrefillWorkspace,     # flashinfer.kda_prefill.RecurrentKDAPrefillWorkspace
    RecurrentKDAPrefillWrapper,       # flashinfer.kda.RecurrentKDAPrefillWrapper（实验性）
    prepare_tf32_kda_prefill,         # flashinfer.kda_prefill.prepare_tf32_kda_prefill
    prepare_bf16_kda_prefill,         # flashinfer.kda_prefill.prepare_bf16_kda_prefill
    KDAPrefillPlanCache,              # flashinfer.kda_prefill.KDAPrefillPlanCache
    kda_prefill_supports_fp32_checkpoints,
)
```

## API 列表

| 名称 | 类型 | 来源模块 | 说明 |
| --- | --- | --- | --- |
| `recurrent_kda` | 函数 | `flashinfer.kda` | 阶段中立的 KDA 门面，每个阶段的规范入口（decode / speculative decode / prefill）。 |
| `recurrent_kda` | 函数 | `flashinfer.kda_decode` | 解码专用门面，**自 0.8 起弃用**，现为共享解码 dispatcher 的 shim。 |
| `RecurrentKDAPrefillWorkspace` | 类 | `flashinfer.kda_prefill` | 调用方拥有的工作区，用于 recurrent-KDA CUDA graph capture。 |
| `RecurrentKDAPrefillWrapper` | 类 | `flashinfer.kda` | **实验性** plan-and-run 包装器，用于 packed recurrent-KDA prefill。 |
| `fused_kda_decode` | 函数 | `flashinfer.kda_decode` | 融合 width-4 causal 卷积、一次 KDA 更新与 gated RMSNorm 的 `T=1` 解码。 |
| `packed_fused_kda_decode` | 函数 | `flashinfer.kda_decode` | 把同一流水线扩展到 packed ragged 投机解码（`T>=1`），保留 per-token checkpoint。 |
| `packed_kda_decode` | 函数 | `flashinfer.kda_decode` | 面向 Kimi K3 的 serving-native packed `T=1,H=12,K=V=128` 解码适配器。 |
| `prepare_tf32_kda_prefill` | 函数 | `flashinfer.kda_prefill` | 在 SM100a/SM103a 上准备一次 TF32 KDA 推理调用，返回可 `launch()`/`close()` 的对象。 |
| `prepare_bf16_kda_prefill` | 函数 | `flashinfer.kda_prefill` | 同上，BF16 版本；支持可选 `plan_cache`。 |
| `KDAPrefillPlanCache` | 类 | `flashinfer.kda_prefill` | 以结构签名为键、有界的 prepared KDA launch LRU 缓存。 |
| `kda_prefill_supports_fp32_checkpoints` | 函数 | `flashinfer.kda_prefill` | 查询本 build 是否为给定设备导出 FP32 中间状态（checkpoint）。 |

## API 详情

### `recurrent_kda`（`flashinfer.kda`）

```python
recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    seq_order: Optional[torch.Tensor] = None,
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_state_indices: Optional[torch.Tensor] = None,
    *,
    disable_state_update: bool = False,
    correction_cache: Optional[torch.Tensor] = None,
    kg_cache: Optional[torch.Tensor] = None,
    backend: Literal[
        "auto", "cute-dsl", "cute-dsl-persistent", "tirx",
        "ptx", "cake", "small-bh", "cudnn",
    ] = "auto",
) -> tuple[torch.Tensor, Optional[torch.Tensor]] | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
```

**功能**：KDA 的公开 API 层，服务 CuTe DSL 实现（`flashinfer.kda_kernels.recurrent_kda`）。支持
单 token 解码、融合投机解码、GQA、可选 `cu_seqlens` packing，以及与后端实现相同的 gate 模式。在
SM120a 上，符合条件的普通多 token prefill 使用架构专属 CuTe DSL 后端。在 SM100a（B200/GB200）与
SM103a（B300/GB300）上，FlashKDA 兼容子集既可使用 frozen Cake schedule，也可使用 source-level
CuTe DSL BT=16 kernel；Cake 后端包含生成的 two-stage BT=16 prepare/chain portfolio。兼容的
equal-head D128 unbounded-softplus `T=1` 解码调用自动使用其 frozen Cake 特化。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[B, T, H, K]`，使用 `cu_seqlens` 时为
  `[1, total_tokens, H, K]`，必须 bfloat16。`T=1` 选择 decode；符合条件的 `T>1` 调用可选为 frozen
  prefill 后端。普通 prefill 接受 strided Q/K：dense-only provider 内部打包它们，cuDNN 保留受支持
  stride。
- `k`（`torch.Tensor`）：key，与 `q` 形状相同，必须 bfloat16。
- `v`（`torch.Tensor`）：value，形状 `[B, T, HV, V]`（packed 时为 `[1, total_tokens, HV, V]`），必须
  bfloat16。`HV != H` 时应用 GQA。
- `g`（`torch.Tensor`）：per-K-dimension gate，形状 `[B, T, HV, K]`（packed 时为
  `[1, total_tokens, HV, K]`），必须 bfloat16。预计算时位于 log 空间；`use_gate_in_kernel=True` 时
  为原始输入。
- `beta`（`torch.Tensor`）：delta-rule 学习率，形状 `[B, T, HV]`（packed 时为
  `[1, total_tokens, HV]`），必须 bfloat16。除非 `beta_is_logit=True`，否则为 pre-sigmoid。符合条件
  的 frozen prefill 接受非重叠 token-row-strided 存储（unit head stride），包括融合投影的视图。
- `A_log`（`torch.Tensor`，可选）：对数衰减参数，形状 `[H]`，必须 float32。`use_gate_in_kernel=True`
  时必需。
- `dt_bias`（`torch.Tensor`，可选）：per-head-K 衰减偏置，形状 `[H*K]` 或 `[H, K]`，必须 float32。
- `scale`（`float`，可选）：query 缩放因子。为 `None` 时默认为 `1 / sqrt(K)`。
- `initial_state`（`torch.Tensor`，可选）：初始状态，形状 `[N, HV, V, K]`。符合条件的 prefill 后端
  接受 bfloat16 或 float32。为 `None` 时零初始化，原地更新。无 `cu_seqlens` 的 batched spec decode
  且省略 `ssm_state_indices` 时，`N` 为 packed checkpoint-slot 数
  `B * (1 + num_spec_tokens)`。对带 `ssm_state_indices` 的 eligible frozen prefill，这是状态池
  `[N_pool, H, 128, 128]`，其 inner slot 连续，池 slot 之间允许 padding。
- `output_final_state`（`bool`，默认 `False`）：是否返回最终状态。
- `use_qk_l2norm_in_kernel`（`bool`，默认 `True`）：是否对 Q/K 做 L2 归一化。
- `use_gate_in_kernel`（`bool`，默认 `False`）：是否在 kernel 内由 `A_log` 与 `g` 计算 gate。
- `lower_bound`（`float`，可选）：若设置，使用
  `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` gate 公式；为 `None` 时使用
  `-exp(A_log) * softplus(g + dt_bias)`。给定的 bound 必须为负。
- `cu_seqlens`（`torch.Tensor`，可选）：连续 CUDA 累积序列长度，形状 `[N+1]`，int32 或 int64。
  frozen prefill 在 graph capture 之外把 int32 offset 转 int64；graph capture 要求调用方提供 int64
  offset。对 frozen prefill，值须从零开始、非递减、以总 token 数结束。Cake 可能读取这些值以准备并
  缓存主机调度元数据。
- `ssm_state_indices`（`torch.Tensor`，可选）：状态缓存索引。标准 decode 为 `[N]` int32，spec decode
  为 `[N, 1+S]` int32（须同时设置 `num_spec_tokens`）。eligible frozen packed prefill 接受连续
  CUDA int32 `[N_seq]` 索引，并直接更新选中的 `initial_state` 池 slot。
- `num_spec_tokens`（`int`，可选）：投机 token 数（S）。设置后在单次融合 kernel launch 中处理
  `1+S` 个 token，必须 `>= 1`。
- `num_accepted_tokens`（`torch.Tensor`，可选）：上一轮 spec decode 的每序列已接受 token 数，形状
  `[N]` int32。为 `None` 时初始状态从 `ssm_state_indices[n, 0]` 加载。大于 `1+S` 的值 clamp 到最终
  checkpoint slot。
- `output`（`torch.Tensor`，可选）：预分配输出，固定布局为 `[B, T, HV, V]`，使用 `cu_seqlens` 时为
  对应 packed/speculative 形状。为 `None` 时新分配。frozen prefill 要求存储与 Q、K、V、G、beta、
  `initial_state` 不相交。
- `initial_state_source`（`torch.Tensor`，可选）：可选只读 committed 状态池
  `[N0, HV, V, K]`。提供时 token 0 从该池而非 `initial_state` 加载。
- `initial_state_indices`（`torch.Tensor`，可选）：每序列 source slot，形状 `[N]` int32，与
  `initial_state_source` 一起必需。
- `beta_is_logit`（`bool`，默认 `False`）：为 `True` 时在 recurrent kernel 内对 `beta` 施加 sigmoid。
- `seq_order`（`torch.Tensor`，可选）：可选 packed-prefill 序列顺序，连续 CUDA int32 的 `[N]`
  置换。CuTe DSL packed engine 调用在省略时于设备上生成稳定的 longest-sequence-first 顺序；CuTe
  DSL decomp 保留原始顺序，除非 graph wrapper 请求设备生成的元数据。Cake 构造并缓存自己的 eager
  主机元数据。fixed-layout prefill 与 decode 调用必须保持为 `None`。
- `prefill_workspace`（`RecurrentKDAPrefillWorkspace`，可选）：调用方拥有、用于 SM100 家族与 SM120
  prefill 后端的工作区。eager 执行可选，CUDA graph capture 必需。capture 前用精确张量在 capture
  stream 上 eager warm。每次 captured `recurrent_kda` 调用使用一个工作区。
- `state_checkpoints`（`torch.Tensor`，可选）：prefill 的 checkpoint 输出或池 `[C, H, 128, 128]`。
  CuTe DSL 接受 BF16 或 FP32，且存在 `initial_state` 时须与其一致；Cake 接受 BF16。无
  `checkpoint_state_indices` 时，每序列第 0 行是其初始状态，后续行是 token 块
  `N, 2N, ...` 之前的状态。
- `checkpoint_cu_starts`（`torch.Tensor`，可选）：连续 CUDA int64 累积 checkpoint 计数
  `[N_seq+1]`。首值必须为零。无 `checkpoint_state_indices` 时每个差值为 `ceil(seq_len / N)`；带
  索引时每个差值为 `floor(seq_len / N)`，计数完成的周期边界（含对齐的最终边界、不含初始状态）。
- `checkpoint_state_indices`（`torch.Tensor`，可选）：仅 CuTe DSL 使用的连续 CUDA int32 目的地行。
  第 `i` 项选择为 packed completed-boundary 第 `i` 项写入 `state_checkpoints` 池的行。kernel 直接写
  池，不使用临时 checkpoint 张量或 scatter。
- `checkpoint_every_n_tokens`（`int`，默认 `0`）：checkpoint 间隔。零禁用；正值必须可被 32 整除，
  唯一例外是 SM100 家族 exact-N16 frozen 路由也接受 16 的倍数。SGLang 通常用 64 或更大的
  cache-page 对齐倍数。
- `disable_state_update`（`bool`，默认 `False`）：frozen / speculative-verify 模式：从 committed
  状态为一序列最多 16 个 token 计算输出，且从不写回任何状态（`final_state` 为 `None`）。见
  `flashinfer.kda_decode.recurrent_kda` 的完整模式契约，包括可选的 slot-indexed
  `correction_cache` / `kg_cache` verify 输出。
- `correction_cache`（`torch.Tensor`，可选）：仅 frozen-verify：slot-indexed float32 per-token
  delta-rule 修正 `[num_slots, HV, T_max, V]`。
- `kg_cache`（`torch.Tensor`，可选）：仅 frozen-verify：slot-indexed bf16（raw key | raw gate）缓存
  `[num_slots, HV, T_max, 2*K]`。
- `backend`（`Literal[...]`，默认 `"auto"`）：实现后端，取值与语义见下文「后端选择」。

**返回**

- `Tuple`：`(output, final_state)`，`output_final_state=False` 时 `final_state` 为 `None`。启用
  checkpointing 时返回三元组 `(output, final_state, state_checkpoints)`。见
  `flashinfer.kda_kernels.recurrent_kda.run_recurrent_kda` 的后端实现。

**注意事项**

- `prefill_workspace` 必须为 `RecurrentKDAPrefillWorkspace`，否则抛 `TypeError`。
- 不支持的 `backend` 值抛 `ValueError`。

**示例**

```python
import torch
import flashinfer

B, T, H, K, V = 1, 1, 32, 128, 128
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
state = torch.zeros(B, H, V, K, device="cuda", dtype=torch.bfloat16)

out, final_state = flashinfer.recurrent_kda(
    q, k, v, g, beta, initial_state=state, output_final_state=True,
)
```

---

### `RecurrentKDAPrefillWorkspace`（类）

```python
class RecurrentKDAPrefillWorkspace(device: torch.device | str)
```

**功能**：recurrent-KDA CUDA graph capture 所需的调用方拥有存储。在图对应的 CUDA 设备上，为每次
captured `flashinfer.kda.recurrent_kda` 调用构造一个 workspace。通过在 capture stream 上以精确
张量 eager 调用该函数来 warm，然后在 capture 前同步该 stream。workspace 在图生命周期内持有可选
最终状态 scratch、后端元数据、TMA descriptor，以及各 schedule 专属 scratch。在 SM100 家族设备上
这包括 beta padding、M64/M128-N32/M128-N16 与 BT64 descriptor storage、small-BH packet-ring
storage，以及 BT16 prepare/chain 元数据、factor 与独立 descriptor storage。Persistent M128 是
eager-only 的 B200/GB200 路由；显式 workspace 使用 non-persistent direct、M64、small-BH 或
eligible BT16 schedule，因此 graph capture 从不同步序列长度以构造主机 task bin。

**参数**

- `device`（`torch.device | str`）：CUDA 设备。非 CUDA 设备抛 `ValueError`；无 index 时解析为当前
  设备。

**生命周期与语义**

- workspace 绑定到其首个 stream。
- 一旦参与 capture，就不能再传给 Python（无论 eager 还是另一次 capture）。
- graph replay 不调用 Python，在 workspace 生命周期内保持有效。
- 构造后无需显式 `close()`；它是纯存储对象。

**示例**

```python
import torch
import flashinfer

q = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
g = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
beta = torch.rand(1, 1024, 8, device="cuda", dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, 1024], device="cuda", dtype=torch.int64)

workspace = flashinfer.RecurrentKDAPrefillWorkspace(device=q.device)
out = torch.empty_like(v)
# 在同一 CUDA stream 上先 warm 再 capture
out, state = flashinfer.recurrent_kda(
    q, k, v, g, beta, cu_seqlens=cu_seqlens, output=out,
    prefill_workspace=workspace, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, beta_is_logit=True,
)
```

---

### `RecurrentKDAPrefillWrapper`（类，实验性）

```python
class RecurrentKDAPrefillWrapper(device: torch.device | str)
RecurrentKDAPrefillWrapper.plan(cu_seqlens: torch.Tensor, *, non_blocking: bool = True) -> None
RecurrentKDAPrefillWrapper.run(
    q, k, v, g, beta,
    A_log=None, dt_bias=None, scale=None, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=False, lower_bound=None, output=None,
    beta_is_logit=False, state_checkpoints=None, checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0, checkpoint_state_indices=None,
    ssm_state_indices=None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]] | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
```

> **警告（实验性）**：`RecurrentKDAPrefillWrapper` 是**实验性**的，不提供兼容性保证，可能在不经
> 弃用的情况下改变或被移除。它尚未出现在任何 release 中；随着 `recurrent_kda` 接口统一，其
> plan-and-run 形态预计会变化。调用 `plan` 或 `run` 本身就是 opt-in，不需要环境变量；planning 实现
> 位于 `flashinfer.experimental.kda_prefill_wrapper`。因此它在上游文档中**没有生成的参考页**，直到
> 其毕业。

**功能**：packed recurrent-KDA prefill 的 plan-and-run 包装器。仅支持 compute capability 10.0 与
10.3。`run` 强制 `backend="cute-dsl"` 且始终传入其规划的 `seq_order`；CC 12.0 后端两者都不支持，
故 CC 12.0 调用方应直接使用 `flashinfer.kda.recurrent_kda`。

**`__init__` 参数**

- `device`（`torch.device | str`）：CUDA 设备。

**`plan(cu_seqlens, *, non_blocking=True)`**

- 在 CUDA Graph capture 之外规划 packed prefill 序列顺序。
- `cu_seqlens` 可位于 CPU 或该 wrapper 的 CUDA 设备，可用 int32 或 int64 存储。其值被拷入稳定的
  int64 CUDA 缓冲；`run` 期间的 GPU 元数据生成会处理零长度序列的重复 offset。

**`run(...)`**

- 使用最近一次 plan 运行 packed recurrent-KDA prefill。
- 参数语义与 `flashinfer.recurrent_kda` 的对应参数一致（`q/k/v/g/beta`、`A_log`、`dt_bias`、
  `scale`、`initial_state`、`output_final_state`、`use_qk_l2norm_in_kernel`、`use_gate_in_kernel`、
  `lower_bound`、`output`、`beta_is_logit`、`state_checkpoints`、`checkpoint_cu_starts`、
  `checkpoint_state_indices`、`checkpoint_every_n_tokens`、`ssm_state_indices`）。

**生命周期与约束**

- `plan` 在 CUDA Graph capture 之外运行，把 `cu_seqlens` 拷入固定地址设备缓冲，而不在主机上读取
  其值。`run` 在 recurrent kernel 消费之前，于 GPU 上生成稳定的 descending-length 序列顺序与累积
  chunk 前缀。
- 序列数与 packed token 张量 extent 在首次 warmup run 后固定，使设备缓冲地址、workspace 容量与
  launch 几何在 CUDA Graph replay 间保持有效。replay 前可再次调用 `plan` 原地更新各长度。
- 仅限 CuTe DSL 后端，并有意使用其 non-persistent schedule。一个 wrapper 实例是单写者资源：不要在
  `run` 的 kernel 仍可能读取其规划缓冲时并发调用 `plan`。

**示例**

```python
import torch
import flashinfer

wrapper = flashinfer.RecurrentKDAPrefillWrapper(device="cuda")
cu_seqlens = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int64)
wrapper.plan(cu_seqlens)

q = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)
# ... k, v, g, beta ...
out, state = wrapper.run(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
```

---

### `fused_kda_decode`

```python
fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    output: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cute-dsl", "cake", "auto"] = "cute-dsl",
    state_indices_mode: Optional[Literal["positive_unique", "unique_or_null", "repeated_positive"]] = None,
) -> torch.Tensor
```

**功能**：运行融合的 Kimi KDA 解码流水线。把 width-4 depthwise causal 卷积（SiLU）、一次
recurrent KDA 更新与 gated RMSNorm 融合在一起。专用于 head 维度 128 以及 8、12、24、32、48 或 96
个 head。`conv_state` 与 `state` 原地更新。slot 0 保留为 null slot；`state_indices` 值非正的行使
输出为零且不更新任一缓存。

**参数**

- `x`（`torch.Tensor`）：packed QKV 投影，形状 `[num_rows, 3 * H * 128]`，bfloat16。channel 维必须
  连续。
- `weight`（`torch.Tensor`）：depthwise 卷积权重，形状 `[3, 4, H * 128]`，float32。
- `conv_state`（`torch.Tensor`）：分页卷积缓存，形状 `[num_slots, 3 * H * 128, 3]`，bfloat16。每个
  slot 须使用 sequence-dimension 缓存布局，stride 为 `[slot_stride, 1, 3 * H * 128]`。
- `raw_gate`（`torch.Tensor`）：原始 per-channel 循环 gate，形状 `[1, num_rows, H, 128]`，bfloat16。
- `raw_beta`（`torch.Tensor`）：原始 delta-rule 学习率 logits，形状 `[1, num_rows, H]`，bfloat16。
- `A_log`（`torch.Tensor`）：对数衰减参数，`H` 个元素，float32。
- `dt_bias`（`torch.Tensor`）：per-channel 衰减偏置，`H * 128` 个元素，float32。
- `state_indices`（`torch.Tensor`）：每个解码行选择的缓存 slot，连续 int32，`num_rows` 个元素。
  活跃索引须在 `[1, num_slots)`；非正索引选择 null 路径。
- `state`（`torch.Tensor`）：分页循环状态，形状 `[num_slots, H, 128, 128]`，float32 或 bfloat16。
  递推以 float32 求值；bfloat16 状态写回时舍入。每个 slot 的 `[H, 128, 128]` 内容须连续。`state`
  与 `conv_state` 必须具有相同的 `num_slots`。
- `output_gate`（`torch.Tensor`）：gated RMSNorm logits，形状 `[num_rows, H, 128]` 或
  `[1, num_rows, H, 128]`，bfloat16。
- `norm_weight`（`torch.Tensor`）：RMSNorm 权重，128 个元素，float32。
- `lower_bound`（`float`，可选，默认 `-5.0`）：递推 gate 使用的负下界。Kimi K3 默认 `-5.0`。传
  `None` 使用原始 Kimi-Linear softplus gate。
- `norm_eps`（`float`，默认 `1e-5`）：非负 RMSNorm epsilon。
- `output`（`torch.Tensor`，可选）：可选预分配连续 bfloat16 输出，形状 `[1, num_rows, H, 128]`。
- `backend`（`Literal["cute-dsl", "cake", "auto"]`，默认 `"cute-dsl"`）：`"cute-dsl"` 保留现有
  FlashInfer 实现；`"cake"` 严格选择导出的 Cake kernel，无匹配路由时抛错；`"auto"` 仅在提供
  `state_indices_mode` 且在 SM100a 或 SM103a 上匹配路由时选择 Cake，否则保留 CuTe DSL。每个架构
  使用各自的编译模块。
- `state_indices_mode`（可选字面量）：仅 Cake dispatcher 使用的主机已知断言。`"positive_unique"`
  表示每个索引为正且唯一；`"unique_or_null"` 表示正索引唯一且可能存在非正 null 行；
  `"repeated_positive"` 表示至少一个正 slot 重复（null 行也可能存在）。`backend="cake"` 时必需。
  dispatcher 从不读取 CUDA 张量来推断该属性。

**返回**

- `torch.Tensor`：bfloat16 输出，形状 `[1, num_rows, H, 128]`。

**注意事项**

- 当底层后端不可用时抛 `NotImplementedError("fused KDA decode backend is unavailable")`。

**示例**

```python
import torch
import flashinfer

H, num_rows, num_slots = 12, 8, 64
x = torch.randn(num_rows, 3 * H * 128, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(3, 4, H * 128, device="cuda", dtype=torch.float32)
conv_state = torch.zeros(num_slots, 3 * H * 128, 3, device="cuda", dtype=torch.bfloat16)
raw_gate = torch.randn(1, num_rows, H, 128, device="cuda", dtype=torch.bfloat16)
raw_beta = torch.randn(1, num_rows, H, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
dt_bias = torch.zeros(H * 128, device="cuda", dtype=torch.float32)
state_indices = torch.arange(1, num_rows + 1, device="cuda", dtype=torch.int32)
state = torch.zeros(num_slots, H, 128, 128, device="cuda", dtype=torch.float32)
output_gate = torch.randn(num_rows, H, 128, device="cuda", dtype=torch.bfloat16)
norm_weight = torch.ones(128, device="cuda", dtype=torch.float32)

out = flashinfer.fused_kda_decode(
    x, weight, conv_state, raw_gate, raw_beta, A_log, dt_bias,
    state_indices, state, output_gate, norm_weight,
)
```

---

### `packed_fused_kda_decode`

```python
packed_fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    output: Optional[torch.Tensor] = None,
    *,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    t1_state_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：运行 packed `T>=1` 的融合 KDA，带 per-token cache checkpoint。token 使用由
`query_start_loc` 描述的 packed ragged 布局。`state_indices` 形状 `[N, T]`，T 是最大验证长度。
`num_accepted_tokens[n] - 1` 选择序列 `n` 的 source checkpoint 与卷积历史 offset；token `t` 把其
recurrent checkpoint 写到 `state_indices[n, t]`。卷积缓存是 `state_indices[n, 0]` 中长度为 `T + 2`
的一个 rolling window。

本门面仅 CuTe：`T=1` 使用 `fused_kda_decode` 底层的 CuTe DSL 实现（无 Cake 或 backend 选择），
含 bfloat16-state 与 softplus-gate 支持。可选 `t1_state_indices` 优化严格限于 `T=1`；对 `T>1`
传入会 fail-closed。`T>1` 使用 SM10x CuTe DSL packed 后端，接受 float32 或 bfloat16 recurrent
state，并要求有限负 `lower_bound`。两条路径都要求 head 维度 128 与卷积宽度 4；`T=1` 接受 8、12、
24、32、48 或 96 个 head，`T>1` 接受 12、24、32、48 或 96 个 head。

**参数**

- `x`（`torch.Tensor`）：packed QKV 投影，形状 `[num_rows, 3 * H * 128]`，bfloat16。
- `weight`（`torch.Tensor`）：depthwise 卷积权重，形状 `[3, 4, H * 128]`，float32。
- `conv_state`（`torch.Tensor`）：分页 bfloat16 卷积缓存。`T=1` 用历史长度 3；`T>1` 用扩展 rolling
  历史长度 `T + 2`。
- `raw_gate`（`torch.Tensor`）：原始循环 gate，形状 `[1, num_rows, H, 128]`，bfloat16。
- `raw_beta`（`torch.Tensor`）：原始 delta-rule 学习率 logits，形状 `[1, num_rows, H]`，bfloat16。
- `A_log`（`torch.Tensor`）：对数衰减参数，`H` 个元素，float32。
- `dt_bias`（`torch.Tensor`）：per-channel 衰减偏置，`H * 128` 个元素，float32。
- `state_indices`（`torch.Tensor`）：连续 int32 缓存 slot，形状 `[N, T]`。`T>1` 时列 `t` 是 token
  `t` 的 recurrent checkpoint 目的地；列 0 也选择 rolling 卷积缓存。
- `state`（`torch.Tensor`）：分页循环状态，形状 `[num_slots, H, 128, 128]`，float32 或 bfloat16。
  `T>1` 在 float32 中保持递推，仅在每个 bfloat16 checkpoint 写回时舍入。
- `output_gate`（`torch.Tensor`）：gated RMSNorm logits，形状 `[num_rows, H, 128]` 或
  `[1, num_rows, H, 128]`，bfloat16。
- `norm_weight`（`torch.Tensor`）：RMSNorm 权重，128 个元素，float32。
- `lower_bound`（`float`，可选，默认 `-5.0`）：负的循环 gate 下界。`T=1` 也接受 `None`（softplus
  gate）；`T>1` 要求有限负值。
- `norm_eps`（`float`，默认 `1e-5`）：非负 RMSNorm epsilon。
- `output`（`torch.Tensor`，可选）：可选预分配连续 bfloat16 输出，形状 `[1, num_rows, H, 128]`。
- `query_start_loc`（`torch.Tensor`，关键字，必需）：连续 int32 packed-row offset，形状 `[N + 1]`。
  必须从零开始，包含在 `x` 内非递减的 offset，且活跃长度不大于 T。
- `num_accepted_tokens`（`torch.Tensor`，关键字，必需）：连续 int32 已接受 token 计数，形状 `[N]`，
  值在 `[1, T]`。
- `t1_state_indices`（`torch.Tensor`，可选）：可选连续 int32 per-packed-row 缓存 slot，形状
  `[num_rows]`。仅 `T=1`；null 与 trailing 行须为零或非正。该张量从不由本函数分配或填充。提供时
  由 direct fused `T=1` 特化使用，`query_start_loc` 仅作为结构元数据保留，其值不被主机或 kernel
  读取。

**返回**

- `torch.Tensor`：packed bfloat16 输出，形状 `[1, num_rows, H, 128]`。

**注意事项**

- `state_indices` 必须至少 2 维（`[N, T]`）。
- 活跃 recurrent 目的地必须为正缓存 slot，且不得与另一活跃序列的目的地别名；非正或零长度行是 null
  行，不改变任一缓存。`T=1` 时，`query_start_loc[-1]` 之外的 packed 行是 trailing 未用容量：输出
  为零且不改缓存。这些值约束由调用方负责，使 CUDA Graph replay 从不执行 device-to-host 校验同步。

**示例**

```python
import torch
import flashinfer

N, H, num_rows, T = 4, 12, 16, 1
x = torch.randn(num_rows, 3 * H * 128, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(3, 4, H * 128, device="cuda", dtype=torch.float32)
conv_state = torch.zeros(64, 3 * H * 128, 3, device="cuda", dtype=torch.bfloat16)
raw_gate = torch.randn(1, num_rows, H, 128, device="cuda", dtype=torch.bfloat16)
raw_beta = torch.randn(1, num_rows, H, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
dt_bias = torch.zeros(H * 128, device="cuda", dtype=torch.float32)
state_indices = torch.randint(1, 64, (N, T), device="cuda", dtype=torch.int32)
state = torch.zeros(64, H, 128, 128, device="cuda", dtype=torch.bfloat16)
output_gate = torch.randn(num_rows, H, 128, device="cuda", dtype=torch.bfloat16)
norm_weight = torch.ones(128, device="cuda", dtype=torch.float32)
query_start_loc = torch.tensor([0, 4, 8, 12, 16], device="cuda", dtype=torch.int32)
num_accepted = torch.ones(N, device="cuda", dtype=torch.int32)

out = flashinfer.packed_fused_kda_decode(
    x, weight, conv_state, raw_gate, raw_beta, A_log, dt_bias,
    state_indices, state, output_gate, norm_weight,
    query_start_loc=query_start_loc, num_accepted_tokens=num_accepted,
)
```

---

### `packed_kda_decode`

```python
packed_kda_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：运行 serving-native 的 packed Kimi K3 recurrent decode。本算子直接消费 post-convolution
packed QKV 行以及原始 gate 与 beta logits，把 Q/K 提取与 L2 归一化、Kimi K3 lower-bound gate 变换、
beta sigmoid 与一次 recurrent 状态更新融合进单个导出的 Cake kernel。它专用于精确 SM100a 与 SM103a
设备上的 `T=1`、`H=12`、`K=V=128`。

固定数值契约使用 `scale=1/sqrt(128)`、L2 epsilon `1e-6`、`lower_bound=-5`。`state` 在调用方当前
PyTorch CUDA stream 上原地更新。batch 小于 32 使用八行 value tile；batch 不小于 32 使用十六行
value tile。

**参数**

- `mixed_qkv`（`torch.Tensor`）：post-convolution packed QKV，形状 `[B, 3 * 12 * 128]`，bfloat16。
  最后一维必须连续；batch 行之间的正 padding 允许。
- `raw_gate`（`torch.Tensor`）：原始 per-channel 循环 gate，形状 `[B, 12 * 128]`，bfloat16。最后一
  维必须连续。
- `raw_beta`（`torch.Tensor`）：原始 delta-rule 学习率 logits，形状 `[B, 12]`，bfloat16。最后一维
  必须连续。
- `A_log`（`torch.Tensor`）：连续 float32 log-decay 参数，形状 `[12]`。
- `dt_bias`（`torch.Tensor`）：连续 float32 per-channel 衰减偏置，形状 `[12 * 128]`。
- `state`（`torch.Tensor`）：调用方拥有的 bfloat16 recurrent 状态池，形状 `[N, 12, 128, 128]`。其
  内三维必须紧凑；外层 slot stride 可含任意正 padding。
- `state_indices`（`torch.Tensor`）：连续 CUDA int32 每行缓存 slot，形状 `[B]`。活跃索引须唯一且在
  界内。`-1` 标记 inactive CUDA-graph padding 行，产生零输出且不访问或更新 `state`。这些值约束不在
  主机校验，以避免设备同步。
- `output`（`torch.Tensor`，可选）：可选调用方拥有的连续 bfloat16 输出，形状 `[B, 1, 12, 128]`。
  提供它可避免分配，且是无分配 CUDA-graph replay 路径所必需的。

**返回**

- `torch.Tensor`：默认形状 `[B, 1, 12, 128]` 的 bfloat16 输出。提供 `output` 时返回的即该确切的
  分配。

**注意事项**

- 与 `recurrent_kda`（输入已拆分为 Q、K、V、gate、beta）以及 `fused_kda_decode`（还执行卷积与
  gated RMSNorm）不同，本算子消费 post-convolution packed 输入。
- 不支持的设备或契约抛错，不回退到其它 KDA 实现。

**示例**

```python
import torch
import flashinfer

B = 8
mixed_qkv = torch.randn(B, 3 * 12 * 128, device="cuda", dtype=torch.bfloat16)
raw_gate = torch.randn(B, 12 * 128, device="cuda", dtype=torch.bfloat16)
raw_beta = torch.randn(B, 12, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(12, device="cuda", dtype=torch.float32)
dt_bias = torch.zeros(12 * 128, device="cuda", dtype=torch.float32)
state = torch.zeros(256, 12, 128, 128, device="cuda", dtype=torch.bfloat16)
state_indices = torch.arange(B, device="cuda", dtype=torch.int32)

out = flashinfer.packed_kda_decode(
    mixed_qkv, raw_gate, raw_beta, A_log, dt_bias, state, state_indices,
)
```

---

### `recurrent_kda`（`flashinfer.kda_decode`，已弃用）

```python
recurrent_kda(
    q, k, v, g, beta,
    A_log=None, dt_bias=None, scale=None, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=False, lower_bound=None, cu_seqlens=None,
    ssm_state_indices=None, num_spec_tokens=None, num_accepted_tokens=None,
    output=None, initial_state_source=None, initial_state_indices=None,
    beta_is_logit=False, *,
    disable_state_update=False, correction_cache=None, kg_cache=None,
    backend: Optional[Literal["cute-dsl", "cake", "auto"]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

> **自 0.8 起弃用**：`flashinfer.kda_decode.recurrent_kda` 已弃用，现为与
> `flashinfer.recurrent_kda` 共享的解码 dispatcher 之上的 shim，后者是规范入口。调用会发出
> `DeprecationWarning`。`fused_kda_decode` 与 `packed_kda_decode` 不受影响。

**功能**：KDA 解码 kernel 的旧门面。支持单 token 解码（`T=1`）与 packed 投机解码（`T>=2`）。
参数语义与 `flashinfer.recurrent_kda` 的解码子集一致，但 `cu_seqlens` 须为 int32，`initial_state`
须为 bfloat16，且 backend 枚举更窄。

**`backend` 参数**（`Optional[Literal["cute-dsl", "cake", "auto"]]`，默认 `None`）：
`"cute-dsl"` 保留现有 FlashInfer 实现；`"cake"` 严格选择导出的 Cake kernel；`"auto"` 仅对
equal-head/D128/T1 unbounded-softplus 契约选择 Cake。省略（`None`）解析为 `"cute-dsl"`，即本门面的
已发布默认值；`None` 被记录为“未请求”而非该选择，以便未来 release 能把省略的调用收敛到
`flashinfer.recurrent_kda` 的 `"auto"`。`backend="cudnn"` 在此入口被拒绝（cuDNN 引擎仅服务普通
多 token prefill）。

**注意事项**

- `disable_state_update=True` 进入 frozen / speculative-verify 模式：最多 16 token/序列，从不写回
  状态，`final_state` 恒为 `None`。`backend="cake"` 在该模式下抛错，`output_final_state=True` 被
  拒绝。要求 `K == V == 128` 与 bf16 状态池。

---

### `prepare_tf32_kda_prefill`

```python
prepare_tf32_kda_prefill(
    q, k, v, g, beta,
    *,
    A_log, dt_bias, out,
    initial_state=None, final_state=None, scale=None, lower_bound=-5.0,
    cu_seqlens=None, sequence_lengths=None, state_indices=None,
    state_checkpoints=None, checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0, beta_is_logit=True,
)
```

**功能**：在 SM100a 或 SM103a 上准备一次 TF32 KDA 推理调用。准备阶段分配 workspace 并在首次使用时
编译。返回对象的 `launch()` 只在当前 stream 上提交工作，并可在调用方拥有的 CUDA graph 中捕获。

**参数**

- `q, k, v, g, beta`：Q/K/V 与输出为 BF16；`initial_state` / `final_state` 必须为 FP32。
  `state_checkpoints` 保持 BF16。packed 调用要求主机 `sequence_lengths` 与 `cu_seqlens` 匹配；
  checkpoint offset 是它们的 canonical chunk-count 前缀和。张量存储必须保持存活且地址不变。
- `A_log`（关键字，必需）：对数衰减参数。
- `dt_bias`（关键字，必需）：per-channel 衰减偏置。
- `out`（关键字，必需）：预分配输出。
- `initial_state`、`final_state`、`scale`、`lower_bound`（默认 `-5.0`）、`cu_seqlens`、
  `sequence_lengths`、`state_indices`、`state_checkpoints`、`checkpoint_cu_starts`、
  `checkpoint_every_n_tokens`（默认 `0`）：可选配置项。
- `beta_is_logit`（`bool`，默认 `True`）：`beta` 可含 BF16 logits 或 active FP32 probabilities。

**返回**

- 一个 prepared launch 对象，提供 `launch()`（在当前 stream 提交）与 `close()`。

**注意事项**

- 不支持训练。

**示例**

```python
import torch
import flashinfer

# 见 docs/api/kda.rst 的 ptx / cute-dsl-persistent 示例；TF32 版本同理：
launch = flashinfer.prepare_tf32_kda_prefill(
    q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, out=out,
)
launch.launch()
```

---

### `prepare_bf16_kda_prefill`

```python
prepare_bf16_kda_prefill(
    q, k, v, g, beta,
    *,
    A_log, dt_bias, out,
    initial_state=None, final_state=None, scale=None, lower_bound=-5.0,
    cu_seqlens=None, sequence_lengths=None, state_indices=None,
    state_checkpoints=None, checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0, beta_is_logit=True,
    plan_cache=None,
)
```

**功能**：在 SM100a 或 SM103a 上准备一次 BF16 KDA 推理调用。Q/K/V 与输出为 BF16；外部
`initial_state` / `final_state` 为 FP32。`state_checkpoints` 为 BF16 观测值，或当其为 FP32 时，是
精确的 FP32 中间状态：fused direct M128 body 随后在每个 chunk 边界保持其 recurrent carrier 为
FP32，并以全精度存储这些行（见 `kda_prefill_supports_fp32_checkpoints`）。

**参数**：与 `prepare_tf32_kda_prefill` 相同，另加：

- `beta_is_logit`（`bool`，默认 `True`）：`beta` 默认含 BF16 logits。`beta_is_logit=False` 时
  `beta` 含 active FP32 probabilities。需要 logits 的路由在每次 `launch()` 时用
  `logit(eps=1e-6)` 转换 beta 并生成 BF16 副本。padded 或 strided beta 存储允许；只转换逻辑 token
  行。
- `plan_cache`（`KDAPrefillPlanCache`，可选）：提供后，与新张量地址具有相同结构签名（shape、
  stride、dtype、alignment、aliasing、`sequence_lengths` 与 checkpoint plan）的既备 launch 会被
  重新绑定到新地址，而非重新准备；只有 TMA descriptor 在当前 stream 上重新编码并上传。缓存 launch
  归缓存所有：不要 `close()` 它们，也不要在多个 stream 上并发使用同一缓存。

**返回**

- 一个 prepared launch 对象，提供 `launch()`（在当前 stream 提交完整操作并支持调用方拥有的 CUDA
  graph capture）与 `close()`。

**注意事项**

- 不支持训练。

---

### `kda_prefill_supports_fp32_checkpoints`

```python
kda_prefill_supports_fp32_checkpoints(device=None, *, lower_bound=None) -> bool
```

**功能**：查询本 build 是否为 `device` 导出 FP32 中间状态。FP32 `state_checkpoints` 选择 fused
direct M128 body 的 FP32-carrier 特化。该特化按 gate 类型导出：`lower_bound=None` 询问 unbounded
softplus gate（Kimi-Linear / Kimi-K3，即 radix prefix caching 恢复所依据的契约），其它值询问
bounded gate。检查会读取该设备架构的导出模块 registry，因此 registry 缺少该变体的安装返回
`False`，调用方则为该 gate 类型保留 BF16 checkpoint 行。

**参数**

- `device`（可选）：目标 CUDA 设备。无法解析架构时返回 `False`。
- `lower_bound`（关键字，可选）：`None` 询问 unbounded softplus gate，否则询问 bounded gate。

**返回**

- `bool`：本 build 是否为该设备与该 gate 类型导出 FP32 中间状态。

---

### `KDAPrefillPlanCache`（类）

```python
class KDAPrefillPlanCache(capacity: int = 64, max_bytes: int | None = None)
KDAPrefillPlanCache.get(inputs, **scalars) -> prepared | None
KDAPrefillPlanCache.put(prepared, inputs, *, retained_bytes: int = 0, **scalars) -> bool
KDAPrefillPlanCache.clear() -> None
len(KDAPrefillPlanCache)
```

**功能**：以结构签名为键、有界的 prepared KDA launch LRU。服务侧每层每个 forward batch 用新的
output、新的 checkpoint 行与 per-layer 状态池调用 prepared export，但同一 batch 的其它层具有相同
token shape 与 `sequence_lengths`。准备（host 校验、route 选择、bin packing、元数据上传、workspace
分配）与地址无关，因此缓存为每个签名保留一个 prepared launch，并在命中时重新绑定其带指针的参数。
命中只做 host 视图构造加一次 descriptor 上传 kernel：无设备同步、无数据相关 host 分支，因此命中可
被 CUDA graph 捕获。

**参数（`__init__`）**

- `capacity`（`int`，默认 `64`）：最大条目数。须为正，否则 `ValueError`。
- `max_bytes`（`int | None`，默认 `None`）：缓存 launch 保留的 workspace 字节预算。为 `None` 时取
  环境变量 `FLASHINFER_KDA_PLAN_CACHE_MAX_BYTES`，否则取默认 `DEFAULT_MAX_BYTES = 8 << 30`
  （8 GiB）。须为正，否则 `ValueError`。约 0.7 GiB 为 16K-token pack 的 affine composite 保留量。

**属性 / 计数**

- `capacity`、`max_bytes`、`bytes`：配置与当前保留字节。
- `hits`、`fast_hits`、`misses`、`uncacheable`、`evictions`：统计计数。
- `DEFAULT_MAX_BYTES`：默认字节预算。

**方法**

- `get(inputs, **scalars)`：返回重新绑定到 `inputs` 的 prepared launch，未命中返回 `None`。descriptor
  重新编码被推迟到 launch 的 stream context，且当没有 TMA-described 张量移动时跳过，除非正在捕获
  CUDA graph（captured launch 总是重新编码）。
- `put(prepared, inputs, *, retained_bytes=0, **scalars)`：记录一个 freshly prepared launch；不可缓存
  时返回 `False`。`retained_bytes` 是 launch 保持存活的设备内存；缓存按 LRU 逐出直到条目数与字节
  预算都满足，始终保留最新条目。
- `clear()`：关闭并清空所有条目。
- `__len__`：当前条目数。

**注意事项**

- 缓存 launch 归缓存所有：不要 `close()` 它们。
- 不要在多个 stream 上并发使用同一缓存。

**示例**

```python
import flashinfer

cache = flashinfer.KDAPrefillPlanCache(capacity=64)
launch = flashinfer.prepare_bf16_kda_prefill(
    q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, out=out, plan_cache=cache,
)
launch.launch()  # 归 cache 所有；不要 close()
```

---

## 后端选择

`backend` 选择的是实现族，而非架构名。`flashinfer.recurrent_kda` 的 `backend` 取值与语义如下：

- `"auto"`（默认）：对符合条件的 ordinary multi-token prefill 选择 source-level CuTe DSL 后端，
  对不支持契约回退到 frozen Cake 后端；并可能为受支持的 plain prefill 契约在 CuTe DSL 与 cuDNN 间
  选择。对 logical batch × head 数不超过设备 SM 数一半的 eligible SM100-family 调用，选择 bundled
  small-BH CuTe DSL kernel；其它情况选择架构合适的 CuTe DSL kernel（含 SM120 后端与 SM100-family
  state checkpoints），否则回退到导出的 frozen Cake 特化。decode 保留现有 KDA decode 路由。
- `"cake"`：严格选择 frozen Cake 后端，契约不支持时抛错（不回退）。Cake prefill 路径根据输入形状
  与物理设备在 direct、persistent、small-BH 与 two-stage BT16 schedule 间选择。
- `"cute-dsl"`：严格选择 BT=16 CuTe DSL prefill kernel，契约不支持时抛错。要求连续 BF16
  Q/K/V/G/beta、共享 head 数且 head 维度 128、in-kernel lower-bound gate、fixed 或 packed-varlen
  布局、BF16 recurrent state、显式 `seq_order`，以及 Cake 相同的 checkpoint 契约。
  `checkpoint_cu_starts` 必须始终为 int64；packed `cu_seqlens` 在 CUDA graph capture 期必须为
  int64。schedule 为非 persistent（Cake 相同的 checkpoint 契约）。
- `"small-bh"`：选择 small-BH CuTe DSL KDA prefill kernel。
- `"cute-dsl-persistent"`：显式选择 SM100/SM103 persistent prefill kernel（改编自
  humanfia/kda-for-kda-release）。要求连续、16 字节对齐的 BF16 Q/K/V/G `[B,T,H,128]` 与 beta
  `[B,T,H]`（`T > 1`，H 为正且可被 8 整除，`B*T*H*128 < 2**31`）、连续 FP32 `A_log[H]` 与
  `dt_bias[H*128]` 或 `[H,128]`、`use_qk_l2norm_in_kernel=True`、`use_gate_in_kernel=True`、
  `beta_is_logit=True`、`-5 <= lower_bound < 0`、可选连续 32 字节对齐 FP32 state
  `[N,H,128,128]`（V-first，原地更新），以及 fixed batch 或 packed `B=1` 序列（连续 CUDA
  int32/int64 `cu_seqlens[N+1]`，从零开始、严格递增、非空）。state pool/indices、checkpoints、
  external sequence ordering、unbounded gates、GQA、speculative decode 与 frozen-state 模式都在此
  后端契约之外，请求它们会抛错。需要 `nvidia-cutlass-dsl>=4.7.0`。
- `"tirx"`：显式选择可选的 SM100/SM103 TIRx prefill kernel（改编自 NVlabs/kda 的 `tirx`）。单序列
  用 BT32 前端与 recurrent chain；其它输入用 fused persistent BT64 kernel 与主机构建的 work list。
  仅在显式选择该后端时才 import TIRx。需要 CUDA-enabled TVM、TIRx Lite 前端与 CUDA 13 toolkit，
  且 `tirx_kernels.tirx_lite` 必须存在。
- `"ptx"`：显式从 NVlabs/kda 选择四个保留的 PTX 程序，目标 SM103a（B300），使用 persistent
  recurrence scheduling 与精确 FP32 state handoff。需要 `nvidia-cuda-nvcc>=13.2` 与
  `apache-tvm-ffi>=0.1.12`。`FLASHINFER_KDA_PTXAS` 选择汇编器；
  `FLASHINFER_KDA_PTX_CACHE_DIR` 覆盖默认 JIT cubin 缓存目录。JIT-only，在
  `FLASHINFER_DISABLE_JIT=1` 的新进程中不可用。
- `"cudnn"`：通过 `flashinfer.cudnn.cudnn_recurrent_kda` 运行 cuDNN 的融合 SM100 linear-attention
  引擎，无法服务时抛携带原因的 `NotImplementedError`。仅覆盖 ordinary multi-token prefill：无
  speculative decode、无 `initial_state_source`、无 state checkpoints。若 cuDNN frontend 支持状态
  池，`ssm_state_indices` 会原地读/更新 `initial_state` 的选中 slot；`output_final_state` 返回该池
  而非 gathered per-sequence copy。状态池调用不会被自动选择为 cuDNN。

**SM120a prefill 子集**：compute capability 12.0 设备选择自己的 prefill 后端
`flashinfer.kda_kernels.sm120_prefill`（CuTe DSL）。`backend` 仍是 `"auto"` / `"cute-dsl"` 可到达该
后端，dispatcher 在 SM100 家族 CuTe DSL prefill 路径之前尝试它；显式 `"cake"` 请求从不探测或运行
SM120。仅在设备为 CC 12.0 且安装的 CuTe DSL 与 CUDA toolkit 能原生目标 `sm_120a`、输入为 ordinary
multi-token prefill、Q/K/V/G 为连续 BF16 `[B,T,H,128]` 且共享 head 数、beta 为连续 BF16
`[B,T,H]`（不支持 GQA 与 `V != K`）、输出满足 INT32 extent、`A_log` 与 `dt_bias` 格式正确、
`use_qk_l2norm_in_kernel=True`、`use_gate_in_kernel=True`、`beta_is_logit=True`、`lower_bound` 在
`[-5.0, 0.0)`、`initial_state` 若给出为连续 BF16 `[N,H,128,128]`（不支持状态池）、`output` 若给出
为连续 BF16 且不与任何输入重叠、且未启用 speculative decode / `seq_order` / prefill checkpoints /
committed-state sources / FP32 gate 或 state 时才使用。SM120a 有两个变体 `decomp`（chunk-parallel
prepare 加 serial recurrence）与 `fused`（单 kernel），按设备 SM 数从实测表选择。CUDA graph
capture 要求调用方拥有的 `RecurrentKDAPrefillWorkspace` 与预分配 `output`，并先在 capture stream 上
eager warm。

**其他语义摘要**

- **输入布局**：ordinary prefill（`auto`、`cute-dsl`、`cake`、`small-bh`）接受 strided Q/K 并在选择
  dense-only kernel 前打包；`cudnn` 保留受支持 stride。V、gates、state 与 output 保持既有布局要求。
- **dense-only prefill 的固定 / packed 约定**：fixed 输入省略 `cu_seqlens`；packed 输入 `B=1` 且接受
  连续 CUDA int32 或 int64 `cu_seqlens`。packed 预填充可选接受 `seq_order`（连续 CUDA int32，`[0, N)`
  的置换，按长度降序可减少尾部 partial wave）。
- **状态与 graph 语义**：显式 `initial_state` 一律原地更新，即使 `output_final_state=False`。使用
  `ssm_state_indices` 时 `initial_state` 是调用方拥有的池，每个 `[H,V,K]` slot 须连续，池基址与 slot
  pitch 须 16 字节对齐，slot id 须唯一且在范围内（调用方契约）。
- **checkpoint**：使用预分配 BF16 `state_checkpoints[C,H,V,K]`、int64 `checkpoint_cu_starts[N+1]` 与
  可被 16 整除的正 `checkpoint_every_n_tokens`。KDA checkpoint 是每个 interval *之前*的状态：每个
  非空序列贡献其初始状态为第 0 行，随后是严格位于其结束之前的一个、两个…… interval 后的状态。
- **CUDA graph capture**：需要调用方拥有的 `RecurrentKDAPrefillWorkspace` 与预分配 `output`，先在
  目标 capture stream 上 eager warm（含精确 Q/K/V/G/beta/output），再同步该 stream。packed graph 还
  必须传入预分配的 int64 `cu_seqlens` 与 int32 `seq_order`。workspace 必须比其 graph 及每次 replay
  存活更久；每个 captured 调用（含同一 graph 中两个 KDA 调用）使用各自独立 workspace。一旦参与
  capture，后续任何 Python 使用（eager 或另一次 capture）都会被拒绝；`graph.replay()` 不重入 Python。

## 相关链接 / 参考

- `flashinfer.kda_decode`（`docs/api/kda.rst` 的 `apikda_decode` 段）：解码专用接口与 Cake 后端细节。
- `flashinfer.kda_prefill`（`docs/api/kda.rst` 的 `apikda_prefill` 段）：prefill 优化后端与工作区契约。
- `flashinfer.cudnn.cudnn_recurrent_kda`：`backend="cudnn"` 的底层融合引擎。
- `#5069 <https://github.com/flashinfer-ai/flashinfer/issues/5069>`_：`RecurrentKDAPrefillWrapper` 的
  毕业计划。

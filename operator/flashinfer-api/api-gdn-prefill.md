# Gated Delta-Rule 预填充（`flashinfer.gdn_prefill`）

> 源码：`flashinfer/gdn_prefill.py`
> 上游文档：`docs/api/gdn_prefill.rst`
> 适用版本：FlashInfer 0.7.2

GDN 的 prefill 侧 kernel。`chunk_gated_delta_rule` 是分块（chunked）GDN scan，在解码循环接管之前
从完整 prompt 初始化循环状态。它支持 GQA / GVA、变长（varlen）packed 输入、状态池索引、上下文
并行（context parallelism, CP）、checkpointing，以及在 SM100 上通过 cuDNN 的融合引擎执行。要求
SM90（Hopper）或 SM100（Blackwell）架构；SM100 路径要求 `head_size == 128`。

## 导入

```python
from flashinfer.gdn_prefill import chunk_gated_delta_rule
# 顶层亦导出：
from flashinfer import chunk_gated_delta_rule
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_gated_delta_rule` | 函数 | 分块 Gated Delta Rule（GDN）prefill；初始化循环状态。 |

## API 详情

### `chunk_gated_delta_rule`

```python
chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Literal["auto"] | bool = "auto",
    state_indices: Optional[torch.Tensor] = None,
    _cp_chunk_len: Optional[int] = None,
    backend: Literal["auto", "flashinfer", "cake_gdn", "cudnn"] = "auto",
    max_seqlen: Optional[int] = None,
    *,
    gate_domain: Literal["linear", "log"] = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：实现门控 delta-rule 线性注意力机制的分块 prefill，面向高效训练与推理。支持 GQA
（grouped query attention）与 GVA（grouped value attention）配置，以及上下文并行、状态池与
checkpointing。

**参数**

- `q`（`torch.Tensor`）：query，形状 `[total_seq_len, num_q_heads, head_size]`，必须连续且在 CUDA 上。
- `k`（`torch.Tensor`）：key，形状 `[total_seq_len, num_k_heads, head_size]`，必须连续且在 CUDA 上。
- `v`（`torch.Tensor`）：value，形状 `[total_seq_len, num_v_heads, head_size]`，必须连续且在 CUDA 上。
- `g`（`torch.Tensor`，可选）：遗忘门（alpha），形状 `[total_seq_len, num_sab_heads]`，其中
  `num_sab_heads = max(num_q_heads, num_v_heads)`。预计算 alpha 时须为 float32；当
  `use_gate_in_kernel=True` 时原始 gate 也可为 float16 或 bfloat16。`None` 时默认为全 1。
- `beta`（`torch.Tensor`，可选）：更新门（beta），形状 `[total_seq_len, num_sab_heads]`。预计算 beta
  时须为 float32；`beta_is_logit=True` 时 logits 也可使用 Q 的 dtype。`None` 时默认为全 1。
- `scale`（`float`，可选）：attention score 缩放因子。为 `None` 时默认为 `1 / sqrt(head_size)`。
- `initial_state`（`torch.Tensor`，可选）：初始 KV 状态。Packed、按序列排列的形状
  `[num_seqs, num_sab_heads, head_size, head_size]`。dtype 可为 float32、bfloat16、float16、
  float8_e4m3fn 或 float8_e5m2。为 `None` 时从零状态开始。当给出 `state_indices`
  （SM90/SM100/SM103/SM120）时，它改为状态**池** `[N_pool, num_sab_heads, head_size, head_size]`，
  序列 `i` 从行 `state_indices[i]` 读取初始状态；在 SM100/SM103 上接受任意 rank-4 positive、
  非重叠的 stride 布局。
- `output_final_state`（`bool`，默认 `False`）：是否输出最终状态。
- `cu_seqlens`（`torch.Tensor`）：累积序列长度，形状 `[num_seqs + 1]`，int32 或 int64，与 `q` 同一
  CUDA 设备。变长（varlen）模式必需，且不得为 `None`。相邻重复 offset 表示合法的零长度序列。
- `use_qk_l2norm_in_kernel`（`bool`，默认 `False`）：是否以 epsilon `1e-6` 对每个 Q/K head 做 L2
  归一化。归一化在 float32 中累积，并在分块 kernel 前舍入回输入 dtype。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `[total_seq_len, num_o_heads, head_size]`，
  其中 `num_o_heads = max(num_q_heads, num_v_heads)`。为 `None` 时自动分配。
- `output_state`（`torch.Tensor`，可选）：预分配输出状态。Packed、按序列排列的形状
  `[num_seqs, num_sab_heads, head_size, head_size]`。dtype 可为 float32、bfloat16、float16、
  float8_e4m3fn 或 float8_e5m2。`output_final_state=True` 时必需。给出 `state_indices` 时，它改为
  输出状态**池** `[N_pool, ...]`，序列 `i` 的最终状态写入行 `state_indices[i]`（当
  `output_state is initial_state` 时原地写）；此时必须由调用方提供（自动分配被拒绝，因为紧凑的
  `[num_seqs, ...]` 缓冲会被池 slot id 越界索引）。SM100/SM103 接受与 `initial_state` 相同的
  positive、非重叠 rank-4 stride 布局。
- `state_checkpoints`（`torch.Tensor`，可选）：预分配 checkpoint，形状 `[total_checkpoints,
  num_sab_heads, head_size, head_size]`。dtype 可为 float32、bfloat16、float16、float8_e4m3fn 或
  float8_e5m2。`checkpoint_every_n_tokens > 0` 时必需。上下文并行 checkpointing 目前支持 SM90、
  SM100 和 SM120。
- `checkpoint_cu_starts`（`torch.Tensor`，可选）：累积 checkpoint 计数，形状 `[num_seqs + 1]`，
  int32 或 int64，与 `q` 同一 CUDA 设备。`checkpoint_cu_starts[i+1] - checkpoint_cu_starts[i]` 是
  序列 `i` 的 checkpoint 数（= `seq_len_i // checkpoint_every_n_tokens`）。
  `checkpoint_every_n_tokens > 0` 时必需。值必须单调且与 `cu_seqlens` 一致；该调用方前置条件在
  启动时**不检查**，以避免 device-to-host 同步。
- `checkpoint_every_n_tokens`（`int`，默认 `0`）：每 N 个 token 存储一个中间状态。必须是 chunk
  大小（64）的倍数。`0` 禁用 checkpointing。
- `use_cp`（`Literal["auto"] | bool`，默认 `"auto"`）：当低并行度启发式匹配时是否使用上下文并行。
  SM100/SM103 对结构上受支持的形状使用生成的 GDN CP-only 四阶段实现；其它合法配置保留
  CuTe-DSL 实现。`"auto"` 启用保守路由，`True` 要求 CP 支持，`False` 禁用 CP。
- `state_indices`（`torch.Tensor`，可选）：int32 或 int64，形状 `[num_seqs]`
  （SM90/SM100/SM103/SM120）。提供后，`initial_state` 与 `output_state` 被视为状态池，其第一维由
  这些 slot id 索引而非按序列排列：序列 `i` 从行 `state_indices[i]` 读取初始状态，并把最终状态写回
  同一行（当 `output_state is initial_state` 时原地写）。池在 SM100/SM103 上可为任意 positive、
  非重叠的 rank-4 stride 布局。`None`（默认）保持序列顺序行映射，且不要求物理视图本身连续。
  id **必须唯一**：与任何索引 scatter 一样，两个序列共享 slot id 会在不同 work tile 上并发写同一
  池行，使其最终状态不确定。唯一性是调用方前置条件（不在启动时检查，以避免每次调用 host sync）。
  `backend="cudnn"` 要求 frontend 1.31+、int32 索引以及具有稠密 `[H, V, K]` 行和 16 字节对齐 slot
  stride 的 float32/bfloat16 池。它支持相同输入/输出池视图或不相交内存，且不支持 checkpoints。
- `_cp_chunk_len`（`int`，可选）：用于测试与调优的内部上下文并行 chunk 长度覆盖。`None` 让 CP 后端
  自动选择长度；显式值必须是 64 的倍数。
- `backend`（`{"auto", "flashinfer", "cake_gdn", "cudnn"}`，默认 `"auto"`）：`auto` 使用原生
  SM90/SM100/SM120 kernel 与上下文并行路由，并可能为实测的 SM100 raw-gate 推理 workload 选择
  cuDNN。自动选择 cuDNN 要求两个 gate 融合标志、BF16 输入/gate、调用方已归一化的 Q/K、分开的紧凑
  float32 输入/输出状态、`output_final_state=True`，以及调用方拥有的 output。显式 CP、池与
  checkpoint 保留原生路由。要求 cuDNN frontend 1.31 GA 与 CuTe DSL 4.7 或更新。Cake kernel 需
  显式 `cake_gdn` 请求：SM100/SM103 上用 `backend="cake_gdn", use_cp=True` 走 Cake CP，
  `use_cp=False` 或 `"auto"` 保留 Cake non-CP。显式 Cake 请求对不支持输入失败且不回退到其它后端。
  `cudnn` 通过 `flashinfer.cudnn.cudnn_chunk_gated_delta_rule` 运行 cuDNN 的融合 SM100
  linear-attention 引擎。
- `max_seqlen`（`int`，可选）：最大逻辑序列长度的安全上界，不大于 `total_seq_len`。CP kernel 用该
  主机侧提示界定每序列启动网格，而无需从 GPU 读回 `cu_seqlens`。省略时 CP 使用 `total_seq_len`
  （对任意 batch 都正确）；传入 batched 调用的确切最大值可让 CP 启动更小网格。
- `gate_domain`（`{"linear", "log"}`，默认 `"linear"`）：预计算 `g` 的域——线性 alpha（默认）或自然
  对数 alpha。`None` 仍表示无 decay。当设置 `use_gate_in_kernel` 时被忽略，因为该标志使 `g` 成为
  原始 gate 输入。
- `use_gate_in_kernel`（`bool`，默认 `False`）：把 `g` 解释为原始 logits 并计算
  `log(alpha) = -exp(A_log) * softplus(g + dt_bias)`。cuDNN 融合该变换；其它后端物化同样的 FP32
  alpha。
- `A_log`、`dt_bias`（`torch.Tensor`，可选）：per-head `[num_sab_heads]` 参数，配合
  `use_gate_in_kernel` 时需要。dtype 为 float32 / float16 / bfloat16，在 `q` 的设备上。
- `beta_is_logit`（`bool`，默认 `False`）：在 FP32 中对 beta 施加 sigmoid，舍入到 beta 的输入
  dtype，再在 FP32 中累积。cuDNN 融合该变换；其它后端物化它。默认接受预计算的 beta。

**返回**

- `torch.Tensor` 或 `Tuple[torch.Tensor, torch.Tensor]`：当 `output_final_state=False` 时返回形状
  `[total_seq_len, num_o_heads, head_size]` 的输出张量；否则返回 `(output, final_state)`，其中
  `final_state` 形状为 `[num_seqs, num_sab_heads, head_size, head_size]`——或当给出 `state_indices`
  时为状态池 `[N_pool, ...]` 本身（即 `output_state`），其由 `state_indices` 命名的行现在持有更新
  后的最终状态。

**注意事项**

- 支持 GQA（`num_q_heads > num_k_heads = num_v_heads`）与 GVA（`num_v_heads > num_q_heads =
  num_k_heads`）。
- 最终状态布局为 `[N, H, V, K]`。
- 要求 SM90（Hopper）或 SM100（Blackwell）架构。SM100 路径要求 `head_size == 128`。在
  SM100/SM103 上，`gdn_cp` 在 CUDA 12.8、CUDA 12.9 和 CUDA 13 上支持结构合法的 equal-head、
  GQA 与 GVA 形状。其它 SM100 CP DSL 路由要求 CUDA 13 与
  `nvidia-cutlass-dsl[cu13]>=4.4.2`（`pip install flashinfer-python[cu13]`）。

**Cake GDN CP 上下文并行后端**

- 通过 `chunk_gated_delta_rule(..., backend="cake_gdn", use_cp=True)` 显式启用 Cake CP。
  `backend="auto"` 与 `backend="flashinfer"` 保持现有 FlashInfer CP 路由。使用
  `backend="cake_gdn"` 时，`use_cp=False` 或 `"auto"` 继续选择非 CP 的 Cake 后端。
- Cake GDN CP 后端覆盖完整的合法 SM100a/SM103a 公开 context-parallel 输入域，保持
  `T precompute -> MN precompute -> state fixup -> CP prefill` 流程，并支持公开 dtype、head
  mapping、可选 gate/scale、int32/int64 cu-seqlens 与 state indices、混合或全空 varlen batch、
  Q/K L2 归一化、checkpoints、output，以及任意 positive、非重叠的 packed/indexed/in-place 状态
  stride。运行时 manifest 仅记录 source inventory 与 loading 元数据。
- 在 SM100a/SM103a 上，`gdn_cp` 路由使用 checked-in CUDA 源码，在 CUDA 12.8、12.9 与 13 下支持
  FP16/BF16 输入以及 FP32/FP16/BF16 状态与 FP32 checkpoints。其它 SM100 context-parallel DSL 路由
  （含 FP8 state 或 checkpoint）要求 CUDA 13 与 `nvidia-cutlass-dsl[cu13]>=4.4.2`。
- 内部分派器缓存形状相关的 plan 与 workspace，但直接启动原生 composite，因此 API 分配的输出与
  轮转输入缓冲可改变地址而不重建 plan。固定地址的内部 prepared 对象可使用 CUDA Graph replay；
  indexed in-place 状态始终留在直接 composite 上，以免 preparation 推进 aliased 循环状态。非法
  输入与不支持的架构 fail-closed；显式 Cake CP 请求不会回退到其它 CP 实现。现有
  `chunk_gated_delta_rule` API 仍是唯一公开入口。每个 TMA-backed 阶段通过 CUDA 的
  `__grid_constant__` kernel 参数 ABI 传递其 descriptor；后端不保留进程生命周期的 descriptor arena。

**示例**

```python
import torch
import flashinfer

total_seq_len, H, D = 4096, 16, 128
seq_lens = [1024, 1024, 1024, 1024]
cu_seqlens = torch.tensor([0, 1024, 2048, 3072, 4096], device="cuda", dtype=torch.int32)

q = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
g = torch.randn(total_seq_len, H, device="cuda", dtype=torch.float32)
beta = torch.rand(total_seq_len, H, device="cuda", dtype=torch.float32)

out, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
```

## 相关链接 / 参考

- `docs/api/gdn_decode.rst`：解码侧 kernel 与状态更新。
- `docs/api/gdn2_prefill.rst`：GDN-2（channel-wise gate）prefill。
- `docs/api/gdp_prefill.rst`：Gated DeltaProduct prefill。

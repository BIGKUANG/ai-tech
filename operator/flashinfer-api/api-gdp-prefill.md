# Gated DeltaProduct 预填充（`flashinfer.gdp_prefill`）

> 源码：`flashinfer/gdp_prefill.py`
> 上游文档：`docs/api/gdp_prefill.rst`
> 适用版本：FlashInfer 0.7.2

Gated DeltaProduct（GDP）的 prefill。`chunk_gated_delta_product` 是分块 GDP scan：GDN 的每个 token
使用一个 beta 门控 Householder 更新，而 GDP 每个 token 使用 `num_householder` 个，在扩展后的
k/v/beta sub-token 时间线上进行。FlashInfer 自身不含 GDP kernel，因此该函数运行在 cuDNN 的融合
SM100 linear-attention 引擎（`flashinfer.cudnn.cudnn_chunk_gated_delta_product`）上。它需要 SM100
家族设备以及带 `cutedsl` extra 的 cudnn-frontend 1.28+；引擎会拒绝任何它无法服务的输入。

## 导入

```python
from flashinfer.gdp_prefill import chunk_gated_delta_product
# 顶层亦导出：
from flashinfer import chunk_gated_delta_product
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_gated_delta_product` | 函数 | 分块 Gated DeltaProduct prefill，经由 cuDNN 融合 SM100 引擎执行。 |

## API 详情

### `chunk_gated_delta_product`

```python
chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    num_householder: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    *,
    backend: Literal["auto", "cudnn"] = "auto",
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：GDP 把 `chunk_gated_delta_rule` 从每 token 一个 beta 门控 Householder 更新推广到
`n = num_householder` 个，每个 token 一个 per-head 标量 decay：decay 在该 token 的更新之前作用，
readout 在最后一次更新之后。对每个真实 token，sub-token 行为 `k` / `v` / `beta` 的
`t*n .. t*n + n - 1`：

```text
S              *= alpha_t
v^{new}_{t,j}   = beta_{t,j} (v_{t,j} - S k_{t,j}),  S += v^{new}_{t,j} ⊗ k_{t,j},  j = 0..n-1
o_t             = scale · S q_t
```

`num_householder == 1` 即精确的 `chunk_gated_delta_rule`。仅接受 varlen（packed / THD）输入，与
`chunk_gated_delta_rule` 一致：传入覆盖真实 token 的 `cu_seqlens`，并把所有序列的 token 首尾相连。

**参数**

- `q`（`torch.Tensor`）：形状 `[total_seq_len, num_q_heads, head_size]`，float16 或 bfloat16，位于
  真实 token 行。stride 被尊重，因此只有最内维必须连续。
- `k, v`（`torch.Tensor`）：形状 `[total_seq_len * num_householder, num_k_heads / num_v_heads,
  head_size]`，float16 或 bfloat16，位于扩展后的 sub-token 时间线上：token `t` 的 `n` 个
  Householder 更新占据行 `t*n .. t*n + n - 1`。`num_k_heads` 必须等于 `num_q_heads` 或
  `num_v_heads`。
- `g`（`torch.Tensor`，可选）：per-head forget gate，位于线性空间（`alpha`，逐元素在 `(0, 1]`），
  形状 `[total_seq_len, num_sab_heads]`，位于真实 token 行，dtype 为 float32、bfloat16 或 float16。
  `None` 时为全 1（无 decay）。
- `beta`（`torch.Tensor`，可选）：per-head、per-Householder update gate，形状 `[total_seq_len *
  num_householder, num_sab_heads]`，post-sigmoid，dtype 为 float32 或 `q.dtype`。`None` 时为全 1。
- `num_householder`（`int`，默认 `1`）：每 token 的 Householder 更新数（`n >= 1`）。
- `scale`（`float`，可选）：query scale；为 `None` 或 `0.0` 时为 `1 / sqrt(head_size)`，与
  `chunk_gated_delta_rule` 一致。
- `initial_state`（`torch.Tensor`，可选）：传入的循环状态 `[num_seqs, num_sab_heads, head_size,
  head_size]`，V-major（`[..., V, K]`），float32 或 bfloat16。`None` 时为零状态。
- `output_final_state`（`bool`，默认 `False`）：是否在输出之外返回最终循环状态。
- `cu_seqlens`（`torch.Tensor`）：累积真实 token 序列长度 `[num_seqs + 1]`，int32 或 int64。必需。
- `use_qk_l2norm_in_kernel`（`bool`，默认 `False`）：把 q/k L2 归一化融合进 kernel，而非要求输入已
  归一化。
- `output`（`torch.Tensor`，可选）：预分配输出 `[total_seq_len, num_o_heads, head_size]`，位于真实
  token 行，原地写入。`None` 时内部自动分配。
- `output_state`（`torch.Tensor`，可选）：预分配最终状态缓冲，原地写入。`None` 且
  `output_final_state` 设置时内部自动分配。在无 `state_indices` 时它不得与 `initial_state` 别名；
  kernel 会把一个序列拆到多个 CTA，读取传入状态的 CTA 会与写传出状态的 CTA 竞争。
- `backend`（`Literal["auto", "cudnn"]`，默认 `"auto"`）：FlashInfer 自身不含 GDP kernel，因此
  `"auto"`（默认）与 `"cudnn"` 都经由 `flashinfer.cudnn.cudnn_chunk_gated_delta_product` 运行
  cuDNN 的融合 SM100 linear-attention 引擎。
- `state_indices`（`torch.Tensor`，可选）：连续 int32 `[num_seqs]` slot，指向 `initial_state` 与
  `output_state` 池，要求 cuDNN frontend 1.31+。池形状为 `[N_pool, H, V, K]`，内层行稠密，slot
  stride 16 字节对齐。slot 值必须唯一且在范围内。当 `output_final_state=True` 时需要显式输出池；
  它可以是完全相同的输入视图或不相交内存。返回的状态是整个输出池，未选中 slot 不变。当为 false
  时两个池都不修改。值不会在 CPU 上读取。

**返回**

- `torch.Tensor` 或 `Tuple[torch.Tensor, torch.Tensor]`：`output`，或 `output_final_state` 时的
  `(output, final_state)`。

**注意事项**

- 需要 SM100 家族（Blackwell）设备以及带 `cutedsl` extra 的 cudnn-frontend 1.29+
  （`pip install 'nvidia-cudnn-frontend[cutedsl]'`）。
- 更细的约束——head 维度、输入 dtype、head 数关系——由引擎决定：它无法服务的图会被 cuDNN 拒绝
  （per-engine 原因记录在前端日志中）。
- 仅 varlen（packed / THD）输入；`cu_seqlens` 必需且须为 int32 或 int64。
- `k` / `v` / `beta` 位于扩展后的 sub-token 时间线上（长度为 `total_seq_len * num_householder`），
  而 `q` / `g` 位于真实 token 行（长度为 `total_seq_len`）。
- `initial_state` 与 `output_state` 的状态布局为 V-major（`[..., V, K]`），与 GDN 的最终状态布局
  `[N, H, V, K]` 一致。
- `g` 位于线性空间（`alpha`，逐元素在 `(0, 1]`），与 GDN 的 `gate_domain="linear"` 约定一致。

**返回值约定**

- `output_final_state=False` 时仅返回 `output`（`torch.Tensor`），位于真实 token 行。
- `output_final_state=True` 时返回 `(output, final_state)` 元组。
- 当提供 `state_indices` 且 `output_final_state=True` 时，返回的 `final_state` 是整个输出状态池，
  未选中的 slot 保持不变。

**示例**

```python
import torch
import flashinfer

total_seq_len, H, D, n = 2048, 16, 128, 2
cu_seqlens = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int32)

q = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(total_seq_len * n, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(total_seq_len * n, H, D, device="cuda", dtype=torch.bfloat16)
g = torch.rand(total_seq_len, H, device="cuda", dtype=torch.float32)
beta = torch.rand(total_seq_len * n, H, device="cuda", dtype=torch.bfloat16)

out, final_state = flashinfer.chunk_gated_delta_product(
    q, k, v, g, beta,
    num_householder=n,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)

# 使用状态池索引（保持分页状态池，避免 gather/scatter）
pool = torch.zeros(64, H, D, D, device="cuda", dtype=torch.float32)
state_indices = torch.arange(2, device="cuda", dtype=torch.int32)
out, pool = flashinfer.chunk_gated_delta_product(
    q, k, v, g, beta,
    num_householder=n,
    cu_seqlens=cu_seqlens,
    initial_state=pool, output_state=pool, state_indices=state_indices,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
```

## 与 GDN 的关系

- `num_householder == 1` 时，GDP 精确等于 `chunk_gated_delta_rule`，sub-token 时间线与真实 token
  时间线重合。
- 每个真实 token 的 `n` 个更新共享同一个 per-head 标量 decay `alpha_t`：decay 在该 token 的所有
  Householder 更新之前施加一次，输出在最后一次更新之后读出。
- `g` / `beta` 为 `None` 时分别按全 1（无 decay）/ 全 1 处理。

## 相关链接 / 参考

- `docs/api/gdn_prefill.rst`：`num_householder == 1` 对应的 GDN 分块 prefill。
- `docs/api/gdn2_prefill.rst`：GDN-2（channel-wise gate）prefill。

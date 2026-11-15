# Gated Delta-Rule 2 预填充（`flashinfer.gdn2_prefill`）

> 源码：`flashinfer/gdn2_prefill.py`
> 上游文档：`docs/api/gdn2_prefill.rst`
> 适用版本：FlashInfer 0.7.2

Gated Delta-Rule 2（GDN-2）的 prefill。`chunk_gated_delta_rule2` 是分块 GDN-2 scan：GDN 的
per-head 标量 forget / erase gate 变为 per-key-channel，并额外用一个 per-value-channel 的 write
gate 缩放传入的 value。FlashInfer 自身不含 GDN-2 kernel，因此该函数运行在 cuDNN 的融合 SM100
linear-attention 引擎（`flashinfer.cudnn.cudnn_chunk_gated_delta_rule2`）上。它需要 SM100 家族设备
以及带 `cutedsl` extra 的 cudnn-frontend 1.28+；引擎会拒绝任何它无法服务的输入。

## 导入

```python
from flashinfer.gdn2_prefill import chunk_gated_delta_rule2
# 顶层亦导出：
from flashinfer import chunk_gated_delta_rule2
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_gated_delta_rule2` | 函数 | 分块 GDN-2 prefill（channel-wise gate），经由 cuDNN 融合 SM100 引擎执行。 |

## API 详情

### `chunk_gated_delta_rule2`

```python
chunk_gated_delta_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
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

**功能**：GDN-2 把 `chunk_gated_delta_rule` 的 per-head 标量 gate 推广到 channel-wise：forget gate
`g` 与 erase gate `beta` 变为 per key channel，第三个 gate `w` 按 value channel 缩放传入 value。
其递推为：

```text
S_t        = diag(e^{g_t}) S_{t-1}
v^{new}_t  = w_t ⊙ v_t - (beta_t ⊙ k_t)^T S_t
S_t       += k_t ⊗ v^{new}_t
o_t        = scale · q_t^T S_t
```

把 `g`、`beta`、`w` 设为 per-channel 常数即可退回 GDN：`w ≡ beta` 且 channel 常量 `g` 使更新成为
标量门控 delta 规则。仅接受 varlen（packed / THD）输入，与 `chunk_gated_delta_rule` 一致：传入
`cu_seqlens`，并把所有序列的 token 首尾相连。

**参数**

- `q, k, v`（`torch.Tensor`）：形状 `[total_seq_len, num_q_heads / num_k_heads / num_v_heads,
  head_size]`，float16 或 bfloat16。`num_k_heads` 必须等于 `num_q_heads` 或 `num_v_heads`，且
  query 与 value head 数须相等或互为倍数。stride 被尊重，因此只有最内维必须连续。
- `g`（`torch.Tensor`，可选）：channel-wise forget gate，作为自然对数 decay（`ln alpha`，逐元素
  `<= 0`），形状 `[total_seq_len, num_sab_heads, head_size]`，dtype 为 float32、bfloat16 或
  float16。`None` 时为全零（无 decay）。
- `beta`（`torch.Tensor`，可选）：channel-wise erase gate，形状 `[total_seq_len, num_sab_heads,
  head_size]`，post-sigmoid，按 `q.dtype` 读取。`None` 时为全 1。
- `w`（`torch.Tensor`，可选）：channel-wise write gate，形状 `[total_seq_len, num_sab_heads,
  head_size]`，按 `q.dtype` 读取。`None` 时为全 1。
- `scale`（`float`，可选）：query scale；为 `None` 或 `0.0` 时为 `1 / sqrt(head_size)`，与
  `chunk_gated_delta_rule` 一致。
- `initial_state`（`torch.Tensor`，可选）：传入的循环状态 `[num_seqs, num_sab_heads, head_size,
  head_size]`，V-major（`[..., V, K]`），float32 或 bfloat16。`None` 时为零状态。
- `output_final_state`（`bool`，默认 `False`）：是否在输出之外返回最终循环状态。
- `cu_seqlens`（`torch.Tensor`）：累积序列长度 `[num_seqs + 1]`，int32 或 int64。必需。
- `use_qk_l2norm_in_kernel`（`bool`，默认 `False`）：把 q/k L2 归一化融合进 kernel，而非要求输入已
  归一化。
- `output`（`torch.Tensor`，可选）：预分配输出 `[total_seq_len, num_o_heads, head_size]`，原地写入。
  `None` 时内部自动分配。
- `output_state`（`torch.Tensor`，可选）：预分配最终状态缓冲，原地写入。`None` 且
  `output_final_state` 设置时内部自动分配。在无 `state_indices` 时它不得与 `initial_state` 别名；
  kernel 会把一个序列拆到多个 CTA，读取传入状态的 CTA 会与写传出状态的 CTA 竞争。
- `backend`（`Literal["auto", "cudnn"]`，默认 `"auto"`）：FlashInfer 自身不含 GDN-2 kernel，因此
  `"auto"`（默认）与 `"cudnn"` 都经由 `flashinfer.cudnn.cudnn_chunk_gated_delta_rule2` 运行 cuDNN
  的融合 SM100 linear-attention 引擎。
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
- `initial_state` 与 `output_state` 的状态布局为 V-major（`[..., V, K]`），与 GDN 的最终状态布局
  `[N, H, V, K]` 一致。
- 门控的 dtype 约定与 GDN 略有不同：`g`、`beta`、`w` 都是 per-channel（形状含 `head_size`），
  而非 per-head 标量；`g` 为自然对数 decay 且逐元素 `<= 0`，`beta` 为 post-sigmoid，`w` 按
  `q.dtype` 读取。
- 当 `use_qk_l2norm_in_kernel=False`（默认）时，调用方须自行保证 Q/K 已归一化；cuDNN 引擎不会
  代替调用方归一化。

**返回值约定**

- `output_final_state=False` 时仅返回 `output`（`torch.Tensor`）。
- `output_final_state=True` 时返回 `(output, final_state)` 元组。
- 当提供 `state_indices` 且 `output_final_state=True` 时，返回的 `final_state` 是整个输出状态池，
  未选中的 slot 保持不变。

**示例**

```python
import torch
import flashinfer

total_seq_len, H, D = 4096, 16, 128
cu_seqlens = torch.tensor([0, 2048, 4096], device="cuda", dtype=torch.int32)

q = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
g = -torch.rand(total_seq_len, H, D, device="cuda", dtype=torch.float32)
beta = torch.rand(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)
w = torch.rand(total_seq_len, H, D, device="cuda", dtype=torch.bfloat16)

out, final_state = flashinfer.chunk_gated_delta_rule2(
    q, k, v, g, beta, w,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)

# 使用状态池索引（保持分页状态池，避免 gather/scatter）
pool = torch.zeros(64, H, D, D, device="cuda", dtype=torch.float32)
state_indices = torch.arange(2, device="cuda", dtype=torch.int32)
out, pool = flashinfer.chunk_gated_delta_rule2(
    q, k, v, g, beta, w,
    cu_seqlens=cu_seqlens,
    initial_state=pool, output_state=pool, state_indices=state_indices,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
```

## 与 GDN 的关系

- 令 `w ≡ beta` 且 `g` 为 channel 常量，GDN-2 的更新即退化为 `chunk_gated_delta_rule` 的标量门控
  delta 规则。
- `g` / `beta` / `w` 为 `None` 时分别按全零（无 decay）/ 全 1 / 全 1 处理，此时行为退化为无门控
  的线性注意力更新。

## 相关链接 / 参考

- `docs/api/gdn_prefill.rst`：GDN 分块 prefill（per-head 标量 gate）。
- `docs/api/gdp_prefill.rst`：Gated DeltaProduct prefill。

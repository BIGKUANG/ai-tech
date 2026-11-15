# cuDNN 后端（`flashinfer.cudnn`）

> 源码：`flashinfer/cudnn/__init__.py`、`flashinfer/cudnn/decode.py`、
> `flashinfer/cudnn/prefill.py`、`flashinfer/cudnn/linear_attention.py`、`flashinfer/cudnn/utils.py`
> 上游文档：`docs/api/cudnn.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.cudnn` 提供由 NVIDIA cuDNN 运行时支撑的注意力内核。这些封装调用 cuDNN 的
batch prefill 与 batch decode，可在主机 GPU 支持 cuDNN 时通过
`BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper` 上的
`backend="cudnn"`（或直接调用）访问。decode 封装后端覆盖 fp16/bf16 GQA（含 `return_lse`）、
CUDA Graph、多 token decode（`q_len_per_req > 1`，bottom-right causal）、左滑动窗口
（`window_left`）与 attention `sinks`；不支持 RoPE、soft-cap 或 fp8/NVFP4 KV。线性注意力
部分提供 cuDNN 融合 SM100 linear-attention 引擎，可在 `flashinfer.chunk_gated_delta_rule`、
`chunk_gated_delta_rule2`、`chunk_gated_delta_product` 与 `flashinfer.recurrent_kda` 上以
`backend="cudnn"` 访问。

## 导入

```python
from flashinfer.cudnn import (
    cudnn_batch_decode_with_kv_cache,
    cudnn_batch_prefill_with_kv_cache,
    cudnn_chunk_gated_delta_product,
    cudnn_chunk_gated_delta_rule,
    cudnn_chunk_gated_delta_rule2,
    cudnn_recurrent_kda,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `cudnn_batch_decode_with_kv_cache` | 函数 | 基于 cuDNN SDPA 的分页 KV batch decode |
| `cudnn_batch_prefill_with_kv_cache` | 函数 | 基于 cuDNN SDPA 的分页 KV batch prefill |
| `cudnn_chunk_gated_delta_rule` | 函数 | cuDNN 融合 SM100 引擎上的 Chunked Gated Delta Rule prefill（GDN） |
| `cudnn_chunk_gated_delta_product` | 函数 | cuDNN 融合 SM100 引擎上的 Chunked Gated DeltaProduct prefill（GDP） |
| `cudnn_chunk_gated_delta_rule2` | 函数 | cuDNN 融合 SM100 引擎上的 Chunked Gated Delta Rule 2 prefill（GDN-2） |
| `cudnn_recurrent_kda` | 函数 | cuDNN 融合 SM100 引擎上的 Kimi Delta Attention prefill（KDA） |

## API 详情

### `cudnn_batch_decode_with_kv_cache`

```python
cudnn_batch_decode_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_sequence_kv: int,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse: Optional[torch.Tensor] = None,
    q_len_per_req: int = 1,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
```

**功能**：由 cuDNN SDPA 支撑的、带分页 KV cache 的批量 decode 注意力。

**参数**

- `q`（`torch.Tensor`）：query，形状 `(batch_size * q_len_per_req, num_heads_qo, head_dim)`
  （普通单 token decode 时为 `(batch_size, num_heads_qo, head_dim)`），`torch.float16` 或
  `torch.bfloat16`（输出使用 `q.dtype`）。`q_len_per_req > 1` 时，同一请求的行是连续的。
  `torch.float16` 要求 cuDNN graph 后端；回退（cubin）路径仅支持 bf16，否则抛
  `NotImplementedError`。
- `k_cache`（`torch.Tensor`）：key cache，形状 `(total_num_pages, num_heads_kv, page_size, head_dim)`。
- `v_cache`（`torch.Tensor`）：value cache，形状 `(total_num_pages, num_heads_kv, page_size, head_dim)`。
- `scale`（`float`）：softmax 缩放因子，通常为 `1 / sqrt(head_dim)`。
- `workspace_buffer`（`torch.Tensor`）：cuDNN 工作区，随 batch 增大；典型 decode 负载 128 MB 足够。
- `max_sequence_kv`（`int`）：batch 中每个 KV 序列的最大 token 数（`s_kv_max`）。
- `actual_seq_lens_kv`（`Optional[torch.Tensor]`）：每个请求的 KV 长度，形状 `(batch_size,)`。
  cuDNN 可用（默认后端）时该张量必须与 `q` 位于同一 CUDA 设备；仅回退的非 cuDNN 路径接受
  （并内部拷贝）CPU 张量。
- `block_tables`（`Optional[torch.Tensor]`）：分页 KV cache 的 page table，形状
  `(batch_size, num_pages_per_seq)`，位于 GPU。
- `is_cuda_graph_compatible`（`bool`）：是否以 CUDA Graph 捕获安全模式规划操作。
- `batch_offsets_q`（`Optional[torch.Tensor]`）：每个请求在 query 张量中的元素偏移，int32 或
  int64，形状 `(batch_size,)` 或 `(batch_size + 1,)`（可选末端偏移），位于 GPU。cuDNN graph
  路径只接受与 query 的 batch stride 匹配的 dense 偏移。建议传 `None` 以避免冗余的设备侧检查。
- `batch_offsets_o`（`Optional[torch.Tensor]`）：同 `batch_offsets_q`，但针对连续输出张量。
  在 cuDNN graph 路径上，非 dense 的 Q/O 偏移会触发异步设备断言，包括在 CUDA Graph 重放前修改的情况。
- `batch_offsets_k`（`Optional[torch.Tensor]`）：每个请求在 key 张量中的偏移，形状 `(batch_size,)`，位于 GPU。
- `batch_offsets_v`（`Optional[torch.Tensor]`）：每个请求在 value 张量中的偏移，形状 `(batch_size,)`，位于 GPU。
- `out`（`Optional[torch.Tensor]`）：预分配输出，形状 `(batch_size, num_heads_qo, head_dim)`，
  dtype 为 `q.dtype`；为 `None` 时内部申请。
- `return_lse`（`bool`）：是否同时返回注意力分数的 log-sum-exp（cuDNN SDPA 的 `Stats` 输出）。
  需要 cuDNN graph 后端；回退（cubin）路径会抛 `NotImplementedError`。
- `lse`（`Optional[torch.Tensor]`）：预分配 LSE，形状 `(batch_size * q_len_per_req, num_heads_qo)`，
  `torch.float32`、连续、与 `q` 同设备；`return_lse=True` 且为 `None` 时内部申请。
- `q_len_per_req`（`int`）：每个请求的 query 行数（投机解码 / 多 token 预测校验）。同一请求的
  行按 bottom-right causal 对角线参与注意力：某个 KV 长度 `kv_len` 的请求，其第 `i` 行看到
  key `0 .. kv_len - q_len_per_req + i`。每个请求需 `kv_len >= q_len_per_req`。默认 `1`（除
  padding 外无掩码）。
- `window_left`（`int`）：FlashInfer 约定下的左滑动窗口边界：某行注意到其对角位置之前的
  `window_left` 个 key 以及该位置本身；`-1`（默认）关闭窗口。
- `sinks`（`Optional[torch.Tensor]`）：每 head 的 attention sink logits，形状 `(num_heads_qo,)`，
  `torch.float32`，位于 `q` 的设备。`sinks[h]` 作为一个零值行的额外 logit 加入每行的 softmax
  分母，与 FlashInfer 其他后端（gpt-oss / Streaming-LLM sinks）一致。cuDNN 栈在
  `q_len_per_req == 1` 时是否服务 sink 由其 SDPA 引擎决定（cudnn-frontend 1.30+ 且启用 FROST
  引擎时可服务；backend 引擎在 graph 构建时抛 not-supported 错误）。

**返回**

- `Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]`：`return_lse=False` 时返回形状
  `(batch_size * q_len_per_req, num_heads_qo, head_dim)` 的输出张量；否则返回 `(output, lse)`，
  其中 `lse` 形状为 `(batch_size * q_len_per_req, num_heads_qo)`、dtype 为 `torch.float32`。

**注意事项**

- 所有张量必须位于同一 CUDA 设备。`q` 可以带任意 batch/head stride（例如 packed QKV 投影的
  切片），只要 `head_dim` 是最内层且连续；`out`/`lse` 必须连续。query 与 KV head 数可以不同
  （`num_heads_qo >= num_heads_kv`，MQA / GQA）。
- LSE 约定：`lse[b, h]` 是折叠了 `scale` 的 pre-softmax 注意力行的 **base-2** log-sum-exp，
  即对有效 KV 位置 `j < actual_seq_lens_kv[b]` 求和，契约与其他 FlashInfer 后端一致
  （`torch.logsumexp(...) * log2(e)`），可馈入 cascade-merge 内核。cuDNN 输出自然对数 stats，
  在此被折为 base-2。

**示例**

```python
import torch
from flashinfer.cudnn import cudnn_batch_decode_with_kv_cache

out = cudnn_batch_decode_with_kv_cache(
    q, k_cache, v_cache, scale,
    workspace_buffer,
    max_sequence_kv=8192,
    actual_seq_lens_kv=seq_lens,
    block_tables=block_tables,
    return_lse=False,
)
```

---

### `cudnn_batch_prefill_with_kv_cache`

```python
cudnn_batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    causal: bool,
    return_lse: bool,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    batch_offsets_stats: Optional[torch.Tensor] = None,
    batch_offsets_units: str = "elements",
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    backend: Optional[str] = None,
    o_data_type: Optional[torch.dtype] = None,
    lse_base: str = "log2",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：由 cuDNN SDPA 支撑的、带分页 KV cache 的批量 prefill 注意力。

**参数**

- `q`（`torch.Tensor`）：packed query，形状 `(total_qo_tokens, num_heads_qo, head_dim_qk)`。
- `k_cache`（`torch.Tensor`）：key cache。分页时为
  `(total_num_pages, num_heads_kv, page_size, head_dim_qk)`；否则
  `(total_kv_tokens, num_heads_kv, head_dim_qk)`。
- `v_cache`（`torch.Tensor`）：value cache。分页时为
  `(total_num_pages, num_heads_kv, page_size, head_dim_vo)`；否则
  `(total_kv_tokens, num_heads_kv, head_dim_vo)`。
- `scale`（`float`）：softmax 缩放因子，通常为 `1 / sqrt(head_dim_qk)`。
- `workspace_buffer`（`torch.Tensor`）：cuDNN 工作区；典型 prefill 负载 128 MB 足够。
- `max_token_per_sequence`（`int`）：每个 query 序列的最大 token 数（`s_qo_max`）。
- `max_sequence_kv`（`int`）：每个 KV 序列的最大 token 数（`s_kv_max`）。
- `actual_seq_lens_q`（`Optional[torch.Tensor]`）：每个请求的 query 长度，形状 `(batch_size, 1, 1, 1)`。
  cuDNN 可用时须与 `q` 同设备；仅回退的非 cuDNN 路径接受 CPU 张量，且该回退路径也是
  `is_cuda_graph_compatible=False` 时唯一要求 CPU 张量的路径。可在
  `batch_offsets_units="tokens"` 且设置 `batch_offsets_q` 时省略（长度由 indptr 隐含）；
  cubin 后端始终要求提供。
- `actual_seq_lens_kv`（`Optional[torch.Tensor]`）：每个请求的 KV 长度，形状 `(batch_size, 1, 1, 1)`。
  设备规则同 `actual_seq_lens_q`。可在 `batch_offsets_units="tokens"` 且设置
  `batch_offsets_k`（非分页 KV）时省略；cubin 后端始终要求提供。
- `block_tables`（`Optional[torch.Tensor]`）：分页 KV block table，形状
  `(batch_size, num_pages_per_seq)`，位于 GPU；非分页 KV 布局传 `None`。
- `causal`（`bool`）：是否应用 causal 掩码。
- `return_lse`（`bool`）：是否返回 log-sum-exp（cubin 后端当前必须为 `True`）。
- `q_scale`（`Optional[torch.Tensor]`）：query 的 FP8 反量化 scale，形状 `(1, 1, 1, 1)`，位于 GPU。
- `k_scale`（`Optional[torch.Tensor]`）：key 的 FP8 反量化 scale，形状 `(1, 1, 1, 1)`，位于 GPU。
- `v_scale`（`Optional[torch.Tensor]`）：value 的 FP8 反量化 scale，形状 `(1, 1, 1, 1)`，位于 GPU。
- `batch_offsets_q`（`Optional[torch.Tensor]`）：packed query 中每个请求的累计起始偏移，形状
  `(batch_size + 1,)`，int32，位于 GPU，单位由 `batch_offsets_units` 给出（元素偏移按 `q`
  自身存储计：`token_offset * q.stride(0)`，对连续 `q` 即 `token_offset * num_heads_qo * head_dim_qk`）。
  cuDNN graph 路径上 `batch_size > 1` 时必需；仅 `batch_size == 1` 时可省略。
- `batch_offsets_o`（`Optional[torch.Tensor]`）：packed 输出张量中每个请求的累计起始偏移，形状
  `(batch_size + 1,)`，int32，位于 GPU，单位由 `batch_offsets_units` 给出（元素偏移为
  `token_offset * num_heads_qo * head_dim_vo`；`out` 连续）。cuDNN graph 路径上
  `batch_size > 1` 时必需；`batch_offsets_units="tokens"` 时默认取 `batch_offsets_q`。
- `batch_offsets_k`（`Optional[torch.Tensor]`）：key 张量中每个请求的累计起始偏移，形状
  `(batch_size + 1,)`，位于 GPU，单位由 `batch_offsets_units` 给出；仅用于非分页（3-D）KV。
- `batch_offsets_v`（`Optional[torch.Tensor]`）：value 张量中每个请求的累计起始偏移，形状
  `(batch_size + 1,)`，位于 GPU，单位由 `batch_offsets_units` 给出；仅用于非分页（3-D）KV；
  `batch_offsets_units="tokens"` 时默认取 `batch_offsets_k`。
- `batch_offsets_stats`（`Optional[torch.Tensor]`）：packed LSE 张量中每个请求的累计起始偏移，
  形状 `(batch_size + 1,)`，单位由 `batch_offsets_units` 给出（元素偏移为
  `token_offset * num_heads_qo`）。省略且 LSE 为 packed 时由 `batch_offsets_q` 推导；与
  padded LSE 一起使用时被拒绝。
- `batch_offsets_units`（`str`，默认 `"elements"`）：`batch_offsets_*` 张量的单位。
  `"elements"`（默认，历史行为）：偏移为预先按张量元素缩放的偏移，例如 query 的
  `cumsum(seq_lens) * num_heads * head_dim`。`"tokens"`：偏移为纯 token 单位前缀和
  （`qo_indptr`/`kv_indptr` 风格，FlashInfer 约定）。对非分页 KV，在 cuDNN 后端 9.24+ 且
  cudnn-frontend 1.25+（fp16/bf16）或后端 9.25+ 且 cudnn-frontend 1.27+（fp8）时，token 单位
  indptr 由内核直接消费、无转换前置遍；否则 FlashInfer 内部将其缩放为元素单位。
- `out`（`Optional[torch.Tensor]`）：预分配连续输出，位于 `q` 设备，形状
  `(total_qo_tokens, num_heads_qo, head_dim_vo)`；为 `None` 时内部申请。
- `lse`（`Optional[torch.Tensor]`）：预分配连续 float32 LSE，位于 `q` 设备，可为 packed
  `(total_qo_tokens, num_heads_qo)`（每 query token 一行，FlashInfer 约定）或 padded
  `(batch_size, max_token_per_sequence, num_heads_qo)`。为 `None` 且 `return_lse=True` 时，
  cuDNN graph 路径申请 packed、`"cubin"` 后端申请 padded（其只写该形式）。
- `is_cuda_graph_compatible`（`bool`）：是否以 CUDA Graph 捕获安全模式规划操作。
- `backend`（`Optional[str]`）：可选的 cuDNN 后端选择器（如 `"cubin"`）。为 `None` 时按 cuDNN
  可用性自动探测。
- `o_data_type`（`Optional[torch.dtype]`）：可选输出 dtype，默认 `q.dtype`。
- `lse_base`（`str`，默认 `"log2"`）：`"log2"` 返回 base-2 LSE，与其他 FlashInfer 后端一致；
  `"ln"` 返回 cuDNN 原生自然对数 stats，跳过转换内核。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor]]`：`(output, lse)`。`output` 形状
  `(total_qo_tokens, num_heads_qo, head_dim_vo)`；`return_lse=True` 时 `lse` 为 packed
  `(total_qo_tokens, num_heads_qo)`（或调用方提供的 padded buffer，或 cubin 后端的 padded 形式），
  否则为 `None`。

**注意事项**

- query 与 KV head 数可以不同（`num_heads_qo >= num_heads_kv`，MQA / GQA）。
- 使用 CUDA Graph 捕获时，`actual_seq_lens_q` 与 `actual_seq_lens_kv` 必须与 `q` 同设备。
- `head_dim_qk` 必须是 128 或 192，`head_dim_vo` 必须是 128。
- `lse` 若非 `None` 必须为 float32、连续、与 `q` 同设备，且形状为 packed 或 padded 之一。

**示例**

```python
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache

output, lse = cudnn_batch_prefill_with_kv_cache(
    q, k_cache, v_cache, scale, workspace_buffer,
    max_token_per_sequence=4096,
    max_sequence_kv=4096,
    actual_seq_lens_q=seq_lens_q,
    actual_seq_lens_kv=seq_lens_kv,
    block_tables=block_tables,
    causal=True,
    return_lse=True,
)
```

---

### `cudnn_chunk_gated_delta_rule`

```python
cudnn_chunk_gated_delta_rule(
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
    batch_invariant: bool = False,
    state_indices: Optional[torch.Tensor] = None,
    *,
    gate_domain: str = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
```

**功能**：cuDNN 融合 SM100 引擎上的 Chunked Gated Delta Rule（GDN）prefill。参数含义与
`flashinfer.chunk_gated_delta_rule` 一致。需要 cudnn-frontend 1.31.0+ 并带 `cutedsl` extra；
其余由引擎自行判定（无法服务的 graph 会被拒绝，per-engine 原因记录在前端日志中）。

**参数**

- `q, k, v`（`torch.Tensor`）：`[total_seq_len, num_q_heads / num_k_heads / num_v_heads, 128]`，
  packed。stride 透传给 cuDNN，因此只有最内层维度必须连续。
- `g`（`torch.Tensor`，可选）：线性空间的逐 head forget gate（`alpha = exp(log_g)`），形状
  `[total_seq_len, num_sab_heads]`，按 `g` 自身 dtype 原样传给 cuDNN（`gate_domain="linear"`），
  cuDNN 可读 float32、bfloat16 或 float16。为 `None` 时全为 1。
- `beta`（`torch.Tensor`，可选）：逐 head update gate `[total_seq_len, num_sab_heads]`，
  post-sigmoid，float32 或 `q.dtype`。为 `None` 时全为 1。
- `scale`（`float`，可选）：query scale；为 `None` 或 `0.0` 时取 `1 / sqrt(head_dim)`，与原生
  GDN 路径一致。
- `initial_state, output_state`（`torch.Tensor`，可选）：状态 `[num_seqs, num_sab_heads, 128, 128]`，
  V-major，float32 或 bfloat16。cuDNN 使用相同布局，因此原样不透传转置，`output_state` 由内核
  原地写。无 `state_indices` 时 `output_state` 不得与 `initial_state` 别名：引擎把一个序列拆到
  多个 CTA 上，读入状态的 chunk-0 CTA 会与写出的最后一个 chunk CTA 竞争。此与
  `state_indices` 依赖的 slot 唯一性一样，是调用方前置条件而非启动期检查。
- `output_final_state`（`bool`）：是否随输出返回传出的循环状态；未设置时不写任何状态。
- `cu_seqlens`（`torch.Tensor`）：`[num_seqs + 1]`，int32 或 int64。必需。
- `use_qk_l2norm_in_kernel`（`bool`）：在前端内以加性 epsilon `1e-6` 归一化 Q/K 并 round 回
  输入 dtype，与公开 GDN API 一致。
- `output`（`torch.Tensor`，可选）：预分配 `[total_seq_len, num_o_heads, 128]`，由内核原地写。
- `batch_invariant`（`bool`）：禁用 split-K 分区，使归约顺序（从而结果）不依赖序列的批处理方式。
  在少量长序列上牺牲 split-K 带来的并行度，在大量短序列上省去其固定调度开销。
- `state_indices`（`torch.Tensor`，可选）：`q` 设备上连续的 int32 `[num_seqs]` pool slot，要求
  cuDNN frontend 1.31+。两个状态均为 pool `[N_pool, H, V, K]`，内层行 dense、slot stride
  16 字节对齐。slot id 必须唯一且在范围内；这些设备值是调用方前置条件、不会被读回。返回状态
  需要一个形状/dtype 相同的显式输出 pool，既可以是相同的输入 view（安全的原地 plan），也可以是
  不相交内存。未被选中的行保持不变。`output_final_state=False` 时不写 pool。
- `gate_domain`（`{"linear", "log"}`，默认 `"linear"`）：预计算的 decay 域，与公开 GDN API 一致。
- `use_gate_in_kernel`（`bool`）：从原始 g logits 融合 `-exp(A_log) * softplus(g + dt_bias)`。
  需要逐 head 的 `A_log` 与 `dt_bias`；此时 `gate_domain` 不适用。
- `A_log, dt_bias`（`torch.Tensor`，可选）：`q` 设备上逐 head 的 `[num_sab_heads]` 原始 gate 参数。
- `beta_is_logit`（`bool`）：在 FP32 使用前融合 sigmoid 并 round 到 `beta` 的输入 dtype。

**返回**

- `torch.Tensor` 或 `Tuple[torch.Tensor, torch.Tensor]`：`output`，或在 `output_final_state`
  时返回 `(output, final_state)`。

**注意事项**

- `cu_seqlens` 必需，为 `None` 时抛 `ValueError`。
- 需要 cudnn-frontend 1.31.0+ 并带 `cutedsl` extra。

**示例**

```python
import torch
from flashinfer.cudnn import cudnn_chunk_gated_delta_rule

cu_seqlens = torch.tensor([0, 2048, 4096], dtype=torch.int32, device="cuda")
out = cudnn_chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu_seqlens)
```

---

### `cudnn_chunk_gated_delta_product`

```python
cudnn_chunk_gated_delta_product(
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
    batch_invariant: bool = False,
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
```

**功能**：cuDNN 融合 SM100 引擎上的 Chunked Gated DeltaProduct（GDP）prefill。GDP 对每个 token
应用 `n = num_householder` 次 beta-gated Householder 更新、每 token 一个逐 head 标量 decay：
即 GDN 递推在扩展 sub-token 时间线上的形式，decay 作用于该 token 的更新之前、readout 在最后一次
更新之后。`num_householder == 1` 即等价于 `cudnn_chunk_gated_delta_rule`。需要 cudnn-frontend
1.31.0+ 并带 `cutedsl` extra。

**参数**

- `q`（`torch.Tensor`）：`[total_seq_len, num_q_heads, head_size]`，在真实 token 行上 packed。
  stride 透传给 cuDNN，只有最内层需连续。
- `k, v`（`torch.Tensor`）：`[total_seq_len * num_householder, num_k_heads / num_v_heads, head_size]`，
  在扩展 sub-token 时间线上 packed：token `t` 的 `n` 次 Householder 更新占据行
  `t*n .. t*n + n - 1`。`num_k_heads` 必须等于 `num_q_heads` 或 `num_v_heads`。
- `g`（`torch.Tensor`，可选）：线性空间逐 head forget gate（`alpha = exp(log_g)`），形状
  `[total_seq_len, num_sab_heads]`，在真实 token 行上，按 `g` 自身 dtype 原样传给 cuDNN
  （`gate_domain="linear"`）。为 `None` 时全为 1。
- `beta`（`torch.Tensor`，可选）：逐 head、逐 Householder 的 update gate
  `[total_seq_len * num_householder, num_sab_heads]`，post-sigmoid，float32 或 `q.dtype`。
  为 `None` 时全为 1。
- `num_householder`（`int`，默认 `1`）：每 token 的 Householder 更新数（`n >= 1`）。
- `scale`（`float`，可选）：query scale；为 `None` 或 `0.0` 时取 `1 / sqrt(head_size)`。
- `initial_state, output_state`（`torch.Tensor`，可选）：状态 `[num_seqs, num_sab_heads, head_size, head_size]`，
  V-major，float32 或 bfloat16。无 `state_indices` 时 `output_state` 不得与 `initial_state` 别名；
  见 `cudnn_chunk_gated_delta_rule`。
- `output_final_state`（`bool`）：是否随输出返回传出的循环状态；未设置时不写任何状态。
- `cu_seqlens`（`torch.Tensor`）：`[num_seqs + 1]`，int32 或 int64，基于真实 token。必需。
- `use_qk_l2norm_in_kernel`（`bool`）：把 q/k 的 L2 归一化融合进内核。
- `output`（`torch.Tensor`，可选）：预分配 `[total_seq_len, num_o_heads, head_size]`，在真实 token
  行上，由内核原地写。
- `batch_invariant`（`bool`）：禁用 split-K 分区；见 `cudnn_chunk_gated_delta_rule`。
- `state_indices`（`torch.Tensor`，可选）：连续的 int32 `[num_seqs]` pool slot，要求 cuDNN
  frontend 1.31+。状态 pool 使用 `[N_pool, H, V, K]`，内层行 dense、slot stride 对齐。返回状态
  需要显式输出 pool（相同输入 view 或不相交内存）。slot 必须唯一且在范围内，值不被读回。
  未选中的行保持不变；`output_final_state=False` 时不写 pool。见
  `cudnn_chunk_gated_delta_rule`。

**返回**

- `torch.Tensor` 或 `Tuple[torch.Tensor, torch.Tensor]`：`output`，或在 `output_final_state`
  时返回 `(output, final_state)`。

**注意事项**

- `cu_seqlens` 必需，为 `None` 时抛 `ValueError`。

**示例**

```python
import torch
from flashinfer.cudnn import cudnn_chunk_gated_delta_product

cu_seqlens = torch.tensor([0, 1024, 2048], dtype=torch.int32, device="cuda")
out = cudnn_chunk_gated_delta_product(q, k, v, g, beta, num_householder=4,
                                      cu_seqlens=cu_seqlens)
```

---

### `cudnn_chunk_gated_delta_rule2`

```python
cudnn_chunk_gated_delta_rule2(
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
    batch_invariant: bool = False,
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
```

**功能**：cuDNN 融合 SM100 引擎上的 Chunked Gated Delta Rule 2（GDN-2）prefill。参数含义与
`flashinfer.chunk_gated_delta_rule2` 一致。GDN-2 将 GDN 的逐 head 标量 gate 泛化为 channel-wise
gate：forget gate `g` 与 erase gate `beta` 是逐 key channel，write gate `w` 是逐 value channel。
递推为：

```
S_t   = diag(e^{g_t}) S_{t-1}
v_new = w_t ⊙ v_t - (beta_t ⊙ k_t)^T S_t
S_t  += k_t ⊗ v_new
o_t   = scale * q_t^T S_t
```

需要 cudnn-frontend 1.31.0+ 并带 `cutedsl` extra。

**参数**

- `q, k, v`（`torch.Tensor`）：`[total_seq_len, num_q_heads / num_k_heads / num_v_heads, 128]`，packed。
- `g`（`torch.Tensor`，可选）：channel-wise forget gate，作为自然对数 decay，形状
  `[total_seq_len, num_sab_heads, 128]`，按 `g` 自身 dtype 原样传给 cuDNN（`gate_domain="log"`）。
  为 `None` 时全为 0。
- `beta`（`torch.Tensor`，可选）：channel-wise erase gate `[total_seq_len, num_sab_heads, 128]`，
  转换为 `q.dtype`。为 `None` 时全为 1。
- `w`（`torch.Tensor`，可选）：channel-wise write gate `[total_seq_len, num_sab_heads, 128]`，
  转换为 `q.dtype`。为 `None` 时全为 1。
- `scale`（`float`，可选）：query scale；为 `None` 或 `0.0` 时取 `1 / sqrt(head_dim)`。
- `initial_state, output_state`（`torch.Tensor`，可选）：状态 `[num_seqs, num_sab_heads, 128, 128]`，
  V-major，float32 或 bfloat16。无 `state_indices` 时 `output_state` 不得与 `initial_state` 别名；
  见 `cudnn_chunk_gated_delta_rule`。
- `output_final_state`（`bool`）：是否随输出返回传出的循环状态；未设置时不写任何状态。
- `cu_seqlens`（`torch.Tensor`）：`[num_seqs + 1]`，int32 或 int64。必需。
- `use_qk_l2norm_in_kernel`（`bool`）：把 q/k 的 L2 归一化融合进内核。
- `output`（`torch.Tensor`，可选）：预分配 `[total_seq_len, num_o_heads, 128]`。
- `batch_invariant`（`bool`）：禁用 split-K 分区；见 `cudnn_chunk_gated_delta_rule`。
- `state_indices`（`torch.Tensor`，可选）：连续 int32 `[num_seqs]` pool slot，要求 cuDNN
  frontend 1.31+。语义同 `cudnn_chunk_gated_delta_rule`。

**返回**

- `torch.Tensor` 或 `Tuple[torch.Tensor, torch.Tensor]`：`output`，或在 `output_final_state`
  时返回 `(output, final_state)`。

**注意事项**

- `cu_seqlens` 必需，为 `None` 时抛 `ValueError`。
- GDN-2 与 GDP 自身没有 FlashInfer 内核，其 `"auto"` 会解析到 cudnn；而 GDN、KDA 不会被隐式选择
  cudnn（二者有自己的 FlashInfer 内核）。

**示例**

```python
import torch
from flashinfer.cudnn import cudnn_chunk_gated_delta_rule2

cu_seqlens = torch.tensor([0, 1024, 2048], dtype=torch.int32, device="cuda")
out = cudnn_chunk_gated_delta_rule2(q, k, v, g, beta, w, cu_seqlens=cu_seqlens)
```

---

### `cudnn_recurrent_kda`

```python
cudnn_recurrent_kda(
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
    beta_is_logit: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    state_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：cuDNN 融合 SM100 引擎上的 Kimi Delta Attention（KDA）prefill。参数含义与
`flashinfer.recurrent_kda` 一致，限于普通多 token prefill 子集：不支持投机解码、不支持
`initial_state_source`、不支持 state checkpoint。需要 cudnn-frontend 1.31.0+ 并带 `cutedsl` extra。

**参数**

- `q, k, v`（`torch.Tensor`）：`[1, total_tokens, H, 128]` 或 `[total_tokens, H, 128]`，bfloat16
  或 float16。
- `g`（`torch.Tensor`）：channel-wise gate `[..., total_tokens, HV, 128]`。除非
  `use_gate_in_kernel`，否则处于 log-space；`use_gate_in_kernel=True` 时它是原始 pre-activation，
  cuDNN 从 `A_log` / `dt_bias` / `lower_bound` 应用 safe-gate 变换。dtype 为 float32、bfloat16
  或 float16；cuDNN 三种都接受，且 gate 的内存格式随之变化，因此原样转发、无拷贝。float16 下
  内核的 chunk-cumulative decay inverse 限制了 decay 强度（每 token 每 channel 约
  `alpha >= 0.9`，超过则溢出）；bfloat16 具 fp32 级指数范围，无此限制。
- `beta`（`torch.Tensor`）：`[..., total_tokens, HV]`。float32 或 `q.dtype` 的 post-sigmoid，
  或 `beta_is_logit` 时 `q.dtype` 的 logits。
- `A_log, dt_bias`（`torch.Tensor`，可选）：safe-gate 参数，`use_gate_in_kernel` 时须同时提供。
- `scale`（`float`，可选）：query scale；为 `None` 时取 `1 / sqrt(head_dim)`。
- `output_final_state`（`bool`）：是否随输出返回最终状态；仅控制返回值，何时写入状态见
  `initial_state`。
- `use_qk_l2norm_in_kernel`（`bool`，默认 `True`）：把 q/k 的 L2 归一化融合进内核。
- `use_gate_in_kernel`（`bool`）：把 `g` 读作原始 pre-activation，并在内核中从 `A_log` /
  `dt_bias` / `lower_bound` 应用 safe-gate 变换。
- `beta_is_logit`（`bool`）：把 `beta` 读作 logits 并在内核中应用 sigmoid。
- `lower_bound`（`float`，可选）：safe-gate 下界，转发为 cuDNN 的 `gate_lower_bound`。
- `cu_seqlens`（`torch.Tensor`）：`[num_seqs + 1]`，int32 或 int64。必需。
- `initial_state, output_state`（`torch.Tensor`，可选）：状态 `[num_seqs, HV, 128, 128]`，V-major，
  float32 或 bfloat16。遵循 Cake 与 CuTe DSL prefill 后端：只要给出 `initial_state` 且未提供单独
  的 `output_state`，`initial_state` 就会推进到最终状态，与 `output_final_state` 无关——后者仅
  控制返回内容。无 `state_indices` 时 `output_state` 不得与 `initial_state` 别名；见
  `cudnn_chunk_gated_delta_rule`。
- `output`（`torch.Tensor`，可选）：预分配输出，由内核原地写。
- `batch_invariant`（`bool`）：禁用 split-K 分区；见 `cudnn_chunk_gated_delta_rule`。
- `state_indices`（`torch.Tensor`，可选）：循环状态 pool 中连续的 int32 `[num_seqs]` slot，要求
  cuDNN frontend 1.31+ 且为普通 packed prefill。pool 形状 `[N_pool, HV, V, K]`，内层行 dense、
  slot stride 16 字节对齐，float32 或 bfloat16。slot id 必须唯一且在范围内，值不被读回。选中的
  输入行会原地推进，除非提供了单独的 `output_state` pool，即使 `output_final_state` 为 false。
  显式输出 pool 必须形状/dtype 相同，且为相同 view 或使用不相交内存。未被触及的 slot 保留原值。
  返回状态时返回整个 pool。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor]]`：`(output, final_state)`，`output_final_state=False`
  时 `final_state` 为 `None`。

**注意事项**

- `cu_seqlens` 必需，为 `None` 时抛 `ValueError`。
- `use_gate_in_kernel=True` 时 `A_log` 与 `dt_bias` 都必需，否则抛 `ValueError`。
- 若 `v.shape[1] < q.shape[1]`（即 `H > HV`），抛 `NotImplementedError`：cuDNN 以 `max(H, HV)`
  heads 承载 KDA 状态，而 FlashInfer 用 `HV`。

**示例**

```python
import torch
from flashinfer.cudnn import cudnn_recurrent_kda

cu_seqlens = torch.tensor([0, 2048, 4096], dtype=torch.int32, device="cuda")
out, final_state = cudnn_recurrent_kda(
    q, k, v, g, beta, cu_seqlens=cu_seqlens, output_final_state=True,
)
```

## 相关链接 / 参考

- decode / prefill 的自动后端选择（`backend="auto"`）、Ragged prefill 几何缓存与 warmup 策略、
  LSE base/layout 特化等设计说明详见上游 rst 正文。
- 状态在此边界不透传转置：FlashInfer 与 cuDNN 都以 V-major `[N, H, V, K]` 保存状态，因此
  `initial_state` / `output_state` 直接透传。GDN 与 GDP 的 forget gate 在此为 linear-space alpha
  并以 `gate_domain="linear"` 传入；GDN-2 与 KDA 的 gate 两侧均为 log-space。


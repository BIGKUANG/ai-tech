# 注意力内核（`flashinfer.decode` / `flashinfer.prefill` / `flashinfer.attention` / `flashinfer.mla` / `flashinfer.xqa`）

> 源码：`flashinfer/decode.py`、`flashinfer/prefill.py`、`flashinfer/attention/_core.py`、
> `flashinfer/mla/`、`flashinfer/xqa.py`、`flashinfer/nvfp4_attention_sm120.py`、
> `flashinfer/cake_dcp.py`、`flashinfer/cake_fmha_request_ordered.py`
> 上游文档：`docs/api/attention.rst`
> 适用版本：FlashInfer 0.7.2

本页覆盖 FlashInfer 的注意力内核：单请求/批量 decode，XQA，单请求/批量 prefill 与 append，
因果+双向区间批量 prefill，统一的 BatchAttention 包装器，SM120 NVFP4 注意力，MLA
（Multi-head Latent Attention）系列，以及 DCP 推测解码辅助。CUDAGraph 兼容的包装器可用于捕获
内核并跨多层复用辅助数据结构。

> **实验性任务调度注意力（PrimTS）**：Blackwell 的任务调度 FMHA context / FMHA decode /
> block-sparse FMHA / MLA decode API 从 `flashinfer.attention.prims_ts` 导入。调度、tile 选择与
> split-KV 归约都是自动实现细节，没有公开调优旋钮。调用它们属显式 opt-in，每个被装饰函数会发出
> 一次 `ExperimentalWarning`，不提供 API 兼容保证；其稳定 API 参考条目延迟到毕业后再生成。
> 目前准确性与性能 signoff 在 SM100a/B200；SM103a/B300 受架构门控但尚未 signoff。

## 导入

```python
from flashinfer import (
    # decode
    single_decode_with_kv_cache, single_decode_with_kv_cache_with_jit_module,
    cudnn_batch_decode_with_kv_cache, trtllm_batch_decode_with_kv_cache,
    xqa_batch_decode_with_kv_cache, BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    get_dcp_spec_workspace_size_bytes, get_dcp_spec_counter_bytes,
    plan_cake_fmha_request_ordered_paged_decode, CakeFmhaRequestOrderedDecodePlan,
    # prefill
    single_prefill_with_kv_cache, single_prefill_with_kv_cache_return_lse,
    single_prefill_with_kv_cache_with_jit_module,
    cudnn_batch_prefill_with_kv_cache, trtllm_batch_context_with_kv_cache,
    trtllm_ragged_attention_deepseek, fmha_v2_prefill_deepseek,
    trtllm_fmha_v2_prefill, fmha_v2_prefill_sm120,
    BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper,
    BatchPrefillWithCausalBidirectionalRangesWrapper,
    BatchAttention, BatchAttentionWithAttentionSinkWrapper,
    nvfp4_attention_sm120_quantize_qkv, nvfp4_attention_sm120_fwd,
)
from flashinfer.xqa import xqa, xqa_mla
from flashinfer.mla import BatchMLAPagedAttentionWrapper, trtllm_batch_decode_with_kv_cache_mla
```

## API 列表

### Decode（`flashinfer.decode`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `single_decode_with_kv_cache` | 函数 | 单请求 decode（带 KV Cache）。 |
| `single_decode_with_kv_cache_with_jit_module` | 函数 | 使用预编译 JIT 模块的单请求 decode。 |
| `cudnn_batch_decode_with_kv_cache` | 函数 | cuDNN SDPA 后端的批量分页 decode。 |
| `trtllm_batch_decode_with_kv_cache` | 函数 | trtllm-gen/xqa 后端的批量分页 decode。 |
| `xqa_batch_decode_with_kv_cache` | 函数 | XQA cookie 后端的批量分页 decode。 |
| `BatchDecodeWithPagedKVCacheWrapper` | 类 | 分页 decode 包装器（`plan`/`run` 复用辅助结构）。 |
| `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | 类 | CUDA Graph 兼容的分页 decode 包装器。 |
| `get_dcp_spec_workspace_size_bytes` | 函数 | Cake FMHA Split-KV BF16 partial-O / FP32 partial-LSE scratch 字节数。 |
| `get_dcp_spec_counter_bytes` | 函数 | v4 完成 ticket 所需字节数。 |
| `plan_cake_fmha_request_ordered_paged_decode` | 函数 | 宿主侧选择 graph-safe 请求排序调度。 |
| `CakeFmhaRequestOrderedDecodePlan` | 类 | 不可变的宿主选择启动调度。 |

### XQA（`flashinfer.xqa`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `xqa` | 函数 | 使用 XQA 内核对分页 KV cache 做注意力。 |
| `xqa_mla` | 函数 | 使用 XQA MLA 内核对分页 KV cache 做注意力。 |

### Prefill（`flashinfer.prefill`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `single_prefill_with_kv_cache` | 函数 | 单请求 prefill/append 注意力。 |
| `single_prefill_with_kv_cache_return_lse` | 函数 | 恒返回 LSE 的便捷包装。 |
| `single_prefill_with_kv_cache_with_jit_module` | 函数 | 使用预编译 JIT 模块的单请求 prefill。 |
| `cudnn_batch_prefill_with_kv_cache` | 函数 | cuDNN SDPA 后端的批量分页 prefill。 |
| `trtllm_batch_context_with_kv_cache` | 函数 | trtllm-gen 后端的批量 context（prefill/append）。 |
| `trtllm_ragged_attention_deepseek` | 函数 | trtllm-gen 的 ragged 注意力（DeepSeek 变体）。 |
| `fmha_v2_prefill_deepseek` | 函数 | FMHA v2 dense prefill（DeepSeek 变体）。 |
| `trtllm_fmha_v2_prefill` | 函数 | trtllm FMHA v2 prefill。 |
| `fmha_v2_prefill_sm120` | 函数 | SM120 FMHA v2 连续 Q/K/V prefill。 |
| `BatchPrefillWithPagedKVCacheWrapper` | 类 | 分页 prefill/append 包装器。 |
| `BatchPrefillWithRaggedKVCacheWrapper` | 类 | ragged KV prefill/append 包装器。 |

### 因果+双向区间 / 统一注意力（`flashinfer.attention`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BatchPrefillWithCausalBidirectionalRangesWrapper` | 类 | 因果掩码 + 逐 query 双向区间的分页批量 prefill。 |
| `BatchAttention` | 类 | 自动在分页 prefill 与 decode 间分发的统一包装器。 |
| `BatchAttentionWithAttentionSinkWrapper` | 类 | 支持 attention sink 的 prefill/decode 包装器。 |

### SM120 NVFP4（`flashinfer.nvfp4_attention_sm120`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `nvfp4_attention_sm120_quantize_qkv` | 函数 | 为 SM120 NVFP4 注意力预处理并量化稠密 Q/K/V。 |
| `nvfp4_attention_sm120_fwd` | 函数 | 在预量化 Q/K/V 上运行 SM120 NVFP4 注意力。 |

### MLA（`flashinfer.mla`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BatchMLAPagedAttentionWrapper` | 类 | MLA 分页注意力包装器（矩阵吸收）。 |
| `trtllm_batch_decode_with_kv_cache_mla` | 函数 | trtllm-gen 的 MLA 批量 decode。 |
| `trtllm_prefill_with_kv_cache_mla` | 函数 | trtllm-gen 的 MLA prefill（decode 兼容语义）。 |
| `trtllm_batch_decode_sparse_mla_dsv4` | 函数 | DeepSeek-V4 稀疏 MLA decode。 |
| `nvfp4_quantize_pack_sparse_mla_cache` | 函数 | NVFP4 打包稀疏 MLA cache。 |
| `nvfp4_quantize_append_sparse_mla_cache` | 函数 | NVFP4 追加写入稀疏 MLA cache。 |
| `dsv41_fp4_quantize_pack_sparse_mla_cache` | 函数 | DeepSeek-V4.1 latent-KV 页量化为 V41_FP4 ABI。 |
| `dsv41_fp4_quantize_append_sparse_mla_cache` | 函数 | V41_FP4 量化追加。 |
| `dsv41_fp8_quantize_pack_sparse_mla_cache` | 函数 | DeepSeek-V4.1 latent-KV 页量化为 528 字节 DSV4_1 FP8 ABI。 |
| `dsv41_fp8_quantize_append_sparse_mla_cache` | 函数 | V41 FP8 量化追加。 |
| `convert_compressed_page_aligned_sparse_indices_to_hca_metadata` | 函数 | 把压缩页对齐稀疏索引转换为 HCA 元数据。 |
| `DSV4HCAMetadata` | 类 | DeepSeek-V4 HCA 元数据。 |
| `xqa_batch_decode_with_kv_cache_mla` | 函数 | XQA 后端的批量 MLA decode。 |
| `supported_sparse_mla_sm120_configs` | 函数 | 枚举已实例化的 SM120 稀疏 MLA decode 配置。 |
| `SparseMLASm120DecodeConfig` | 类 | 某一 SM120 稀疏 MLA 内核族的已实例化 decode 集。 |
| `SparseMLASm120Wrapper` | 类 | SM120 稀疏 MLA 包装器。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_decode` | 函数 | Cake SM120 NVFP4 稀疏 MLA decode。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_prefill` | 函数 | Cake SM120 NVFP4 稀疏 MLA prefill。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel` | 函数 | 选择 decode 还是 prefill 内核。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill` | 函数 | 规划 prefill 启动。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles` | 函数 | 规划 head tile 数。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits` | 函数 | 规划 split 数。 |
| `cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes` | 函数 | split-KV partial 与 LSE scratch 字节数。 |
| `cake_dsv4_nvfp4_rope_quantize_insert` | 函数 | 滑窗池的 RoPE + NVFP4 量化 + 分页写入（融合）。 |
| `cake_dsv4_nvfp4_kv_rope_quantize_insert` | 函数 | 压缩池的 RoPE + NVFP4 量化 + 写入（融合）。 |

## API 详情

### Decode

#### `single_decode_with_kv_cache`

```python
single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：单请求 KV-Cache decode 注意力，返回注意力输出。`return_lse=True` 时额外返回 logsumexp。
`q` 形状 `[num_qo_heads, head_dim]`，`k`/`v` 为 `[kv_len, num_kv_heads, head_dim]`（NHD）或
`[num_kv_heads, kv_len, head_dim]`（HND）。

**参数**：`kv_layout`（`"NHD"`/`"HND"`）、`pos_encoding_mode`（`"NONE"`/`"ROPE_LLAMA"`/`"ALIBI"`）、
`use_tensor_cores`、`q_scale`/`k_scale`/`v_scale`（FP8 标定）、`window_left`、`logits_soft_cap`、
`sm_scale`（默认 `1/sqrt(head_dim)`）、`rope_scale`、`rope_theta`、`return_lse`。

**返回**：`return_lse=False` 时返回输出 `[num_qo_heads, head_dim]`；为 `True` 时返回
`(输出, lse)`。

**注意事项**：低层入口由 `single_decode_with_kv_cache_with_jit_module` 在 backend 分发后调用；
一般用户直接调用本函数。

---

#### `single_decode_with_kv_cache_with_jit_module`

```python
single_decode_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    window_left: int = -1,
    return_lse: bool = False,
)
```

**功能**：使用预编译 JIT 模块的单请求 decode。这是 `single_decode_with_kv_cache` 在 backend 分发后
使用的低层入口；面向已持有编译好 JIT 模块的高级用户（例如通过额外内核参数自定义注意力变体）。

**参数**

- `jit_module`（`Any`）：由某个 `gen_*_module` 工厂返回的编译 JIT 模块。
- `q`（`torch.Tensor`）：query，形状 `[num_qo_heads, head_dim]`。
- `k`（`torch.Tensor`）：key，形状 `[kv_len, num_kv_heads, head_dim]`（NHD）或
  `[num_kv_heads, kv_len, head_dim]`（HND）。
- `v`（`torch.Tensor`）：value，layout 同 `k`。
- `*args`：转发给 JIT 模块 `run` 符号的额外位置参数。
- `kv_layout`（`str`，默认 `"NHD"`）：`k`/`v` 的 layout。
- `window_left`（`int`，默认 `-1`）：滑窗左窗口；`-1` 禁用。
- `return_lse`（`bool`，默认 `False`）：为内核分配 LSE 缓冲（`(num_qo_heads,)`，float32）。
  **注意：缓冲被分配并填充但不返回给调用方**——本函数始终只返回 `o`。

**返回**：`torch.Tensor`：输出，形状同 `q`。

---

#### `cudnn_batch_decode_with_kv_cache`

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

**功能**：由 cuDNN SDPA 后端支持的批量分页 KV-cache decode 注意力。

**参数**

- `q`（`torch.Tensor`）：形状 `(batch_size * q_len_per_req, num_heads_qo, head_dim)`
  （普通单 token decode 时为 `(batch_size, num_heads_qo, head_dim)`），fp16 或 bf16（输出用
  `q.dtype`）。`q_len_per_req > 1` 时同一请求的行连续。fp16 需要 cuDNN graph 后端；回退（cubin）
  路径仅 bf16 并会抛 `NotImplementedError`。
- `k_cache`/`v_cache`（`torch.Tensor`）：形状 `(total_num_pages, num_heads_kv, page_size, head_dim)`。
- `scale`（`float`）：softmax 缩放，通常 `1 / sqrt(head_dim)`。
- `workspace_buffer`（`torch.Tensor`）：cuDNN workspace。随 batch 增大；通常 128 MB 足够。
- `max_sequence_kv`（`int`）：批内每 KV 序列最大 token 数（`s_kv_max`）。
- `actual_seq_lens_kv`（`Optional[torch.Tensor]`）：每请求 KV 长度，形状 `(batch_size,)`。cuDNN
  可用（默认）时须与 `q` 同 CUDA 设备；仅回退路径接受（并内部拷贝）CPU 张量。
- `block_tables`（`Optional[torch.Tensor]`）：分页 KV cache 的页表，形状
  `(batch_size, num_pages_per_seq)`，在 GPU 上。
- `is_cuda_graph_compatible`（`bool`，默认 `False`）：是否以 CUDA-graph 捕获安全模式规划。
- `batch_offsets_q/o/k/v`（`Optional[torch.Tensor]`）：逐请求的元素偏移（int32 或 int64）；cuDNN
  graph 路径只接受与 batch stride 匹配的稠密 Q/O 偏移。建议传 `None`。
- `out`（`Optional[torch.Tensor]`）：输出张量；不提供时内部申请。
- `return_lse`（`bool`，默认 `False`）/`lse`（`Optional[torch.Tensor]`）：是否返回/写入 LSE。
- `q_len_per_req`（`int`，默认 `1`）：每请求 query token 数（推测/MTP 深度）。
- `window_left`（`int`，默认 `-1`）：滑窗左窗口；`-1` 禁用。
- `sinks`（`Optional[torch.Tensor]`）：每个头的额外 softmax 分母项。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

---

#### `trtllm_batch_decode_with_kv_cache`

```python
trtllm_batch_decode_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
    bmm1_scale=1.0, bmm2_scale=1.0, window_left=-1, out=None, out_dtype=None,
    o_sf_scale=None, o_sf_vec_size=None, sinks=None, kv_layout="HND",
    enable_pdl=None, backend="auto", q_len_per_req=1, o_scale=1.0, mask=None,
    max_q_len=None, cum_seq_lens_q=None, skip_softmax_threshold_scale_factor=None,
    kv_cache_sf=None, uses_shared_paged_kv_idx=True, lse=None, return_lse=False,
    bmm1_scale_log2=None, multi_ctas_kv_counter_buffer=None,
    enable_block_sparse_attention=False, cp_world=1, cp_rank=0,
    causal_seqlens_kv_global=None, bf16q_fp8kv_transform_mode=None,
    request_order=None, request_order_plan=None,
) -> Union[torch.Tensor, FP4Tensor, Tuple[..., torch.Tensor]]
```

**功能**：trtllm-gen / XQA backend 的批量分页 decode（当 `backend="auto"` 时亦可能选 cudnn）。
除覆盖 trtllm-gen 与 xqa 外，还承载 Cake FMHA DCP 推测解码路由：通过调用方持有的 scratch buffer，
预热后的调用可被捕获进 CUDA Graph；也可经 `flashinfer.cake_fmha.cake_batch_decode_with_kv_cache`
到达，非空的 `causal_seqlens_kv_global` 是显式的 add-on 选择键。

**主要参数**

- `query`（`torch.Tensor`）：`[num_tokens, num_heads, head_dim]`。
- `kv_cache`（`Union[torch.Tensor, Tuple]`）：单张量形状 `[num_pages, 1 或 2, num_kv_heads,
  page_size, head_dim]`（HND）或 `[num_pages, 1 或 2, page_size, num_kv_heads, head_dim]`（NHD）；
  或 (k, v) 元组。**trtllm-gen 后端要求 `head_dim`（last dim）stride 为 1**（TMA 硬件约束），
  head/batch(page) 维可为任意 stride。
- `workspace_buffer`（`torch.Tensor`）：trtllm-gen softmax stats/scratch 或 xqa scratch。
- `block_tables`（`torch.Tensor`）：`uses_shared_paged_kv_idx=True`（默认）时形状
  `[batch_size, max_num_pages_per_seq]`；为 `False` 时 `[batch_size, 2, max_num_pages_per_seq]`
  （dim 1 区分 K(0)/V(1)）；`enable_block_sparse_attention=True` 时形状
  `[num_kv_heads, batch_size, max_num_pages_per_seq]`（连续）。
- `seq_lens`（`torch.Tensor`）：uint32 一维，`[batch_size]`。
- `max_seq_len`（`int`）：KV cache 最大序列长度。
- `bmm1_scale`/`bmm2_scale`（`Union[float, torch.Tensor]`，默认 `1.0`）：bmm1/bmm2 融合 scale；
  trtllm-gen 后端可为 float32 张量。`bmm1_scale_log2` 为可选预计算 log2 形式。
- `out`/`out_dtype`（可选）：输出与其 dtype（nvfp4 用字符串 `"nvfp4"`）。
- `sinks`（`Optional[List[torch.Tensor]]`）：每头 softmax 分母附加值。
- `kv_layout`（`str`，默认 `"HND"`）：`"NHD"` 或 `"HND"`。
- `backend`（`str`，默认 `"auto"`）：后端选择。
- `q_len_per_req`（`int`，默认 `1`）：每请求 query token 数。
- `skip_softmax_threshold_scale_factor`（`Optional[float]`）：启用 skip-softmax 稀疏。
- `return_lse`/`lse`：是否返回/写入 LSE。
- `enable_block_sparse_attention`（`bool`，默认 `False`）：启用块稀疏解码注意力。
- `cp_world`/`cp_rank`（`int`）：context-parallel world/rank。
- `causal_seqlens_kv_global`（`Optional[torch.Tensor]`）：DCP 推测解码显式选择键。
- `request_order`/`request_order_plan`：SM103 Cake 请求排序张量/预先规划的调度。

**返回**：注意力输出（或 `FP4Tensor`）；`return_lse=True` 时返回元组。

**注意事项**

- 完整参数与返回值语义见源码 docstring；DCP Cake 路由见下文的 DCP 小节。

---

#### `xqa_batch_decode_with_kv_cache`

```python
xqa_batch_decode_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
    bmm1_scale=1.0, bmm2_scale=1.0, window_left=-1, out=None, sinks=None,
    kv_layout="NHD", enable_pdl=None, q_len_per_req=1, o_scale=1.0, mask=None,
    kv_cache_sf=None, q_cu_seq_lens=None,
) -> torch.Tensor
```

**功能**：XQA cookie 后端的批量分页 decode。

**主要参数**：`query`（`[num_tokens, num_heads, head_dim]`，`num_tokens = batch_size *
q_len_per_request`，或给 `q_cu_seq_lens` 时的逐请求 draft 长度之和）、`kv_cache`（分页，NHD/HND）、
`workspace_buffer`（首次使用须初始化为 0）、`block_tables`（`[batch_size, num_pages]`）、
`seq_lens`（uint32 `[batch_size]`）、`max_seq_len`、`bmm1_scale`/`bmm2_scale`、`window_left`、
`out`、`sinks`、`kv_layout`、`enable_pdl`、`q_len_per_req`、`o_scale`（fp8 输出 scale）、`mask`
（xqa 推测解码的 draft-block 掩码）、`kv_cache_sf`（NVFP4 KV cache 时必须提供）、`q_cu_seq_lens`
（`[batch_size + 1]` int32，启用 ragged Q 推测解码）。

**返回**：`torch.Tensor`：注意力输出。

**注意事项**：SM90 上 fp8 KV cache 且非负滑窗的推测解码走通用内核而非 Hopper fp8 内核。

---

#### `BatchDecodeWithPagedKVCacheWrapper`

```python
BatchDecodeWithPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    use_tensor_cores: bool = False,
    paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
    paged_kv_indices_buffer: Optional[torch.Tensor] = None,
    paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    backend: str = "auto",
    jit_args: Optional[List[Any]] = None,
) -> None
```

**功能**：分页 KV-cache 的批量 decode 注意力包装器（decode attention，最早见于 vLLM）。为加速
计算会创建辅助数据结构，可跨多次 decode 调用（如不同 Transformer 层）复用；本包装器管理这些数据
结构的生命周期。

**生命周期**：`__init__` → `plan`（= `begin_forward`）→ 可选 `prewarm_paged_kv_stride_variant` →
`run`（= `forward`）/`run_return_lse`。

#### `__init__` 参数

- `float_workspace_buffer`（`torch.Tensor`）：float workspace 缓冲，用于 split-k 中间结果。
- `kv_layout`（`str`，默认 `"NHD"`）：`"NHD"` 或 `"HND"`。
- `use_cuda_graph`（`bool`，默认 `False`）：是否启用 CUDA Graph；启用时辅助结构存于提供的缓冲，
  `batch_size` 在生命周期内不能改变。
- `use_tensor_cores`（`bool`，默认 `False`）：是否使用 tensor core 版本。
- `paged_kv_indptr_buffer`/`paged_kv_indices_buffer`/`paged_kv_last_page_len_buffer`
  （`Optional[torch.Tensor]`）：CUDA Graph 模式下的预留缓冲。
- `backend`（`str`，默认 `"auto"`）：后端选择（如 `auto`/`fa2`/`cudnn`/`trtllm-gen`/`xqa`）。
- `jit_args`（`Optional[List[Any]]`）：提供给 JIT 模块的参数。

#### `plan`

```python
plan(
    indptr: torch.Tensor, indices: torch.Tensor, last_page_len: torch.Tensor,
    num_qo_heads: int, num_kv_heads: int, head_dim: int, page_size: int,
    *deprecated_positional_args, **kwargs,
) -> None
```

**功能**：为给定问题规格规划批量 decode。必须在任何 `run` 之前调用；辅助结构在此创建并缓存。
`num_qo_heads` 须是 `num_kv_heads` 的倍数，否则使用 GQA。`plan` 不能在 CUDA Graph 或
`torch.compile` 内使用。

**主要参数**：`indptr`/`indices`/`last_page_len`（分页 kv cache 的 CSR 元数据）、`num_qo_heads`、
`num_kv_heads`、`head_dim`、`page_size`，以及关键字参数 `pos_encoding_mode`、`data_type`/
`q_data_type`/`kv_data_type`、`window_left`、`sm_scale`、`rope_scale`、`rope_theta`、
`logits_soft_cap`、`non_blocking`、`backend`、`q_len_per_req` 等（随后端不同）。

#### `run`（= `forward`）

```python
run(
    q: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *args,
    q_scale=None, k_scale=None, v_scale=None, out=None, lse=None,
    return_lse=False, enable_pdl=None, window_left=None, sinks=None,
    q_len_per_req=None, skip_softmax_threshold_scale_factor=None, kv_cache_sf=None,
    ...
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 query 与分页 kv cache 之间的批量 decode 注意力，返回注意力输出。`q` 形状
`[batch_size, num_qo_heads, head_dim]`（或带 `q_len_per_req` 的推测形状），`paged_kv_cache` 为
tuple `(k_cache, v_cache)` 或单个 5-D 堆叠张量（`paged_kv_cache[:, 0]` 为 K，`[:, 1]` 为 V）。

**主要参数**：`out`（预分配输出）、`lse`/`return_lse`、`enable_pdl`、`window_left`、`sinks`、
`q_len_per_req`（推测解码深度）、`kv_cache_sf`（NVFP4 KV cache 时）、FP8 scale 等。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
decode_wrapper.plan(kv_page_indptr, kv_page_indices, kv_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    pos_encoding_mode="NONE", data_type=torch.float16)
o = decode_wrapper.run(q, kv_cache)
```

---

#### `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`

```python
CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer: torch.Tensor,
    indptr_buffer: torch.Tensor,
    indices_buffer: torch.Tensor,
    last_page_len_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_tensor_cores: bool = False,
) -> None
```

**功能**：CUDA Graph 兼容的分页 decode 包装器，继承 `BatchDecodeWithPagedKVCacheWrapper`。所有
辅助缓冲在构造时提供，`batch_size` 固定。除 `__init__` 外还提供 `paged_run`（低层运行入口）与
`_paged_run`、`_plan` 等。

**参数**

- `workspace_buffer`（`torch.Tensor`）：float workspace 缓冲。
- `indptr_buffer`/`indices_buffer`/`last_page_len_buffer`（`torch.Tensor`）：分页 kv cache 的
  indptr/indices/last_page_len 预留缓冲，大小分别为 `[batch_size + 1]`、`[max_num_pages]`、
  `[batch_size]`。
- `kv_layout`（`str`，默认 `"NHD"`）：`"NHD"` 或 `"HND"`。
- `use_tensor_cores`（`bool`，默认 `False`）：是否使用 tensor core 版本。

**注意事项**：`paged_run` 不经 `@flashinfer_api`（避免重复日志），其签名包含 workspace 与
aux 缓冲、plan info、q/k/v cache、indptr 等；CUDA Graph 模式下 `batch_size` 固定。

---

#### `get_dcp_spec_workspace_size_bytes`

```python
get_dcp_spec_workspace_size_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_split: int = 16,
    *,
    head_dim: int = 128,
) -> int
```

**功能**：返回 Cake FMHA Split-KV 的 BF16 partial-O 与 FP32 partial-LSE scratch 所需字节数。

**参数**：`batch_size`、`q_len_per_req`、`num_qo_heads`、`num_split`、`head_dim`。

**返回**：`int`：字节数。用于为 `trtllm_batch_decode_with_kv_cache` 的 DCP Cake 路由分配
`workspace_buffer`。

---

#### `get_dcp_spec_counter_bytes`

```python
get_dcp_spec_counter_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_kv_heads: int,
) -> int
```

**功能**：返回 v4 完成 ticket 所需字节数（初始化时清零一次，之后自重置）。

**参数**：`batch_size`、`q_len_per_req`、`num_kv_heads`。

**返回**：`int`：字节数。

---

#### `plan_cake_fmha_request_ordered_paged_decode`

```python
plan_cake_fmha_request_ordered_paged_decode(
    kv_lens: Sequence[int],
    q_len: int,
    *,
    request_order_case: Literal["identity", "length_desc"] = "length_desc",
    real_batch_size: int | None = None,
    write_lse: bool = False,
) -> CakeFmhaRequestOrderedDecodePlan
```

**功能**：从宿主可见的不可变元数据选择一条导出的调度。返回的 plan 不含设备数据。在 CUDA Graph
捕获前调用，然后在重放之间就地更新设备 `request_order` 张量的内容。精确导出的路由（two-wave 与
low-Q1 cluster 调度、split LSE 行）仅在其导出的请求长度下被选中；其他一切使用单 split 持久程序。

**参数**

- `kv_lens`（`Sequence[int]`）：各请求的 KV 长度。
- `q_len`（`int`）：query 长度。
- `request_order_case`（`Literal["identity", "length_desc"]`，默认 `"length_desc"`）：请求排序情形。
- `real_batch_size`（`int | None`，默认 `None`）：真实 batch 大小。
- `write_lse`（`bool`，默认 `False`）：是否写 LSE。

**返回**：`CakeFmhaRequestOrderedDecodePlan`：不可变启动调度。

**注意事项**：SM103 上同一 Cake 入口接受设备 `request_order` 张量，用于 BF16-query、FP8-E4M3
分页 decode 且 head_dim 256。页表行须填充到 `4 * ceil(max_seq_len / 256)` 项，且精确的张量/
workspace 绑定必须在捕获前急切调用一次以初始化其 TMA 描述符；此后仅改变 order 张量内容无需重新
捕获。

---

#### `CakeFmhaRequestOrderedDecodePlan`

**功能**：graph-safe 请求排序的不可变、宿主选择的启动调度。由
`plan_cake_fmha_request_ordered_paged_decode` 返回。

### XQA

#### `xqa`

```python
xqa(
    q, k_cache, v_cache, page_table, seq_lens, output, workspace_buffer,
    semaphores, num_kv_heads, page_size, sinks=None, q_scale=1.0, kv_scale=1.0,
    sliding_win_size=0, kv_layout="NHD", sm_count=None, enable_pdl=None,
    rcp_out_scale=1.0, q_seq_len=1, mask=None, *,
    q_cu_seq_lens=None, k_sf_cache=None, v_sf_cache=None,
) -> None
```

**功能**：使用 XQA 内核，对分页 KV cache 做注意力。就地修改 `output`。

**主要参数**

- `q`（`torch.Tensor`）：不使用推测解码时形状 `[batch_size, beam_width, num_q_heads, head_dim]`，
  使用推测解码时 `[batch_size, beam_width, q_seq_len, num_q_heads, head_dim]`。dtype fp16 或
  bf16。当前仅支持 `beam_width = 1`。
- `k_cache`/`v_cache`（`torch.Tensor`）：分页，`NHD` 为
  `[num_pages, page_size, num_kv_heads, head_dim]`，`HND` 为
  `[num_pages, num_kv_heads, page_size, head_dim]`。dtype 与 query 匹配或 `torch.float8_e4m3fn`
  （走 fp8 计算）。NVFP4 KV 时 dtype 为 `uint8`、last dim 为 `head_dim / 2`。
- `k_sf_cache`/`v_sf_cache`（`Optional[torch.Tensor]`）：NVFP4 KV 时的 scale factor cache，形状
  `[num_pages, ..., head_dim / 16]`（NHD）或 `[num_pages, num_kv_heads, page_size, head_dim/16]`
  （HND），dtype `uint8`。
- `page_table`（`torch.Tensor`）：`[batch_size, nb_pages_per_seq]`，int32，K 与 V 共用。
- `seq_lens`（`torch.Tensor`）：`[batch_size, beam_width]`，uint32。
- `output`（`torch.Tensor`）：形状同 query；就地修改。
- `workspace_buffer`（`torch.Tensor`）：临时计算 workspace，uint8。
- `semaphores`（`torch.Tensor`）：同步 semaphore，uint32。
- `num_kv_heads`（`int`）：KV 头数。
- `page_size`（`int`）：页大小。
- `sinks`、`q_scale`、`kv_scale`、`sliding_win_size`、`kv_layout`、`sm_count`、`enable_pdl`、
  `rcp_out_scale`、`q_seq_len`、`mask`、`q_cu_seq_lens`：见源码 docstring。

**返回**：`None`（就地修改 `output`）。

---

#### `xqa_mla`

```python
xqa_mla(
    q, k_cache, v_cache, page_table, seq_lens, output, workspace_buffer,
    semaphores, page_size, q_scale=1.0, kv_scale=1.0, sm_count=None, enable_pdl=None,
) -> None
```

**功能**：使用 XQA MLA（Multi-Head Latent Attention）内核，对分页 KV cache 做注意力。

**主要参数**

- `q`（`torch.Tensor`）：`[batch_size, beam_width, num_q_heads, head_dim]`，`torch.float8_e4m3fn`。
  当前仅支持 `beam_width = 1`。
- `k_cache`/`v_cache`（`torch.Tensor`）：`[total_num_cache_heads, head_dim]`，`torch.float8_e4m3fn`。
- `page_table`（`torch.Tensor`）：`[batch_size, nb_pages_per_seq]`，int32，K 与 V 共用。
- `seq_lens`（`torch.Tensor`）：`[batch_size, beam_width]`，uint32。
- `output`（`torch.Tensor`）：`[batch_size, beam_width, num_q_heads, head_dim]`，`torch.bfloat16`；
  就地修改。
- `workspace_buffer`（`torch.Tensor`）：uint8 workspace。
- `semaphores`（`torch.Tensor`）：uint32 semaphore。
- `page_size`（`int`）：页大小，须为 `[16, 32, 64, 128]` 之一。
- `q_scale`/`kv_scale`（`Union[float, torch.Tensor]`，默认 `1.0`）：缩放因子。
- `sm_count`（`Optional[int]`）：使用的 SM 数；`None` 时从设备推断。
- `enable_pdl`（`Optional[bool]`）：PDL 优化开关；`None` 时若硬件支持则启用。

**返回**：`None`（就地修改 `output`）。

**注意事项**：函数从张量形状自动推断若干参数：`batch_size`（`q.shape[0]`）、`head_dim`
（`q.shape[-1]`）、`head_group_ratio` 固定为 128、`max_seq_len`（`page_table.shape[-1] * page_size`）。

### Prefill

#### `single_prefill_with_kv_cache`

```python
single_prefill_with_kv_cache(
    q, k, v,
    scale_q=None, scale_k=None, scale_v=None, o_dtype=None,
    custom_mask=None, packed_custom_mask=None, causal=False,
    kv_layout="NHD", pos_encoding_mode="NONE", use_fp16_qk_reduction=False,
    sm_scale=None, window_left=-1, logits_soft_cap=None, rope_scale=None,
    rope_theta=None, backend="auto", return_lse=False,
    kv_cache_sf=None, k_scale=None, v_scale=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：单请求 prefill/append 注意力，返回注意力输出。`q` 形状 `[qo_len, num_qo_heads,
head_dim]`，`k`/`v` 为 `[kv_len, num_kv_heads, head_dim]`（NHD）或 `[num_kv_heads, kv_len,
head_dim]`（HND）。

**主要参数**

- `scale_q`/`scale_k`/`scale_v`、`o_dtype`、`kv_cache_sf`、`k_scale`、`v_scale`：FP8/NVFP4 量化
  scale 与输出 dtype。
- `custom_mask`/`packed_custom_mask`：稠密/位打包自定义掩码。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码。
- `kv_layout`、`pos_encoding_mode`、`use_fp16_qk_reduction`、`sm_scale`、`window_left`、
  `logits_soft_cap`、`rope_scale`、`rope_theta`、`backend`、`return_lse`。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

---

#### `single_prefill_with_kv_cache_return_lse`

```python
single_prefill_with_kv_cache_return_lse(...)  # functools.partial(..., return_lse=True)
```

**功能**：`single_prefill_with_kv_cache` 的便捷包装，恒返回 LSE。等价于以 `return_lse=True` 调用
`single_prefill_with_kv_cache`；接受相同参数并原样转发。

**返回**：`Tuple[torch.Tensor, torch.Tensor]`：`(output, lse)`。

---

#### `single_prefill_with_kv_cache_with_jit_module`

```python
single_prefill_with_kv_cache_with_jit_module(
    jit_module: Any, q, k, v, *args,
    kv_layout="NHD", mask_mode=<MaskMode.NON_CAUSAL.value>,
    window_left=-1, return_lse=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：使用预编译 JIT 模块的单请求 prefill。低层入口，供已持有编译 JIT 模块的高级用户使用。

**参数**：`jit_module`、`q`/`k`/`v`、`*args`（转发给 JIT 模块）、`kv_layout`、`mask_mode`、
`window_left`、`return_lse`。

**返回**：注意力输出，或 `return_lse` 时的 `(输出, lse)`。

---

#### `cudnn_batch_prefill_with_kv_cache`

```python
cudnn_batch_prefill_with_kv_cache(
    q, k_cache, v_cache, scale, workspace_buffer, *,
    max_token_per_sequence: int, max_sequence_kv: int,
    actual_seq_lens_q=None, actual_seq_lens_kv=None, block_tables=None,
    causal: bool, return_lse: bool, q_scale=None, k_scale=None, v_scale=None,
    batch_offsets_q=None, batch_offsets_o=None, batch_offsets_k=None,
    batch_offsets_v=None, batch_offsets_stats=None, batch_offsets_units="elements",
    out=None, lse=None, is_cuda_graph_compatible=False, backend=None,
    o_data_type=None, lse_base="log2",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

**功能**：由 cuDNN SDPA 后端支持的批量分页 KV-cache prefill 注意力。

**主要参数**

- `q`（`torch.Tensor`）：打包 query `(total_qo_tokens, num_heads_qo, head_dim_qk)`。
- `k_cache`/`v_cache`（`torch.Tensor`）：分页时 `(total_num_pages, num_heads_kv, page_size,
  head_dim_qk/vo)`，非分页时 `(total_kv_tokens, num_heads_kv, head_dim)`。
- `scale`（`float`）：softmax 缩放，通常 `1 / sqrt(head_dim_qk)`。
- `workspace_buffer`（`torch.Tensor`）：cuDNN workspace，通常 128 MB 足够。
- `max_token_per_sequence`/`max_sequence_kv`（`int`）：每 query/KV 序列最大 token 数。
- `actual_seq_lens_q`/`actual_seq_lens_kv`（`Optional[torch.Tensor]`）：形状 `(batch_size, 1, 1, 1)`
  的逐请求 query/KV 长度。
- `block_tables`（`Optional[torch.Tensor]`）：分页映射。
- `causal`（`bool`）：是否因果。
- `return_lse`（`bool`）：是否返回 LSE。
- `q_scale`/`k_scale`/`v_scale`、`batch_offsets_*`、`batch_offsets_units`、`out`、`lse`、
  `is_cuda_graph_compatible`、`backend`、`o_data_type`、`lse_base`。

**返回**：`(output, lse_or_None)`。

---

#### `trtllm_batch_context_with_kv_cache`

```python
trtllm_batch_context_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len,
    max_kv_len, bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q,
    cum_seq_lens_kv, window_left=-1, out=None, out_dtype=None, o_sf_scale=None,
    o_sf_vec_size=None, kv_layout="HND", enable_pdl=None, sinks=None,
    kv_cache_sf=None, skip_softmax_threshold_scale_factor=None,
    uses_shared_paged_kv_idx=True, causal=True, lse=None, return_lse=False,
    multi_ctas_kv_counter_buffer=None, use_fp16_softmax=None,
    uses_spcompress=None, backend="trtllm-gen",
) -> Union[torch.Tensor, FP4Tensor, Tuple[..., torch.Tensor]]
```

**功能**：trtllm-gen 后端的批量 context（prefill/append）注意力。

**主要参数**

- `query`（`torch.Tensor`）：`[num_tokens, num_heads, head_dim]`。
- `kv_cache`：单张量或 (k, v) 元组；trtllm-gen 要求 `head_dim`（last dim）stride 为 1。
- `block_tables`：`uses_shared_paged_kv_idx=True` 时 `[batch_size, max_num_pages_per_seq]`，否则
  `[batch_size, 2, max_num_pages_per_seq]`。
- `seq_lens`（uint32 `[batch_size]`）、`max_q_len`、`max_kv_len`、`bmm1_scale`/`bmm2_scale`、
  `batch_size`、`cum_seq_lens_q`/`cum_seq_lens_kv`（`[batch_size + 1]`）。
- `window_left`、`out`/`out_dtype`、`o_sf_scale`/`o_sf_vec_size`（nvfp4 输出）、`kv_layout`、
  `enable_pdl`、`sinks`、`kv_cache_sf`、`skip_softmax_threshold_scale_factor`、`causal`、
  `lse`/`return_lse`、`use_fp16_softmax`、`uses_spcompress`、`backend`。

**返回**：注意力输出（或 `FP4Tensor`）；`return_lse=True` 时返回元组。

---

#### `trtllm_ragged_attention_deepseek`

```python
trtllm_ragged_attention_deepseek(
    query, key, value, workspace_buffer, seq_lens, max_q_len, max_kv_len,
    bmm1_scale, bmm2_scale, o_sf_scale, batch_size, window_left, cum_seq_lens_q,
    cum_seq_lens_kv, enable_pdl, is_causal, return_lse,
    attention_sinks=None, skip_softmax_threshold_scale_factor=None, out=None,
    lse=None, sage_attn_sfs=(None, None, None, None),
    num_elts_per_sage_attn_blk=(0, 0, 0, 0), backend="trtllm-gen",
    q_seq_lens_cpu=None, kv_seq_lens_cpu=None, use_fp16_softmax=None,
    uses_spcompress=None, skip_all_rows_active_check=True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：trtllm-gen 的 ragged 注意力（DeepSeek 变体）。

**主要参数**

- `query`/`key`/`value`（`torch.Tensor`）：`[num_tokens, num_heads, head_dim]`。
- `workspace_buffer`、`seq_lens`、`max_q_len`、`max_kv_len`。
- `bmm1_scale`（`scale_q * scale_k / sqrt(head_dim_qk)`）、`bmm2_scale`（`scale_v`）、`o_sf_scale`。
- `batch_size`、`window_left`、`cum_seq_lens_q`/`cum_seq_lens_kv`、`enable_pdl`、`is_causal`、
  `return_lse`（为 `True` 时返回 `(out, lse)`）。
- `attention_sinks`、`skip_softmax_threshold_scale_factor`（启用 skip-softmax 稀疏；
  实际阈值 = 提供值 / 上下文长度）、`out`、`lse`。
- `sage_attn_sfs`（SageAttention 四个子块的 scale 因子 `(q_sf, k_sf, p_sf, v_sf)`）、
  `num_elts_per_sage_attn_blk`、`use_fp16_softmax`、`uses_spcompress`。

**返回**：注意力输出；`return_lse=True` 时返回 `(out, lse)`。

---

#### `fmha_v2_prefill_sm120`

```python
fmha_v2_prefill_sm120(
    query, key, value, out, num_heads, head_dim, seq_len, scale_softmax,
    scale_bmm1=None, scale_bmm2=None, causal=True, return_lse=False, lse=None,
    scale_bmm1_d=None, scale_bmm2_d=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：用连续、独立的 Q/K/V 张量运行 SM120 FMHA v2。支持 FP8 E4M3 标准 MHA 自注意力（Q/K/V
头数相等，head_dim 为 64 或 128）与 BF16 输出。**此入口不支持 GQA 与 MQA**。`scale_bmm1` 合并 Q/K
反量化 scale 与注意力 scale；`scale_bmm2` 为 V 反量化 scale。`causal=False` 用于稠密/双向。
`out` 为预分配输出张量。

**返回**：注意力输出；`return_lse=True` 时返回 `(out, lse)`。

---

#### `fmha_v2_prefill_deepseek`

```python
fmha_v2_prefill_deepseek(
    query, key, value, out, num_heads, head_dim, seq_len, scale_softmax,
    scale_bmm1=None, scale_bmm2=None, return_lse=False, lse=None,
    scale_bmm1_d=None, scale_bmm2_d=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：FMHA v2 dense prefill（DeepSeek 变体）。参数同 `fmha_v2_prefill_sm120`（无 `causal`）。

**返回**：注意力输出；`return_lse=True` 时返回 `(out, lse)`。

---

#### `trtllm_fmha_v2_prefill`

```python
trtllm_fmha_v2_prefill(
    qkv, input_layout, workspace_buffer, seq_lens, max_q_len, max_kv_len,
    bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q, cum_seq_lens_kv,
    block_tables=None, out=None, out_dtype=None, sinks=None,
    pos_encoding_mode=None, logits_soft_cap_scale=None, mask_mode="causal",
    window_left=-1, chunked_attention_size=0, save_softmax_stats=False,
    skip_softmax_threshold_scale_factor=0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：trtllm FMHA v2 prefill。

**主要参数**

- `qkv`（`Union[torch.Tensor, Tuple]`）：qkv 张量，或二/三元组。
- `input_layout`（`str`）：输入布局。
- `workspace_buffer`、`seq_lens`、`max_q_len`、`max_kv_len`、`bmm1_scale`、`bmm2_scale`、
  `batch_size`、`cum_seq_lens_q`/`cum_seq_lens_kv`。
- `block_tables`、`out`/`out_dtype`、`sinks`、`pos_encoding_mode`、`logits_soft_cap_scale`、
  `mask_mode`（默认 `"causal"`）、`window_left`、`chunked_attention_size`、`save_softmax_stats`、
  `skip_softmax_threshold_scale_factor`。

**返回**：注意力输出，或 `(输出, lse)`。

---

#### `BatchPrefillWithPagedKVCacheWrapper`

```python
BatchPrefillWithPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    qo_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indices_buf: Optional[torch.Tensor] = None,
    paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
    custom_mask_buf: Optional[torch.Tensor] = None,
    mask_indptr_buf: Optional[torch.Tensor] = None,
    backend: str = "auto",
    jit_args: Optional[List[Any]] = None,
    jit_kwargs: Optional[Dict[str, Any]] = None,
    variant_owns_mask: bool = False,
    max_total_num_rows: Optional[int] = None,
) -> None
```

**功能**：分页 KV-cache 的批量 prefill/append 注意力包装器。辅助数据结构可跨多次 prefill/append
调用（如不同 Transformer 层）复用；包装器管理其生命周期。

**生命周期**：`__init__` → `plan`（= `begin_forward`）→ 可选 `prewarm_paged_kv_stride_variant` →
`run`（= `forward`）/`run_return_lse`。

#### `__init__` 参数

- `float_workspace_buffer`、`kv_layout`、`use_cuda_graph`、`backend`、`jit_args`、`jit_kwargs`、
  `variant_owns_mask`、`max_total_num_rows`：workspace/layout/图捕获/后端/JIT 配置。
- `qo_indptr_buf`/`paged_kv_indptr_buf`/`paged_kv_indices_buf`/`paged_kv_last_page_len_buf`
  （`Optional[torch.Tensor]`）：CUDA Graph 模式下的预留缓冲。
- `custom_mask_buf`/`mask_indptr_buf`（`Optional[torch.Tensor]`）：自定义掩码及 indptr 预留缓冲。

#### `plan`（= `begin_forward`）

```python
plan(
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim_qk, page_size, head_dim_vo=None,
    custom_mask=None, packed_custom_mask=None, causal=False,
    pos_encoding_mode="NONE", use_fp16_qk_reduction=False, sm_scale=None,
    window_left=-1, logits_soft_cap=None, rope_scale=None, rope_theta=None,
    q_data_type="float16", kv_data_type=None, o_data_type=None,
    non_blocking=True, prefix_len_ptr=None, token_pos_in_items_ptr=None,
    token_pos_in_items_len=0, max_item_len_ptr=None, seq_lens=None,
    seq_lens_q=None, block_tables=None, max_token_per_sequence=None,
    max_sequence_kv=None, fixed_split_size=None, disable_split_kv=False,
) -> None
```

**功能**：为给定问题规格创建批量分页 prefill/append 的辅助数据结构。必须在任何 `run` 之前调用；
`num_qo_heads` 须是 `num_kv_heads` 的倍数，否则使用 GQA。`plan` 不能在 CUDA Graph 或
`torch.compile` 内使用。

**主要参数**：`qo_indptr`/`paged_kv_indptr`/`paged_kv_indices`/`paged_kv_last_page_len`（分页
元数据）、`num_qo_heads`/`num_kv_heads`/`head_dim_qk`/`head_dim_vo`/`page_size`、`custom_mask`/
`packed_custom_mask`、`causal`、`pos_encoding_mode`、`use_fp16_qk_reduction`、`sm_scale`、
`window_left`、`logits_soft_cap`、`rope_scale`/`rope_theta`、`q_data_type`/`kv_data_type`/
`o_data_type`、`prefix_len_ptr`、`token_pos_in_items_ptr`/`token_pos_in_items_len`、
`max_item_len_ptr`、`seq_lens`/`seq_lens_q`/`block_tables`、`max_token_per_sequence`/
`max_sequence_kv`、`fixed_split_size`、`disable_split_kv`。

#### `run`（= `forward`）

```python
run(
    q: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *args, q_scale=None, k_scale=None, v_scale=None, out=None, lse=None,
    return_lse=False, lse_base="log2", lse_layout="NH", enable_pdl=None,
    window_left=None, sinks=None, kv_cache_sf=None,
    skip_softmax_threshold_scale_factor=None, use_fp16_softmax=None,
    uses_spcompress=None,
    ...
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 query 与分页 kv cache 之间的批量 prefill/append 注意力。`q` 形状
`[qo_indptr[-1], num_qo_heads, head_dim]`，`paged_kv_cache` 为 tuple 或单个 5-D 堆叠张量。

**主要参数**：`out`、`lse`/`return_lse`（`lse_base` 为 `"log2"`/`"basee"`，`lse_layout` 为
`"NH"` 等）、`enable_pdl`、`window_left`、`sinks`、`kv_cache_sf`、
`skip_softmax_threshold_scale_factor`、`use_fp16_softmax`、`uses_spcompress`、FP8 scale 等。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

#### `prewarm_paged_kv_stride_variant`

```python
prewarm_paged_kv_stride_variant(variant: str = "independent") -> None
```

**功能**：在 `plan` 之后加载惰性 tensor-core paged-KV-stride 变体。CUDA Graph 捕获前调用。
`variant` 仅支持 `"independent"`。

---

#### `BatchPrefillWithRaggedKVCacheWrapper`

```python
BatchPrefillWithRaggedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    qo_indptr_buf: Optional[torch.Tensor] = None,
    kv_indptr_buf: Optional[torch.Tensor] = None,
    custom_mask_buf: Optional[torch.Tensor] = None,
    mask_indptr_buf: Optional[torch.Tensor] = None,
    backend: str = "auto",
    jit_args: Optional[List[Any]] = None,
    jit_kwargs: Optional[Dict[str, Any]] = None,
    variant_owns_mask: bool = False,
) -> None
```

**功能**：ragged（非分页）KV-cache 的批量 prefill/append 注意力包装器。生命周期同分页版本。

#### `plan`（= `begin_forward`）

```python
plan(
    qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk,
    head_dim_vo=None, custom_mask=None, packed_custom_mask=None, causal=False,
    pos_encoding_mode="NONE", use_fp16_qk_reduction=False, window_left=-1,
    logits_soft_cap=None, sm_scale=None, rope_scale=None, rope_theta=None,
    q_data_type="float16", kv_data_type=None, o_data_type=None,
    non_blocking=True, prefix_len_ptr=None, token_pos_in_items_ptr=None,
    token_pos_in_items_len=0, max_item_len_ptr=None, fixed_split_size=None,
    disable_split_kv=False, seq_lens=None, seq_lens_q=None,
    max_token_per_sequence=None, max_sequence_kv=None, v_indptr=None,
    o_indptr=None, qo_indptr_cpu=None, kv_indptr_cpu=None,
) -> None
```

**功能**：为 ragged KV 的批量 prefill/append 创建辅助数据结构。

#### `run`（= `forward`）

```python
run(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args,
    out=None, lse=None, return_lse=False, lse_base="log2", lse_layout="NH",
    enable_pdl=None, kv_cache_sf=None, ...
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 query 与 ragged K/V 之间的批量 prefill/append 注意力。`q`/`k`/`v` 为 ragged 布局。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

### 因果+双向区间 / 统一注意力

#### `BatchPrefillWithCausalBidirectionalRangesWrapper`

```python
BatchPrefillWithCausalBidirectionalRangesWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    qo_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indices_buf: Optional[torch.Tensor] = None,
    paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
    pos_encoding_mode: str = "NONE",
    q_data_type: torch.dtype = torch.bfloat16,
    kv_data_type: torch.dtype = torch.bfloat16,
    o_data_type: Optional[torch.dtype] = None,
    head_dim_qk: int = 128,
    head_dim_vo: int = 128,
) -> None
```

**功能**：带因果掩码 + 逐 query 双向区间的分页批量 prefill。掩码为
`(causal AND causal_window) OR (in_range AND range_window)`，其中 `in_range` 由调用方给出的
inclusive `[start, end]` key 区间逐 query token 决定。掩码**从不物化**：一个自定义 fa2 注意力变体
在自己的每个 KV tile 上从紧凑的 range 张量求值，因此不随 `qo_len * kv_len` 增长。

区间以一个连续的 `int32` 张量经 `run` 传入，形状 `[total_q, 2]`，位于 query 设备上，按与 query
张量相同的顺序逐有效 query token 一行。行 `i` 是 query `i` 在其请求中绝对 key 位置的 inclusive
`[start, end]`；`(-1, -1)` 标记无区间的 query，其保持普通因果掩码。

本包装器仅支持 fa2（fa2 是唯一在每个 KV tile 上求值自定义 `LogitsMask` 的后端），与
`prefix_len_ptr`（multi-item scoring）无关且不能组合使用。JIT 模块在构造函数中按 query/KV/输出
dtype、两个 head 维度与位置编码模式特化；`plan` 与 `workspace_size` 拒绝与编译结果不同的值，并
拒绝那些否则会被接受却静默忽略的继承选项。

**生命周期**：`__init__` → `plan` → `run`。

#### `plan`

```python
plan(
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim_qk, page_size, head_dim_vo=None,
    custom_mask=None, packed_custom_mask=None, causal=False,
    pos_encoding_mode="NONE", use_fp16_qk_reduction=False, sm_scale=None,
    window_left=-1, logits_soft_cap=None, q_data_type=None, kv_data_type=None,
    o_data_type=None, non_blocking=True, prefix_len_ptr=None,
    token_pos_in_items_ptr=None, token_pos_in_items_len=0, max_item_len_ptr=None,
    max_token_per_sequence=None, max_sequence_kv=None, fixed_split_size=None,
    disable_split_kv=False,
) -> None
```

**功能**：为因果+双向区间批量 prefill 创建辅助数据结构。会校验特化一致性并拒绝被继承但无意义的
选项。

#### `run`（= `forward`）

```python
run(
    q: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    bidirectional_ranges: torch.Tensor,
    causal_window_left: int = -1,
    range_window_left: int = -1,
    q_scale=None, k_scale=None, v_scale=None, out=None, lse=None,
    return_lse=False, enable_pdl=None, kv_cache_sf=None, ...
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 query 与分页 kv cache 之间的因果+双向区间批量 prefill。

**参数**

- `q`（`torch.Tensor`）：query。
- `paged_kv_cache`（`Union[torch.Tensor, Tuple]`）：分页 KV cache。
- `bidirectional_ranges`（`torch.Tensor`）：连续的 int32 `[total_q, 2]`，逐 query token 的
  inclusive `[start, end]`；`(-1, -1)` 表示无区间。
- `causal_window_left`（`int`，默认 `-1`）：因果部分的滑窗左边界；`-1` 禁用。
- `range_window_left`（`int`，默认 `-1`）：区间部分的滑窗左边界；`-1` 禁用。
- `q_scale`/`k_scale`/`v_scale`、`out`、`lse`/`return_lse`、`enable_pdl`、`kv_cache_sf`。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

**示例**

```python
import torch, flashinfer
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
    workspace, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    head_dim_qk=128, head_dim_vo=128,
)
wrapper.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
             num_qo_heads, num_kv_heads, 128, page_size)
o = wrapper.run(q, paged_kv_cache, bidirectional_ranges)
```

---

#### `BatchAttention`

```python
BatchAttention(
    kv_layout: str = "NHD",
    device: str = "cuda",
)
```

**功能**：整体式批量注意力包装器，把分页 prefill 与分页 decode 请求融合进单次内核启动。
`BatchAttention` 依据 `plan` 时提供的 `qo_indptr`/`kv_indptr` 区间，逐请求在 prefill 式与 decode 式
执行之间分发，使服务栈能提交混合批（如一些 prompt 在 prefill，另一些在 decode）而无需拆成两个
包装器。workspace 缓冲由实例拥有并在 `plan`/`run` 之间复用。这是混合 prefill/decode 请求、单次
内核启动服务栈的推荐入口。

**参数**

- `kv_layout`（`str`，默认 `"NHD"`）：分页 KV-cache 张量的 layout，`"NHD"`（token 主序）或
  `"HND"`（head 主序）。
- `device`（`str`，默认 `"cuda"`）：拥有内部 workspace 缓冲的 CUDA 设备，如 `"cuda"` 或 `"cuda:0"`。

#### `plan`

```python
plan(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    causal: bool = False,
    sm_scale: float = None,
    logits_soft_cap: Optional[float] = None,
    q_data_type: torch.dtype = torch.bfloat16,
    kv_data_type: torch.dtype = torch.bfloat16,
    use_profiler: bool = False,
) -> None
```

**功能**：规划混合批量注意力。

**参数**

- `qo_indptr`/`kv_indptr`/`kv_indices`（`torch.Tensor`）：query 与 KV 的 indptr/索引。
- `kv_len_arr`（`torch.Tensor`）：每请求 KV-cache 长度（token），形状 `[batch_size]`，int32。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数，必须整除 `num_qo_heads`。
- `head_dim_qk`（`int`）：query/key 每头维度。
- `head_dim_vo`（`int`）：value/output 每头维度。
- `page_size`（`int`）：分页 KV-cache 页大小。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码。
- `sm_scale`（`float`，默认 `None`）：softmax scale；`None` 时 `1/sqrt(head_dim_qk)`。
- `logits_soft_cap`（`Optional[float]`，默认 `None`）：logits soft-cap；`None`/`0` 禁用。
- `q_data_type`（`torch.dtype`，默认 `torch.bfloat16`）：query dtype。
- `kv_data_type`（`torch.dtype`，默认 `torch.bfloat16`）：key/value dtype。
- `use_profiler`（`bool`，默认 `False`）：是否编译 profiler 变体内核。

**注意事项**：`head_dim > 256` 不支持（holistic 持久内核限制），会抛 `ValueError`。

#### `prewarm_paged_kv_stride_variant`

```python
prewarm_paged_kv_stride_variant(variant: str = "independent") -> None
```

**功能**：预热惰性 paged-KV-stride 变体；CUDA Graph 捕获前调用。

#### `run`

```python
run(
    q: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    logits_soft_cap: float = 0.0,
    profiler_buffer: Optional[torch.Tensor] = None,
    kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：计算混合批量注意力，返回 `(out, lse)`。

---

#### `BatchAttentionWithAttentionSinkWrapper`

```python
BatchAttentionWithAttentionSinkWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    qo_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indptr_buf: Optional[torch.Tensor] = None,
    paged_kv_indices_buf: Optional[torch.Tensor] = None,
    paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
    custom_mask_buf: Optional[torch.Tensor] = None,
    mask_indptr_buf: Optional[torch.Tensor] = None,
    backend: str = "auto",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    q_data_type: torch.dtype = torch.bfloat16,
    kv_data_type: torch.dtype = torch.bfloat16,
    head_dim_qk: int = 128,
    head_dim_vo: int = 128,
    window_left: int = -1,
) -> None
```

**功能**：支持 attention sink 的 prefill/decode 分页 KV-cache 注意力包装器。继承
`BatchPrefillWithPagedKVCacheWrapper`，在 prefill 或 decode 注意力期间便捷地使用 attention sink。

**参数**：同 `BatchPrefillWithPagedKVCacheWrapper`，另加：

- `backend`（`str`，默认 `"auto"`）：仅 `["fa2", "fa3", "auto"]`；`auto` 时在初始化 JIT 模块前
  分发后端。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）、`use_fp16_qk_reduction`（`bool`，默认 `False`）、
  `q_data_type`/`kv_data_type`（`torch.dtype`，默认 `torch.bfloat16`）、`head_dim_qk`/`head_dim_vo`
  （`int`，默认 `128`）、`window_left`（`int`，默认 `-1`）。

**注意事项**：`plan`/`run` 复用父类实现；通过 `run(..., sinks=...)` 传入 attention sink（额外
per-head 张量，float），它被编译进内核的 `additional_tensor_names=["sink"]`。

### SM120 NVFP4

#### `nvfp4_attention_sm120_quantize_qkv`

```python
nvfp4_attention_sm120_quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]
```

**功能**：为 SM120 NVFP4 注意力预处理并量化稠密 Q/K/V。输入 layout：Q 为
`[batch, num_qo_heads, M, head_dim]`，K/V 为 `[batch, num_kv_heads, N, head_dim]`。输入须为连续
CUDA 张量，且 batch 大小、head 维度、dtype、设备一致。`num_qo_heads` 须大于等于且整除
`num_kv_heads`。Q 独立填充到 `M_pad = round_up(M, 128)`；K/V 填充到 `N_pad = round_up(N, 128)`。

**参数**

- `q, k, v`（`torch.Tensor`）：稠密 Q/K/V，dtype fp16 或 bf16。
- `per_block_mean`（`bool`，默认 `True`）：是否每 128-token 块对 Q 中心化。为 `False` 时 Q 在全
  序列上一次性中心化。

**返回**：`Tuple[torch.Tensor, ...]`：`q_fp4`、`k_fp4`、转置的 `v_fp4_t`、scale 张量 `q_scale`、
`k_scale`、`v_scale_t`，以及紧凑 FP32 QK 修正，形状 `[batch, num_qo_heads, M_pad / 128, N_pad]`
（`per_block_mean=False` 时为 `[batch, num_qo_heads, 1, N_pad]`）。

---

#### `nvfp4_attention_sm120_fwd`

```python
nvfp4_attention_sm120_fwd(
    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction,
    sm_scale=None, causal=False, per_block_mean=True, out=None, lse=None,
    out_dtype=torch.bfloat16, softmax_scale=None, return_lse=True,
    unpadded_k_len=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：在预量化 Q/K/V 上运行 SM120 NVFP4 注意力。打包张量应由
`nvfp4_attention_sm120_quantize_qkv` 产生。`q_fp4` layout 为
`[batch, num_qo_heads, M_pad, head_dim / 2]`，`k_fp4` 为 `[batch, num_kv_heads, N_pad,
head_dim / 2]`。`v_fp4_t` 与 `v_scale_t` 以 `num_kv_heads` 转置存储。

**参数**

- `q_fp4, k_fp4, v_fp4_t`（`torch.Tensor`）：打包 NVFP4 Q/K/V。
- `q_scale, k_scale, v_scale_t`（`torch.Tensor`）：Q/K/V 的逐向量 FP8 scale。
- `qk_correction`（`torch.Tensor`）：`nvfp4_attention_sm120_quantize_qkv` 返回的紧凑 FP32 修正项，
  每 128-token Q 块一行。
- `sm_scale`（`Optional[float]`）：softmax 前施加在 QK 分数上的 scale；省略时 `1/sqrt(head_dim)`。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码。
- `per_block_mean`（`bool`，默认 `True`）：须与 `nvfp4_attention_sm120_quantize_qkv` 一致。
- `out`、`lse`（`Optional[torch.Tensor]`）：可选输出与 log-sum-exp 缓冲。
- `out_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：未提供 `out` 时使用的输出 dtype。
- `softmax_scale`（`Optional[float]`）：`sm_scale` 的废弃别名。
- `return_lse`（`bool`，默认 `True`）：是否计算并返回 LSE（为兼容旧的 `(out, lse)` 契约默认
  `True`）；设为 `False` 时只返回注意力输出。
- `unpadded_k_len`（`Optional[int]`）：逻辑 K/V 序列长度；位于或超过此位置的值在 softmax 前被
  掩码；默认为物理 `N_pad` 范围以向后兼容。

**返回**：`return_lse=False` 时返回注意力输出；否则返回 `(输出, lse)`。

### MLA

MLA（Multi-head Latent Attention）是 DeepSeek 系列模型提出的注意力机制。

> **NVFP4 缓存与后端说明（摘录自 rst）**：`kv_cache_format="nvfp4"` 指每 token 384 字节的
> DeepSeek-V4 NVFP4 稀疏缓存（448 个 NoPE 值以 E2M1 表示、每 16 值一个 E4M3 scale，64 个 BF16
> RoPE 值；每页 `page_size * 352` 数据字节后接 `page_size * 32` scale 字节），由 SM120/SM121 上的
> `backend="sparse"`，以及 SM100/SM103（B200 / GB300）上的 `backend="cake"` 消费。CAKE 路由接受
> BF16 query、两张独立表（`sparse_indices` 覆盖 `swa_kv_cache`，`extra_sparse_indices` 覆盖
> `compressed_kv_cache`，各带自己的 `*_topk_lens`；`-1` 条目被掩码）、`sinks`，以及由
> `flashinfer.mla.cake_dsv4.get_cake_dsv4_workspace_bytes` 定尺寸的调用方持有 `workspace_buffer`，
> 且 CUDA-Graph safe。缓存的构建见 `nvfp4_quantize_pack_sparse_mla_cache` /
> `nvfp4_quantize_append_sparse_mla_cache`。`backend="auto"` 在 SM100/SM103 上仍选 TRTLLM-GEN；
> 欲用 Cake 须显式传 `backend="cake"`。
>
> `backend="cute-dsl"` 时，把 `hca_swa_indices` 作为绝对行传入展平的 SWA cache，
> `hca_compressed_block_tables` 作为物理压缩缓存页 ID。SWA 表形状 `[B * Q, 128]`，可表达环形
> 旋转/回绕。`hca_sparse_indices_format="compressed-page-aligned"` 可对压缩段为规范页扩展的合并表
> 做兼容性转换。

#### `BatchMLAPagedAttentionWrapper`

```python
BatchMLAPagedAttentionWrapper(
    float_workspace_buffer: torch.Tensor,
    use_cuda_graph: bool = False,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_len_arr: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> None
```

**功能**：DeepSeek 模型的 MLA 分页注意力包装器。面向矩阵吸收（Matrix Absorption）形式的 MLA 的
decode 与增量 prefill，其中 query/key 与 value/output 投影在注意力之前被吸收。非吸收 MLA prefill
路径请用相应 prefill 包装器。

规划的包装器面拥有 FA2、FA3、CUTLASS、cuTile、TRTLLM-GEN、XQA 与 CuTe DSL 的规划与执行。用规范
元数据调用一次 `plan`，再调用 `run`；plan 捕获受支持的输入/输出契约与具体后端的元数据表示。

**生命周期**：`__init__` → `plan` → `run`。

#### `plan`

`plan` 有多个重载：一个以 `metadata=MLAPlanMetadata` 的纯关键字形式，以及若干传统位置参数形式
（`qo_indptr`/`kv_indptr`/`kv_indices`/`kv_len_arr`，或以 `cum_seq_lens_q`/`block_tables`/
`seq_lens` 表示）。公共参数包括：

```python
plan(
    ..., num_heads, head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale,
    q_data_type, kv_data_type, use_profiler=False, *,
    metadata=None, cum_seq_lens_q=None, block_tables=None, seq_lens=None,
    max_q_len=None, query_layout="split", kv_cache_layout="split",
    lse_mode="none", output_dtype=None, output_scale="none",
    scale_mode="default", skip_softmax=False, enable_pdl=None, use_sinks=False,
)
```

**主要参数**：`num_heads`、`head_dim_ckv`、`head_dim_kpe`、`page_size`、`causal`、`sm_scale`、
`q_data_type`、`kv_data_type`；`query_layout`/`kv_cache_layout`（`"packed"` 或 `"split"`）、
`lse_mode`（`"none"`/`"base2"`/`"basee"`）、`output_dtype`、`output_scale`、`scale_mode`
（`"default"`/`"kv-per-tensor"`/`"bmm-scalar"`/`"bmm-tensor"`）、`skip_softmax`、`enable_pdl`、
`use_sinks`、`use_profiler`。

#### `run`

```python
run(
    *, query=None, kv_cache=None, q_nope=None, q_pe=None, ckv_cache=None,
    kpe_cache=None, out=None, lse=None, return_lse=False,
    profiler_buffer=None, kv_len=None, page_table=None,
    return_lse_base_on_e=False, o_scale=None, ckv_scale=None,
    ckv_scale_arr=None, kpe_scale=None, sinks=None,
    skip_softmax_threshold_scale_factor=None, bmm1_scale=None, ...
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：运行 MLA 分页注意力。支持打包 query/KV（`query`/`kv_cache`）或分离的
`q_nope`/`q_pe`/`ckv_cache`/`kpe_cache` 形式。

**主要参数**：`out`、`lse`/`return_lse`、`profiler_buffer`、`kv_len`、`page_table`、
`return_lse_base_on_e`、`o_scale`、`ckv_scale`/`ckv_scale_arr`/`kpe_scale`、`sinks`、
`skip_softmax_threshold_scale_factor`、`bmm1_scale`。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

---

#### `trtllm_batch_decode_with_kv_cache_mla`

```python
trtllm_batch_decode_with_kv_cache_mla(
    query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank,
    qk_rope_head_dim, block_tables, seq_lens, max_seq_len, sparse_mla_top_k=0,
    out=None, bmm1_scale=1.0, bmm2_scale=1.0, sinks=None,
    skip_softmax_threshold_scale_factor=None, enable_pdl=None, backend="auto",
    is_var_seq=True, uses_shared_paged_kv_idx=True, lse=None, return_lse=False,
    cute_dsl_impl="auto", kv_scale_format="auto", cum_seq_lens_q=None,
    max_q_len=None, multi_ctas_kv_counter_buffer=None, sparse_mla_top_k_lens=None,
    enable_dcp=False, cp_world=1, cp_rank=0, causal_seqlens_kv_global=None,
    use_fp16_softmax=None, return_lse_base=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：trtllm-gen 后端的 MLA 批量 decode。参数文档见 `_mla_with_kv_cache_impl`。

**主要参数**：`query`、`kv_cache`、`workspace_buffer`、`qk_nope_head_dim`（将在 1.0 移除）、
`kv_lora_rank`、`qk_rope_head_dim`、`block_tables`、`seq_lens`、`max_seq_len`、`sparse_mla_top_k`、
`out`、`bmm1_scale`/`bmm2_scale`、`sinks`、`backend`、`is_var_seq`、`uses_shared_paged_kv_idx`、
`lse`/`return_lse`、`cute_dsl_impl`、`kv_scale_format`、`cum_seq_lens_q`、`max_q_len`、
`sparse_mla_top_k_lens`、`enable_dcp`、`cp_world`/`cp_rank`、`causal_seqlens_kv_global`、
`use_fp16_softmax`、`return_lse_base`（`"basee"`/`"base2"`）。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

---

#### `trtllm_prefill_with_kv_cache_mla`

```python
trtllm_prefill_with_kv_cache_mla(...)  # 签名与 trtllm_batch_decode_with_kv_cache_mla 相同
```

**功能**：以 decode 兼容语义运行 MLA prefill。与 `trtllm_batch_decode_with_kv_cache_mla` 共享实现、
参数与返回契约、异常与后端选择。`q_len_per_request == 1` 时为 decode；compatible 后端下
`q_len_per_request > 1` 时为增量 prefill 与多 token 预测（MTP）。

**注意事项**：XQA 仅支持 `q_len_per_request == 1`。由于 `backend="auto"` 可能选 XQA，需要多 token
prefill 的调用方必须显式选择兼容后端。该入口也以 `flashinfer.prefill.trtllm_prefill_with_kv_cache_mla`
提供。

---

#### `trtllm_batch_decode_sparse_mla_dsv4`

```python
trtllm_batch_decode_sparse_mla_dsv4(
    query, swa_kv_cache, workspace_buffer, sparse_indices=None,
    compressed_kv_cache=None, sparse_topk_lens=None, seq_lens=None, out=None,
    bmm1_scale=1.0, bmm2_scale=1.0, sinks=None, kv_layout="HND",
    cum_seq_lens_q=None, max_q_len=None, enable_pdl=None, swa_topk_lens=None,
    extra_sparse_indices=None, extra_sparse_topk_lens=None, backend="auto",
    hca_swa_indices=None, hca_compressed_block_tables=None, hca_seq_lens=None,
    hca_is_causal=True, hca_use_persistent=False, hca_sparse_indices_format=None,
    remapped_sparse_indices_buffer=None, sparse_indices_are_storage_offsets=None,
    dsv4_inv_rope_cos_sin_cache=None, dsv4_output_scale=None,
    sparse_topk_lens_offset=0, multi_ctas_kv_counter_buffer=None, *,
    kv_cache_format="fp8",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：DeepSeek V4 稀疏 MLA decode。实现由 query 设备架构选择。

- 在 SM100/SM103/SM107 上调用预构建的 DeepSeek V4 稀疏 MLA cubin。query 与两个 KV 池使用
  head dim 512。query 可为 BF16 或 per-tensor FP8 E4M3，默认输出 BF16。提供
  `dsv4_inv_rope_cos_sin_cache` 时，固定 TRTLLM-GEN RopeQuant 尾声改写 group-major FP8 E4M3 值
  与打包 UE8M0 scale。`sparse_indices` 前 128 列为 `swa_kv_cache` 的 SWA 条目，其余列为
  `compressed_kv_cache` 的压缩条目；`sparse_topk_lens` 给出每个 query token 的总活跃稀疏长度
  （须含固定 128 个 SWA 条目）；`seq_lens` 提供 SWA 有效性窗口的原始 KV 序列长度。
- 在 SM120/SM121 上调用 packed sparse 后端。`swa_kv_cache` 为必需的打包 uint8 SWA 池：FP8 每
  token 584 字节，FP8 DSv4.1（`kv_cache_format="fp8_dsv41"`）528 字节，或 group-16 NVFP4 384
  字节（由 `kv_cache_format` 选择）。
- `backend="cute-dsl"`（SM100/SM103）调用 DeepSeek V4 HCA 内核。
- `backend="cake"`（SM120/SM121）且 `kv_cache_format="nvfp4"` 时经 SM120 `"sparse"` 调用面调用
  Cake SM120 NVFP4 稀疏 MLA 内核。

**主要参数**：`query`、`swa_kv_cache`、`workspace_buffer`、`sparse_indices`、
`compressed_kv_cache`、`sparse_topk_lens`、`seq_lens`、`out`、`bmm1_scale`/`bmm2_scale`、
`sinks`、`kv_layout`、`cum_seq_lens_q`、`max_q_len`、`enable_pdl`、`swa_topk_lens`、
`extra_sparse_indices`/`extra_sparse_topk_lens`、`backend`（`"auto"`/`"trtllm-gen"`/`"cute-dsl"`/
`"sparse"`/`"cake"`）、HCA 相关参数、`dsv4_inv_rope_cos_sin_cache`、`dsv4_output_scale`、
`sparse_topk_lens_offset`、`kv_cache_format`（`"fp8"`/`"nvfp4"`/`"fp8_dsv41"`/`"fp8_dsv41_fp4_ca"`）。

**返回**：注意力输出；`return_lse=True` 时返回 `(输出, lse)`。

---

#### 稀疏 MLA 缓存量化写入函数

```python
nvfp4_quantize_pack_sparse_mla_cache(latent_kv: torch.Tensor, *, kv_layout: str = "HND") -> torch.Tensor
nvfp4_quantize_append_sparse_mla_cache(latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor) -> None
dsv41_fp4_quantize_pack_sparse_mla_cache(latent_kv: torch.Tensor, *, kv_layout: str = "HND") -> torch.Tensor
dsv41_fp4_quantize_append_sparse_mla_cache(latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor) -> None
dsv41_fp8_quantize_pack_sparse_mla_cache(latent_kv: torch.Tensor, *, kv_layout: str = "HND") -> torch.Tensor
dsv41_fp8_quantize_append_sparse_mla_cache(latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor) -> None
```

**功能**：把 MLA latent-KV 页量化并打包/追加到稀疏 MLA cache。

- `nvfp4_quantize_pack_sparse_mla_cache`：打包（pack）为 NVFP4 缓存（对全部四种架构统一实现）。
- `nvfp4_quantize_append_sparse_mla_cache`：按 `slot_mapping` 追加写入 `cache`。
- `dsv41_fp4_quantize_pack_sparse_mla_cache`：把完整 DeepSeek-V4.1 latent-KV 页量化为 V41_FP4
  ABI。`latent_kv` 为连续 CUDA BF16/FP16 张量，形状 `[num_pages, page_size, 512]`（HND/NHD 位置的
  单例 latent-head 轴也可接受）。全 512 值按 16 一组量化为打包 E2M1 与 E4M3 scale
  （`scale = amax/6`）。返回不透明 uint8 分页缓存，逻辑形状 `[num_pages, 1, page_size, 288]`
  （HND）或 `[num_pages, page_size, 1, 288]`（NHD）；每物理页存 `page_size * 256` 数据字节后接
  `page_size * 32` scale 字节。**消费者不得把 last dim 解释为连续的逐 token 记录。**
- `dsv41_fp8_quantize_pack_sparse_mla_cache`：量化为 528 字节 DSV4_1 FP8 ABI。`latent_kv` 同上；
  全 512 值按 32 一组量化为 E4M3 与每组的 UE8M0（2 的幂）scale
  （`scale = 2**ceil(log2(max(amax/448, 1e-4)))`，`amax` 下限 `1e-4`，值除以 scale 后
  round-to-nearest-even E4M3、饱和于 448）。组内 NaN 写 `0xFF` scale 字节与 `0x7F` code；无穷组写
  `0xFF` scale 字节、无穷值 `0x7F`、有限值为带符号零 code。返回不透明 uint8 缓存，逻辑形状
  `[num_pages, 1, page_size, 528]`（HND）或 `[num_pages, page_size, 1, 528]`（NHD）；每页
  `page_size * 512` E4M3 数据字节后接 `page_size * 16` UE8M0 scale 字节。

**参数**

- `latent_kv`（`torch.Tensor`）：连续 CUDA BF16/FP16，形状 `[num_pages, page_size, 512]`。
- `slot_mapping`（`torch.Tensor`）：追加写入的槽位映射。
- `cache`（`torch.Tensor`）：目标缓存（原地更新）。
- `kv_layout`（`str`，默认 `"HND"`）：输出 layout，`"HND"` 或 `"NHD"`。

**返回**：`pack` 变体返回不透明 uint8 分页缓存；`append` 变体返回 `None`（原地更新）。

---

#### `convert_compressed_page_aligned_sparse_indices_to_hca_metadata`

```python
convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    *,
    q_len: int,
    kv_layout: Literal["HND", "NHD"] = "HND",
) -> DSV4HCAMetadata
```

**功能**：把合并的稀疏索引转换为可复用的 HCA 元数据。活跃的 SWA 条目仍是任意的绝对 token 行索引，
可表达环形缓冲旋转或回绕。压缩段必须是规范扩展 `page_id * page_size + page_offset`。对于稠密因果
query，该转换要求 `visible_raw_len = seq_len - q_len + query_position + 1` 且
`sparse_topk_len = 128 + floor(visible_raw_len / 128)`。

**参数**

- `sparse_indices`（`torch.Tensor`）：合并的稀疏索引。
- `sparse_topk_lens`（`torch.Tensor`）：每个 query token 的活跃稀疏长度。
- `seq_lens`（`torch.Tensor`）：KV 序列长度。
- `swa_kv_cache`（`torch.Tensor`）：SWA KV cache。
- `compressed_kv_cache`（`torch.Tensor`）：压缩 KV cache。
- `q_len`（`int`）：query 长度。
- `kv_layout`（`Literal["HND", "NHD"]`，默认 `"HND"`）：cache layout。

**返回**：`DSV4HCAMetadata`：可复用的 HCA 元数据。

**注意事项**

- 该转换校验张量值、分配元数据并同步设备。它**不是 CUDA Graph 捕获安全**的，且不得在延迟敏感的
  decode 循环中重复。应在 setup 期间运行，然后用返回的索引与 compressed block table 配合
  `backend="cute-dsl"` 复用。

---

#### `DSV4HCAMetadata`

**功能**：CuTe DSL 因果 HCA 后端的可复用元数据容器。

**属性**

- `hca_swa_indices`：进入展平 SWA cache 的绝对行。
- `hca_compressed_block_tables`：物理压缩缓存页 ID。
- `hca_seq_lens`：描述 TMA 调度的后端 HCA 足迹，可能超过逐行有效 `sparse_topk_lens`。

前两者按展平的 query 行（`B * Q`）排序。由
`convert_compressed_page_aligned_sparse_indices_to_hca_metadata` 产出，并作为
`trtllm_batch_decode_sparse_mla_dsv4` 的输入。

---

#### `xqa_batch_decode_with_kv_cache_mla`

```python
xqa_batch_decode_with_kv_cache_mla(
    query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank,
    qk_rope_head_dim, block_tables, seq_lens, max_seq_len, out=None,
    bmm1_scale=1.0, bmm2_scale=1.0, sinks=None, enable_pdl=None,
) -> torch.Tensor
```

**功能**：XQA 后端的批量 MLA decode。针对 SM120a/SM121a tensor core 优化的单 query（MTP 感知）
MLA decode 内核，接受 DeepSeek-V3/R1 推理所用的拼接 `(q_nope || q_rope)` query 与
`(ckv || kpe)` 分页 KV cache 布局。

**参数**

- `query`（`torch.Tensor`）：`[batch_size, q_len_per_request, num_heads, head_dim_qk]`，
  `head_dim_qk = kv_lora_rank + qk_rope_head_dim`，须为 `[q_nope, q_rope]` 拼接。
  `q_len_per_request` 当前必须为 `1`。
- `kv_cache`（`torch.Tensor`）：分页 KV cache，3-D
  `[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]` 或 4-D
  `[num_pages, 1, page_size, ...]`，last dim 为 `[ckv_cache, kpe_cache]` 拼接。
- `workspace_buffer`（`torch.Tensor`）：预分配后端 scratch workspace。
- `qk_nope_head_dim`（`int`）：非 RoPE head 维度，必须为 `128`（将在 1.0 移除，改用 `kv_lora_rank`）。
- `kv_lora_rank`（`int`）：latent KV 投影秩，必须为 `512`。
- `qk_rope_head_dim`（`int`）：RoPE head 维度。
- `block_tables`、`seq_lens`、`max_seq_len`、`out`、`bmm1_scale`/`bmm2_scale`、`sinks`、
  `enable_pdl`。

**返回**：`torch.Tensor`：MLA decode 输出。

---

#### `supported_sparse_mla_sm120_configs`

```python
supported_sparse_mla_sm120_configs(*, kv_cache_format: str = "fp8") -> dict[str, SparseMLASm120DecodeConfig]
```

**功能**：枚举已实例化的 SM120 稀疏 MLA decode 内核配置。让调用方在初始化时校验服务配置，而不是
在首个 decode 形态请求时才发现未实例化的 `(num_heads, topk)` 对。答案来自族级静态 Python 表——
一个粗粒度预过滤，从不编译内核或初始化 CUDA，故可在无 GPU 主机及 `FLASHINFER_DISABLE_JIT` 下工作。
C++ dispatcher 仍是权威：落在包络内的形状仍可能在 dispatch 时被元数据规则（layout、精度、双缓存
配对）拒绝。

**参数**

- `kv_cache_format`（`{"fp8", "nvfp4"}`，默认 `"fp8"`）：请求其独立标定内核包络的存储格式。

**返回**

- `dict[str, SparseMLASm120DecodeConfig]`：从内核族到其已实例化 decode 集的映射。FP8 返回 DSv4、
  DSv3.2、GLM-NSA、GLM53_NOPE、DOTS3_SWA 与 DSV4.1 条目；NVFP4 目前返回独立标定的 DSv4 条目。

**示例**

```python
import flashinfer
configs = flashinfer.mla.supported_sparse_mla_sm120_configs()
configs["dsv4"].supports_decode(num_heads=64, topk=256)  # True
nvfp4 = flashinfer.mla.supported_sparse_mla_sm120_configs(kv_cache_format="nvfp4")
nvfp4["dsv4"].bytes_per_token  # 384
```

---

#### `SparseMLASm120DecodeConfig`

**功能**：某一 SM120 稀疏 MLA 内核族的已实例化 decode 内核集。decode 形态调用（`num_tokens <=
max_num_tokens`）在形状匹配这里描述的某个实例时优先使用独立 decode 内核。对 FP8，prefill
orchestrator 可在其自身包络中服务剩余 decode 形态；NVFP4 目前两内核使用相同的精确 head/top-k
包络。crossover 标定可能把合格形状路由到 prefill。本配置仅描述 decode。值来自静态 Python 表
（用于 init-time 探测的粗预过滤）；C++ dispatcher 是权威。

**属性**

- `d_qk`（`int`）：该族服务的 query/key head 维度（DSv4 / GLM53_NOPE 为 `512`，DSv3.2 /
  GLM-NSA 为 `576`，DOTS3_SWA 滑窗族为 `1088`，其 `d_v` 为 1024）。
- `page_block_size`（`int`）：默认页大小；除非 `page_block_sizes` 列出更多实例化大小或
  `page_block_size_is_runtime` 置位。
- `page_block_sizes`（`frozenset[int]`）：精确实例化的主页大小集。
- `page_block_size_is_runtime`（`bool`）：是否接受独立的正页大小而非固定主页大小。
- `max_num_tokens`（`int`）：路由到 decode 内核的最大 `num_tokens`（含）。
- `topks`（`frozenset[int]`）：标定的 top-k 值。
- `min_topk`（`int`）：最小合法 `topk`（索引行宽）。滑窗族为 `513`，其他为 `1`。
- `max_num_heads`（`int`）：head 数包络上界。
- `kv_cache_format`（`str`）：本条目描述的打包缓存格式（`"fp8"` 或 `"nvfp4"`）。
- `bytes_per_token`（`int`）：每 token 的逻辑打包缓存字节数。
- `head_counts`（`Optional[frozenset[int]]`）：内核无 runtime-head 回退时的精确实例化 head 数。
- `topk_is_runtime`（`bool`）：是否接受每个 `topk >= min_topk`。
- `extra_page_block_sizes`（`frozenset[int]`）：可选次级缓存接受的精确页大小。

---

#### `SparseMLASm120Wrapper`

```python
SparseMLASm120Wrapper(
    max_num_tokens: Optional[int] = None,
    max_num_heads: Optional[int] = None,
    *,
    d_v: int = 512,
    kv_scale_format: str = "auto",
    kv_cache_format: str = "fp8",
    extra_kv_fp4: bool = False,
    compute_precision: str = "default",
    device: Optional[torch.device] = None,
    backend: str = "auto",
) -> None
```

**功能**：SM120 稀疏 MLA 分页注意力实现 runner。提供规划的（plan/run）SM120 稀疏 MLA 调用面，
支持 `fp8` 与 `nvfp4` 缓存格式、主/次（SWA/compressed）双缓存及 HND/NHD/3-D 打包 cache 布局。

`max_num_tokens` 与 `max_num_heads` 为可选校验上界。Eager 调用缓存精确形状的执行计划与专用最终
LSE 缓冲。未捕获的调用共享 grow-only 的 mid-output 与 mid-LSE scratch arena。某预热形状的首次
捕获会在其内核调用前分配专用 scratch；该形状随后在形状变化间保留其 pinned 缓冲。Decode scratch
也可经 `run(mid_out=..., mid_lse=...)` 以具有足够 plan 定义字节容量的连续缓冲提供。**先预热每一个
被捕获的元数据形状。共享 wrapper 自有 scratch 的调用不得在不同流上并发执行。** Eager tuning 可能
在 profile 变化时替换某形状的 plan 与资源；此后须重新捕获图。另一进程的 calibration-store 更新也
可能在下一次 eager 调用时改变 plan workspace 并要求重新捕获。wrapper 不拥有 graph 对象，也不维护
历史 profile 资源。

**参数**

- `max_num_tokens`（`Optional[int]`，默认 `None`）：wrapper 接受的最坏情形 `num_tokens`。必须与
  `max_num_heads` 一同提供。
- `max_num_heads`（`Optional[int]`，默认 `None`）：最坏情形 `num_heads`。
- `d_v`（`int`）：value head 维度。DSV3_2 / DSV4 / GLM 变体为 `512`，DOTS3_SWA 为 `1024`。须与
  model type 所选 `d_qk` 一致。
- `kv_scale_format`（`str`，默认 `"auto"`）：scale 语义，用于区分共享 query 宽度的族。`"auto"` 与
  `"pow2_fp32"` 在 `d_qk=576` 选 DSv3.2 的 2 的幂 FP32 内联 scale；`"arbitrary_fp32"` 选 GLM 风格
  任意 FP32 内联 scale（GLM_NSA 在 `d_qk=576`，GLM53_NOPE 在 `d_qk=512`）；`"auto"` 在 `d_qk=512`
  选 DSV4；`"ue8m0_g32"` 在 `d_qk=512` 选 DSV4_1（DeepSeek-V4.1：对全 FP8 的 512 宽 K 使用 32 宽
  UE8M0 分组）。
- `kv_cache_format`（`{"fp8", "nvfp4"}`，默认 `"fp8"`）：打包缓存格式。两种格式复用本 wrapper 及其
  `run` 签名；每种格式保有自己的 planner 与内部内核。
- `extra_kv_fp4`（`bool`，默认 `False`）：为 `True` 时，双缓存调用的额外（压缩）缓存持有 288 字节
  V41_FP4 行，数值路由在片内解码。要求 `kv_cache_format="fp8"` 且 `kv_scale_format="ue8m0_g32"`。
- `compute_precision`（`{"default", "fp8", "bf16", "nvfp4"}`，默认 `"default"`）：本 wrapper 固定。
  `default` 保留旧的 family/shape 混合与 DSV4 NVFP4 缓存路由。
- `device`（`Optional[torch.device]`，默认 `None`）：目标设备。
- `backend`（`str`，默认 `"auto"`）：后端选择。

#### `run`

```python
run(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    prefill_impl: Optional[str] = None,
    return_lse: bool = False,
    lse_scale: float = 1.0,
) -> Optional[torch.Tensor]
```

**功能**：运行 SM120 稀疏 MLA 分页注意力。`q`、`kv_cache`、`indices`、`output`、`sm_scale` 为
主要参数；`topk_length`、`attn_sink`、`extra_kv_cache`/`extra_indices`/`extra_topk_length`（双缓存
的压缩段）、`out_lse`、`mid_out`/`mid_lse`（可选的 split scratch）、`prefill_impl`、`return_lse`、
`lse_scale`（`1.0` 为自然对数 LSE，`1/log2e` 为 base-2 LSE）控制细节。

**返回**：`return_lse=True` 时返回 LSE 张量，否则返回 `None`（输出就地写入 `output`）。空有效 KV 行
产生零输出；packed 后端的最终 base-2 LSE 与有限 sink 语义见源码 docstring。

---

#### Cake SM120 NVFP4 稀疏 MLA 函数族

```python
cake_sparse_mla_sm120_dsv4_nvfp4_decode(
    q, kv_cache, indices, output, out_lse, sm_scale, *,
    topk_length=None, attn_sink=None, extra_kv_cache=None, extra_indices=None,
    extra_topk_length=None, mid_out=None, mid_lse=None, lse_scale=1.0,
    num_splits=None, max_splits=16, head_tiles=None,
) -> Dict[str, int]

cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
    q, kv_cache, indices, output, out_lse, sm_scale, *,
    topk_length=None, attn_sink=None, extra_kv_cache=None, extra_indices=None,
    extra_topk_length=None, lse_scale=1.0, head_tiles=None,
) -> Dict[str, int]

cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
    *, num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0, num_sms: int,
) -> str

cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
    *, num_tokens, num_heads, topk, extra_topk=0, num_sms,
) -> int

cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles(
    *, num_tokens, num_heads, topk, extra_topk=0, num_sms,
) -> int

cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits(
    *, num_tokens, num_heads, topk, extra_topk=0, num_sms,
    max_splits=16, head_tiles=1,
) -> Tuple[int, int]

cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0,
) -> int
```

**功能**：Cake SM120 NVFP4 稀疏 MLA 内核族。

- `..._decode`：split decode 内核（split-KV partial 写入 `mid_out`/`mid_lse`）；返回统计字典。
- `..._prefill`：单次启动 prefill 内核（每 token 与 head 块一个 CTA，覆盖其全部候选，无 split
  scratch）；返回统计字典。
- `..._select_kernel`：依据实测 crossover 选择 `"decode"` 或 `"prefill"` 内核，返回内核名字符串。
- `..._plan_prefill`：规划 prefill 启动，返回整数。
- `..._plan_head_tiles`：规划 head tile 数，返回整数。
- `..._plan_splits`：规划 `(num_splits, head_tiles)` 等，返回二元组。
- `..._scratch_bytes`：返回 split-KV partial 与 LSE scratch 字节数，用于为
  `trtllm_batch_decode_sparse_mla_dsv4` 的 `workspace_buffer` 定尺寸。

**注意事项**：SM120/SM121 且 `kv_cache_format="nvfp4"` 时，一个 query token 运行 Cake split
decode 内核；多个 token 时依 `..._select_kernel` 运行 decode 或单次启动 prefill 内核
（DeepSeek-V4 sparse MLA SM120 tracker：flashinfer#4254）。

---

#### `cake_dsv4_nvfp4_rope_quantize_insert`

```python
cake_dsv4_nvfp4_rope_quantize_insert(
    q, kv, cache, slot_mapping, positions, cos_sin_cache, *,
    q_head_padded=0, apply_q_rope=True, q_inplace=False,
) -> torch.Tensor
```

**功能**：融合的滑窗池写入器：GPT-J RoPE（对 query 与 latent KV）、head 填充的 `q_out`（或
`q_inplace=True` 且无 head padding 时原地旋转 query）、NVFP4 量化与分页写入，一次启动完成。

**参数**

- `q`、`kv`、`cache`、`slot_mapping`、`positions`、`cos_sin_cache`（`torch.Tensor`）：query、latent
  KV、目标 cache、槽位映射、位置与 cos/sin cache。
- `q_head_padded`（`int`，默认 `0`）：query head 填充。
- `apply_q_rope`（`bool`，默认 `True`）：是否对 query 施加 RoPE。
- `q_inplace`（`bool`，默认 `False`）：是否原地旋转 query（无 head padding）。

**返回**：`torch.Tensor`：旋转/填充后的 query（`q_out` 或原地结果）。

**注意事项**：产出的字节与对 BF16 舍入后已 rope 的行施加 `nvfp4_quantize_append_sparse_mla_cache`
一致。

---

#### `cake_dsv4_nvfp4_kv_rope_quantize_insert`

```python
cake_dsv4_nvfp4_kv_rope_quantize_insert(
    kv, cache, slot_mapping, positions, cos_sin_cache, *, compress_ratio=1,
) -> None
```

**功能**：融合的压缩池写入器：对 latent KV 施加 RoPE、NVFP4 量化与写入，一次启动完成。
`compress_ratio` 为 1 或 2，或在 ratio 1 的 spec context 下使用。

**参数**

- `kv`、`cache`、`slot_mapping`、`positions`、`cos_sin_cache`（`torch.Tensor`）。
- `compress_ratio`（`int`，默认 `1`）：压缩比，1 或 2。

**返回**：`None`（原地更新 `cache`）。

**注意事项**：产出的字节与 `nvfp4_quantize_append_sparse_mla_cache` 作用于 BF16 舍入后的 roped
行一致。

## 相关链接 / 参考

- 上游 `docs/api/attention.rst`
- 源码：`flashinfer/decode.py`、`flashinfer/prefill.py`、`flashinfer/attention/`、
  `flashinfer/mla/`、`flashinfer/xqa.py`、`flashinfer/nvfp4_attention_sm120.py`、
  `flashinfer/cake_dcp.py`、`flashinfer/cake_fmha_request_ordered.py`
- 实验性任务调度注意力 PrimTS 指南：`flashinfer/attention/prims_ts/README.md`
- MLA Page Layout 文档与 FlashInfer MLA 博客：http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html
- DeepSeek-V2/V3/R1：https://arxiv.org/abs/2405.04434 、https://arxiv.org/abs/2412.19437 、
  https://arxiv.org/abs/2501.12948
- 分页 MQA logits 见 `docs-api/api-attn-scores.md`

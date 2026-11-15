# 块稀疏注意力（`flashinfer.sparse`）

> 源码：`flashinfer/sparse.py`，以及 `flashinfer/cake_vsa.py`、`flashinfer/msa_ops/`
> 上游文档：`docs/api/sparse.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.sparse` 提供块稀疏（block sparse）flash attention 支持：以 BSR 块稀疏矩阵作为注意力
掩码，支持任意块大小 `(R, C)`，也支持每个 `kv_head` 使用不同稀疏模式的变长块稀疏。除通用
Wrapper 外，本页还覆盖 Cake VSA（`flashinfer.cake_vsa`）入口，以及 Minimax Sparse Attention
（MSA，`flashinfer.msa_ops`）的稀疏 prefill/decode、top-k 选择与 proxy 打分——MSA 在 SM100/SM103/
SM107 与 SM120/SM121 Blackwell GPU 上分发（proxy 打分仅 SM120/SM121）。

## 导入

```python
from flashinfer import BlockSparseAttentionWrapper, VariableBlockSparseAttentionWrapper
from flashinfer.cake_vsa import plan_cake_vsa, run_cake_vsa
from flashinfer.msa_ops import (
    msa_proxy_score, msa_proxy_score_fp4, MSASparseAttentionWorkspace, supports_packed_kv,
    msa_sparse_attention, msa_sparse_decode_attention, msa_topk_select,
    prepare_msa_nvfp4_sparse_decode,
)
# 或
from flashinfer.sparse import BlockSparseAttentionWrapper
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BlockSparseAttentionWrapper` | 类 | 以块稀疏矩阵为掩码的注意力 Wrapper，支持任意块大小 `(R, C)`。 |
| `VariableBlockSparseAttentionWrapper` | 类 | 支持变长块与逐 kv_head 稀疏模式的 Wrapper。 |
| `plan_cake_vsa` | 函数 | 为 Cake VSA 源码级后端创建元数据与 workspace。 |
| `run_cake_vsa` | 函数 | 运行一条显式的 Cake VSA 源码级路由。 |
| `msa_proxy_score` | 函数 | SM120/SM121 的 MSA 稠密 proxy 打分（逐 KV 块最大 logits）。 |
| `msa_proxy_score_fp4` | 函数 | `msa_proxy_score` 的 NVFP4 版本。 |
| `MSASparseAttentionWorkspace` | 类 | SM100/SM103/SM107 的 MSA CUDA Graph 捕获所需调用方存储。 |
| `supports_packed_kv` | 函数 | 返回某设备上 MSA 是否接受打包分页 K/V 视图。 |
| `msa_sparse_attention` | 函数 | MSA 稀疏 prefill 前向。 |
| `msa_sparse_decode_attention` | 函数 | MSA 稀疏 decode 注意力。 |
| `msa_topk_select` | 函数 | 依据 proxy 分数挑选 top-K KV 块。 |
| `prepare_msa_nvfp4_sparse_decode` | 函数 | 用生成的 Cake 程序准备 NVFP4 分页-KV 稀疏 decode。 |

## API 详情

### `BlockSparseAttentionWrapper`

```python
BlockSparseAttentionWrapper(
    float_workspace_buffer: torch.Tensor,
    backend: str = "auto",
) -> None
```

**功能**：以块稀疏矩阵作为注意力掩码的注意力 Wrapper。块稀疏矩阵定义见 SciPy 的
`bsr_matrix`。支持任意块大小 `(R, C)`。

**生命周期**：构造 `__init__` → `plan`（创建辅助数据结构）→ `run`（可跨多次调用复用辅助结构）。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲，用于 split-k 中间
  结果。推荐 128MB，设备须与输入张量一致。
- `backend`（`str`，默认 `"auto"`）：实现后端，可为 `auto`/`fa2`/`fa3` 或 `cake`。为 `auto` 时
  根据设备架构与内核可用性自动选择。

#### `plan`

```python
plan(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    M: int,
    N: int,
    R: int,
    C: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
    packed_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Union[str, torch.dtype] = "float16",
    non_blocking: bool = True,
    block_mask: Optional[torch.Tensor] = None,
    kv_block_lens: Optional[torch.Tensor] = None,
    q2k_indices: Optional[torch.Tensor] = None,
    q2k_num: Optional[torch.Tensor] = None,
    kv_splits: Optional[Union[int, str]] = None,
    use_clc: Optional[bool] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None
```

**功能**：为块稀疏注意力创建辅助数据结构。

**参数**

- `indptr`（`Optional[torch.Tensor]`）：块稀疏矩阵行维的块索引指针，形状 `(MB + 1,)`，`MB` 为
  行维块数。除 `cake`、`vsa_sm100_blk128`、`vsa_sm100_blk64`、`vsa_sm120_blk64` 且提供
  `block_mask` 的情况外，所有后端都必需。
- `indices`（`Optional[torch.Tensor]`）：列维块索引，形状 `(nnz,)`，元素应小于 `NB`（列维块数）。
  同上，除若干后端外必需。
- `M`（`int`）：块稀疏矩阵行数，`MB = ceil_div(M, R)`。
- `N`（`int`）：列数，`NB = N // C`，`N` 须能被 `C` 整除。
- `R`（`int`）：每块的行数。
- `C`（`int`）：每块的列数。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：每头维度。
- `mask`（`Optional[torch.Tensor]`）：形状 `(nnz, R, C)` 的掩码张量；若每块都是满块则无需提供。
- `packed_mask`（`Optional[torch.Tensor]`）：一维打包掩码，提供时忽略 `mask`。由
  `flashinfer.quantization.packbits` 生成。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码（仅在未提供自定义掩码时生效）。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）：`NONE`/`ROPE_LLAMA`/`ALIBI`。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否用 f16 做 qk reduction。
- `logits_soft_cap`（`Optional[float]`）：logits soft cap 值；不提供时为 0。
- `sm_scale`（`Optional[float]`）：softmax scale；默认 `1.0 / sqrt(head_dim)`。
- `rope_scale`（`Optional[float]`）：RoPE 插值 scale；默认 `1.0`。
- `rope_theta`（`Optional[float]`）：RoPE theta；默认 `1e4`。
- `q_data_type`（`Union[str, torch.dtype]`，默认 `"float16"`）：query 数据类型。
- `kv_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：key/value 数据类型；默认
  `q_data_type`。
- `o_data_type`（`Union[str, torch.dtype]`，默认 `"float16"`）：输出数据类型。量化时无法由输入
  dtype 推断输出 dtype。
- `non_blocking`（`bool`，默认 `True`）：是否异步拷贝到设备。
- `block_mask`（`Optional[torch.Tensor]`）：逐头块级注意力掩码，dtype bool。形状可为
  `(num_qo_heads, MB, NB)` 或 `(num_kv_heads, MB, NB)`。`block_mask[h,i,j] = True` 表示 Q 块 `i`
  对 KV 块 `j` 在头 `h` 上做注意力。GQA 时（`num_qo_heads > num_kv_heads`），提供
  `(num_qo_heads, MB, NB)` 时使用每个 KV 头组的第一个 QO 头（共享 KV 头的 QO 头稀疏性必须相同）。
  由 `cake`、`vsa_sm100_blk128`、`vsa_sm100_blk64`、`vsa_sm120_blk64` 后端支持；提供时
  `indptr`/`indices` 无需给出且被忽略。
- `kv_block_lens`（`Optional[torch.Tensor]`）：每个 KV 块的有效 token 数，形状 `(NB,)`，取值须在
  `[1, C]`。由 `cake` block-64 路由支持；省略时每块按 `C` 个有效 token。
- `q2k_indices`（`Optional[torch.Tensor]`）：逐头直接 KV 块选择，连续 int32，形状
  `(num_qo_heads, MB, topk)`。由 `cake` block-64 路由支持，且与 `block_mask` 及 BSR 元数据互斥。
- `q2k_num`（`Optional[torch.Tensor]`）：每行有效选择数，连续 int32，形状 `(num_qo_heads, MB)`；
  省略时每行使用完整 `topk` 维。
- `kv_splits`（`Optional[Union[int, str]]`）：split-KV 合并路径的 KV 分割数，或 `"auto"` 由稀疏度
  启发式选。仅 `vsa_sm100_blk64` 后端支持，其他后端必须为 `None`。默认 `None` 禁用分割（等价于
  显式传 `1`）。
- `use_clc`（`Optional[bool]`）：覆盖 SM100 blk64 调度器：`True` 强制 CLC 持久调度器，`False`
  强制静态调度器，`None`（默认）用基于形状的启发式。仅 `vsa_sm100_blk64` 后端支持。
- `q_scale`（`Optional[torch.Tensor]`）：Sage FP8 的 `q` 量化 scale，形状
  `(1, num_qo_heads, seqlen_q)`，float32。仅 `vsa_sm100_blk64` 后端支持，且仅在 `q`/`k`/`v` 为
  `float8_e4m3fn` 时。必须与 `k_scale`/`v_scale` 同时提供或都不提供。
- `k_scale`（`Optional[torch.Tensor]`）：Sage FP8 的 `k` 量化 scale，形状
  `(1, num_qo_heads, ceil(seqlen_k / 16))`，float32。
- `v_scale`（`Optional[torch.Tensor]`）：Sage FP8 的 `v` 量化 scale，形状
  `(num_qo_heads, head_dim)`，float32。

#### `run`

```python
run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    enable_pdl: Optional[bool] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 Q/K/V 之间的块稀疏注意力。

**参数**

- `q`（`torch.Tensor`）：query，形状 `(M, num_qo_heads, head_dim)`。
- `k`（`torch.Tensor`）：key，形状 `(N, num_kv_heads, head_dim)`。
- `v`（`torch.Tensor`）：value，形状 `(N, num_kv_heads, head_dim)`。
- `scale_q`/`scale_k`/`scale_v`（`Optional[torch.Tensor]`）：逐头 FP8 量化 scale，形状分别为
  `[num_qo_heads]`/`[num_kv_heads]`/`[num_kv_heads]`。不提供时为 `1.0`。`cake` 后端不接受 FP8
  scale 张量。
- `out`（`Optional[torch.Tensor]`）：输出张量；不提供时内部申请。
- `lse`（`Optional[torch.Tensor]`）：注意力 logits 的 log-sum-exp；不提供时内部申请。
- `return_lse`（`bool`，默认 `False`）：是否返回 log-sum-exp。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL（>= sm90，当前仅 FA2 与 CUDA core
  decode）。

**返回**

- `return_lse=False` 时返回注意力输出，形状 `[M, num_qo_heads, head_dim]`；为 `True` 时返回元组
  `(输出, logsumexp)`，logsumexp 形状 `[M, num_qo_heads]`。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")
indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")
bsr_wrapper.plan(indptr, indices, 3, 3, 1, 1, num_qo_heads, num_kv_heads, head_dim)
q = torch.randn((3, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")
k = torch.randn((3, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
v = torch.randn((3, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
o = bsr_wrapper.run(q, k, v)
```

---

### `VariableBlockSparseAttentionWrapper`

```python
VariableBlockSparseAttentionWrapper(
    float_workspace_buffer: torch.Tensor,
    backend: str = "auto",
) -> None
```

**功能**：以块稀疏矩阵为注意力掩码的注意力 Wrapper，支持由 `block_row_sz` 与 `block_col_sz` 给出的
变长块。此外每个 `kv_head` 可指定自己的稀疏模式而不必使用相同掩码。

**生命周期**：构造 `__init__` → `plan` → `run`。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲（推荐 128MB），设备
  须与输入张量一致。
- `backend`（`str`，默认 `"auto"`）：实现后端，可为 `auto`/`fa2`/`fa3`，或显式的 `cake`（生成的
  Cake 内核；仅 Hopper SM90：BF16 HND、head_dim 128、Q/KV 头数相等、非因果 64-token 块、每
  query 块 1..64 个选中 KV 块），或 `cake_cute`（相同内核与 planner，用其 CuTe DSL 渲染构建；需
  可选的 `nvidia-cutlass-dsl` 包）。自动选择**从不**选 `cake`/`cake_cute`。

#### `plan`

```python
plan(
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = False,
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
) -> None
```

**功能**：为块稀疏注意力创建辅助数据结构。必须在任何 `run` 之前调用。`num_qo_heads` 必须是
`num_kv_heads` 的倍数；不等时使用 GQA。

**参数**

- `block_mask_map`（`torch.Tensor`）：块掩码图（bool），形状 `(num_kv_heads, MB, NB)`，`MB` 为行维
  块数，`NB` 为列维块数。
- `block_row_sz`（`torch.Tensor`）：块行大小，形状 `(num_kv_heads, MB)`。
- `block_col_sz`（`torch.Tensor`）：块列大小，形状 `(num_kv_heads, NB)`。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。一组 `qo_heads` 共享对应 `kv_heads` 的同一稀疏模式。
- `head_dim`（`int`）：每头维度。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）：`NONE`/`ROPE_LLAMA`/`ALIBI`。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否用 f16 做 qk reduction。
- `logits_soft_cap`（`Optional[float]`）：logits soft cap 值；不提供时为 0。
- `sm_scale`（`Optional[float]`）：softmax scale；默认 `1.0 / sqrt(head_dim)`。
- `rope_scale`（`Optional[float]`）：RoPE 插值 scale；默认 `1.0`。
- `rope_theta`（`Optional[float]`）：RoPE theta；默认 `1e4`。
- `non_blocking`（`bool`，默认 `True`）：是否异步拷贝到设备。
- `q_data_type`（`Union[str, torch.dtype]`，默认 `"float16"`）：query dtype，用于特化 JIT 内核。
- `kv_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：key/value dtype；`None` 时默认
  `q_data_type`。

#### `run`

```python
run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    enable_pdl: Optional[bool] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：计算 Q/K/V 之间的块稀疏注意力。

**参数**

- `q`（`torch.Tensor`）：query，形状 `(num_qo_heads, qo_len, head_dim)`。
- `k`（`torch.Tensor`）：key，形状 `(num_kv_heads, kv_len, head_dim)`。
- `v`（`torch.Tensor`）：value，形状 `(num_kv_heads, kv_len, head_dim)`。
- `out`（`Optional[torch.Tensor]`）：输出；不提供时内部申请。
- `lse`（`Optional[torch.Tensor]`）：log-sum-exp；不提供时内部申请。
- `return_lse`（`bool`，默认 `False`）：是否返回 log-sum-exp。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：是否启用 PDL（>= sm90，当前仅 FA2 与 CUDA core
  decode）。

**返回**

- `return_lse=False` 时返回注意力输出 `[M, num_qo_heads, head_dim]`；为 `True` 时返回
  `(输出, logsumexp)`。对 `backend="cake"`/`"cake_cute"`，输出遵循同一 DPS ABI（`out` 为
  `[H*M, 1, head_dim]`），返回值为 HND 视图 `[H, M, head_dim]`；`out` 必须连续且不得与
  `q`/`k`/`v` 重叠。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
wrapper = flashinfer.VariableBlockSparseAttentionWrapper(workspace_buffer)
block_mask_map = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device="cuda:0")
block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda:0")
block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda:0")
wrapper.plan(block_mask_map, block_row_sz, block_col_sz, num_qo_heads, num_kv_heads, head_dim)
o = wrapper.run(q, k, v)
```

---

### `plan_cake_vsa`

```python
plan_cake_vsa(
    indptr, indices, block_mask, kv_block_lens, q2k_indices, q2k_num, *,
    M, N, R, C, num_qo_heads, num_kv_heads, head_dim,
    q_data_type, sm_scale, device,
) -> dict[str, Any]
```

**功能**：为 Cake VSA 源码级后端创建稳定元数据与 workspace。

**参数**

- `indptr`（`Optional[torch.Tensor]`）：所有头共享的块模式 CSR 行指针，形状 `(M // R + 1,)`。当
  `block_mask` 与 `q2k_indices` 均未提供时，与 `indices` 一同必需。
- `indices`（`Optional[torch.Tensor]`）：对应 `indptr` 的列索引。重复或非紧凑行会在固定 stride
  源码内核使用前被规范化。
- `block_mask`（`Optional[torch.Tensor]`）：bool 块掩码，形状 `(num_qo_heads, M // R, N // C)` 或
  `(num_kv_heads, M // R, N // C)`。
- `kv_block_lens`（`Optional[torch.Tensor]`）：每个 KV 块的有效 token 数，形状 `(N // C,)`。仅
  64-token 块支持。
- `q2k_indices`（`Optional[torch.Tensor]`）：直接的 block-64 选择，连续 int32，形状
  `(num_qo_heads, M // R, topk)`。
- `q2k_num`（`Optional[torch.Tensor]`）：每行活跃的 `q2k_indices` 数目，连续 int32，形状
  `(num_qo_heads, M // R)`。
- `M`（`int`）：query 序列长度。
- `N`（`int`）：key/value 序列长度。
- `R`（`int`）：query 块大小。Cake 支持 64 与 128。
- `C`（`int`）：key/value 块大小，必须等于 `R`。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：每头维度。Cake 支持 64、96、128。
- `q_data_type`（`torch.dtype`）：规划的 Q/K/V dtype，`torch.float16` 或 `torch.bfloat16`。
- `sm_scale`（`Optional[float]`）：softmax scale。`None` 选 `1 / sqrt(head_dim)`。
- `device`（`torch.device`）：将执行该 plan 的 SM100 或 SM103 CUDA 设备。

**返回**

- `dict[str, Any]`：经校验的元数据与可复用 workspace，供 `run_cake_vsa` 消费。所有设备端 reduce
  与每条路由专属的元数据张量都在此构建，`run_cake_vsa` 只做张量元数据校验与启动。

**注意事项**

- `R != C` 或 `R not in (64, 128)` 会抛 `ValueError`；`kv_block_lens`/`q2k_indices` 仅 block-64
  支持；`M`/`N` 必须能被块大小整除；`head_dim` 须在 `(64, 96, 128)`；`q_data_type` 须为 fp16/bf16；
  `num_qo_heads` 须能被 `num_kv_heads` 整除。
- cake blk64 支持 native-head BF16 D128；D64/D96 路由仅支持 native-head BF16；BF16 GQA 路由要求
  `Hq=8` 且组大小为 2、4 或 8。

---

### `run_cake_vsa`

```python
run_cake_vsa(
    plan: dict[str, Any], q, k, v, *,
    out, lse, return_lse, backend,
)
```

**功能**：运行一条显式的源码级路由；没有外部回退。

**参数**

- `plan`（`dict[str, Any]`）：`plan_cake_vsa` 返回的元数据。
- `q`（`torch.Tensor`）：连续的 query，形状 `(M, num_qo_heads, head_dim)`。
- `k`（`torch.Tensor`）：连续的 key，形状 `(N, num_kv_heads, head_dim)`。
- `v`（`torch.Tensor`）：与 `k` 同形状同 dtype 的连续 value。
- `out`（`Optional[torch.Tensor]`）：可选输出缓冲，与 `q` 匹配。
- `lse`（`Optional[torch.Tensor]`）：可选 float32 log-sum-exp，形状 `(M, num_qo_heads)`。仅当
  `return_lse` 为真时接受。
- `return_lse`（`bool`）：是否随输出返回 log-sum-exp。D64/D96、BF16 GQA、ultrasparse、长序列路由
  不支持该选项。
- `backend`（`str`）：必须为 `"cake"`。

**返回**

- `torch.Tensor` 或 `tuple[torch.Tensor, torch.Tensor]`：注意力输出，或 `return_lse` 为真时的
  `(output, lse)`。

**注意事项**

- `backend != "cake"` 抛 `ValueError`；D64/D96、BF16 GQA、ultrasparse 路由在 `return_lse=True`
  时抛 `ValueError`；ultrasparse 路由要求恰好选中 6 个块。

---

### `msa_proxy_score`

```python
msa_proxy_score(
    q, k, cu_seqlens_q, cu_seqlens_k=None, *,
    page_table=None, seqused_k=None, causal=True,
    max_seqlen_q=None, max_k_tiles=None, output=None,
    reduce_heads=False, q_offset=None,
) -> torch.Tensor
```

**功能**：SM120/SM121 的 MSA 稠密 proxy 趟：逐 KV 块最大注意力 logits。计算
`max_score[h, t, q]`——在 KV 块 `t` 的 128 个 token 上、逐 query token 与 query 头、对未缩放且
causal 掩码的 `Q K^T` logits 取最大。输出直接喂给 `msa_topk_select`。超出序列有效范围的 KV 块，
或完全在因果上限之上的 KV 块，产生 `-inf`。无 softmax、无 V——这是 MSA 流水线第 1 阶段。

**参数**

- `q`（`torch.Tensor`）：`(total_q, num_qo_heads, 128)`，bf16 或 fp16（廉价 proxy Q）。当 `k` 为
  fp8 时可为 fp8 E4M3（decode 形状原生消费，其他形状内部一次性上转）。
- `k`（`torch.Tensor`）：扁平 `(total_k, num_kv_heads, 128)`（配 `cu_seqlens_k`），或分页
  `(num_pages, num_kv_heads, 128, 128)`（配 `page_table` + `seqused_k`）。可为 fp8 E4M3。
  `num_qo_heads` 必须是 `num_kv_heads` 的倍数。
- `cu_seqlens_q`（`torch.Tensor`）：`(batch_size + 1,)` int32 累积 query 长度。
- `cu_seqlens_k`（`Optional[torch.Tensor]`）：`(batch_size + 1,)` int32 累积 KV 长度。扁平 KV
  layout 必需，分页 layout 不用。
- `page_table`（`Optional[torch.Tensor]`）：分页 KV layout 的页表映射，与 `seqused_k` 一同必需。
- `seqused_k`（`Optional[torch.Tensor]`）：逐序列有效 KV token 数，分页 layout 与 `page_table`
  一同必需。
- `causal`（`bool`，默认 `True`）：右对齐因果掩码，在块最大**之前**施加。
- `max_seqlen_q`（`Optional[int]`）：最大 query 长度。某些 CUDA Graph/编译路径需要。
- `max_k_tiles`（`Optional[int]`）：输出的 KV 块列数；默认批量上最大 `ceil(seqlen_k / 128)`。
- `output`（`Optional[torch.Tensor]`）：预分配 float32 输出。一般形状
  `(num_qo_heads, max_k_tiles, total_q)`，`reduce_heads=True` 时为 `(1, max_k_tiles, total_q)`。
- `reduce_heads`（`bool`，默认 `False`）：为 `True` 时对 query 头轴做 max-reduce，返回单个
  `(1, max_k_tiles, total_q)` 分数。默认 `False` 为逐头分数。
- `q_offset`（`int` 或 `torch.Tensor`，可选）：因果对齐逻辑使用的 query 位置偏移。

**返回**

- `torch.Tensor`：float32 `max_score`，供 `msa_topk_select`：
  `(num_qo_heads, max_k_tiles, total_q)`，`reduce_heads=True` 时为 `(1, max_k_tiles, total_q)`。

**注意事项**

- 需要 SM120 或 SM121 且 CUDA >= 12.8，否则抛 `RuntimeError`。`head_dim` 必须为 128。`fp8 q`
  要求 `fp8 k`。

---

### `msa_proxy_score_fp4`

```python
msa_proxy_score_fp4(
    q_fp4, k_fp4, q_scale, k_scale, q_global_scale, k_global_scale,
    cu_seqlens_q, cu_seqlens_k=None, *,
    page_table=None, seqused_k=None, causal=True,
    max_seqlen_q=None, max_k_tiles=None, output=None,
    reduce_heads=False, q_offset=None,
) -> torch.Tensor
```

**功能**：SM120/SM121 的 NVFP4 MSA 稠密 proxy 趟（`msa_proxy_score` 的 FP4 版本）。契约与输出
相同（逐 KV 块、未缩放、因果掩码 `Q K^T` 的最大值），但 Q/K 以打包 NVFP4 预量化到达
（`e2m1` + 每 16 的 `e4m3` 块 scale + 每张量全局 scale），因此 index K 以约 4 bit/元素从 HBM 读
取——这是省带宽路径。数值等于对同一打包输入的 torch 反量化（不等于 bf16 参考，后者因 fp4 舍入
而不同）。两个全局 scale 在块最大之前折入 logits 为 `q_global_scale * k_global_scale`。

**参数**

- `q_fp4`（`torch.Tensor`）：`(total_q, num_qo_heads, 64)` uint8，打包 e2m1（2 半字节/字节）。
- `k_fp4`（`torch.Tensor`）：扁平 `(total_k, num_kv_heads, 64)` 或分页
  `(num_pages, num_kv_heads, 128, 64)`。`num_qo_heads` 必须是 `num_kv_heads` 的倍数。
- `q_scale`、`k_scale`（`torch.Tensor`）：cuBLAS 128x4 tiled layout 的扁平 uint8 e4m3 块 scale，
  按逻辑行 `token*num_heads + head` 索引（分页 K：`(page*num_kv_heads + kv_head)*128 +
  token_in_page`）。由 `nvfp4_quantize`（128x4 layout，`sf_vec_size=16`）产生。
- `q_global_scale`、`k_global_scale`（`float`）：每张量反向全局 scale（`1 / global_scale`）。
- `cu_seqlens_q`、`cu_seqlens_k`、`page_table`、`seqused_k`、`causal`、`max_seqlen_q`、
  `max_k_tiles`、`output`、`reduce_heads`、`q_offset`：同 `msa_proxy_score`。

**返回**

- `torch.Tensor`：float32 `max_score`，`(num_qo_heads, max_k_tiles, total_q)` 或
  `reduce_heads=True` 时 `(1, max_k_tiles, total_q)`。

**注意事项**

- 需要 SM120/SM121 且 CUDA >= 12.8。`q_fp4`/`k_fp4` 必须为打包 uint8，`q_scale`/`k_scale` 必须为
  uint8。

---

### `MSASparseAttentionWorkspace`

```python
MSASparseAttentionWorkspace(device: torch.device | str) -> None
```

**功能**：SM100/SM103/SM107 的 MSA CUDA Graph 捕获所需的调用方存储。为每次被捕获的稀疏注意力调用
构造一个 workspace。用将要捕获的精确张量、选项与 CUDA 流**急切预热**（eager warm），然后在捕获前
同步该流。workspace 持有输出与临时张量，其地址须在图重放期间保持稳定。

**参数**

- `device`（`torch.device | str`）：CUDA 设备。

**注意事项**

- workspace 绑定到其首个流。一旦参与捕获，便不能再经 Python 传入；图重放在 workspace 生命周期内
  保持有效。
- 多请求长 prefill 路由在 eager warm 调用中把请求边界从 `cu_seqlens_q` 与
  `cu_seqlens_k`/`seqused_k` 读到 host 并保留给该绑定：后续每次调用传同一批张量对象（重建的
  `cu_seqlens` 张量——新对象，即使地址相同——会被重新读取，代价是一次同步），并只通过 torch 操作
  改写其内容。

---

### `supports_packed_kv`

```python
supports_packed_kv(device: torch.device | str) -> bool
```

**功能**：返回该设备上 MSA 是否接受打包分页 K/V 视图。在跨这些架构集成 cache manager 时，用当前
设备调用它；遗留的聚合标志 `SUPPORTS_PACKED_KV` 描述 SM120/SM121 后端。

**参数**

- `device`（`torch.device | str`）：目标设备。

**返回**

- `bool`：当设备为 CUDA 且计算能力为 `(12, 0)` 或 `(12, 1)` 时为 `True`。

---

### `msa_sparse_attention`

```python
msa_sparse_attention(
    q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k=None,
    causal=False, softmax_scale=None, page_table=None, seqused_k=None,
    return_softmax_lse=False, k_scale=None, v_scale=None,
    k_global_scale=None, v_global_scale=None, q_offset=None,
    return_temperature_lse=False, lse_temperature_scale=1.0,
    workspace=None,
)
```

**功能**：SM100/SM103 与 SM120/SM121 GPU 的 Minimax Sparse Attention 前向。每个 query 只对
`q2k_indices` 中选中的 top-K KV 块做注意力。Query token 按 tile 处理：每个 tile 对其 token 选中
块的**并集**做一次在线 softmax，直接写出最终输出。并集元数据由 `q2k_indices` 内部构建。

`q` 为 bf16/fp16；`k`/`v` 为 bf16/fp16、fp8（E4M3）或打包 NVFP4（uint8 配 `k_scale`/`v_scale`），
GQA 或 MHA layout，扁平或分页。打包 NVFP4 每字节存两个值，故其 K/V last dim 为 `head_dim // 2`
而非 `head_dim`。`q2k_indices` 为 `(num_kv_heads, total_q, topk)` int32（升序，`-1` 填充）。

**参数**

- `q`（`torch.Tensor`）：query，形状 `(total_q, num_qo_heads, head_dim)`，bf16/fp16。
- `k`（`torch.Tensor`）：key。bf16/fp16/fp8 时扁平形状 `(total_k, num_kv_heads, head_dim)`，分页
  形状 `(num_pages, num_kv_heads, 128, head_dim)`。打包 NVFP4 两种 layout 的 last dim 均为
  `head_dim // 2`。
- `v`（`torch.Tensor`）：value，layout 约定同 `k`。
- `q2k_indices`（`torch.Tensor`）：int32，形状 `(num_kv_heads, total_q, topk)`，升序、`-1` 填充。
- `cu_seqlens_q`（`torch.Tensor`）：int32 累积 query 长度，形状 `(batch_size + 1,)`。
- `cu_seqlens_k`（`Optional[torch.Tensor]`，默认 `None`）：int32 累积 KV 长度，形状
  `(batch_size + 1,)`，用于稠密路径。提供 `seqused_k` 时可省略。
- `causal`（`bool`，默认 `False`）：是否施加因果掩码。
- `softmax_scale`（`Optional[float]`，默认 `None`）：softmax scale；默认 `head_dim**-0.5`。
- `page_table`（`Optional[torch.Tensor]`，默认 `None`）：启用分页 KV 路径，形状
  `(batch_size, max_pages)`，把 batch-local KV 块索引映射到页。需要 `seqused_k`。
- `seqused_k`（`Optional[torch.Tensor]`，默认 `None`）：int32，形状 `(batch_size,)`，分页路径中每
  序列的有效 KV 长度。
- `return_softmax_lse`（`bool`，默认 `False`）：是否额外返回自然对数 LSE，形状
  `(total_q, num_qo_heads)`。
- `k_scale`、`v_scale`（`Optional[torch.Tensor]`，默认 `None`）：NVFP4 块 scale 的 uint8 字节视图，
  由 `nvfp4_quantize`（`sf_vec_size=16`、swizzled 128x4 layout）产生。
- `k_global_scale`、`v_global_scale`（`Optional[float]`，默认 `None`）：全局反量化 scale。K 折入
  softmax scale，V 作用于输出。
- `q_offset`（可选）：特定 MSA 工作流使用的逐 query 偏移张量。
- `return_temperature_lse`（`bool`，默认 `False`）：是否额外返回 MSA 温度 LSE。启用时返回
  `(out, lse, lse_t)`。
- `lse_temperature_scale`（`float`，默认 `1.0`）：计算温度 LSE 时施加于指数的 scale。
- `workspace`（`MSASparseAttentionWorkspace`，可选）：compute capability 10.0/10.3/10.7 上 CUDA
  Graph 捕获所需的调用方存储。捕获前用精确张量、选项与捕获流急切预热。SM120/SM121 后端不使用。

**返回**

- `torch.Tensor` 或张量元组：`out` 形状 `(total_q, num_qo_heads, head_dim)`。若
  `return_temperature_lse=True`，返回 `(out, lse, lse_t)`（无论 `return_softmax_lse`）；否则
  `return_softmax_lse=True` 返回 `(out, lse)`；两个标志皆假时返回 `out`。每个 LSE 张量形状
  `(total_q, num_qo_heads)`，dtype float32。

**注意事项**

- compute capability 10.0/10.3 支持 bf16/fp16 K/V 与 E4M3 K/V（bf16 Q），加恰好一条打包 NVFP4
  路由（分页 K/V 原地作为平面页的 strided 视图，配 `k_scale`/`v_scale` 与两个全局 scale、causal、
  无 LSE、用 `seqused_k` 而非 `cu_seqlens_k`），且要求 `topk == 16`。
- `workspace` 在非 10.0/10.3 后端传入会抛 `ValueError`。

---

### `msa_sparse_decode_attention`

```python
msa_sparse_decode_attention(
    q, k, v, q2k_indices, *,
    page_table=None, seqused_k=None, cu_seqlens_k=None,
    seqlen_q=1, causal=True, softmax_scale=None, return_softmax_lse=False,
    k_scale=None, v_scale=None, k_global_scale=None, v_global_scale=None,
    q_offset=None, partial_dtype=None, force_fused=None, workspace=None, out=None,
)
```

**功能**：SM100/SM103/SM107 与 SM120/SM121 GPU 的稀疏 decode 注意力。为 decode step 计算注意力：
每个请求贡献 `seqlen_q` 个 query token（批内统一），只对 `q2k_indices` 中选中的 KV 块做注意力。
decode token 右对齐：请求的 token `i` 位于位置 `seqlen_k - seqlen_q + i`。

**参数**

- `q`（`torch.Tensor`）：`(batch_size * seqlen_q, num_qo_heads, 128)`，bf16 或 fp16。
- `k`、`v`（`torch.Tensor`）：分页 `(num_pages, num_kv_heads, 128, 128)` 配 `page_table` 与
  `seqused_k`；或扁平 varlen `(total_k, num_kv_heads, 128)` 配 `cu_seqlens_k`。可为 fp8 E4M3。
- `q2k_indices`（`torch.Tensor`）：`(num_kv_heads, batch_size * seqlen_q, topk)` int32，升序、`-1`
  尾部填充（`msa_topk_select` 产生的格式）。
- `seqlen_q`（`int`，默认 `1`）：每请求统一的 query 长度（如推测解码可 >1）。
- `page_table`（`Optional[torch.Tensor]`）：分页 KV layout 的页表映射。
- `seqused_k`（`Optional[torch.Tensor]`）：分页 KV layout 的逐序列有效 KV token 数。
- `cu_seqlens_k`（`Optional[torch.Tensor]`）：ragged KV layout 的 `(batch_size + 1,)` int32 累积
  KV 长度。
- `causal`（`bool`，默认 `True`）：右对齐因果掩码。
- `softmax_scale`（`Optional[float]`）：覆盖默认注意力缩放。
- `return_softmax_lse`（`bool`，默认 `False`）：是否额外返回逐 query log-sum-exp。
- `k_scale`、`v_scale`（`Optional[torch.Tensor]`）：NVFP4 only：e4m3 块 scale，每 16 元素一个。
  layout 逐架构不同。
- `k_global_scale`、`v_global_scale`（`Optional[float]`）：全局反量化 scale。
- `q_offset`（`int` 或 `torch.Tensor`）：因果对齐用的可选 query 位置偏移。
- `partial_dtype`（`Optional[torch.dtype]`）：受支持内核的累加器/部分结果 dtype 覆盖。10.0/10.3/
  10.7 后端始终用其原生 float32 split 存储并忽略此覆盖。
- `force_fused`（`Optional[bool]`）：覆盖自适应 split-K 决策。`True`/`False` 强制 fused/split，
  `None`（默认）自适应。10.0/10.3/10.7 上为 API 兼容而接受；生产 direct-M16 路由不 split。
- `workspace`（`MSASparseAttentionWorkspace`，可选）：10.0/10.3/10.7 上 CUDA Graph 捕获所需的
  调用方存储。SM120/SM121 后端不使用。
- `out`（`Optional[torch.Tensor]`）：注意力输出目标。给定时内核直接写入并原样返回——不分配临时、
  不拷贝。必须连续、BF16、在 `q` 的设备上且恰好为 `q.shape`。仅 10.0/10.3/10.7 的打包 NVFP4 分页
  KV 路由支持；其他路由传入时抛 `NotImplementedError`。

**返回**

- `torch.Tensor` 或 `(torch.Tensor, torch.Tensor)`：`(batch_size * seqlen_q, num_qo_heads, 128)`，
  为 `q` 的 dtype，但 uniform FP8 Q/K/V 返回 BF16；`return_softmax_lse` 时附带自然对数 LSE。给出
  `out` 时它就是返回的张量。

**注意事项**

- 10.0/10.3/10.7 上 `topk` 必须为 16。`head_dim` 必须为 128。分页 decode 需要 `seqused_k`；扁平
  decode 需要 `cu_seqlens_k`。

---

### `msa_topk_select`

```python
msa_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[Union[int, torch.Tensor]] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor
```

**功能**：基于注意力分数为每个 query token 挑选 top-K KV 块。实现 Minimax Sparse Attention 的
块打分趟：给定廉价 proxy prefill 的逐块最大注意力分数，为每个 (query token, head) 对挑选 `topk`
个最重要的 KV 块并返回其排序后的索引。分发支持 compute capability 10.0/10.3 与 SM120/SM121。

**参数**

- `max_score`（`torch.Tensor`）：形状 `(num_qo_heads, max_k_tiles, total_qo_len)`，float32。无效
  tile（超出实际 KV 长度）的条目须由调用方置为 `-inf`。
- `topk`（`int`）：每个 (query token, head) 挑选的 KV 块数。**必须为 16**。
- `num_valid_pages`（`int` 或 `torch.Tensor`，可选）：实际有效 KV 页数（`<= max_k_tiles`）。索引
  `>= num_valid_pages` 替换为 -1 并排到尾部。默认 `max_k_tiles`（禁用钳制）。也可为 int32 张量
  `(total_qo_len,)`，给每个 query token 其自身的有效页数。**张量形式仅 SM120/SM121 支持**；在
  compute capability 10.0/10.3 上请传标量或 `None`（张量会在分发前被拒绝）。
- `output`（`torch.Tensor`，可选）：预分配输出，形状 `(total_qo_len, num_qo_heads, topk)`，int32。
  未提供时内部申请。
- `force_begin_blocks`（`int`，默认 `0`）：总是包含的起始 KV 块数（sink token）。
- `force_end_blocks`（`int`，默认 `0`）：总是包含的末尾 KV 块数（局部窗口）。

**返回**

- `torch.Tensor`：形状 `(total_qo_len, num_qo_heads, topk)`，int32。升序 KV 块索引；`-1` 为尾部
  填充的无效槽。

**注意事项**

- 当接近平局的分值竞争最后槽位时，可能选中任一块；确切索引可能随问题规模与版本变化，但选中的
  分值是匹配的。
- `force_begin_blocks + force_end_blocks` 必须 `<= topk`（张量 `num_valid_pages` 时 `<=` 该
  计数）。非张量 `num_valid_pages` 须在 `(0, max_k_tiles]`。

---

### `prepare_msa_nvfp4_sparse_decode`

```python
prepare_msa_nvfp4_sparse_decode(
    q, k, v, q2k_indices, *,
    k_scale, v_scale, page_table, seqused_k, k_global_scale, v_global_scale,
    seqlen_q=1, softmax_scale=None, out=None, lse=None, backend="cake",
)
```

**功能**：用生成的 Cake 程序准备 NVFP4 分页-KV 稀疏 decode（SM100/SM103/SM107）。实验性 Cake
后端服务与 `msa_sparse_decode_attention` 打包 NVFP4 分页-KV 路由相同的问题面——平面页池、
top-k 16、右对齐因果 decode token——使用持久内核保持 K tile 常驻 tensor memory，并以交换后的方向
运行两个 MMA，另有一个面向四页以内请求批的短项 cluster 程序。准备阶段校验并绑定张量；返回的
runner 启动时无分配、无 host 同步，可被捕获进 CUDA Graph。

**参数**

- `q`（`torch.Tensor`）：BF16 `[batch * seqlen_q, num_q_heads, 128]`；请求 `b` 拥有行
  `[b * seqlen_q, (b + 1) * seqlen_q)`，query 头 `h` 关注 KV 头 `h // (num_q_heads //
  num_kv_heads)`（每 KV 头至多 16 个 query 头）。
- `k`、`v`（`torch.Tensor`）：uint8 `[num_pages, num_kv_heads, 128, 64]`，平面页池的 strided 视图
  （打包 E2M1，每字节两个值）。
- `q2k_indices`（`torch.Tensor`）：int32 `[num_kv_heads, batch * seqlen_q, 16]`，每个 KV 头与
  query token 选中的页索引，升序、`-1` 填充。
- `k_scale`、`v_scale`（`torch.Tensor`）：`[num_pages, num_kv_heads, 128, 8]` E4M3 块 scale 的
  strided 视图（uint8 或 float8_e4m3fn）；K scale 线性，V scale 为 cache writer 的 swizzled 顺序。
- `page_table`（`torch.Tensor`）：int32 `[batch, max_pages]`，每请求的物理页 ID。
- `seqused_k`（`torch.Tensor`）：int32 `[batch]`，每请求 KV token 数（含新 token）。
- `k_global_scale`、`v_global_scale`（`float`）：正的每侧全局 scale；K scale 折入 softmax scale，
  V scale 在尾声施加。
- `seqlen_q`（`int`，默认 `1`）：每请求 query token 数，在 `[1, 32]`。
- `softmax_scale`（`Optional[float]`）：默认 `1 / sqrt(128)`。
- `out`、`lse`（`Optional[torch.Tensor]`）：可选调用方持有的 BF16 输出
  `[batch * seqlen_q, num_q_heads, 128]` 与 float32 自然对数归一化子
  `[batch * seqlen_q, num_q_heads]`。
- `backend`（`str`，默认 `"cake"`）：仅支持 `"cake"`。

**返回**

- `MSANvfp4DecodeRunner`：调用它在当前流上启动 decode 并返回 `out`。

**注意事项**

- 该 API 带 `flashinfer_experimental_api` 装饰，属实验性。

## 相关链接 / 参考

- 上游 `docs/api/sparse.rst`
- 源码 `flashinfer/sparse.py`、`flashinfer/cake_vsa.py`、`flashinfer/msa_ops/`
- `flashinfer.mla.cake_dsv4.get_cake_dsv4_workspace_bytes`
- SciPy `bsr_matrix`：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html

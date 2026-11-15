# FlashAttention-4 (CuTeDSL) 注意力 API

- 源码：`flash_attn/cute/interface.py`
- 导入：`import flash_attn.cute as fa4`（或 `from flash_attn.cute import flash_attn_func, flash_attn_varlen_func`）
- 目标架构：Hopper (SM90) / Blackwell (SM100, SM110)
- 包名：`flash-attn-4`（`__version__` 从包元数据读取，未安装时回退为 `"0.0.0"`）
- 实现：Python + CuTeDSL（NVIDIA CUTLASS DSL），运行时 JIT 编译为 PTX/CUBIN。

## 公共导出

`flash_attn/cute/__init__.py` 仅导出两个入口：

```python
from .interface import flash_attn_func, flash_attn_varlen_func

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
```

`interface.py` 中另有 `flash_attn_combine` 与 `get_scheduler_metadata`，可从 `flash_attn.cute.interface` 直接导入。

张量布局要求：`(batch, seqlen, num_heads, head_dim)`，最后一维连续，**16 字节对齐**。

## 函数总览

| 函数 | 说明 |
| --- | --- |
| `flash_attn_func` | 标准注意力（支持 MLA `qv`、稀疏、mask/score modifier） |
| `flash_attn_varlen_func` | 变长注意力 |
| `flash_attn_combine` | 合并 split-KV 部分结果 |
| `get_scheduler_metadata` | 为 varlen tile 调度器准备元数据 |

---

## flash_attn_func

```python
flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    gather_bwd_recompute_p: bool = False,
    gather_bwd_token_chunk: Optional[int] = None,
)
```

标准（dense）注意力。支持 GQA/MQA、MLA、softcap、局部/因果掩码、learnable sink、score/mask 自定义函数、块稀疏。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `q` | `(batch, seqlen_q, nheads, hdim)` | 查询 |
| `k` | `(batch, seqlen_k, nheads_k, hdim)` | 键 |
| `v` | `(batch, seqlen_k, nheads_k, hdim_v)` | 值 |
| `qv` | `(batch, seqlen_q, nheads, hdim_v)`，可选 | MLA 吸收形式的 `q_nope` |
| `gather_kv_indices` | `(batch, seqlen_q, gather_kv_length)`，可选 | topk 稀疏（MLA 吸收内核）使用的 KV 索引 |
| `softmax_scale` | float，可选 | 缩放，默认 `1/sqrt(hdim)` |
| `causal` | bool | 因果掩码 |
| `window_size` | `(left, right)`，元素可为 `None` | 滑动窗口局部注意力 |
| `learnable_sink` | Tensor，可选 | 可学习的 attention sink |
| `softcap` | float | `> 0` 启用 softcapping |
| `num_splits` | int | split-KV 切分数 |
| `pack_gqa` | bool，可选 | GQA 打包 |
| `deterministic` | bool | 确定性反向 |
| `score_mod` / `score_mod_bwd` | `@cute.jit` 可调用，可选 | 自定义分数修改函数（前向 / 反向） |
| `mask_mod` | `@cute.jit` 可调用，可选 | 自定义掩码修改函数 |
| `aux_tensors` | list，可选 | 传给 score/mask modifier 的辅助张量 |
| `aux_scalars` | tuple，可选 | 传给 score/mask modifier 的辅助标量 |
| `block_sparse_tensors` / `block_sparse_tensors_bwd` | `BlockSparseTensorsTorch`，可选 | 块稀疏掩码（前向 / 反向） |
| `return_lse` | bool | 是否返回 logsumexp |
| `gather_bwd_recompute_p` | bool | 稀疏 MLA 反向：不保存 `p`/`row_max`，反向内核内重算 |
| `gather_bwd_token_chunk` | int，可选 | 稀疏 MLA 反向：按 token 分块运行以限制 dS 临时显存 |

**返回**

- `out`：`(batch, seqlen_q, nheads, hdim_v)`
- `lse`（`return_lse=True`）：`(nheads, total_q)` 或 `(batch, nheads, seqlen_q)`；当有 `qv` 时形状为 `(total_q, nheads)` / `(batch, seqlen_q, nheads)`

> `gather_bwd_token_chunk` 要求 varlen 或 `batch_size == 1`；否则会告警并按不分块执行。

---

## flash_attn_varlen_func

```python
flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
    gather_bwd_recompute_p: bool = False,
    gather_bwd_token_chunk: Optional[int] = None,
)
```

变长注意力，同时兼容带 batch 维与不带 batch 维的输入。

**张量参数**

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `q` | `(total_q, nheads, hdim)` 或 `(batch, seqlen_q, nheads, hdim)` | 查询 |
| `k` | `(total_k, nheads_k, hdim)` 或 `(batch, seqlen_k, nheads_k, hdim)` | 键 |
| `v` | `(total_k, nheads_k, hdim_v)` 或 `(batch, seqlen_k, nheads_k, hdim_v)` | 值 |
| `qv` | `(total_q, nheads, hdim_v)` 或 `(batch, seqlen_q, nheads, hdim_v)` | MLA |
| `cu_seqlens_q` | `(batch + 1,)` | 累积 query 长度（或改用 `seqused_q`） |
| `cu_seqlens_k` | `(batch + 1,)` | 累积 key 长度 |
| `gather_kv_indices` | `(total_q, gather_kv_length)` 或 `(batch, seqlen_q, gather_kv_length)` | topk 稀疏索引 |
| `page_table` | `(batch, max_num_pages_per_seq)` | paged KV 页表 |

**返回**

- `out`：`(total_q, nheads, hdim_v)` 或 `(batch, seqlen_q, nheads, hdim_v)`
- `lse`（`return_lse=True`）：
  - 无 `qv`：`(nheads, total_q)` 或 `(batch, nheads, seqlen_q)`
  - 有 `qv`：`(total_q, nheads)` 或 `(batch, seqlen_q, nheads)`（`nheads` 连续，便于向量化；MLA/GQA 中 `nheads` 通常至少可被 4 整除）

**关键可选参数**

- `qv`：MLA 吸收形式写作 `O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V`，其中 `Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache`。
- `gather_kv_indices`：用于 MLA 吸收内核的 topk 稀疏。
- `page_table`：paged KV。当页由 TMA 加载（一页容纳整个 key tile）时，会加载该块的每一行，包括序列最后一页中超过 `seqused_k` 的行；这些行分数被 mask，但 `P = 0` 乘以 NaN 的 V 行仍是 NaN，因此这些行必须是有限值（其数值无所谓）。序列未使用的页可含任意值（含 NaN），`seqused_k = 0` 的序列同理。
- `max_seqlen_q` / `max_seqlen_k`：可选的长度上界标量。Blackwell 累积长度路径下不在 host 读取张量提示；HD256 对省略/张量提示使用 flat grid，对 host 整数使用矩形 grid。整型上界必须覆盖每次 CUDA graph replay。
- `min_seqlen_k`：变长时指定任意 batch 的最小 KV 序列长度，配合 `gather_kv_indices` 判断是否需要 OOB 掩码。
- `scheduler_metadata`：特定 tile 调度器使用的张量，由 `get_scheduler_metadata` 计算。
- `seqlen_k_per_split`：使用动态（逐 batch）`num_splits` 时，可固定每个 split 覆盖的 `seqlen_k` 以获得位级可复现性。
- `disable_scheduler_metadata`：为 `True` 时忽略传入的 `scheduler_metadata` 且不重新计算。
- `gather_bwd_recompute_p`：稀疏 MLA（1..128 个 Q head）前向完全不保存 `p`/`row_max`（128 head、gather 宽 2048 时每 token 约 520 KiB，从前向一直保留到反向）；反向主内核从 `(q, qv, k, v, lse)` 在 kernel 内重算 `P = exp2(scale*S - lse*log2e)`。训练前向也更快（无 p 存储）。梯度与默认路径非位级一致（少一次 P 的舍入），但相对 fp32 参考至少同样精确。该模式支持 `retain_graph`。
- `gather_bwd_token_chunk`：仅稀疏 MLA 反向，按该大小的 token 分块运行反向内核，把 dS 临时显存限制在 chunk 而非完整序列（128 head、gather 宽 2048 时每 token 约 520 KiB）。可与 `gather_bwd_recompute_p` 组合，使整个反向临时显存受 chunk 限制。必须是正整数（否则抛错）；要求 varlen 或 batch 1（否则告警并按不分块执行）。有小幅启动开销。`None` 表示禁用。

---

## flash_attn_combine

```python
flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

合并 split 注意力的部分输出与 logsumexp。

**参数**

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `out_partial` | `(num_splits, batch, seqlen, nheads, headdim)` 或 `(num_splits, total_q, nheads, headdim)` | 部分输出 |
| `lse_partial` | `(num_splits, batch, seqlen, nheads)` 或 `(num_splits, total_q, nheads)` | 部分 LSE |
| `out` | 可选 | 若 `None` 自动创建 |
| `out_dtype` | 可选 | 若 `None` 使用输入的 dtype |
| `cu_seqlens` | 可选 | 变长累积长度 |
| `seqused` | 可选 | 每个 batch 实际使用长度 |
| `virtual_batch_idx` | `(batch,)`，`int32`，可选 | 虚拟 batch 下标到真实 batch 的映射（persistent 调度器重排 batch 以负载均衡时使用） |
| `return_lse` | bool | 是否返回合并后的 LSE，默认 True |

**返回**

- `out`：`(batch, seqlen, nheads, headdim)` 或变长 `(total_q, nheads, headdim)`
- `lse`：`(batch, seqlen, nheads)` 或变长 `(total_q, nheads)`；`return_lse=False` 时为 `None`

> 期望输入为 split 注意力计算产出的格式，第一维是 `num_splits`；从用户格式到内核格式的 permute 已在 kernel 内完成。

---

## get_scheduler_metadata

```python
get_scheduler_metadata(
    max_seqlen_q: int,
    max_seqlen_k: int,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    num_splits: int,
    headdim_v: Optional[int] = None,
    pack_gqa: Optional[int] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqlen_k_new: int = 0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqlen_k_per_split: Optional[int] = None,
    _arch: Optional[int] = None,
) -> SchedulerMetadataTensorsTorch
```

为变长 tile 调度器（`SingleTileVarlenScheduler` 与 `DynamicPersistentVarlenScheduler`）准备元数据张量。要求至少提供 `cu_seqlens_q` / `cu_seqlens_k` / `seqused_q` / `seqused_k` 之一。

**关键参数**

- `num_splits`：prepare kernel 每个 batch 条目可产生的最大 split 数。
- `seqlen_k_per_split`：为前后向位级可复现可固定每 split 的精确 `seqlen_k`；此时 `num_splits` 相应计算。

**返回** `SchedulerMetadataTensorsTorch`（named tuple）：

- `num_splits_dynamic_ptr`：逐 batch 的 `num_splits`
- `num_nheads_in_l2_ptr`：head swizzle 用，避免 L2 cache 抖动
- `tile_count_semaphore`：`DynamicPersistentVarlenScheduler` 原子递增用的全局信号量
- `cu_total_m_blocks`：统计总 m_block 的前缀和，用于大 batch 的二分 batch 搜索
- `cu_total_splits_m_blocks`：互补前缀和，用于二分 batch 搜索及在无 `num_splits_dynamic_ptr` 时提取动态 num splits

---

## 块稀疏张量结构

`BlockSparseTensorsTorch`（`flash_attn/cute/block_sparsity.py`）为 `NamedTuple`：

```python
class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    cu_total_m_blocks: torch.Tensor | None = None
    cu_block_idx_offsets: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    spt: bool | None = None
```

---

## 自定义 score/mask 函数

`score_mod` / `score_mod_bwd` / `mask_mod` 是用户定义的 `@cute.jit` 可调用对象，编译期注入内核；辅助输入通过 `aux_tensors` / `aux_scalars` 传入。相关测试见 `tests/cute/test_mask_mod.py`、`tests/cute/test_score_mod.py`。

## 编译与缓存

FA4 内核 JIT 编译，缓存 key 包含 dtype、head_dim、causal、mask/score_mod 哈希、架构、块大小等。缓存层级：内存 LRU + 可选磁盘缓存（`get_jit_cache()`）。

相关环境变量：

- `CUTE_CUBIN_PATH`：导出 CUBIN/SASS
- `CUTE_DSL_KEEP_PTX=1`：保留 PTX
- `CUTE_DSL_PTXAS_PATH`：自定义 ptxas
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`：启用磁盘缓存

**示例**

```python
import torch
import flash_attn.cute as fa4

q = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out = fa4.flash_attn_func(q, k, v, causal=True)
out.sum().backward()
```

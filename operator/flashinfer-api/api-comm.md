# 通信原语（`flashinfer.comm`）

> 源码：`flashinfer/comm/__init__.py` 及各子模块（`cuda_ipc.py`、`dlpack_utils.py`、
> `mapping.py`、`all_gather_matmul/`、`trtllm_ar.py`、`allreduce.py`、`vllm_ar.py`、
> `quantized_allreduce.py`、`pcie_ipc_*.py`、`pcie_ipc_collectives/`、`ulysses*.py`、
> `qkv_norm_rope_exchange_overlap.py`、`mnnvl.py`、`trtllm_mnnvl_ar.py`、
> `trtllm_moe_alltoall.py`、`dcp_alltoall.py`、`dcp_lse_reduce.py`、`mixed_comm.py`）
> 上游文档：`docs/api/comm.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.comm` 提供分布式通信原语与工具，包括 CUDA IPC、AllReduce、MNNVL
（Multi-Node NVLink）、PCIe IPC、Ulysses 上下文并行 all-to-all、MoE all-to-all、DCP
（decode context-parallel）all-to-all 与混合通信。多数集合操作是 **collective**：同一
group 内所有 rank 必须以相同顺序、相同配置调用。模块以多个后端（TRT-LLM、MNNVL、
vLLM、PCIe IPC、Cake、NCCL）覆盖不同架构（SM80/SM90/SM100/SM103/SM120）。

## 导入

```python
import flashinfer.comm as comm
from flashinfer.comm import (
    CudaRTLibrary, create_shared_buffer, free_shared_buffer, pack_strided_memory, Mapping,
    all_gather_matmul, prepare_all_gather_matmul,
    allreduce_fusion, create_allreduce_fusion_workspace, AllReduceFusionWorkspace,
    TRTLLMAllReduceFusionWorkspace, MNNVLAllReduceFusionWorkspace,
    quantized_all_reduce,
    PcieIpcAllReduceWorkspace, PcieIpcLaunchConfig, PcieIpcVariant,
    get_pcie_ipc_launch_config, probe_pcie_ipc_rank_topology, resolve_pcie_ipc_profile,
    PcieIpcAllGatherWorkspace, PcieIpcReduceScatterWorkspace,
    UlyssesCommunicator, UlyssesWorkspace, QkvNormRopeExchangeOverlap,
    resolve_ulysses_backend, decide_ulysses_backend, probe_ulysses_rank_topology,
    init_ulysses_a2a, dispose_ulysses_a2a, ulysses_a2a,
    MnnvlMemory, McastGPUBuffer,
    moe_a2a_initialize, moe_a2a_dispatch, moe_a2a_combine, MoeAlltoAll,
    decode_cp_a2a_alltoall, decode_cp_a2a_lse_reduce,
    MixedCommOp, MixedCommMode, MixedCommHandler, run_mixed_comm,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `CudaRTLibrary` | 类 | CUDA Runtime 的 ctypes 封装（IPC 句柄等） |
| `create_shared_buffer` | 函数 | 通过 CUDA IPC 分配并跨 group 共享缓冲区 |
| `free_shared_buffer` | 函数 | 释放 `create_shared_buffer` 创建的缓冲区 |
| `pack_strided_memory` | 函数 | 把跨步设备内存包装为 PyTorch 张量视图（DLPack） |
| `Mapping` | 类 | 描述 TP/PP/CP/MoE 并行拓扑 |
| `all_gather_matmul` | 函数 | push-wait 融合 all-gather matmul |
| `prepare_all_gather_matmul` | 函数 | 预绑定权重的 Cake all-gather matmul 启动器 |
| `AllReduceFusionOp` | 类 | TRT-LLM allreduce fusion 操作码 |
| `AllReduceFusionPattern` | 类 | 融合模式常量 |
| `AllReduceStrategyConfig` | 类 | allreduce 策略配置位 |
| `AllReduceStrategyType` | 类 | allreduce 策略类型 |
| `QuantizationSFLayout` | 类 | 量化 scale factor 布局 |
| `trtllm_allreduce_fusion` | 函数 | TRT-LLM AllReduce + RMSNorm 融合 |
| `trtllm_moe_allreduce_fusion` | 函数 | MoE reduction + AllReduce + RMSNorm 融合 |
| `trtllm_moe_finalize_allreduce_fusion` | 函数 | MoE finalize + AllReduce + RMSNorm 融合 |
| `trtllm_create_ipc_workspace_for_all_reduce_fusion` | 函数 | 创建 allreduce fusion IPC 工作区 |
| `trtllm_destroy_ipc_workspace_for_all_reduce` | 函数 | 释放旧式 allreduce 工作区引用 |
| `trtllm_destroy_ipc_workspace_for_all_reduce_fusion` | 函数 | 释放 fusion 工作区引用 |
| `trtllm_lamport_initialize` | 函数 | 初始化单个 Lamport 缓冲区为负零 |
| `trtllm_lamport_initialize_all` | 函数 | 初始化三个 Lamport 缓冲区为负零 |
| `compute_fp4_swizzled_layout_sf_size` | 函数 | 计算 FP4 swizzled 布局填充后大小 |
| `allreduce_fusion` | 函数 | 统一 AllReduce + RMSNorm（+量化）入口 |
| `create_allreduce_fusion_workspace` | 函数 | 按后端创建 AllReduce fusion 工作区 |
| `AllReduceFusionWorkspace` | 类 | AllReduce fusion 工作区抽象基类 |
| `TRTLLMAllReduceFusionWorkspace` | 类 | TRT-LLM 后端工作区 |
| `MNNVLAllReduceFusionWorkspace` | 类 | MNNVL 后端工作区 |
| `quantized_all_reduce` | 函数 | FP8 量化 two-shot AllReduce（per-block scaling） |
| `vllm_all_reduce` | 函数 | vLLM 自定义 all-reduce |
| `vllm_dispose` | 函数 | 释放 vLLM all-reduce 句柄 |
| `vllm_init_custom_ar` | 函数 | 初始化 vLLM 自定义 all-reduce |
| `vllm_register_buffer` | 函数 | 注册对端 IPC 共享缓冲区 |
| `vllm_register_graph_buffers` | 函数 | 跨 all-reduce world 注册 graph 缓冲区 |
| `vllm_get_graph_buffer_ipc_meta` | 函数 | 返回 graph 缓冲区的 IPC 元数据 |
| `vllm_meta_size` | 函数 | 返回 vLLM all-reduce 元数据结构大小 |
| `PcieIpcAllReduceWorkspace` | 类 | PCIe IPC all-reduce 工作区 |
| `PcieIpcLaunchConfig` | 类 | PCIe IPC 启动配置（`blocks/threads/variant`） |
| `PcieIpcVariant` | 类 | PCIe IPC all-reduce 内核变体枚举 |
| `get_pcie_ipc_launch_config` | 函数 | 为给定形状返回启动配置或 `None` |
| `probe_pcie_ipc_rank_topology` | 函数 | 探测本 rank GPU 的 PCIe 拓扑 |
| `resolve_pcie_ipc_profile` | 函数 | 集体探测并商定 fabric profile |
| `PcieIpcAllGatherWorkspace` | 类 | PCIe IPC all-gather 工作区 |
| `PcieIpcAllGatherLaunchConfig` | 类 | all-gather 启动配置 |
| `PcieIpcAllGatherVariant` | 类 | all-gather 内核变体 |
| `get_pcie_ipc_all_gather_launch_config` | 函数 | all-gather 启动种子 |
| `PcieIpcReduceScatterWorkspace` | 类 | PCIe IPC reduce-scatter 工作区 |
| `PcieIpcReduceScatterLaunchConfig` | 类 | reduce-scatter 启动配置 |
| `PcieIpcReduceScatterVariant` | 类 | reduce-scatter 内核变体 |
| `get_pcie_ipc_reduce_scatter_launch_config` | 函数 | reduce-scatter 启动种子 |
| `UlyssesCommunicator` | 类 | Ulysses 上下文并行 all-to-all 通信器 |
| `UlyssesWorkspace` | 类 | NCCL Ulysses 集合的可复用 staging 存储 |
| `pack_ulysses_qkv_head_chunk` | 函数 | 打包一个 Q/K/V head band |
| `merge_ulysses_output_head_chunk` | 函数 | 把紧凑输出 head band 合并回完整输出 |
| `QkvNormRopeExchangeOverlap` | 类 | 融合 QKV 投影 + RMSNorm + RoPE + head exchange |
| `UlyssesBackendDecision` | 类 | 后端选择结果 |
| `UlyssesRankTopology` | 类 | 每 rank 拓扑探测结果 |
| `UlyssesBackendError` | 类 | 强制 nvlink 但拓扑不支持时抛出 |
| `resolve_ulysses_backend` | 函数 | 组一致的 Ulysses 后端选择 |
| `decide_ulysses_backend` | 函数 | 纯决策函数：探测结果 → 后端 |
| `probe_ulysses_rank_topology` | 函数 | 探测 rank 的 P2P/NVLink 可达性 |
| `init_ulysses_a2a` | 函数 | 初始化融合转置 Ulysses NVLink-P2P 后端 |
| `dispose_ulysses_a2a` | 函数 | 释放 `init_ulysses_a2a` 句柄 |
| `ulysses_a2a` | 函数 | 融合转置 Ulysses all-to-all 内核入口 |
| `MnnvlMemory` | 类 | MNNVL 内存分配/映射 |
| `McastGPUBuffer` | 类 | MNNVL 多播/单播缓冲区封装 |
| `trtllm_mnnvl_all_reduce` | 函数 | 旧的指针式 MNNVL all-reduce（已废弃） |
| `trtllm_mnnvl_allreduce` | 函数 | MNNVL all-reduce sum |
| `trtllm_mnnvl_fused_allreduce_add_rmsnorm` | 函数 | MNNVL AllReduce + Residual + RMSNorm |
| `trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant` | 函数 | 上者 + FP8/NVFP4 量化 |
| `trtllm_mnnvl_fused_allreduce_rmsnorm` | 函数 | 旧的 MNNVL TwoShot AllReduce + RMSNorm（已废弃） |
| `mpi_barrier` | 函数 | MPI barrier |
| `moe_a2a_initialize` | 函数 | 初始化 MoE all-to-all 工作区 |
| `moe_a2a_dispatch` | 函数 | MoE all-to-all dispatch 阶段 |
| `moe_a2a_combine` | 函数 | MoE all-to-all combine 阶段 |
| `moe_a2a_sanitize_expert_ids` | 函数 | 清洗无效 expert id slot |
| `moe_a2a_get_workspace_size_per_rank` | 函数 | 计算 MoE A2A 每 rank 工作区大小 |
| `moe_a2a_wrap_payload_tensor_in_workspace` | 函数 | 把工作区切片包装为张量视图 |
| `MoeAlltoAll` | 类 | MoE All-to-All 管理器（已废弃） |
| `decode_cp_a2a_workspace_size` | 函数 | DCP A2A 每 rank 工作区字节数 |
| `decode_cp_a2a_allocate_mnnvl_workspace` | 函数 | 分配 MNNVL DCP 工作区 |
| `decode_cp_a2a_init_workspace` | 函数 | 初始化 DCP 工作区 FIFO |
| `decode_cp_a2a_alltoall` | 函数 | DCP all-to-all 交换 |
| `decode_cp_a2a_lse_reduce_workspace_size` | 函数 | NCCL LSA DCP 融合 op 工作区大小 |
| `decode_cp_a2a_lse_reduce_create_workspace` | 函数 | 创建并 rendezvous 融合 op 工作区 |
| `decode_cp_a2a_lse_reduce` | 函数 | 融合 LSE 加权 reduce 的 DCP A2A |
| `MixedCommOp` | 类 | 混合通信操作枚举 |
| `MixedCommMode` | 类 | 混合通信执行模式枚举 |
| `MixedCommHandler` | 类 | 混合通信处理器 |
| `run_mixed_comm` | 函数 | 执行混合通信操作 |

## API 详情

### `CudaRTLibrary`

```python
class CudaRTLibrary:
    def __init__(self, so_file: Optional[str] = None)
```

**功能**：CUDA Runtime 的 ctypes 封装。默认从当前进程已加载的 `libcudart` 解析库文件；
暴露 `cudaSetDevice`、`cudaDeviceSynchronize`、`cudaDeviceReset`、`cudaGetErrorString`、
`cudaMalloc`、`cudaFree`、`cudaMemset`、`cudaMemcpy`、`cudaIpcGetMemHandle`、
`cudaIpcOpenMemHandle`、`cudaIpcCloseMemHandle` 等函数，并用 `CUDART_CHECK` 统一检查
返回错误码。

**参数**

- `so_file`（`Optional[str]`）：CUDA Runtime 共享库路径；为 `None` 时自动查找已加载的
  `libcudart`。

**注意事项**

- 各函数内部通过 `CUDART_CHECK` 在错误码非 0 时抛 `RuntimeError`。

---

### `create_shared_buffer`

```python
create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]
```

**功能**：分配缓冲区并通过 CUDA IPC 在 process group 内共享。本地 rank 执行 `cudaMalloc`
后执行 `cudaIpcGetMemHandle`；其他 rank 打开句柄获得映射到自身地址空间的指针。

**参数**

- `size_in_bytes`（`int`）：每个 rank 分配的缓冲区大小。
- `group`（`torch.distributed.ProcessGroup`，可选）：交换 IPC 句柄的 process group；默认
  `dist.group.WORLD`。

**返回**

- `list[int]`：每 rank 的设备指针（本 rank 位置为 rank-local 指针，其他为 IPC 映射指针）。

**注意事项**

- 集体操作。

---

### `free_shared_buffer`

```python
free_shared_buffer(pointers: List[int], group: Optional[ProcessGroup] = None) -> None
```

**功能**：释放 `create_shared_buffer` 创建的共享缓冲区。**集体**：group 内所有 rank 必须
一起调用；调用前须确保没有内核仍在使用这些缓冲区（同步相关 stream）。

**参数**

- `pointers`（`list[int]`）：`create_shared_buffer` 返回的每 rank 指针列表。
- `group`（`ProcessGroup`，可选）：分配时使用的 process group；默认 `dist.group.WORLD`。

**注意事项**

- 拆除顺序对 CUDA IPC 很重要：每个 rank 先关闭自己用 `cudaIpcOpenMemHandle` 打开的对端映射，
  barrier 确认所有映射关闭后，各 rank 才 `cudaFree` 自己的分配——对仍被 peer 进程 IPC 映射的
  内存执行 free 属未定义行为。

---

### `pack_strided_memory`

```python
pack_strided_memory(
    ptr: int,
    segment_size: int,
    segment_stride: int,
    num_segments: int,
    dtype: torch.dtype,
    dev_id: int,
) -> torch.Tensor
```

**功能**：把跨步设备分配包装为 PyTorch 张量视图。每次调用都会新建一个 DLPack capsule，
因此该函数返回的每个张量各自消费自己的 capsule。

**参数**

- `ptr`（`int`）：设备指针（例如 `cudaMalloc` 返回的）。
- `segment_size`（`int`）：每段字节数。
- `segment_stride`（`int`）：相邻段之间的字节步长。
- `num_segments`（`int`）：暴露的段数。
- `dtype`（`torch.dtype`）：结果张量的元素 dtype。
- `dev_id`（`int`）：`ptr` 所在的 CUDA 设备 id。

**返回**

- `torch.Tensor`：查看所提供设备内存的张量。

---

### `Mapping`

```python
class Mapping:
    def __init__(
        self,
        world_size=1,
        rank=0,
        gpus_per_node=8,
        *,
        cp_size=1,
        cp_config=None,
        tp_size=1,
        pp_size=1,
        moe_cluster_size=-1,
        moe_tp_size=-1,
        moe_ep_size=-1,
        attn_tp_size=-1,
        attn_cp_size=-1,
        auto_parallel=False,
        enable_attention_dp=False,
    )
```

**功能**：描述一个节点/集群的并行拓扑，并推导 TP/PP/CP/MoE-TP/MoE-EP/MoE-cluster 等
rank 分组。类 docstring 给出多个示例（8 GPU、tp=4、pp=2；moe_tp=2、moe_ep=4；多节点
16 GPU；tp=2/pp=2/cp=2 等）。

**`__init__` 参数**

- `world_size`、`rank`、`gpus_per_node`：全局 rank 数、当前 rank、每节点 GPU 数。
- `cp_size`、`cp_config`、`tp_size`、`pp_size`：上下文并行 / TP / PP 大小。
- `moe_cluster_size`、`moe_tp_size`、`moe_ep_size`（默认 `-1` 表示无 MoE）：MoE 并行维度。
- `attn_tp_size`、`attn_cp_size`（默认 `-1`）：注意力 TP/CP 大小。
- `auto_parallel`（`bool`）：自动并行模式。
- `enable_attention_dp`（`bool`）：是否启用 attention DP。

**关键属性 / 方法**

- `tp_rank` / `pp_rank` / `cp_rank`、`moe_tp_rank` / `moe_cluster_rank` / `moe_ep_rank`：
  各维度上的本地 rank。
- `tp_group` / `pp_group` / `cp_group`、`moe_tp_group` / `moe_cluster_group` / `moe_ep_group`：
  对应的 rank 列表。
- `node_rank` / `local_rank`、`get_node_rank(rank)` / `get_local_rank(rank)`。
- `has_tp()` / `has_pp()` / `has_cp()` / `has_moe_tp()` / `has_moe_ep()` / `has_moe_cluster()`、
  `is_multi_node()`、`is_first_pp_rank()` / `is_last_pp_rank()` / `is_second_last_pp_rank()`、
  `prev_pp_rank()` / `next_pp_rank()`。
- `pp_layers(num_layers)`：按 PP rank 切分层。
- `ep_experts(num_experts)`：按 MoE EP rank 切分专家。
- `from_dict(mapping)` / `to_dict()`：序列化。
- `__eq__` / `__hash__`：按拓扑字段比较。

**注意事项**

- `attn_cp_size` 目前必须为 1，否则抛 `ValueError`。
- 非 auto parallel 时须满足 `tp_size * pp_size * cp_size == world_size`、
  `tp_size == moe_tp_size * moe_ep_size * moe_cluster_size`、
  `tp_size * cp_size == attn_tp_size * attn_cp_size`，否则抛 `ValueError`。
- `moe_ep_size != 1` 且 `cp_size > 1` 时抛 `NotImplementedError`。

**示例**

```python
from flashinfer.comm import Mapping

mapping = Mapping(world_size=8, rank=0, gpus_per_node=8, tp_size=4, pp_size=2)
print(mapping.tp_group, mapping.pp_group)
```

---

### `all_gather_matmul`

```python
all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    verbose: bool = False,
)
```

**功能**：运行 push-wait all-gather matmul。每个 rank 持有 `(M, K)` 的本地输入与 `(K, N)`
权重，计算输出 `(M * world_size, N)`，其中 slice `[i]` 为 `input[i] @ weight`。算法分两阶段：
push（每 rank 把本地输入分块写入每个 peer 的对称内存缓冲区并置就绪信号）与 wait-matmul
（matmul 内核遍历所有 peer，对本地块立即计算，对远程块先自旋等待信号）。

**参数**

- `inp`（`torch.Tensor`）：本地输入，`(M, K)`。
- `w`（`torch.Tensor`）：权重，`(K, N)`。
- `group`（`dist.ProcessGroup`）：process group。
- `backend`（`str`，默认 `"auto"`）：`"auto"` 保持既有的 Blackwell cuTile / 其他平台
  Triton 路由；`"cake"` 选择精确源码构建的 Blackwell 实现并拒绝不支持的输入。
- `verbose`（`bool`，默认 `False`）：是否打印调试信息。

**返回**

- 输出张量，形状 `(M * world_size, N)`。

**注意事项**

- `backend` 非 `"auto"`/`"cake"` 时抛 `ValueError`。
- 路由：SM >= 100（Blackwell+）走 cuTile 实现；SM < 100 走 Triton 实现。

**示例**

```python
import torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from flashinfer.comm import all_gather_matmul

device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
dist.init_process_group("nccl", rank=rank, world_size=world_size)
inp = symm_mem.empty(16384, 8192, device=device, dtype=torch.bfloat16).normal_()
w = torch.randn(8192, 2048, device=device, dtype=torch.bfloat16)
out = all_gather_matmul(inp, w, dist.group.WORLD)  # (16384 * world_size, 2048)
```

---

### `prepare_all_gather_matmul`

```python
prepare_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    max_rows: Optional[int] = None,
    verbose: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**功能**：准备一个容量受限的 Cake all-gather matmul 启动器。返回的可调用对象绑定了 `w` 与
`group`，接受任意具有 `inp` 的 dtype、设备与 `K`、且行数不超过 `max_rows`（默认
`inp.shape[0]`）的连续输入；对称 scratch 在这唯一一次集体调用中按该容量确定大小，因此后续
行数更少的调用不执行任何集体。`backend="auto"` 与 `"cake"` 都会选择 Cake 路由。启动器准备
完成后可被 CUDA Graph 捕获。不支持的输入在准备阶段抛出，而不是回退到其他实现。

**参数**

- `inp`、`w`、`group`：同上。
- `backend`（`str`，默认 `"auto"`）：`"auto"` 或 `"cake"`。
- `max_rows`（`int`，可选）：行容量上限；默认 `inp.shape[0]`。
- `verbose`（`bool`，默认 `False`）。

**返回**

- `Callable[[torch.Tensor], torch.Tensor]`：接受连续输入并返回结果的启动器。

**注意事项**

- `w` 为逻辑 `[8192, N]` 权重，可连续，也可为连续 `[N, 8192]` 参数的转置视图，`N` 为 256 的
  正倍数。Cake 路由支持 SM100/SM103、bfloat16/float16、2/4/8-rank NCCL group，以及 torch
  默认 CUDA 或 NVSHMEM 对称内存后端。

---

### AllReduce 枚举（TRT-LLM）

**`AllReduceFusionOp`**：TRT-LLM allreduce fusion 操作码。
`NONE=0`、`RESIDUAL_RMS_NORM=1`、`LAST_PROCESS_FOR_UB=2`、`RESIDUAL_RMS_PREPOST_NORM=3`、
`RESIDUAL_RMS_NORM_QUANT_FP8=4`、`RESIDUAL_RMS_NORM_QUANT_NVFP4=5`、
`RESIDUAL_RMS_NORM_OUT_QUANT_FP8=6`、`RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4=7`、
`MOE_ALLREDUCE_RESIDUAL_RMS_NORM=8`、`MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM=9`。

**`AllReduceFusionPattern`**：`trtllm_allreduce_fusion` 的融合模式。
`kAllReduce=0`、`kARResidualRMSNorm=1`、`kARResidualRMSNormFP8Quant=2`、
`kARResidualRMSNormFP4Quant=3`、`kARResidualRMSNormOutFP8Quant=4`、
`kARResidualRMSNormOutFP4Quant=5`、`kMoEReductionARResidualRMSNorm=6`、
`kMoEFinalizeARResidualRMSNorm=7`、`kARResidualRMSNormPerTokenGroupFP8PackedQuant=8`、
`kARResidualRMSNormOutPerTokenGroupFP8PackedQuant=9`、`kARResidualRMSNormDynamicFP8Quant=10`、
`kARResidualRMSNormOutDynamicFP8Quant=11`。

**`AllReduceStrategyConfig`**：`USE_MEMCPY = 1 << 0`、`PUSH_MODE = 1 << 1`。

**`AllReduceStrategyType`**：`NCCL=0`、`MIN_LATENCY=1`、`UB=2`、`AUTO=3`、`ONESHOT=4`、
`TWOSHOT=5`、`LOWPRECISION=6`。

**`QuantizationSFLayout`**：量化 scale factor 布局。`SWIZZLED_128x4=0`、`SWIZZLED_8x4=1`、
`LINEAR=2`。swizzled 布局把 scale factor block 按 512 字节组织在全局内存中，每个 block 含
128x4 个 FP8 值，因此 SF 矩阵维度被填充（行向上取整到 128、列向上取整到 4）；SF 行 `i` 映射到
数据 block 行 `(i % 4) * 32 + (i / 4)`。

---

### `trtllm_allreduce_fusion`

```python
trtllm_allreduce_fusion(
    allreduce_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    pattern_code: AllReduceFusionPattern,
    use_oneshot: Optional[bool],
    allreduce_out: Optional[torch.Tensor],
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    rms_eps: Optional[float],
    scale_factor: Optional[Union[torch.Tensor, float]],
    layout_code: Optional[QuantizationSFLayout],
    metadata: Optional[dict] = None,
    block_quant_group_size: Optional[int] = None,
    weight_bias: Optional[float] = None,
) -> None
```

**功能**：执行 TRT-LLM AllReduce fusion。

**参数**

- `allreduce_in`：输入张量，`[token_num, hidden_dim]`。
- `world_size` / `world_rank`：process group 大小 / 当前 rank。
- `token_num` / `hidden_dim`：token 数 / hidden 维度。
- `workspace_ptrs`：工作区指针。
- `launch_with_pdl`：是否以 PDL 启动。
- `use_oneshot`：是否使用 oneshot；为 `None` 时用内部启发式。
- `trigger_completion_at_end`：是否在末尾触发完成。
- `fp32_acc`：是否使用 fp32 累加。
- `pattern_code`：融合模式（`AllReduceFusionPattern`）。
- `allreduce_out` / `residual_in` / `residual_out` / `norm_out` / `quant_out`：输出/输入张量，均为 `[token_num, hidden_dim]`。
- `scale_out`：scale 输出；初始化参考 `tests/comm/test_trtllm_allreduce_fusion.py`。
- `rms_gamma`：`[hidden_dim]`。
- `rms_eps`：RMS epsilon。
- `scale_factor`：scale factor；为 CUDA Graph 安全起见应传张量。
- `layout_code`：量化布局码。
- `metadata`（可选）：来自 `create_ipc_workspace_for_all_reduce_fusion` 的工作区元数据，提供时
  校验 `token_num <= max_token_num`、`world_size == tp_size`、`hidden_dim` 一致，失败抛 `ValueError`。
- `block_quant_group_size`（可选）：per-token-group 块量化模式的组大小（沿 hidden_dim 的连续
  元素数共享一个 scale factor），须 `> 0` 且能整除 `hidden_dim`（当模式需要时）；不进行块量化
  的模式忽略（视为 0/未用）。
- `weight_bias`（可选）：在缩放前加到 `rms_gamma` 的偏置。`None`/`0.0` 为标准 RMSNorm；`1.0`
  为 Gemma / Qwen3.5 RMSNorm（`out = (1 + gamma) * x * rsqrt(...)`）；对 `kAllReduce` 与不应用
  RMSNorm 的 quant-only 模式忽略。

**返回**

- `None`。

**注意事项**

- 非 oneshot 时断言 `token_num > world_size`。

---

### `trtllm_moe_allreduce_fusion`

```python
trtllm_moe_allreduce_fusion(
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    layout_code: Optional[QuantizationSFLayout],
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    weight_bias: Optional[float] = None,
    *,
    backend: Literal["trtllm", "cake"] = "trtllm",
) -> None
```

**功能**：MoE reduction + AllReduce + RMSNorm（+可选量化）融合，把 expert 加权归约与 allreduce、
norm 融合进单个内核。

**参数**

- `world_size` / `world_rank` / `token_num` / `hidden_dim` / `workspace_ptrs` / `launch_with_pdl`：同上。
- `residual_in`：`[token_num, hidden_dim]`。
- `rms_gamma`：`[hidden_dim]`。`rms_eps`：RMS epsilon。
- `scale_factor`：scale factor。
- `moe_reduction_device_num_experts`：专家数。
- `moe_reduction_scale_input` / `moe_reduction_active_experts_token_input` /
  `moe_reduction_token_input`：`[token_num, hidden_dim]`。
- `layout_code`：量化布局码。
- `moe_allreduce_out` / `residual_out` / `norm_out`：`[token_num, hidden_dim]`；
  `quant_out`：`[token_num // 4, hidden_dim]`，fp16/bf16 → fp4。
- `scale_out`：scale 输出；初始化参考 `tests/comm/test_trtllm_moe_allreduce_fusion.py`。
- `weight_bias`：同 `trtllm_allreduce_fusion`。
- `backend`（关键字限定，默认 `"trtllm"`）：`"trtllm"` 或受限的 `"cake"` SM100/SM103 后端。Cake
  可选后端支持连续 FP16/BF16 张量、world size 2/4/8、`hidden_dim=7168`、payload 在既有
  Lamport `MAX_COMM_SIZE` 字节限制内、且提供 residual 与 norm 输出；不支持量化。`weight_bias`
  仍是运行期值，`None` 以 0.0 传给内核。

**返回**

- `None`。

**注意事项**

- 仅支持 one-shot；`required_lamport_comm_size > MAX_COMM_SIZE` 时抛 `ValueError`。

---

### `trtllm_moe_finalize_allreduce_fusion`

```python
trtllm_moe_finalize_allreduce_fusion(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: Optional[torch.Tensor],
    expert_scale_factor: Optional[torch.Tensor],
    routed_scaling_factor: Optional[float],
    weight_bias: Optional[float] = None,
    *,
    backend: Literal["trtllm", "cake"] = "trtllm",
) -> None
```

**功能**：MoE finalize + AllReduce + RMSNorm 融合——把 top-k expert gather/scale 与 allreduce、
norm 融合进单个内核，支持共享专家输出相加（如 DeepSeek 架构）。

**参数**

- `allreduce_in`：permuted/padded 的 MoE expert 输出，形状 `[num_permuted_rows, hidden_dim]`；
  `num_permuted_rows` 可因 expert padding 大于 `token_num * top_k`。行由
  `expanded_idx_to_permuted_idx` 引用。
- `residual_in` / `norm_out` / `residual_out`：`[token_num, hidden_dim]`。
- `norm_weight`：`[hidden_dim]`。
- `expanded_idx_to_permuted_idx`：`[token_num, top_k]`。
- `quant_out`：packed FP4 输出缓冲，`token_num * hidden_dim // 2` 字节。
- `scale_out`：SWIZZLED_128x4 布局的 E4M3 scale 输出，
  `round_up(token_num, 128) * round_up(hidden_dim // 16, 4)` 字节。
- `workspace_ptrs` / `launch_with_pdl` / `world_rank` / `world_size` / `eps`：同上。
- `shared_expert_output`：`[token_num, hidden_dim]`。
- `expert_scale_factor`：`[token_num, top_k]`。
- `routed_scaling_factor`：routed scaling factor。
- `weight_bias`：同前。
- `backend`（默认 `"trtllm"`）：`"trtllm"` 保留既有实现；`"cake"` 选择源码构建的 SM100/SM103
  内核，适用于 hidden size 7168 且 TP size 为 2/4/8。

**返回**

- `None`。

---

### `trtllm_create_ipc_workspace_for_all_reduce_fusion`

```python
trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
    group: Optional[ProcessGroup] = None,
    create_metadata: bool = False,
    comm_backend: Optional[CommBackend] = None,
    use_symm_dev_mem: bool = False,
) -> Union[
    Tuple[List[List[int]], torch.Tensor],
    Tuple[List[List[int]], torch.Tensor, dict],
    Tuple[List[List[int]], torch.Tensor, List[SymmDeviceMemory], dict],
]
```

**功能**：为 allreduce fusion 创建 3 个 IPC 缓冲区，大小分别为
`[buffer_size, flag_size, lamport_buffer_size * 3]`。

**参数**

- `tp_rank` / `tp_size`：当前 rank / group 大小。
- `max_token_num`：序列最大 token 数。
- `hidden_dim`：hidden 维度。
- `use_fp32_lamport`（默认 `False`）：是否使用 fp32 数据类型。
- `group`：使用的 process group。
- `create_metadata`（默认 `False`）：是否返回 metadata 字典作为第三个元素。
- `comm_backend`：使用的通信后端。
- `use_symm_dev_mem`（默认 `False`）：是否使用对称设备内存。

**返回**

- `create_metadata=False`：`(ipc_handles, workspace_tensor)`。
- `create_metadata=True` 且 `use_symm_dev_mem=False`：`(ipc_handles, workspace_tensor, metadata)`，
  metadata 含 `tp_rank/tp_size/max_token_num/hidden_dim/use_fp32_lamport/buffer_size/flag_size/lamport_comm_size/lamport_buffer_size`。
- `create_metadata=True` 且 `use_symm_dev_mem=True`：`(ipc_handles, workspace_tensor, mem_handles, metadata)`，
  metadata 额外含 `control_flag_ptr`；`mem_handles` 为 `SymmDeviceMemory` 列表。

**注意事项**

- 缓冲区大小为 `buffer_size = tp_size * max_token_num * hidden_dim * (4 if use_fp32_lamport else 2)`、
  `flag_size = tp_size * BarrierFlagCount * 4`、`lamport_buffer_size = lamport_comm_size * 3`，
  其中 `lamport_comm_size = tp_size * max_token_num * hidden_dim * (2 或 4)`，并被限制在
  `MAX_COMM_SIZE`（`MAX_INT32` 向下取整到 2MB）以内。
- `use_symm_dev_mem=True` 时必须 `create_metadata=True`，否则抛 `ValueError`。
- 此处 `tp_size` 与 `world_size` 可互换（allReduceFusion）。

---

### `trtllm_destroy_ipc_workspace_for_all_reduce`

```python
trtllm_destroy_ipc_workspace_for_all_reduce(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None
```

**功能**：释放为一个 IPC 工作区持有的对称内存引用，为向后兼容保留。与之配对的旧
`trtllm_create_ipc_workspace_for_all_reduce` 已被移除；对由
`trtllm_create_ipc_workspace_for_all_reduce_fusion` 创建的工作区请改用
`trtllm_destroy_ipc_workspace_for_all_reduce_fusion`（二者行为一致）。释放后不应再使用该
workspace 列表。

---

### `trtllm_destroy_ipc_workspace_for_all_reduce_fusion`

```python
trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
    workspace: List[List[int]], group: Optional[ProcessGroup] = None
) -> None
```

**功能**：销毁由 `trtllm_create_ipc_workspace_for_all_reduce_fusion` 创建的工作区，释放内部
持有的对称内存引用。释放后不应再使用该 workspace 列表。`group` 未使用，仅为 API 兼容保留。

---

### `trtllm_lamport_initialize`

```python
trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None
```

**功能**：把单个 Lamport 风格缓冲区初始化为负零。

**参数**

- `buffer_ptr`（`int`）：缓冲区设备指针。
- `size`（`int`）：缓冲区元素数。
- `dtype`（`torch.dtype`）：元素 dtype。

---

### `trtllm_lamport_initialize_all`

```python
trtllm_lamport_initialize_all(
    buffer_0_ptr: int,
    buffer_1_ptr: int,
    buffer_2_ptr: int,
    size: int,
    dtype: torch.dtype,
) -> None
```

**功能**：把三个 Lamport 缓冲区初始化为负零。

**参数**

- `buffer_0_ptr` / `buffer_1_ptr` / `buffer_2_ptr`（`int`）：三个缓冲区的设备指针。
- `size`（`int`）：每个缓冲区的元素数。
- `dtype`（`torch.dtype`）：每个缓冲区的元素 dtype。

---

### `compute_fp4_swizzled_layout_sf_size`

```python
compute_fp4_swizzled_layout_sf_size(total_row: int, total_column: int) -> int
```

**功能**：计算 FP4 swizzled 布局填充后的大小（行数乘列数）。

**参数**

- `total_row`（`int`）：未填充布局的逻辑行数。
- `total_column`（`int`）：未填充布局的逻辑列数。

**返回**

- `int`：swizzled 布局的 `padded_row * padded_column`（`padded_row` 向上取整到 128，
  `padded_column` 向上取整到 4）。

---

### `allreduce_fusion`

```python
allreduce_fusion(
    input: torch.Tensor,
    workspace: AllReduceFusionWorkspace,
    pattern: int,
    launch_with_pdl: bool = False,
    trigger_completion_at_end: bool = True,
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    residual_in: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: float = 1e-6,
    scale_factor: Optional[Union[torch.Tensor, float]] = None,
    layout_code: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
    moe_reduction_device_num_experts: Optional[int] = None,
    moe_reduction_scale_input: Optional[torch.Tensor] = None,
    moe_reduction_active_experts_token_input: Optional[torch.Tensor] = None,
    moe_reduction_token_input: Optional[torch.Tensor] = None,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    expert_scale_factor: Optional[torch.Tensor] = None,
    shared_expert_output: Optional[torch.Tensor] = None,
    block_quant_group_size: Optional[int] = None,
    weight_bias: float = 0.0,
    *,
    moe_finalize_backend: Literal["trtllm", "cake"] = "trtllm",
) -> torch.Tensor
```

**功能**：AllReduce + RMSNorm 融合操作，对支持的后端可选 FP8/NVFP4 量化。AllReduce 后端由
workspace 类型自动决定；如需其他后端，请创建对应后端的工作区。`moe_finalize_backend` 仅在
TRT-LLM 工作区上选择 MoE finalize 模式的实现。支持的模式：仅 AllReduce；AllReduce +
Residual + RMSNorm；AllReduce + Residual + RMSNorm + 量化（FP8/NVFP4）。

**参数**

- `input`：标准 allreduce 模式为 `[token_num, hidden_dim]`；`kMoEFinalizeARResidualRMSNorm`
  时为 permuted/padded 的 MoE expert 输出 `[num_permuted_rows, hidden_dim]`，token 输出形状由
  `residual_in` 决定。
- `workspace`：由通用工厂或后端专用构造器创建的工作区，其具体类型决定后端。
- `pattern`：融合模式（`AllReduceFusionPattern` 常量）。MNNVL 支持标准 FP8/NVFP4 量化模式
  （2-5）与 dynamic FP8 模式（10-11）；packed group 量化模式仅 TRT-LLM 支持。
  `kMoEFinalizeARResidualRMSNorm` 可经 TRT-LLM、TRT-LLM 工作区上的 Cake，以及显式 MNNVL CuTe
  DSL 后端使用。
- `launch_with_pdl`：使用 PDL。MNNVL CuTe DSL preset 决定其编译期 PDL 模式并在与本值不一致时警告。
- `trigger_completion_at_end`（仅 TRT-LLM）：控制 PDL 完成信号时机，`True`（默认）在内核结束后
  信号（安全、无重叠）；`False` 提前信号以允许下一个 PDL-aware 内核重叠；MNNVL 忽略。
- `output`：预分配 AllReduce 输出 `[token_num, hidden_dim]`。
- `residual_out`：预分配 norm 前输出（residual add 后、norm 前）。
- `norm_out`：预分配归一化输出。
- `quant_out`：预分配量化输出。FP8 形状 `[token_num, hidden_dim]`；NVFP4 形状
  `[token_num, hidden_dim / 2]`。
- `scale_out`：预分配量化 scale factor 缓冲。Dynamic FP8 形状 `[token_num, 1]`、dtype
  `torch.float32`；NVFP4 用 `layout_code` 选择的布局；per-tensor FP8 不使用。
- `residual_in`：要相加的 residual，`[token_num, hidden_dim]`。
- `rms_gamma`：RMSNorm 权重，`[hidden_dim]`。`rms_eps`：RMS epsilon。
- `scale_factor`：FP8/NVFP4 量化使用的输出 scale。
- `layout_code`：NVFP4 scale factor 布局（`QuantizationSFLayout`）。MNNVL 支持
  `SWIZZLED_128x4` 与 `LINEAR`；`SWIZZLED_8x4` 仅 TRT-LLM。
- `use_oneshot`、`fp32_acc`：控制参数。
- MoE reduction 参数（`pattern=kMoEReductionARResidualRMSNorm`）与 MoE finalize 参数
  （`pattern=kMoEFinalizeARResidualRMSNorm`）：见同名底层函数。
- `block_quant_group_size`：group quant 参数。
- `weight_bias`（默认 `0.0`）：RMSNorm 变体。

**返回**

- `torch.Tensor`：AllReduce（或融合）结果。

**注意事项**

- 只要 `workspace.is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype)` 返回
  `True`，即可用不同 `(num_tokens, hidden_dim)` 复用同一工作区。

---

### `create_allreduce_fusion_workspace`

```python
create_allreduce_fusion_workspace(
    backend: Literal["trtllm", "mnnvl", "auto"] = "auto",
    world_size: int = None,
    rank: int = None,
    max_token_num: int = None,
    hidden_dim: int = None,
    dtype: torch.dtype = None,
    gpus_per_node: int = None,
    comm_backend: Optional[CommBackend] = None,
    force_oneshot_support: bool = False,
    group: Optional[ProcessGroup] = None,
) -> AllReduceFusionWorkspace
```

**功能**：为 AllReduce fusion 操作创建工作区，后端选择使用基于拓扑的检查与启发式。

**参数**

- `backend`（默认 `"auto"`）：`"auto"` 用基于拓扑的启发式在 `"trtllm"` 与 `"mnnvl"` 间选择。
- `world_size` / `rank`：group 中 rank 数 / 当前 rank id。
- `max_token_num`：工作区需支持的最大 token 数。
- `hidden_dim`：hidden 维度。
- `dtype`：通信张量的元素 dtype。
- `gpus_per_node`（可选）：每节点 GPU 数（多节点拓扑决策）；默认
  `min(torch.cuda.device_count(), world_size)`。
- `comm_backend`（可选）：rendezvous 使用的通信后端；默认 process group 默认值。
- `force_oneshot_support`（默认 `False`）：`True` 时按最大请求问题大小分配 oneshot 策略工作区；
  `False` 时为所有问题大小分配 twoshot 工作区、并为 oneshot 分配到启发式阈值。
- `group`（可选）：对称内存 rendezvous 使用的 process group（仅 TRT-LLM 后端）。

**返回**

- `AllReduceFusionWorkspace`：`TRTLLMAllReduceFusionWorkspace` 或
  `MNNVLAllReduceFusionWorkspace`，其类型决定 `allreduce_fusion` 分派到哪个后端。

**注意事项**

- 无可用后端或问题大小不受支持时抛 `ValueError`；显式 `backend` 不匹配已知实现时抛 `RuntimeError`。
- MNNVL 后端在 NVIDIA Confidential Computing 下不可用（需要 NVLink multicast），此时抛 `ValueError`。

**示例**

```python
ws = create_allreduce_fusion_workspace(
    backend="auto", world_size=8, rank=0,
    max_token_num=2048, hidden_dim=4096, dtype=torch.bfloat16,
)
print(ws.backend)  # "trtllm"
```

---

### `AllReduceFusionWorkspace`

```python
class AllReduceFusionWorkspace(ABC):
    def __init__(self, world_size: int, rank: int)
```

**功能**：AllReduce fusion 工作区的抽象基类。

**关键成员**

- `world_size` / `rank`：group 大小 / 当前 rank。
- `backend`（抽象属性）：返回后端名称。
- `destroy()`（抽象方法）：销毁工作区并释放资源。
- `is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype, use_oneshot=None)`（抽象方法）：
  检查工作区是否足够。
- `__del__`：若未显式 `destroy()`，发出 `ResourceWarning` 并尝试清理。

---

### `TRTLLMAllReduceFusionWorkspace`

```python
class TRTLLMAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float16,
        comm_backend: Optional[CommBackend] = None,
        group: Optional[ProcessGroup] = None,
    )
```

**功能**：TensorRT-LLM 的 AllReduce fusion 工作区。

**`__init__` 参数**

- `tp_size` / `tp_rank`：tensor parallel 大小 / rank。
- `max_token_num` / `hidden_dim`：最大 token 数 / hidden 维度。
- `dtype`（默认 `torch.float16`）：数据类型。
- `comm_backend`：通信后端。
- `group`：对称内存 rendezvous 的 process group；默认 `torch.distributed.group.WORLD`。

**关键属性 / 方法**

- `backend`：恒为 `"trtllm"`。
- `ipc_handles` / `workspace_tensor` / `mem_handles` / `metadata`：内部工作区暴露的属性。
- `is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype, use_oneshot=None)`：按元数据校验。
- `checkpoint_prepare()`：分离物理 backing（释放不可 checkpoint 的句柄），对
  `SymmDeviceMemory` 支撑的工作区不支持。重复成功调用为 no-op。
- `checkpoint_restore(comm_backend)`：恢复物理 backing。`comm_backend` 必须复现原 rank 与 world
  size。重复成功调用为 no-op。
- `destroy()`：销毁工作区并释放资源（幂等）。

**注意事项**

- 两个 checkpoint 方法都是 **collective**，所有 rank 必须按相同顺序调用。若在 detach/reattach
  开始后发生异常，不要重试或复用，重启受影响 rank。torch 对称内存支撑的工作区不支持该生命周期。
- NVIDIA Confidential Computing 下禁用对称设备内存（需要无 multicast 的 IPC 工作区）。

---

### `MNNVLAllReduceFusionWorkspace`

```python
class MNNVLAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        buffer_size_in_bytes: Optional[int] = None,
        comm_backend: Optional[CommBackend] = None,
    )
```

**功能**：MNNVL AllReduce fusion 工作区。按问题大小分配初始化；若 `max_num_tokens` 超过
one-shot 阈值，则按「阈值处所需 one-shot 大小」与「所需 two-shot 大小」的最大值创建工作区。
工作区不绑定具体问题大小，只要分配大小足够即可复用。

**`__init__` 参数**

- `mapping`：含 rank 信息的 Mapping 配置。
- `max_num_tokens` / `hidden_dim` / `dtype`：要归约的输入最大 token 数 / hidden 维度 / 数据类型。
- `buffer_size_in_bytes`：每 lamport 缓冲区的请求字节数；实际可用大小为
  `NUM_LAMPORT_BUFFERS * actual_buffer_size_per_lamport_buffer`。覆盖高级用法（例如强制 oneshot
  策略、忽略启发式）。
- `comm_backend`：用于创建工作区与同步；未提供时使用 `MPIBackend`（使用 `COMM_WORLD`）。

**注意事项**

- 必须提供 `max_num_tokens`（并带 `hidden_dim`、`dtype`）或 `buffer_size_in_bytes` 之一。
- `buffer_size_in_bytes` 超过 `UINT32_MAX` 时抛 `ValueError`。
- 类常量：`NUM_LAMPORT_BUFFERS = 3`、`TWOSHOT_NUM_STAGES = 2`、`PACKED_ALIGNMENT_BYTES = 16`。
- 相关策略枚举 `MNNVLAllreduceFusionStrategy`：`ONESHOT=0`、`TWOSHOT=1`、`AUTO=99`；
  `MNNVLQuantType`：`NONE=0`、`FP8=1`、`NVFP4=2`、`DYNAMIC_FP8=3`。二者未列于 rst 的
  autosummary，但为这些函数的参数类型。

---

### `quantized_all_reduce`

```python
quantized_all_reduce(
    inp: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    scale_group: int = SCALE_GROUP_DEFAULT,
    block_size: int | None = None,
    num_warps: int | None = None,
    max_num_blocks: int | None = None,
    p2p_phase3: bool | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor
```

**功能**：带 per-block scaling 的 FP8 量化 two-shot AllReduce。在经对称内存做 P2P 传输前，把
激活量化为带 per-block scale 的 FP8，使集合期间的 GPU 间带宽减半。

**参数**

- `inp`：输入张量（BF16、连续、numel 可被 8 整除）。
- `group`：集合的 process group。
- `scale_group`（默认 256）：每个 scale group 的元素数；更小值精度更好、带宽代价略增。
- `block_size`：计算 tile 大小；`None` 自动（`<= 64MB` 为 4096，更大为 16384）。
- `num_warps`：每线程块 warp 数；`None` 自动（16）。
- `max_num_blocks`：最大使用 SM 数；`None` 按张量大小自动调优。
- `p2p_phase3`：Phase 3 读取策略；`None` 自动（`>= 32MB` 为 `True`）。
- `output`：可选预分配输出；`None` 时新申请。

**返回**

- 归约后的输出张量（形状/dtype 与输入相同）。

**注意事项**

- 需要 SM90+（Hopper 或更新）以支持 FP8 E4M3 和 NVSwitch 对称内存。world size 必须整除
  `block_size`（默认 4096），因此 world size 需为 2 的幂、最高 16。
- 最适合 `>= 2MB` 的张量；更小的张量下 `torch.ops.symm_mem.multimem_all_reduce_` 或
  `flashinfer.comm.allreduce_fusion` 可能因固定开销更低而更快。
- 首次调用会分配对称内存缓冲（跨所有 rank 的集体）。在 warmup 时用最大预期张量调用一次，
  避免推理期间重新分配。
- 通过 `flashinfer.comm.quantized_all_reduce` 惰性导出（模块级 `__getattr__`）。

---

### vLLM AllReduce

```python
vllm_init_custom_ar(ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool) -> int
vllm_dispose(fa: int) -> None
vllm_all_reduce(fa: int, inp: torch.Tensor, out: torch.Tensor, reg_buffer: int, reg_buffer_sz_bytes: int, num_ctas: int) -> None
vllm_get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]
vllm_register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None
vllm_register_graph_buffers(fa: int, handles: List[List[int]], offsets: List[List[int]]) -> None
vllm_meta_size() -> int
```

**功能**：vLLM 自定义 all-reduce 后端。

- `vllm_init_custom_ar`：初始化后端，返回不透明句柄 `fa`。`ipc_tensors` 为每 rank 通信缓冲的
  IPC 指针；`rank_data` 为元数据交换的 scratch 张量；`full_nvlink=True` 表示每对 rank 都经
  NVLink 连接（启用完全 NVLink 优化内核路径）。
- `vllm_dispose(fa)`：释放句柄资源。
- `vllm_all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas)`：通过 vLLM 自定义
  内核执行 out-of-place all-reduce。`reg_buffer` 为内核使用的注册缓冲设备指针；`num_ctas`
  为启动的 CTA 数（上限 36）。
- `vllm_get_graph_buffer_ipc_meta(fa)`：返回 `(output_bytes, output_offsets)`，描述已注册的
  graph-capture 缓冲，用于向 peer rank 发布 IPC 句柄。
- `vllm_register_buffer(fa, fake_ipc_ptrs)`：把 peer 的 IPC-shared 缓冲注册到本地句柄。
- `vllm_register_graph_buffers(fa, handles, offsets)`：跨 all-reduce world 注册 graph 缓冲。
- `vllm_meta_size()`：返回 vLLM all-reduce 元数据结构字节数。

**注意事项**

- 这些名称是 `flashinfer.comm` 对 `flashinfer.comm.vllm_ar` 的别名：`vllm_all_reduce` =
  `vllm_ar.all_reduce`，`vllm_dispose` = `vllm_ar.dispose`，`vllm_init_custom_ar` =
  `vllm_ar.init_custom_ar`，`vllm_register_buffer` = `vllm_ar.register_buffer`，
  `vllm_register_graph_buffers` = `vllm_ar.register_graph_buffers`，
  `vllm_get_graph_buffer_ipc_meta` = `vllm_ar.get_graph_buffer_ipc_meta`，`vllm_meta_size` =
  `vllm_ar.meta_size`。

---

### `PcieIpcAllReduceWorkspace`

```python
class PcieIpcAllReduceWorkspace:
    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = AR_MAX_BLOCKS,
        profile: Optional[str] = None,
        tune_batches: Optional[Sequence[int]] = None,
        tune_cache: Optional[str] = None,
    )
```

**功能**：PCIe IPC all-reduce 的共享工作区。为每 rank 分配一块 slab，通过 CUDA IPC 共享并
绑定到内核。工作区一次性定容、不可增长，因此 `max_numel` 必须覆盖将发出的最大集合；超过则
必须回退到其他后端。

**`__init__` 参数**

- `group`（`ProcessGroup`）：共享工作区的 process group；所有 rank 必须以相同参数构建。
- `max_numel`（`int`）：将被 all-reduce 的最大元素数。
- `dtype`（`torch.dtype`，默认 `torch.bfloat16`）：bfloat16 或 float16（只绑定元素*大小*，故
  一个工作区同时服务两者）。
- `max_blocks`（`int`）：任一次启动可请求的 block 上限，决定 barrier 与 epoch slot 大小。
- `profile`（`str`，可选）：强制互联标签（`"rootcplx"` 或 `"pcieswitch"`）而非探测；该标签不选
  内核，只分区 tune cache，避免两种拓扑互相读取测量结果。探测是集体的，在任何分配前运行。
- `tune_batches`（`Sequence[int]`，可选）：调优用的 batch 桶。
- `tune_cache`（`str`，可选）：调优配置读写位置；默认 `FLASHINFER_AUTOTUNE_DIR`（或工作区目录）。

**关键方法**

- `handle`（属性）：不透明后端句柄。
- `launch_config(inp)`：`inp` 的确定性配置。
- `supports(inp)`：形状/dtype 是否受支持（纯函数，不涉及速度）。
- `all_reduce(inp, *, out=None, config=None, enable_pdl=False) -> torch.Tensor`：out-of-place
  all-reduce。`inp` 为字节数可被 16 整除的连续 CUDA 张量；`out` 省略时申请；`config` 省略时
  从 tune cache 或种子解析。`enable_pdl` **当前被拒绝**（TP8 block 内核在写入 island ack 与
  barrier flag 前就触发 launch 完成）。形状不受支持时抛 `ValueError`（先查 `supports()`）。
- `tuned_launch_config(inp)`：命中缓存时返回测量配置，否则返回种子。
- `prepare(shapes, *, dtype=torch.bfloat16)`：**集体**——立即为每个形状解析启动配置，之后这些
  形状都从进程内缓存服务，使其可被 CUDA Graph 捕获。所有 rank 必须传相同列表（同顺序）。
- `tune(hiddens, *, dtype=torch.bfloat16, cache=None, tune_group=None, warmup=TUNE_WARMUP, repeat=TUNE_REPEAT)`：
  **集体**——为每个调优形状测量启动配置并持久化。
- `rebind_stream()`：在排定两条 stream 顺序后迁移该工作区绑定（一个工作区只服务一条 CUDA stream）。
- `destroy()`：销毁工作区（集体）。
- 也实现上下文管理器协议（`__enter__`/`__exit__`）。

**注意事项**

- 这是 **collective** 且异常严格：内核无超时、无元数据交换地自旋等待 peer flag，因此每 rank
  必须以相同顺序、相同形状/dtype/启动配置发出相同调用序列。跳过、重排或传入不同显式 `config`
  会导致组 hang（而非报错）。`launch_config` 是形状/dtype 与工作区不可变属性的纯函数，正是
  为了让每 rank 无需运行期协商即可得出相同答案。
- 一个工作区只服务 **一条 CUDA stream**；第二条 stream 会被拒绝。
- `max_numel` 应按真实负载设置而非取整：epoch 双缓冲两半相距 `world_size * max_numel` 个元素，
  过大的工作区会拉远两半而带来可测的时间开销。

**示例**

```python
import flashinfer.comm as comm

ws = comm.PcieIpcAllReduceWorkspace(group=group, max_numel=128 * 6144)
for x in activations:                 # 所有 rank 相同形状、相同顺序
    if ws.supports(x):
        y = ws.all_reduce(x)          # out-of-place
    else:
        y = x.clone()
        dist.all_reduce(y, group=group)
ws.destroy()                          # 集体
```

---

### `PcieIpcLaunchConfig`

```python
@dataclass(frozen=True)
class PcieIpcLaunchConfig:
    blocks: int
    threads: int
    variant: IpcVariant
```

**功能**：PCIe IPC all-reduce 的启动几何与内核选择（在 `flashinfer.comm` 中别名为
`PcieIpcLaunchConfig`，源码类型为 `IpcLaunchConfig`）。

---

### `PcieIpcVariant`

```python
class IpcVariant(IntEnum):
    UNSTAGED = 0
    STAGED = 1
    STAGED_RING = 2
    FLAT_STAGED = 3
    COPY_ENGINE_RING = 4
    COPY_ENGINE_ISLAND = 5
    COPY_ENGINE_RING_MEMOP = 6
```

**功能**：选择启动哪个内核；镜像头文件中的 `fi::Variant`。值跨 FFI 边界以整数传输，因此仅追加。
`FLAT_STAGED` 仅在 world size 8 接受。`COPY_ENGINE_RING`、`COPY_ENGINE_ISLAND`、
`COPY_ENGINE_RING_MEMOP` 为 copy-engine 数据平面变体。

**注意事项**

- copy-engine 变体中 `blocks` 不是 grid 大小，而是 ring 的 sub-chunk 深度。
- `COPY_ENGINE_ISLAND` 仅在 world size 8 且仅在 `rootcplx-noswitch` fabric 上可达。

---

### `get_pcie_ipc_launch_config`

```python
get_pcie_ipc_launch_config(
    world_size: int,
    numel: int,
    elem_size: int,
    max_blocks: int = MAX_BLOCKS,
) -> Optional[IpcLaunchConfig]
```

**功能**：为单个形状返回启动配置，形状不受支持时返回 `None`。`None` 表示内核无法运行该形状，
调用方必须使用其他后端；它绝不表示「未调优」——未调优形状会获得种子。仅依赖其参数，因此
group 内每 rank 得到相同答案（这是前提，因为一个 rank 退出而其 peer 参与会死锁集合）。

**参数**

- `world_size`（`int`）：group 大小。
- `numel`（`int`）：元素数。
- `elem_size`（`int`）：每元素字节数。
- `max_blocks`（`int`，默认 `MAX_BLOCKS`）：block 上限。

**返回**

- `Optional[IpcLaunchConfig]`：启动配置或 `None`。

---

### `probe_pcie_ipc_rank_topology`

```python
probe_pcie_ipc_rank_topology(
    rank: int, device: Optional[torch.device] = None
) -> PcieIpcRankTopology
```

**功能**：把本 rank 的 GPU 与其他可见 GPU 进行探测，记录失败。集体解析器先交换 group UUID，
以便也能查询被本进程 `CUDA_VISIBLE_DEVICES` 隐藏的 peer。

**参数**

- `rank`（`int`）：要记录的 rank id。
- `device`（`torch.device`，可选）：CUDA 设备。

**返回**

- `PcieIpcRankTopology`：每 rank 探测结果（`rank`、`hostname`、`device_index`、`device_uuid`、
  `peer_switch_local`、`pair_errors`、`probe_error`）。该数据类未列于 rst autosummary。

---

### `resolve_pcie_ipc_profile`

```python
resolve_pcie_ipc_profile(
    group: ProcessGroup,
    requested: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PcieIpcProfileDecision
```

**功能**：探测每个 rank 并商定一个 profile。**集体**。在任何工作区分配或 JIT 构建之前运行，
因此不支持的拓扑零成本；并收集每 rank 探测结果，使每 rank 从相同证据得出相同决策。

**参数**

- `group`（`ProcessGroup`）：process group。
- `requested`（`str`，可选）：显式请求的 profile 标签。
- `device`（`torch.device`，可选）：CUDA 设备。

**返回**

- `PcieIpcProfileDecision`：`profile` 与 `reason`。

**注意事项**

- 显式 `requested` 总是优先；否则仅当某 rank 正向观察到 switch-local peer 时组才为
  switch-paired，未知或无法探测的情况回退到 `rootcplx-noswitch`。
- CUDA IPC 无法跨主机，因此跨主机的 group 会抛 `ValueError`。

---

### `PcieIpcAllGatherWorkspace`

```python
class PcieIpcAllGatherWorkspace(_PcieIpcWorkspace):
    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = PCIE_IPC_MAX_BLOCKS,
        tune_batches: Sequence[int] = PCIE_IPC_TUNE_BATCHES,
        tune_cache: Optional[str] = None,
    )
```

**功能**：out-of-place PCIe IPC all-gather 工作区。`max_numel` 是最大的 rank-local **输入**
shard，而非 gather 后输出大小。输入形状 `[local_rows, hidden]`，输出为 rank-major 的
`[world_size * local_rows, hidden]`。输入与调用方提供的输出须 16 字节对齐。支持 BF16/FP16/FP32，
所有变体移动不透明 16 字节 pack，故输入位模式被精确保留。默认启动在可用时使用实测的精确形状
cache，否则回退到保守种子。

**关键方法**

- `launch_config(inp)` / `supports(inp)` / `tuned_launch_config(inp)`。
- `tune(hiddens, *, cache=None, warmup=..., repeat=...)`：在工作区的 tune batch 上测量。
- `all_gather(inp, *, out=None, config=None) -> torch.Tensor`：把一个 rank-local shard gather 成
  rank-major 输出。`inp` 为 `[local_rows, hidden]` 连续 rank-local 输入；`out` 为
  `[world_size * local_rows, hidden]`，省略时申请，提供时须连续、16 字节对齐、且不与 `inp` 重叠。

**注意事项**

- 构造、调用与销毁都是 collective；每 rank 必须发出相同调用序列与启动配置。
- 一个工作区绑定到一条有序 CUDA stream，拥有独立于其他 PCIe IPC 工作区的 slab。
- 捕获的 CUDA Graph 会继续使用该工作区的句柄与 IPC slab；须保持工作区存活至所有 replay 完成，
  且不得与其他使用同一工作区的调用并发重放。

---

### `PcieIpcAllGatherLaunchConfig` / `PcieIpcAllGatherVariant`

```python
@dataclass(frozen=True)
class PcieIpcAllGatherLaunchConfig:
    blocks: int
    threads: int
    variant: PcieIpcAllGatherVariant

class PcieIpcAllGatherVariant(IntEnum):
    FLAT_PUSH = 0
    RECURSIVE_DOUBLING = 1
    COPY_ENGINE = 2
```

**功能**：all-gather 的启动配置与内核变体（跨 FFI 边界选择）。`COPY_ENGINE` 使用递归倍增，
仅 world size 8 且 `ordered_4plus4` 时可用（`blocks == 1`、`threads == 32`）；`FLAT_PUSH` 仅
world size 4；`RECURSIVE_DOUBLING` 作为通用回退。

---

### `get_pcie_ipc_all_gather_launch_config`

```python
get_pcie_ipc_all_gather_launch_config(
    world_size: int,
    shard_numel: int,
    max_blocks: int = AG_RS_MAX_BLOCKS,
    element_size: int = 2,
    ordered_4plus4: bool = False,
) -> Optional[PcieIpcAllGatherLaunchConfig]
```

**功能**：返回确定性的 all-gather 启动种子。

**参数**

- `world_size`（`int`）：rank 数，2/4/8。
- `shard_numel`（`int`）：每 rank 输入元素数，须占正数个完整 16 字节 pack。
- `max_blocks`（`int`，默认 `AG_RS_MAX_BLOCKS`）：工作区 block 容量与 launch-grid 上限，1..64。
- `element_size`（`int`，默认 `2`）：每元素字节数，2（BF16/FP16）或 4（FP32）。
- `ordered_4plus4`（`bool`，默认 `False`）：拓扑发现是否验证了 TP8 的 0-3 与 4-7 两个 island 布局；
  决定是否启用 copy-engine 变体，否则 TP8 使用递归倍增。

**返回**

- `Optional[PcieIpcAllGatherLaunchConfig]`：保守启动配置，或 `None`（不支持）。这是种子策略而非
  设备实测结果。

---

### `PcieIpcReduceScatterWorkspace`

```python
class PcieIpcReduceScatterWorkspace(_PcieIpcWorkspace):
    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = PCIE_IPC_MAX_BLOCKS,
        tune_batches: Sequence[int] = PCIE_IPC_TUNE_BATCHES,
        tune_cache: Optional[str] = None,
    )
```

**功能**：基于 PCIe IPC 的 SUM reduce-scatter 工作区。`max_numel` 是最大的 rank-local **输出**
shard。输入形状 `[world_size * local_rows, hidden]`，每个 rank 收到 rank-major 偏移处形状
`[local_rows, hidden]` 的归约 shard。支持 BF16/FP16/FP32；每个内核以 FP32 累加并转换为输出
dtype；TP8 变体还会在最终累加前转换一个四-rank partial，因此结果不承诺与 NCCL 按位一致。

**关键方法**

- `launch_config(inp)` / `supports(inp)` / `tuned_launch_config(inp)` / `tune(...)`。
- `reduce_scatter(inp, *, out=None, config=None) -> torch.Tensor`：`inp` 为
  `[world_size * local_rows, hidden]` 连续输入；每个输出 shard 须适合 `max_numel` 且含完整
  16 字节 pack。`out` 为 `[local_rows, hidden]`，省略时申请，提供时须连续、16 字节对齐、不与
  `inp` 重叠。

**注意事项**

- 归约顺序由所选内核固定；TP8 变体交换一个输出 dtype 的 island partial，故调用方不应依赖与
  NCCL 的按位一致。
- 与 AllGather 工作区相同的 collective 与 stream 约束。

---

### `PcieIpcReduceScatterLaunchConfig` / `PcieIpcReduceScatterVariant`

```python
@dataclass(frozen=True)
class PcieIpcReduceScatterLaunchConfig:
    blocks: int
    threads: int
    variant: PcieIpcReduceScatterVariant

class PcieIpcReduceScatterVariant(IntEnum):
    FLAT_CYCLIC = 0
    FLAT_ONE_PACK = 1
    TOPOLOGY_CYCLIC = 2
    TOPOLOGY_ONE_PACK = 3
```

**功能**：reduce-scatter 的启动配置与内核变体。`TOPOLOGY_*` 变体要求 world size 8 且
`ordered_4plus4`；`FLAT_*` 用于 world size 2/4。`ONE_PACK` 表示每个线程一个 16 字节 pack、以
grid 大小的 batch 处理。

---

### `get_pcie_ipc_reduce_scatter_launch_config`

```python
get_pcie_ipc_reduce_scatter_launch_config(
    world_size: int,
    shard_numel: int,
    max_blocks: int = AG_RS_MAX_BLOCKS,
    element_size: int = 2,
    ordered_4plus4: bool = False,
) -> Optional[PcieIpcReduceScatterLaunchConfig]
```

**功能**：返回确定性的 reduce-scatter 启动种子。

**参数**

- `world_size`（`int`）：rank 数，2/4/8。
- `shard_numel`（`int`）：每 rank 的**输出**元素数（并非完整归约输入），须占正数个完整 16 字节 pack。
- `max_blocks`（`int`，默认 `AG_RS_MAX_BLOCKS`）：1..64。
- `element_size`（`int`，默认 `2`）：2 或 4。
- `ordered_4plus4`（`bool`，默认 `False`）：TP8 必需；TP2/TP4 不需要。

**返回**

- `Optional[PcieIpcReduceScatterLaunchConfig]`：保守启动配置，或 `None`（不支持或 TP8 布局不满足）。


---

### `UlyssesCommunicator`

```python
class UlyssesCommunicator:
    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        *,
        max_elems: int,
        dtype: torch.dtype,
        backend: str = "auto",
        device: Optional[Union[torch.device, str, int]] = None,
    )
```

**功能**：Ulysses 上下文并行 all-to-all 通信器，提供 4-D 布局 `[B, S, H, D]` 上的两个布局变换
（典型注意力层发起四次集合调用：q/k/v 经 `scatter_heads`，输出经 `gather_heads`）：

- `scatter_heads`：`[B, S_local, H, D] -> [B, S_global, H_local, D]`（每 rank 保留*完整*序列的
  head slice）；
- `gather_heads`：`[B, S_global, H_local, D] -> [B, S_local, H, D]`（每 rank 拿回其*本地*序列
  shard 的所有 head）。

其中 `H` 为全局 head 数，`H_local = H // world_size`，`S_global = S_local * world_size`。两个
后端产生按位相同的结果。

**`__init__` 参数**

- `group`（`Optional[ProcessGroup]`）：参与 rank 的 process group；`None` 用 `dist.group.WORLD`。
- `max_elems`（`int`）：单个集合调用每 rank 通信元素数上限，用于确定后端工作区大小。
- `dtype`（`torch.dtype`）：集合操作数的元素 dtype，须为 `torch.float16` / `torch.bfloat16` /
  `torch.float32` 之一。
- `backend`（`str`，默认 `"auto"`）：后端选择策略。`"auto"` 探测拓扑，支持时优先 NVLink，否则
  回退 NCCL；`"nvlink"` 强制 NVLink 后端、不可用时抛错；`"nccl"` 强制 NCCL 路径。
- `device`（`torch.device`/`str`/`int`，可选）：绑定到该 rank 的 CUDA 设备；`None` 用当前设备。

**关键方法 / 属性**

- `backend`（有效后端）、`fallback_reason`、`decision` / `topology_decision`：后端选择结果。
- `create_workspace(*, max_elems=None) -> UlyssesWorkspace`：分配可复用 NCCL send/receive staging。
  本操作 rank-local、非集体。
- `scatter_heads(x, *, out=None, workspace=None) -> torch.Tensor`：`[B, S_local, H, D] ->
  [B, S_global, H_local, D]`。`out` 为预分配 `[B, S_local * world_size, H // world_size, D]`；
  `workspace` 为可复用 NCCL pack/receive 存储（NVLink 后端校验但不用，因为其自有 IPC staging）。
- `scatter_qkv_head_chunk(query, key, value, *, head_offset, head_count, out=None, workspace=None)`：
  打包并用一次集合 scatter 一个 Q/K/V head band，输出融合 `[B, S_global, head_count, 3 * D]`。
- `gather_heads(...) -> torch.Tensor`、`gather_output_head_chunk(...)`：逆变换与紧凑输出 head band
  合并。
- `close()`：释放通信器。NVLink 后端被激活时为 collective，纯 NCCL 后端本地且幂等。
- 也实现上下文管理器协议（`__enter__`/`__exit__`）。

**注意事项**

- 构造函数始终是 collective；`close()` 仅在 NVLink 后端被激活时为 collective。NVLink 初始化
  或 collective `close` 内的 rank-local 失败会作为组结果交换，使所有 rank 共同清理并（或回退）
  抛出而非死锁；失败的 `close` 可重试。
- NCCL 后端在 `world_size > 1` 时要求 `group` 支持 CUDA all-to-all（NCCL process group），
  构造期检查。`world_size == 1` 为 passthrough（两个集合都原样返回输入，无拷贝）。
- 操作数须为构造 dtype 的连续 4-D CUDA 张量，每个维度为正，元素数不超过 `max_elems`
  （≤ 2^31 − 1）；`scatter_heads` 要求 `H % world_size == 0`，`gather_heads` 要求
  `S_global % world_size == 0`。
- 集合在当前 CUDA stream 上运行；每 rank 必须发出相同调用序列、形状一致；每个通信器同一时刻
  至多一个集合在飞。
- 所有 rank 必须请求相同 `backend`、并在 `max_elems` 与 `dtype` 上一致；每 rank 可绑定不同设备
  （`device` 接受 `torch.device`、`str` 或 `int` ordinal，如 `cuda:rank`）。

**示例**

```python
with UlyssesCommunicator(group, max_elems=B * S_local * H * D,
                         dtype=torch.bfloat16) as comm:
    q_ = comm.scatter_heads(q)   # [B,S_local,H,D] -> [B,S_global,H_local,D]
    k_ = comm.scatter_heads(k)
    v_ = comm.scatter_heads(v)
    o_ = attention(q_, k_, v_)
    o = comm.gather_heads(o_)    # [B,S_global,H_local,D] -> [B,S_local,H,D]
```

---

### `UlyssesWorkspace`

```python
class UlyssesWorkspace:
    def __init__(
        self,
        *,
        max_elems: int,
        dtype: torch.dtype,
        device: Optional[Union[torch.device, str, int]] = None,
    )
```

**功能**：NCCL Ulysses 集合的可复用 staging 存储。工作区拥有两块扁平 CUDA 缓冲（packed
all-to-all 输入、接收结果）。把它传给 `scatter_heads`/`gather_heads` 可去除每次调用的 NCCL
staging 分配。NVLink 后端已自有 IPC staging 存储，故它校验但不用该对象。

**参数**

- `max_elems`（`int`）：每 staging 缓冲的元素容量，须为正常数且不超过 int32 范围。
- `dtype`（`torch.dtype`）：元素 dtype，float16/bfloat16/float32。
- `device`（`torch.device`/`str`/`int`，可选）：CUDA 设备；`None` 用当前设备。

**属性**

- `send_buffer` / `recv_buffer`（`torch.Tensor`）：扁平 send/receive staging 缓冲（高级/调试用）。

**注意事项**

- 工作区是本地分配：不拥有 process group、不创建 stream。可被串行的 scatter/gather 调用共享，
  但不得被并发集合使用。

---

### `pack_ulysses_qkv_head_chunk`

```python
pack_ulysses_qkv_head_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    world_size: int,
    head_offset: int,
    head_count: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：打包一个 destination-local 的 Q/K/V head band。三个输入使用 `[B, S_local, H, D]`，
可为独立的正步长视图（包括融合投影产生的视图）；`H` 须能被 `world_size` 整除。选中的本地 band
应用于每个目的 rank 的 head slice 内部，结果为连续融合 payload
`[B, S_local, world_size * head_count, 3 * D]`。该 payload 可直接传给
`UlyssesCommunicator.scatter_heads`，其结果形状为 `[B, S_global, head_count, 3 * D]`；对末维切片
可得 Q/K/V 视图。

**参数**

- `query` / `key` / `value`（`torch.Tensor`）：形状 `[B, S_local, H, D]` 的正步长 CUDA 张量
  （同形状、dtype、设备）。
- `world_size`（`int`）：Ulysses rank 数；`H` 须能被其整除。
- `head_offset`（`int`）：选中 band 在每个目的 rank 的 `H // world_size` 本地 head 内的起点。
- `head_count`（`int`）：选中 band 中连续本地 head 数。
- `out`（`torch.Tensor`，可选）：预分配连续结果
  `[B, S_local, world_size * head_count, 3 * D]`，不得与 query/key/value 别名。

**返回**

- `torch.Tensor`：连续融合 Q/K/V payload `[B, S_local, world_size * head_count, 3 * D]`。

**注意事项**

- 在调用方当前 CUDA stream 上运行，仅在省略 `out` 时分配。

---

### `merge_ulysses_output_head_chunk`

```python
merge_ulysses_output_head_chunk(
    received: torch.Tensor,
    *,
    world_size: int,
    local_heads: int,
    head_offset: int,
    out: torch.Tensor,
) -> torch.Tensor
```

**功能**：把一个紧凑的 Ulysses 输出 head band 合并进完整输出。`received` 是 gather 分块注意力
输出后的结果，形状 `[B, S_local, world_size * head_count, D]`；`out` 是完整的
`[B, S_local, world_size * local_heads, D]` 目标。只写每个源 rank 的 band
`[rank * local_heads + head_offset, + head_count)`，其他输出元素保持不变。

**参数**

- `received`（`torch.Tensor`）：连续 gathered head band
  `[B, S_local, world_size * head_count, D]`。
- `world_size`（`int`）：`received` 与 `out` 中表示的 Ulysses rank 数。
- `local_heads`（`int`）：每个 rank 贡献给完整输出的注意力输出 head 数。
- `head_offset`（`int`）：每个 rank 的 `local_heads` 内选中 band 的起点。
- `out`（`torch.Tensor`）：预分配连续目标 `[B, S_local, world_size * local_heads, D]`，不得与
  `received` 别名；band 外的元素保持不变。

**返回**

- `torch.Tensor`：合并选中 head band 后的 `out`。

**注意事项**

- 对一个 head count 之和等于 `local_heads` 的调度中的每个不重叠 chunk 各调用一次，即可重建
  普通 whole-head Ulysses 结果。在调用方当前 CUDA stream 上运行，不分配。

---

### `QkvNormRopeExchangeOverlap`

```python
class QkvNormRopeExchangeOverlap:
    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        *,
        hidden: int,
        num_heads: int,
        head_dim: int = HEAD_DIM,
        dtype: torch.dtype = torch.bfloat16,
        layout: str = "bshd",
        max_tokens: int,
        num_slots: int = 2,
        backend: str = "auto",
        device: Optional[Union[torch.device, str, int]] = None,
    )
```

**功能**：把序列并行（Ulysses）层的注意力前端融合进一个 prepared session：BF16 QKV 投影、
Q 与 K 的 per-head RMSNorm（各一个 `[D]` scale）、来自 per-token `[rows, D]` cos/sin 表的
interleaved RoPE，以及 head exchange。Cake 后端（`backend="nvlink"`；SM90、SM100、SM103，
单节点 NVLink group 的 2/4/8 rank，`head_dim=128`，BF16）把每个目的地的 head band 从 GEMM
epilogue 直接写入该 rank 的对称接收 slot 并发布 per-chunk 就绪字，使 exchange 与投影重叠；
consumer stream 上一个单 CTA wait 内核获取这些字，调用方 stream 汇入它。输出是当前接收 slot
的视图（`layout="bshd"`：`(q, k, v)` 各 `[B, S_global, H_local, D]`；`layout="fused"`：一个
`[B, S_global, H_local, 3 * D]` payload），在 slot 于 `num_slots` 次调用后被复用前有效；需要
自有拷贝时传 `out=`。

**`__init__` 参数**

- `group`（可选）：process group；`None` 用 `dist.group.WORLD`。
- `hidden` / `num_heads`：hidden 维度 / head 数；须满足 `hidden == num_heads * head_dim`。
- `head_dim`（默认 128）：head 维度，须为 128。
- `dtype`（默认 `torch.bfloat16`）：只支持 bfloat16。
- `layout`（默认 `"bshd"`）：布局，见 `LAYOUTS`。
- `max_tokens`（`int`）：`max_tokens` 行的 tile plan 容量。
- `num_slots`（默认 2）：对称接收 slot 数，须至少为 2。
- `backend`（默认 `"auto"`）：`"auto"` 在 `resolve_ulysses_backend` 验证 NVLink mesh 且设备架构
  受支持时选 Cake 路径，否则选 NCCL 路径。
- `device`（可选）：本 rank 的 CUDA 设备。

**关键方法**

- `__call__(x, w_qkv, q_norm_weight, k_norm_weight, cos_sin, *, out=None)`：投影、归一化、旋转
  并交换一个激活 shard；返回接收视图（或 `out`）。`x` 为 `[B, S_local, hidden]` bf16；
  `w_qkv` 为 `[3 * hidden, hidden]` bf16（`torch.nn.Linear` 布局，行按 Q、K、V 排序）；
  norm 权重为 `[head_dim]` bf16；`cos_sin` 为 `[B * S_local, head_dim]` fp32（pairs 0..63 的 cos
  后接 interleaved RoPE 的 sin）。
- `last_outputs()`：最近一次调用的输出视图。
- `bound_programs(batch, s_local)`：Cake 后端——`(batch, s_local)` 每次调用各 stage 启动的
  程序名（eager 预置行）。
- `poison_receive_slots()`：用 NaN 模式填充本 rank 接收区（仅 eager），使下次调用必须重写它。
- `graph_capturable`（属性）：Cake 路径可捕获；NCCL 路径为 `False`。
- `describe()` / `close()`；也支持上下文管理器（`__enter__`/`__exit__`）。

**注意事项**

- 一个 session 由 `resolve_ulysses_backend` 选择后端；`backend="nvlink"` 在不可用时抛
  `UlyssesBackendError`。
- session 被 eager 准备（对称 slot、readiness pad、barrier flag、JIT 程序、`max_tokens` 行的
  tile plan），prepared 调用可被 CUDA Graph 捕获；行数超过 `max_tokens` 或首次在捕获内见到的
  行数会抛错而非分配。`backend="auto"` 回退的 NCCL 路径不可被 graph 捕获。
- 需已初始化的 NCCL process group。

**示例**

```python
from flashinfer.comm import QkvNormRopeExchangeOverlap

session = QkvNormRopeExchangeOverlap(
    group, hidden=8192, num_heads=64, head_dim=128, layout="bshd",
    max_tokens=65536, backend="auto",
)
q, k, v = session(x, w_qkv, q_norm_weight, k_norm_weight, cos_sin)
session.close()
```

---

### `UlyssesBackendDecision`

```python
@dataclass(frozen=True)
class UlyssesBackendDecision:
    backend: str
    reason: str
```

**功能**：后端选择的结果。`backend` 为实际选择的后端（`"nvlink"` 或 `"nccl"`）；`reason` 为
*选择*原因：对 NVLink 记录验证了什么，对 NCCL 仅在 `backend="auto"` 时为回退原因（显式
`backend="nccl"` 仅报告 "requested"）。

---

### `UlyssesRankTopology`

```python
@dataclass
class UlyssesRankTopology:
    rank: int
    hostname: str = ""
    device_index: int = -1
    device_uuid: str = ""
    pci_bus_id: str = ""
    peer_p2p: Dict[str, bool] = field(default_factory=dict)
    peer_nvlink: Dict[str, bool] = field(default_factory=dict)
    pair_errors: Dict[str, str] = field(default_factory=dict)
    probe_error: Optional[str] = None
```

**功能**：每 rank 拓扑探测结果，跨 group 交换。`peer_p2p` / `peer_nvlink` 以 *peer GPU 的 UUID*
为键，使决策层可跨 rank 拼接结果，而不受各进程 `CUDA_VISIBLE_DEVICES` 顺序影响；对本 rank 不可见
的 peer GPU 只是从字典中缺失（视为无 P2P）。`pair_errors` 区分「该 pair 探测失败」与「已验证该
pair 无 NVLink」。

---

### `UlyssesBackendError`

```python
class UlyssesBackendError(RuntimeError):
    ...
```

**功能**：当强制 `backend="nvlink"` 但拓扑不支持时抛出。

---

### `probe_ulysses_rank_topology`

```python
probe_ulysses_rank_topology(
    device: Optional[Union[torch.device, str, int]], rank: int
) -> UlyssesRankTopology
```

**功能**：探测本 rank 的 GPU 身份以及其对每个本进程可见的其他 CUDA 设备的 P2P/NVLink 可达性。
从不抛出：整个探测（含 hostname 与设备解析）在异常包络内运行，任何失败都会落入 `probe_error`，
（保守的）决策层随后回退 NCCL。

**参数**

- `device`（可选）：本 rank 的 CUDA 设备；`None` 用当前设备。
- `rank`（`int`）：要记录到返回拓扑对象中的 rank id。

**返回**

- `UlyssesRankTopology`：每 rank 拓扑信息，包含任何探测错误而非本地抛出。

---

### `decide_ulysses_backend`

```python
decide_ulysses_backend(
    requested: str,
    topologies: List[UlyssesRankTopology],
    supported_world_sizes: Sequence[int] = SUPPORTED_WORLD_SIZES,
) -> UlyssesBackendDecision
```

**功能**：纯决策函数：gathered 的每 rank 探测结果 → `(backend, reason)`。对其输入确定，因此
gather 到相同拓扑列表的每 rank 计算相同决策。保守：未知、不一致或无法验证的一律选 NCCL；仅在
`requested == "nvlink"` 且 NVLink 路径不可用时抛 `UlyssesBackendError`。

**参数**

- `requested`（`str`）：请求的后端策略：`"auto"`、`"nvlink"` 或 `"nccl"`。
- `topologies`（`list[UlyssesRankTopology]`）：从 process group gather 的每 rank 拓扑探测结果。
- `supported_world_sizes`（`Sequence[int]`，可选）：融合 NVLink 内核实例化的 world size。

**返回**

- `UlyssesBackendDecision`：后端选择与选择/回退原因。

---

### `resolve_ulysses_backend`

```python
resolve_ulysses_backend(
    backend: str = "auto",
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[Union[torch.device, str, int]] = None,
) -> UlyssesBackendDecision
```

**功能**：组一致的后端选择。必须在任何 IPC 分配或 JIT 编译*之前*运行。它不分配 IPC 工作区、
不编译任何东西；`all_gather_object` 元数据集合本身可能在 NCCL process group 上经 CUDA 缓冲
staging。序列：gather 每 rank 请求的后端并共同拒绝无效/不一致请求（至多三次
`all_gather_object` 的固定前缀）→ gather 每 rank 探测结果 → 每 rank 对同一 gathered 列表评估
同一纯决策并交叉校验（任何不一致都保守选 NCCL，或对 `backend="nvlink"` 抛错）。

**参数**

- `backend`（`str`，默认 `"auto"`）：请求的后端策略。
- `group`（可选）：必须做出一致后端决策的 process group；默认 `torch.distributed.group.WORLD`。
- `device`（可选）：用于探测与元数据集合 guard 的 CUDA 设备。

**返回**

- `UlyssesBackendDecision`：组一致的后端选择与选择/回退原因。

---

### `init_ulysses_a2a`

```python
init_ulysses_a2a(
    out_ipc_ptrs: List[int],
    signal_ipc_ptrs: List[int],
    rank: int,
    world_size: int,
    full_nvlink: bool,
) -> int
```

**功能**：初始化融合转置 Ulysses NVLink-P2P all-to-all 后端。

> **注意（高级/内部 API）**：优先使用 `UlyssesCommunicator`，它在任何 IPC 分配或 JIT 编译前
> 依据实际 GPU 拓扑选择后端、拥有 IPC 工作区生命周期并校验操作数。本原始入口假设调用方已
> 验证 all-pairs NVLink P2P。

内核为 *push* 模型：每 rank 把发往 peer 的 head/sequence block 直接经 NVLink 写入 peer 的
IPC-shared 输出 staging 缓冲，Ulysses 布局置换折入写地址。只有输出 staging 缓冲与信号缓冲需
IPC-shared（用 `flashinfer.comm.create_shared_buffer` 分配）；输入张量本地读取、无需注册。

**参数**

- `out_ipc_ptrs`（`list[int]`）：按 rank 排序的输出 staging 缓冲设备指针（经 CUDA IPC 打开），
  每个须至少能容纳该 group 的 all-to-all 输出。
- `signal_ipc_ptrs`（`list[int]`）：用于 GPU 间 barrier 的信号缓冲设备指针，每个须为
  `flashinfer.comm.vllm_meta_size` 字节。
- `rank`（`int`）：Ulysses group 内当前 rank。
- `world_size`（`int`）：Ulysses group 大小，须为 `(2, 4, 6, 8)` 之一。
- `full_nvlink`（`bool`）：每对 rank 是否经 NVLink 连接。push 内核要求 all-pairs P2P，调用方
  须据此 gate。

**返回**

- `int`：不透明句柄 `fa`，传给后续 `ulysses_a2a` 调用，用 `dispose_ulysses_a2a` 释放。

**注意事项**

- `init` 用 `cudaMemset` 清零本 rank 信号缓冲（对 host 异步），故本包装在返回前同步 *当前*
  CUDA 设备。调用方仍须在所有 rank 返回 init 后、首次 all-to-all 前发出 process-group barrier。
- `world_size` 不在支持集合、或 `full_nvlink=False` 时抛 `ValueError`。

---

### `dispose_ulysses_a2a`

```python
dispose_ulysses_a2a(fa: int) -> None
```

**功能**：释放 `init_ulysses_a2a` 返回的句柄。`fa` 是 C++ `UlyssesA2A*` 重新解释为整数
（`fptr_t`）的不透明句柄，非设备指针或 Python 对象，只对本模块有意义。调用后句柄悬空，不得再
传给 `ulysses_a2a`。

---

### `ulysses_a2a`

```python
ulysses_a2a(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    B: int,
    S_local: int,
    H: int,
    D: int,
    mode: int,
) -> None
```

**功能**：融合转置 Ulysses all-to-all。

> **注意（高级/内部 API）**：优先使用 `UlyssesCommunicator.scatter_heads`（`mode == 0`）与
> `UlyssesCommunicator.gather_heads`（`mode == 1`），它们从张量形状推导几何并校验操作数。

**参数**

- `fa`（`int`）：`init_ulysses_a2a` 返回的不透明后端句柄。
- `inp` / `out`（`torch.Tensor`）：连续 CUDA 张量，dtype 相同（float32/float16/bfloat16）。
- `B` / `S_local` / `H` / `D`：几何参数，`H` 为*全局* head 数，`H_local = H // world_size`，
  `S_global = S_local * world_size`。
- `mode`（`int`）：`0`（输入 a2a）`inp [B, S_local, H, D] -> out [B, S_global, H_local, D]`；
  `1`（输出 a2a）`inp [B, S_global, H_local, D] -> out [B, S_local, H, D]`。

**返回**

- `None`（结果写入 `out`，与等价 NCCL all-to-all 加布局置换按位一致）。

**注意事项**

- 所有 rank 须以一致几何、相同顺序调用；不一致是集合失败（hang 或损坏）。


---

### `MnnvlMemory`

```python
class MnnvlMemory:
    def __init__(self, mapping: Mapping, size: int)
```

**功能**：MNNVL 内存分配/映射。构造时通过 `open_mnnvl_memory` 获得 `self.ptr` 与
`self.rank_stride`。类属性包括 `initialized`、`current_mem_offset`、`current_rank_stride`、
`current_start_address`、`allocation_granularity`、`fabric_page_size`（512 MB）、`comm`、
`dev_id`、`allocated_map`、`address_refcnt`、`config`。

**参数**

- `mapping`（`Mapping`）：rank 信息。
- `size`（`int`）：段大小（每 rank 的 segment 大小）。

**关键成员**

- `mapped`（属性）：是否已映射。
- `as_torch_strided_tensor(dtype) -> torch.Tensor`：经 `pack_strided_memory` 返回查看该内存的
  跨步张量（每 rank 一个 segment）。
- `@staticmethod initialize()`：触发 CUDA context 与 NVML 初始化。
- `@staticmethod set_comm_from_config(mapping, config=None)`、`get_comm(mapping)`：设置/获取通信器。
- `@classmethod open_mnnvl_memory(mapping, size)` / `close_mnnvl_memory(ptr)`：打开/关闭。
- `__del__`：未显式关闭时尝试 `close_mnnvl_memory`。

---

### `McastGPUBuffer`

```python
class McastGPUBuffer:
    def __init__(
        self,
        buf_size: int,
        group_size: int,
        group_rank: int,
        device: torch.device,
        comm_backend_for_handle_transfer: Optional[CommBackend] = None,
    )
```

**功能**：`SymmDeviceMemory` 的封装，便于 PyTorch 张量创建；管理可经 unicast 或 multicast
访问的缓冲，用于多节点通信（TensorRT-LLM `McastGPUBuffer` 的 Python 移植）。

**参数**

- `buf_size`（`int`）：请求的缓冲字节数（实际可用大小可能因对齐而不同）。
- `group_size` / `group_rank`：通信 group 的 rank 数 / 本地 process 在 group 中的 rank。
- `device`（`torch.device`）：分配缓冲的 CUDA 设备。
- `comm_backend_for_handle_transfer`（可选）：handle 传输的通信后端。

**关键方法**

- `lamport_initialize(rank, dtype)`：初始化 Lamport 缓冲。
- `get_multicast_buffer(sizes, dtype, storage_offset=0) -> torch.Tensor`：**未实现**（抛
  `NotImplementedError`）。
- `get_unicast_buffer(sizes, dtype, storage_offset=0) -> torch.Tensor`：**未实现**（抛
  `NotImplementedError`）。
- `get_multicast_ptr() -> int`：原始 multicast 指针。
- `get_unicast_ptr(rank) -> int`：指向给定 rank 的原始 unicast 指针。
- `get_buffer_ptrs_dev() -> int`：缓冲指针设备数组。

---

### `mpi_barrier`

```python
mpi_barrier()
```

**功能**：MPI barrier（`MPI.COMM_WORLD.Barrier()`），可能被 `dist.barrier()` 取代。

---

### `trtllm_mnnvl_allreduce`

```python
trtllm_mnnvl_allreduce(
    input: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    launch_with_pdl: bool,
    output: Optional[torch.Tensor] = None,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
) -> torch.Tensor
```

**功能**：跨 tensor-parallel rank 执行 MNNVL all-reduce sum。`input` 须为 2-D 本地 shard
`[num_tokens, hidden_dim]`；结果形状相同，写入 `output`（提供时）或新申请张量。`workspace` 须为
同一 tensor-parallel group 创建、且缓冲容量足够所选策略的 `MNNVLAllReduceFusionWorkspace`。

支持的策略：`ONESHOT`——每 rank 把本地 shard 存入 peer 可见工作区、各 rank 本地归约（小 payload
的低延迟路径）；`TWOSHOT`——scatter token slice、在目的 rank 归约每个 slice、再广播结果（大
payload 的吞吐路径）。`AUTO` 时 FlashInfer 按 payload 大小选择。两种策略跨 rank 完全确定。

**参数**

- `input`：本地输入 shard `[num_tokens, hidden_dim]`。
- `workspace`：`MNNVLAllReduceFusionWorkspace`。
- `launch_with_pdl`：是否以 PDL 启动。
- `output`：存放结果的输出张量；未提供时创建空张量。
- `strategy`：`MNNVLAllreduceFusionStrategy`；未提供时用内部启发式。

**返回**

- `output`：归约后的张量 `[num_tokens, hidden_dim]`。

**注意事项**

- 纯 all-reduce helper，不做量化；需要 RMSNorm 后量化输出请用
  `trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant` 或统一 `allreduce_fusion` API。
- `input` 非 2-D、`output` 非 2-D、或缓冲不足时抛 `ValueError`。

---

### `trtllm_mnnvl_fused_allreduce_add_rmsnorm`

```python
trtllm_mnnvl_fused_allreduce_add_rmsnorm(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    epsilon: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
    weight_bias: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：执行 MNNVL AllReduce + Residual + RMSNorm。先对 shard 输入做多节点 all-reduce（sum），
再从 multicast 缓冲直接读取结果做 residual 加法与 RMSNorm（当前 rank 的 multicast 缓冲与 unicast
缓冲相同）。

**参数**

- `input`：`[num_tokens, hidden_dim]`。
- `residual_in`：`[num_tokens, hidden_dim]`。
- `gamma`：`[hidden_dim]`。
- `workspace`：`MNNVLAllReduceFusionWorkspace`。
- `epsilon`（可选）：RMSNorm epsilon；未提供时用 `torch.finfo.eps`。
- `output`：归一化结果 `[num_tokens, hidden_dim]`；未提供时创建空张量。
- `residual_out`：residual 输出 `[num_tokens, hidden_dim]`；未提供时创建空张量。
- `launch_with_pdl`、`strategy`、`weight_bias`：同上。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(output, residual_out)`。

---

### `trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant`

```python
trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    epsilon: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    output_scale: Union[torch.Tensor, float, None] = None,
    layout_code: int = QuantizationSFLayout.SWIZZLED_128x4,
    quant_type: int = MNNVLQuantType.NONE,
    launch_with_pdl: bool = False,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
    weight_bias: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]
```

**功能**：执行 MNNVL AllReduce + Residual + RMSNorm + FP8/NVFP4 量化。量化在 RMSNorm 之后应用；
`output` 可选，仅在同时需要归一化但未量化的张量时传入；量化结果总以 `quant_out` 返回。

**参数**

- `input` / `residual_in`：`[num_tokens, hidden_dim]`。
- `gamma`：`[hidden_dim]`。
- `workspace`：`MNNVLAllReduceFusionWorkspace`。
- `epsilon`：RMSNorm epsilon；默认 `torch.finfo(input.dtype).eps`。
- `output`（可选）：归一化输出 `[num_tokens, hidden_dim]`。
- `residual_out`（可选）：residual 输出 `[num_tokens, hidden_dim]`。
- `quant_out`（可选）：量化输出。FP8 形状 `[num_tokens, hidden_dim]`、dtype
  `torch.float8_e4m3fn`；NVFP4 形状 `[num_tokens, hidden_dim // 2]`、dtype `torch.uint8` 或
  `torch.float4_e2m1fn_x2`。
- `scale_out`（可选）：scale 输出。Dynamic FP8 为 `[num_tokens, 1]` float32；NVFP4 用 `LINEAR`
  布局时 `[num_tokens, hidden_dim // 16]`，或对填充后的 `SWIZZLED_128x4` 布局足够大的一维张量；
  Static FP8 忽略该参数。
- `output_scale`：量化输出 scale，标量 float 或 float32 张量；默认 `1.0`。
- `layout_code`（默认 `QuantizationSFLayout.SWIZZLED_128x4`）：NVFP4 scale 布局；MNNVL 支持
  `SWIZZLED_128x4` 与 `LINEAR`，不支持 `SWIZZLED_8x4`。
- `quant_type`（默认 `MNNVLQuantType.NONE`）：`MNNVLQuantType.FP8`、`NVFP4` 或 `DYNAMIC_FP8`。
- `launch_with_pdl`、`strategy`、`weight_bias`：同上。

**返回**

- `Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]`：
  `(output, residual_out, quant_out, scale_out)`。

---

### `trtllm_mnnvl_all_reduce`

```python
trtllm_mnnvl_all_reduce(
    inp: torch.Tensor,
    multicast_buffer_ptr: int,
    buffer_ptrs_dev: int,
    buffer_M: int,
    buffer_flags_mnnvl: torch.Tensor,
    nranks: int,
    rank: int,
    wait_for_results: bool,
    launch_with_pdl: bool,
    out: Optional[torch.Tensor] = None,
) -> None
```

**功能**：**已废弃**的指针式 MNNVL all-reduce API。请改用 `trtllm_mnnvl_allreduce` 配
`MNNVLAllReduceFusionWorkspace`。执行跨 GPU 的多节点 NVLink all-reduce：把每个 GPU 的输入
shard scatter 到正确的 unicast 缓冲、在每个 GPU 上做 all-reduce、再把结果广播到所有 GPU。

**参数**

- `inp`：本地输入 shard。
- `multicast_buffer_ptr`（`int`）：multicast 缓冲指针（整数）。
- `buffer_ptrs_dev`（`int`）：设备缓冲指针（整数）。
- `buffer_M`（`int`）：最大元素数 // hidden_dim。
- `buffer_flags_mnnvl`（`torch.Tensor`）：缓冲状态标志。
- `nranks` / `rank`：参与 rank 总数 / 当前进程 rank。
- `wait_for_results`（`bool`）：为 `True` 时把结果存入 `out`。
- `launch_with_pdl`（`bool`）：是否以 PDL 启动。
- `out`（可选）：存放结果的输出张量（`wait_for_results=True` 时必需）。

**返回**

- `None`。

---

### `trtllm_mnnvl_fused_allreduce_rmsnorm`

```python
trtllm_mnnvl_fused_allreduce_rmsnorm(
    prenorm_output: torch.Tensor,
    normed_output: torch.Tensor,
    shard_input: torch.Tensor,
    multicast_buffer_ptr: int,
    buffer_ptrs_dev: int,
    unicast_ptr: int,
    buffer_M: int,
    buffer_flags_mnnvl: torch.Tensor,
    nranks: int,
    rank: int,
    gamma: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
    launch_with_pdl: bool,
) -> None
```

**功能**：**已废弃**——执行 MNNVL TwoShot AllReduce + RMSNorm；请改用
`trtllm_mnnvl_fused_allreduce_add_rmsnorm`。先对 `shard_input` 做多节点 all-reduce（sum），
再从 multicast 缓冲直接读取结果做 RMSNorm。

**参数**

- `prenorm_output`：prenorm 结果输出张量。
- `normed_output`：归一化结果输出张量。
- `shard_input`：输入张量 shard。
- `multicast_buffer_ptr` / `buffer_ptrs_dev` / `unicast_ptr`（`int`）：multicast 缓冲、设备缓冲
  指针数组、本地 unicast 缓冲的指针（整数）。
- `buffer_M`：最大元素数 // hidden_dim。
- `buffer_flags_mnnvl`：同步用 buffer flag。
- `nranks` / `rank`：TP group rank 数 / 当前 rank。
- `gamma`：RMSNorm 权重。`epsilon`：RMSNorm epsilon。`residual`：要相加的 residual。
- `launch_with_pdl`：是否以 PDL 启动。

**返回**

- `None`。

---

### `moe_a2a_initialize`

```python
moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
    eplb_stats_num_experts: int = 0,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
)
```

**功能**：初始化 MoE all-to-all 工作区并返回 metainfo 张量。metainfo 编码每 rank 偏移与
`moe_a2a_dispatch`/`moe_a2a_combine` 所需记录，须回传给那些例程。`moe_a2a_initialize` 幂等，
必须为每个工作区分配在任何 dispatch/combine 前调用一次。

**参数**

- `workspace`（`torch.Tensor`）：`[ep_size, size_per_rank]` 共享工作区张量。
- `ep_rank`（`int`）：当前 expert-parallel rank。
- `ep_size`（`int`）：expert-parallel world size。
- `max_num_tokens`（`int`）：任 rank 单次调用可 dispatch 的最大 token 数，用于确定 metainfo 大小。
- `eplb_stats_num_experts`（默认 `0`）：为 EPLB gathered-stats 区预留的专家数；`0` 禁用，
  非零时须与传给 `moe_a2a_dispatch` 的 `eplb_local_stats` 长度一致。
- `backend`（关键字限定，默认 `"trtllm"`）：`"trtllm"` 或 `"cake"`（CC 10.0/10.3）。对同一
  工作区及所有 rank 必须用同一后端。

**返回**

- `torch.Tensor`：对调用方不透明的 metainfo 张量，传给后续 `moe_a2a_*` 调用。

---

### `moe_a2a_wrap_payload_tensor_in_workspace`

```python
moe_a2a_wrap_payload_tensor_in_workspace(
    workspace: torch.Tensor,
    leading_shape: list[int],
    slice_start: int,
    slice_end: int,
    dtype: torch.dtype,
) -> torch.Tensor
```

**功能**：把共享工作区的一段切片包装为带类型的张量视图。

**参数**

- `workspace`（`torch.Tensor`）：`[ep_size, size_per_rank]`（或 `[size_per_rank]`）工作区张量。
- `leading_shape`（`list[int]`）：结果视图的 leading shape；末维由 `slice_end - slice_start`
  与 `dtype` 推断。
- `slice_start` / `slice_end`（`int`）：切片起止偏移（字节），须落在单个 rank 内。
- `dtype`（`torch.dtype`）：结果视图的元素 dtype。

**返回**

- `torch.Tensor`：形状 `leading_shape + [-1]`、由工作区支撑的张量。

---

### `moe_a2a_dispatch`

```python
moe_a2a_dispatch(
    token_selected_experts: torch.Tensor,
    input_payloads: list[torch.Tensor],
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    num_experts: int,
    enable_pdl: Optional[bool] = None,
    eplb_local_stats: Optional[torch.Tensor] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
    recv_view_cache: Optional[dict] = None,
)
```

**功能**：运行 MoE all-to-all 的 dispatch 阶段，把每个 token 的 payload 送到其目标 expert 所在
rank。`backend="trtllm"` 为默认（含 Blackwell）；仅在与此后端初始化/派发的工作区配合时才用
`backend="cake"`，且所有 peer rank 必须相同选择。

**参数**

- `token_selected_experts`（`torch.Tensor`）：`[local_num_tokens, top_k]` int32。
- `input_payloads`（`list[torch.Tensor]`）：每个 token 的 payload 张量，各形如
  `[local_num_tokens, *]`。
- `workspace` / `metainfo`：`moe_a2a_initialize` 返回的对象。
- `runtime_max_tokens_per_rank`（`int`）：本批次每 rank 最大 token 数，须 `<=` 初始化时的
  `max_num_tokens`。
- `ep_rank` / `ep_size` / `top_k` / `num_experts`：EP 配置。
- `enable_pdl`（可选）：是否使用 PDL；`None` 自动探测。
- `eplb_local_stats`（可选）：EPLB 本地统计张量。
- `enable_rank_mask`（默认 `False`）：是否使用 `active_rank_mask`。
- `active_rank_mask`（可选）：活跃 rank mask 张量。
- `backend`（关键字限定，默认 `"trtllm"`）。
- `recv_view_cache`（可选，关键字限定）：接收视图缓存字典。

**返回**

- `list[torch.Tensor]`：接收到的 payload 张量列表。

---

### `moe_a2a_combine`

```python
moe_a2a_combine(
    payload: torch.Tensor,
    local_num_tokens: int,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    combine_payload_offset: int,
    payload_in_workspace: bool = False,
    output_dtype: Optional[torch.dtype] = None,
    output_scales: Optional[torch.Tensor] = None,
    output_scalar_scale: float = 1.0,
    sf_layout: SfLayout = SfLayout.layout_linear,
    output: Optional[torch.Tensor] = None,
    *,
    use_low_precision: bool = False,
    enable_pdl: Optional[bool] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    backend: MoeAlltoAllBackend = "trtllm",
) -> torch.Tensor
```

**功能**：把 per-expert 输出合并回源 rank。是 `moe_a2a_dispatch` 的逆操作：把 rank-local expert
输出行散回提供原始 token 的 rank。

**参数**

- `payload` / `local_num_tokens` / `workspace` / `metainfo` / `runtime_max_tokens_per_rank` /
  `ep_rank` / `ep_size` / `top_k`：见 dispatch 与 `MoeAlltoAll.dispatch`。
- `combine_payload_offset`（`int`）：combine payload 在工作区中的偏移。
- `payload_in_workspace`（默认 `False`）：payload 是否已位于工作区中。
- `output_dtype`（可选）：输出 dtype。
- `output_scales`（可选）：输出 scale。
- `output_scalar_scale`（默认 `1.0`）：标量 scale。
- `sf_layout`（默认 `SfLayout.layout_linear`）：scale factor 布局。
- `output`（可选）：预分配输出。
- `use_low_precision`（关键字限定，默认 `False`）：是否使用低精度。
- `enable_pdl` / `enable_rank_mask` / `active_rank_mask` / `backend`：见 dispatch。

**返回**

- `torch.Tensor`：合并后的输出。

---

### `moe_a2a_sanitize_expert_ids`

```python
moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
    enable_pdl: Optional[bool] = None,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
)
```

**功能**：把没有 token 路由到本 rank 的无效 slot 的 expert id 设为 `invalid_expert_id`。

**参数**

- `expert_ids`（`torch.Tensor`）：`[local_num_tokens, top_k]` int32 expert 分配（原地修改）。
- `workspace` / `metainfo`：`moe_a2a_initialize` 返回的对象。
- `ep_rank`（`int`）：当前 EP rank。
- `invalid_expert_id`（`int`）：写入未收到 token 的 slot 的值（依 `recv_counters`，例如超过某源
  rank 有效计数的 padding）。
- `enable_pdl`（可选）：是否使用 PDL；`None` 从设备自动探测。
- `backend`（关键字限定，默认 `"trtllm"`）：通信后端；仅对 Cake 后端初始化的 workspace 才用
  `"cake"`。

---

### `moe_a2a_get_workspace_size_per_rank`

```python
moe_a2a_get_workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    total_dispatch_payload_size_per_token: int,
    combine_payload_size_per_token: int,
    eplb_stats_num_experts: int = 0,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
)
```

**功能**：计算 MoE all-to-all 原语的每 rank 工作区大小。

**参数**

- `ep_size`（`int`）：EP world size。
- `max_num_tokens`（`int`）：所有 rank 的最大 token 数。
- `total_dispatch_payload_size_per_token`（`int`）：dispatch 阶段每 token 发送的所有 payload 之和（字节）。
- `combine_payload_size_per_token`（`int`）：combine 阶段每 token 回送的 payload 大小（字节）。
- `eplb_stats_num_experts`（默认 `0`）：EPLB gathered-stats 区预留专家数。
- `backend`（关键字限定，默认 `"trtllm"`）。

**返回**

- `int`：每 rank 所需工作区字节数（各段向上取整到 128 字节对齐）。

---

### `MoeAlltoAll`

```python
class MoeAlltoAll:
    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        workspace_size_per_rank: int = None,
        hidden_size: int = None,
        mnnvl_config: Optional[MnnvlConfig] = None,
        eplb_stats_num_experts: int = 0,
        enable_rank_mask: bool = False,
        *,
        backend: MoeAlltoAllBackend = "trtllm",
    )
```

**功能**：管理 MoE All-to-All 操作（含工作区分配与同步）。提供吞吐优化的后端，支持每次集合
多个 payload、显式 dispatch/combine 阶段与工作区支撑的张量。

> **已废弃（0.7.1）**：请改用 `flashinfer.moe_ep.NVLinkOneSidedAlltoAll`（`backend="cake"`
> 时为 `flashinfer.moe_ep.CakeAlltoAll`）；本类的实现将迁入它们。

`backend="trtllm"` 为默认（含 Blackwell）。要在每个 rank 构造时显式启用生成的 Blackwell 内核，
传 `backend="cake"`；不支持的架构会被拒绝，而非静默路由到别的后端。

**`__init__` 参数**

- `mapping`（`Mapping`）：描述并行布局（须暴露 `moe_ep_rank` 与 `moe_ep_size`）。
- `max_num_tokens`（`int`）：本 rank 单次调用将 dispatch 的最大 token 数。
- `top_k`（`int`）：每 token 分配的专家数。
- `num_experts`（`int`）：总专家数（跨所有 rank）。
- `workspace_size_per_rank`（`int`，可选）：预计算的每 rank 工作区字节数；为 `None` 时须提供
  `hidden_size`，并经 `get_moe_workspace_size_per_rank` 确定大小。
- `hidden_size`（`int`，可选）：hidden 维度。
- `mnnvl_config`（`MnnvlConfig`，可选）：底层 MNNVL 通信后端配置。
- `eplb_stats_num_experts`（默认 `0`）：EPLB gathered-stats 区预留专家数；非零时向 `dispatch`
  传该长度的 `eplb_local_stats`。
- `enable_rank_mask`（默认 `False`）：`dispatch`/`combine` 是否可带 `active_rank_mask`；实例生命
  周期内固定（镜像底层内核的编译期特化）。
- `backend`（关键字限定，默认 `"trtllm"`）：`"trtllm"` 或 `"cake"`（需 CC 10.0/10.3），所有
  参与 rank 须相同。

**关键成员**

- `@classmethod get_workspace(workspace_size_per_rank, ep_rank, ep_size, max_num_tokens, mapping,
  eplb_stats_num_experts=0, *, backend="trtllm") -> dict`：分配/缓存工作区。所有 rank 须选相同
  后端；不同后端的工作区从不共享。
- `get_moe_workspace_size_per_rank(...)`：按分布尺寸计算每 rank 工作区大小。
- `eplb_gathered_stats`（属性）：最近一次 `dispatch` 的 EPLB 统计；工作区支撑的
  `[ep_size, eplb_stats_num_experts]` int32 视图，或禁用时为 `None`。
- `dispatch(token_selected_experts, input_payloads, runtime_max_tokens_per_rank,
  invalid_token_expert_id=None, expert_id_payload_index=None, eplb_local_stats=None,
  active_rank_mask=None) -> list[torch.Tensor]`：运行 dispatch 阶段。
- `combine(...)`、`get_combine_payload_tensor_in_workspace(...)`：combine 阶段与工作区中的
  combine payload 张量获取。
- `checkpoint_prepare()`：为 checkpoint 解除映射 MNNVL 句柄；重复调用为 no-op。活动 all-to-all
  阶段期间调用抛 `RuntimeError`。
- `checkpoint_restore(comm_backend)`：恢复后重映射 MNNVL 句柄；重复调用为 no-op。`comm_backend`
  必须复现原 rank 与 world size，否则抛 `RuntimeError`（含 metainfo 变化检测）。

**注意事项**

- 两个 checkpoint 方法都是 **collective**，所有 rank 必须按相同顺序调用，且每 rank 在释放
  backing 前须停止使用工作区（内部 `torch.cuda.synchronize()` + barrier）。若在物理句柄
  unmap/remap 开始后发生异常，不要重试或复用，重启受影响 rank。
- 类代码保留一个跨进程的共享 `_WORKSPACE_CACHE`。

**示例**

```python
moe_a2a = MoeAlltoAll(mapping, max_num_tokens=2048, top_k=2, num_experts=8)
recv = moe_a2a.dispatch(experts, [hidden, ids, scales], batch_size)
output = moe_a2a.combine(processed, batch_size)
```

---

### `decode_cp_a2a_workspace_size`

```python
decode_cp_a2a_workspace_size(cp_size: int) -> int
```

**功能**：返回给定 CP group 大小下每 rank 的工作区字节数。

**参数**

- `cp_size`（`int`）：上下文并行 group 大小（rank 数）。

**返回**

- `int`：每 rank 工作区字节数。

**示例**

```python
>>> decode_cp_a2a_workspace_size(4)
16778240
```

---

### `decode_cp_a2a_allocate_mnnvl_workspace`

```python
decode_cp_a2a_allocate_mnnvl_workspace(
    mapping: Mapping,
    *,
    mnnvl_config: Optional[MnnvlConfig] = None,
) -> torch.Tensor
```

**功能**：分配形状 `[cp_size, ws_elems_per_rank]` 的 MNNVL 支撑工作区。DCP A2A 内核要求一个
横跨所有 CP rank 的统一 VA，因此工作区分配必须经 MNNVL fabric 内存；本函数是唯一受支持的分配器。

**参数**

- `mapping`（`Mapping`）：MNNVL 分配用 Mapping，携带 `cp_size` 与 `cp_rank`（通信器用
  `mapping.pp_rank`、`mapping.cp_rank`、`mapping.tp_rank` 做 split）。
- `mnnvl_config`（`MnnvlConfig`，可选）：MNNVL 通信后端配置；用 `torch.distributed` 时必需
  （传 `MnnvlConfig(comm_backend=TorchDistBackend(group))`）。

**返回**

- `torch.Tensor`：形状 `[cp_size, ws_elems_per_rank]` 的 `torch.int64` 张量。

**注意事项**

- 分配后须调用 `decode_cp_a2a_init_workspace`，并在首次 `decode_cp_a2a_alltoall` 前做一个跨 rank
  barrier。

---

### `decode_cp_a2a_init_workspace`

```python
decode_cp_a2a_init_workspace(
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> None
```

**功能**：初始化工作区 FIFO 缓冲（首次 alltoall 前调用一次）。重置**本地**工作区行
（`workspace[cp_rank]`）中的 FIFO 缓冲。本函数**同步**：返回时 GPU memset 保证已完成。

**参数**

- `workspace`（`torch.Tensor`）：来自 `decode_cp_a2a_allocate_mnnvl_workspace` 的
  `[cp_size, ws_elems_per_rank]` int64 张量。
- `cp_rank`（`int`）：本 rank 在 CP group 中的位置。
- `cp_size`（`int`）：上下文并行 group 大小。

**注意事项**

- 使用 MNNVL 工作区时，**所有 rank** 必须完成 `decode_cp_a2a_init_workspace` 并执行跨 rank
  barrier（如 `dist.barrier(group)`）后，**任何** rank 才能调用 `decode_cp_a2a_alltoall`；否则
  某 rank 可能在 peer 尚未初始化完时就写入其 FIFO，导致死锁。

---

### `decode_cp_a2a_alltoall`

```python
decode_cp_a2a_alltoall(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    enable_pdl: Optional[bool] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

**功能**：执行 DCP all-to-all 交换。每个 rank 把 `partial_o[..., peer, :]` 切片发给对应 peer，
并把所有 peer 的贡献接收到输出张量。

**参数**

- `partial_o`（`torch.Tensor`）：`[..., cp_size, D]` 的 `half` 或 `bfloat16` 张量；
  `D * element_size` 须 16 字节对齐。
- `softmax_stats`（`torch.Tensor`）：`[..., cp_size, S]` float32，`S >= 2` 且为偶数；batch 维度
  须与 `partial_o` 匹配。
- `workspace`（`torch.Tensor`）：已初始化的 `[cp_size, ws_elems_per_rank]` int64 张量。
- `cp_rank` / `cp_size`：CP group 配置。
- `enable_pdl`（可选）：启用 PDL（SM90+）；SM90+ 默认为 `True`，否则 `False`。
- `out`（可选）：预分配 `(partial_o_out, softmax_stats_out)`，形状/dtype/设备同输入。提供时写入
  并返回这些张量，使 CUDA Graph 捕获与稳态 decode 循环不每次分配。

**返回**

- `tuple[torch.Tensor, torch.Tensor]`：交换后的 `(partial_o_out, softmax_stats_out)`。

---

### `decode_cp_a2a_lse_reduce_workspace_size`

```python
decode_cp_a2a_lse_reduce_workspace_size(
    max_tokens: int,
    local_heads: int,
    cp_size: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int
```

**功能**：返回融合 op 所需的 NCCL 对称工作区字节数。

**参数**

- `max_tokens`（`int`）：后续调用 token/batch 维度上界。
- `local_heads`（`int`）：reduce-scatter 后本 rank 保留的 head 数。
- `cp_size`（`int`）：上下文并行 group 大小。
- `head_dim`（`int`）：每 head 元素数。
- `dtype`（`torch.dtype`）：`partial_o`/`output` 的存储类型（fp16 或 bf16）。

**返回**

- `int`：所需字节数；返回值为 `payload_offset + 2 * cp_size * max_tokens * local_heads *
  (head_dim * itemsize + 4)`。

**注意事项**

- 任一几何参数 `<= 0`、`cp_size > 64`、dtype 非 fp16/bf16、或 `head_dim * itemsize` 非 16 的倍数
  时抛 `ValueError`/`TypeError`。

---

### `decode_cp_a2a_lse_reduce_create_workspace`

```python
decode_cp_a2a_lse_reduce_create_workspace(
    max_tokens: int,
    local_heads: int,
    cp_size: int,
    head_dim: int,
    dtype: torch.dtype,
    group: Any,
) -> torch.Tensor
```

**功能**：创建并 rendezvous 融合 op 的 NCCL 对称工作区。`group` 中所有 rank 必须集体调用。
group 须位于一个 NCCL LSA（load/store-accessible）NVLink 域内；跨 LSA 域的多节点 group 不支持。
每个 group 分配一个工作区并对每次调用与 CUDA Graph 重放复用；一个工作区只能从一条有序 CUDA
stream 使用，并发 stream 需各分配一个。

**参数**

- `max_tokens` / `local_heads` / `cp_size` / `head_dim` / `dtype`：见工作区大小函数。
- `group`（`torch.distributed.ProcessGroup` 或 `str`）：CP team 的 process group（或 group 名）。

**返回**

- `torch.Tensor`：已 rendezvous 的 NCCL 对称内存 `uint8` 张量；在 CUDA Graph 捕获前分配一次并
  复用。

**注意事项**

- PyTorch 的 NCCL 对称内存通信器注册表按具体 CUDA 设备为键，内部使用带索引的
  `cuda:<local_rank>` 设备。

---

### `decode_cp_a2a_lse_reduce`

```python
decode_cp_a2a_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    lse_mode: Literal["base2", "basee"] = "base2",
) -> torch.Tensor
```

**功能**：把一个 NCCL LSA DCP A2A 交换与 LSE 加权 reduce 融合。发送、接收同步与归约在一个
协作 CUDA 内核中执行。

**参数**

- `partial_o`（`torch.Tensor`）：`[batch, heads, cp_size, head_dim]`（fp16 或 bf16），或更一般地
  `[..., cp_size, head_dim]`。`partial_o[..., peer, :]` 是发往该 CP rank 的切片。
- `partial_lse`（`torch.Tensor`）：`[batch, heads, cp_size]` float32，或更一般地 `[..., cp_size]`；
  leading 维度须与 `partial_o` 匹配。`heads` 只是输入中的 head 数，可本地或全局，因本操作不分片
  head 轴。
- `workspace`（`torch.Tensor`）：来自 `decode_cp_a2a_lse_reduce_create_workspace` 的已 rendezvous
  NCCL 对称内存张量；仅从一条有序 CUDA stream 复用。
- `cp_rank`（`int`）：本 rank 在 CP group 中的索引。
- `cp_size`（`int`）：上下文并行 group 大小。
- `lse_mode`（`Literal["base2", "basee"]`，默认 `"base2"`）：`partial_lse` 的对数底。`"base2"`
  是 FlashInfer MLA 产生的默认值；`"basee"` 选择自然对数 LSE。

**返回**

- `torch.Tensor`：`[batch, heads, head_dim]`，或更一般地 `[..., head_dim]`，dtype 与
  `partial_o` 相同。

**注意事项**

- 这是 collective：每 rank 必须以相同顺序、相同次数调用，包括 CUDA Graph 重放。所有 rank 须
  属于一个 NCCL LSA/NVLink 域。
- `lse_mode` 非 `"base2"`/`"basee"` 时抛 `ValueError`。

---

### `MixedCommOp`

```python
class MixedCommOp(enum.IntEnum):
    ALLREDUCE = 0
    ALLGATHER = enum.auto()
    REDUCESCATTER = enum.auto()
    ALLREDUCE_ALLGATHER = enum.auto()
    REDUCESCATTER_ALLREDUCE = enum.auto()
```

**功能**：混合通信操作类型枚举。值须与 `mixed_comm.cuh` 中的 `MixedCommOp` 对齐。

---

### `MixedCommMode`

```python
class MixedCommMode(enum.IntEnum):
    FUSED_OPT_WAITS_MC = 0
    FUSED_OPT_WAITS_UC = enum.auto()
    FUSED_OPT_BYTES1_MC = enum.auto()
    FUSED_OPT_BYTES1_UC = enum.auto()
    FUSED_OPT_BYTES2_MC = enum.auto()
    FUSED_OPT_BYTES2_UC = enum.auto()
    NCCL_ONE = enum.auto()
    NCCL_TP_DP = enum.auto()
    AUTOTUNE = enum.auto()
```

**功能**：混合通信执行模式枚举。融合模式（`FUSED_*`）用虚拟内存（节点内）与 nvshmem（节点间）
运行单个融合内核；NCCL 模式委派给一个或多个 NCCL 集合调用；`AUTOTUNE` 依据 profiling 选择最佳
模式。融合值须与 `mixed_comm.cuh` 中的 `MixedCommMode` 对齐。

---

### `MixedCommHandler`

```python
class MixedCommHandler:
    def __init__(
        self,
        world_rank: int,
        world_size: int,
        local_rank: int,
        local_size: int,
        inter_rank: int,
        inter_size: int,
        local_tp_size: int | None,
        local_dp_size: int | None,
        inter_tp_size: int | None,
        inter_dp_size: int | None,
        dtype: torch.dtype,
        device: torch.device,
        grid_size: int | None = None,
        max_block_size: int | None = None,
        min_block_size: int = 256,
        min_num_steps: int = 4,
        ib_enable_ibgda: bool = True,
        should_init_nvshmem: bool = True,
        use_autotune: bool = True,
    )
```

**功能**：混合通信处理器。初始化时建立虚拟内存、nvshmem 并运行 autotune。

**`__init__` 参数**

- `world_rank` / `world_size` / `local_rank` / `local_size` / `inter_rank` / `inter_size`：全局/本地/
  节点间 rank 拓扑。
- `local_tp_size` / `local_dp_size` / `inter_tp_size` / `inter_dp_size`（`int | None`）：各维度
  并行大小；为 `None` 时按 `local_size`/`inter_size` 推导。
- `dtype`（`torch.dtype`）：须为 `torch.float16` 或 `torch.bfloat16`。
- `device`（`torch.device`）：CUDA 设备。
- `grid_size`（可选）：网格大小，默认设备 SM 数。
- `max_block_size` / `min_block_size`（默认 `256`）/ `min_num_steps`（默认 `4`）：block 配置。
- `ib_enable_ibgda`（默认 `True`）：设置 `NVSHMEM_IB_ENABLE_IBGDA`。
- `should_init_nvshmem`（默认 `True`）：是否初始化 nvshmem。
- `use_autotune`（默认 `True`）：是否运行 autotune。

**关键方法 / 属性**

- `para_info`：`ParallelInfo`（并行拓扑，含各 rank/size 属性与创建 process group 的辅助方法）。
- `get_valid_op_list()`：当前并行拓扑下有效通信操作列表。混合拓扑返回
  `[ALLREDUCE_ALLGATHER, REDUCESCATTER_ALLREDUCE]`；纯 TP 返回 `[ALLREDUCE]`；纯 DP 返回
  `[ALLGATHER, REDUCESCATTER]`。
- `get_valid_mode_list()`：当前拓扑下有效执行模式列表。
- `select_autotune_mode(op, x_in)`：自动调优选择模式。
- `init_virtual_memory()` / `init_nvshmem()` / `shutdown()` / `checkpoint_prepare()` /
  `checkpoint_restore(comm_backend)`。

**注意事项**

- 断言 `torch.distributed` 已初始化、`local_size > 1`、dtype 为 fp16/bf16。

---

### `run_mixed_comm`

```python
run_mixed_comm(
    op: MixedCommOp,
    handler: MixedCommHandler,
    x_in: torch.Tensor,
    x_out: torch.Tensor | None = None,
    mode: MixedCommMode | None = None,
) -> torch.Tensor
```

**功能**：执行一个混合通信操作。这是通过混合通信处理器运行通信集合的主入口，支持融合 GPU 内核
（节点内用虚拟内存、节点间用 nvshmem）、基于 NCCL 的回退，以及 autotuned 模式选择。

**参数**

- `op`（`MixedCommOp`）：要执行的通信操作。
- `handler`（`MixedCommHandler`）：已初始化的处理器。
- `x_in`（`torch.Tensor`）：输入张量，须至少 2-D 且匹配处理器的 dtype/device。
- `x_out`（`torch.Tensor`，可选）：预分配输出张量；为 `None` 时自动分配。
- `mode`（`MixedCommMode`，可选）：执行模式。为 `None` 时用 autotune（若启用），否则回退到
  NCCL 模式（`NCCL_TP_DP` 若有效，否则 `NCCL_ONE`）。

**返回**

- `torch.Tensor`：集合操作的结果张量。

**注意事项**

- `mode == MixedCommMode.AUTOTUNE` 时先经 `handler.select_autotune_mode(op, x_in)` 解析具体模式。

---

## 相关链接 / 参考

- 上游 rst 的 `flashinfer.comm` 正文详细描述了 All-Gather matmul 的 Cake 路由约束、PCIe IPC
  两种数据平面（SM 路径与 copy-engine 路径）、Ulysses 后端策略、Head-Chunk 布局原语、DPC 与
  NCCL LSA DCP all-to-all、以及 MNNVL A2A 的废弃说明。
- 未列于 rst autosummary 但由本模块导出的辅助符号（如 `gen_trtllm_comm_module`、
  `gen_vllm_comm_module`、`gen_pcie_ipc_comm_module`、`get_pcie_ipc_comm_module`、
  `pcie_ipc_default_cache_path`、`PCIE_IPC_CUSTOM_OP`、`ULYSSES_BACKENDS`、
  `gen_ulysses_a2a_module`、`get_ulysses_a2a_module`、`CakeFusedNormCombineWorkspace`、
  `cake_fused_norm_combine*`、`moe_a2a_active_rank_mask`、`MNNVLAllreduceFusionStrategy`、
  `MNNVLQuantType`、`get_allreduce_mnnvl_workspace` 等）未在本文档展开。
- `flashinfer.comm.MoeAlltoAll` 已被 `flashinfer.moe_ep.NVLinkOneSidedAlltoAll` /
  `flashinfer.moe_ep.CakeAlltoAll` 取代。


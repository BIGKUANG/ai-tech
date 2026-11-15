# Mega MoE API

- 源码：`csrc/apis/mega_moe.hpp`，Python 封装 `deep_gemm/mega/__init__.py`
- Python 导入：`import deep_gemm`
- 目标架构：SM100（多进程 + 对称内存）

Mega MoE 把 EP dispatch、linear 1 / linear 2（FP8xFP4 或 FP8xFP8）、SwiGLU、EP combine 融合并重叠到单个 mega-kernel 中，让 NVLink 通信与张量核心计算互相重叠。需要多进程启动并使用对称内存（symmetric memory）。

相关函数：

| 函数 | 说明 |
| --- | --- |
| `get_symm_buffer_for_mega_moe(...)` | 创建对称缓冲区（`SymmBuffer`） |
| `SymmBuffer(...)` | 对称缓冲区类，持有 buffer 与输入视图 |
| `transform_weights_for_mega_moe(l1, l2)` | 权重 / SF 布局变换（必须调用） |
| `fp8_fp4_mega_moe(...)` | FP8/FP4 融合 mega MoE 内核 |
| `bf16_mega_moe(...)` | BF16 融合 mega MoE 内核 |
| `get_block_m_for_mega_moe(...)` | 查询给定配置下的 `block_m` |
| `get_token_alignment_for_mega_moe()` | 查询 token 对齐要求 |

---

## SymmBuffer

```python
class SymmBuffer:
    def __init__(self, group, num_experts,
                 num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden,
                 num_shared_experts=0,
                 mma_type='fp8xfp4',
                 activation='swiglu',
                 base=None)
```

对称缓冲区封装。构造时：

1. 把 `num_max_tokens_per_rank` 按 `get_token_alignment_for_mega_moe()` 对齐。
2. 计算所需字节数并分配对称内存（`group.size() == 1` 时用普通 `torch` 分配，否则用 `torch.distributed._symmetric_memory` 的 `rendezvous`）。
3. 清零并跨 rank `barrier` + `cuda.synchronize`。
4. 通过 `slice_input_buffers` 创建输入视图。

**参数**

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `group` | `dist.ProcessGroup` | EP 进程组；`group.size()` 即 `num_ranks` |
| `num_experts` | int | 全局 expert 数，须能被 `num_ranks` 整除 |
| `num_max_tokens_per_rank` | int | 每 rank 最大 token 数 |
| `num_topk` | int | 每个 token 选择的 expert 数 |
| `hidden` | int | 隐藏维 |
| `intermediate_hidden` | int | FFN 中间维 |
| `num_shared_experts` | int | shared expert 数，默认 0 |
| `mma_type` | str | `'fp8xfp4'`（默认）、`'fp8xfp8'`、`'bf16xbf16'` |
| `activation` | str | 仅支持 `'swiglu'` |
| `base` | `SymmBuffer`，可选 | 复用已有 buffer（要求 `base.group is group` 且容量足够） |

**公开属性**

- 配置：`group`、`num_experts`、`num_max_tokens_per_rank`、`num_topk`、`hidden`、`intermediate_hidden`、`num_shared_experts`、`mma_type`、`activation`
- 原始缓冲：`buffer`（`int8` 对称缓冲区）、`handle`
- 输入视图：`x`、`x_sf`、`topk_idx`、`topk_weights`
- shared 视图：`shared_l1_acts`、`shared_l1_acts_sf`、`shared_l2_acts`、`shared_l2_acts_sf`
- 中间（routed）视图：`l1_acts`、`l1_acts_sf`、`l2_acts`、`l2_acts_sf`

**方法**

- `destroy()`：释放 `handle` / `buffer` / `group` / `x` / `x_sf` 引用。

**输入视图的形状**

| 视图 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `x` | `[num_max_tokens_per_rank, hidden]` | FP8 时为 `float8_e4m3fn`，否则 BF16 |
| `x_sf` | `[num_max_tokens_per_rank, hidden//128]`，`int32` | 仅 FP8 路径；K-major |
| `topk_idx` | `[num_max_tokens_per_rank, num_topk]`，`int64` | — |
| `topk_weights` | `[num_max_tokens_per_rank, num_topk]`，`float32` | — |
| `l1_acts_sf` / `l2_acts_sf` | `[num_sf_ring_tokens, ...]`，`int32` | M-major |
| `shared_l1_acts_sf` | `[num_max_shared_sf_tokens, hidden//128]`，`int32` | 转置布局 |

> 每次调用内核前，必须把输入复制进 `buffer`（见 `DG_COMM_KERNEL_DEBUG` 说明）。

---

## get_symm_buffer_for_mega_moe

```python
get_symm_buffer_for_mega_moe(
    group, num_experts,
    num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    num_shared_experts=0,
    use_fp8_dispatch=None,   # 已废弃
    mma_type='fp8xfp4',
    activation='swiglu',
) -> SymmBuffer
```

`SymmBuffer` 的便捷工厂。`use_fp8_dispatch` 已废弃：若提供，会校验它与 `mma_type` 一致并发出 `DeprecationWarning`，请改用 `mma_type`。

---

## transform_weights_for_mega_moe

```python
transform_weights_for_mega_moe(l1_weights, l2_weights, activation='swiglu')
    -> (transformed_l1, transformed_l2)
```

权重 / 缩放因子布局变换。**调用 Mega MoE 前必须执行**，因为内核只接受该变换后的布局：

- **FP8/FP4 路径**（输入为 `(data, sf)` 元组）：
  - L1：按 8 行粒度交错 gate / up 权重与 SF（`_interleave_weights`）。
  - L1 与 L2：对 SF 施加 UTCCP intra-128-row 转置（`reshape(-1, 4, 32, packed_sf_k).transpose(2, 3)`，要求 `mn % 128 == 0`）。
- **BF16 路径**（输入为普通张量）：仅对 L1 交错 gate / up，L2 不变。
- `activation` 目前仅支持 `'swiglu'`。

---

## fp8_fp4_mega_moe

```python
fp8_fp4_mega_moe(
    y,
    l1_weights, l2_weights,
    sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32),
    activation='swiglu',
    activation_clamp=None,
    fast_math=True,
)
```

**参数**

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `y` | `[num_tokens, hidden]`，`torch.bfloat16` | 输出；`y.size(0)` 即本次参与计算的 `num_tokens` |
| `l1_weights` | `(weight, sf)` | routed L1 权重（FP4 或 FP8），已通过 `transform_weights_for_mega_moe` 变换 |
| `l2_weights` | `(weight, sf)` | routed L2 权重 |
| `sym_buffer` | `SymmBuffer` | — |
| `shared_l1_weights` / `shared_l2_weights` | `(weight, sf)` 对，可选 | shared expert 权重，**必须是 FP8 E4M3**；两者同时提供 |
| `cumulative_local_expert_recv_stats` | `[num_experts_per_rank]`，`int`，可选 | 本地 expert 接收计数（就地更新） |
| `recipe` | `(1, 1, 32)` | gran_k 固定为 32 |
| `activation` | str | 仅支持 `'swiglu'` |
| `activation_clamp` | float，可选 | 激活截断值（`>= 0`）；`None` 表示 `+inf` |
| `fast_math` | bool | 快速数学近似 |

**权重 / SF 契约**

- L1 权重逻辑形状 `[num_experts_per_rank, 2*intermediate_hidden, hidden]`，L2 为 `[num_experts_per_rank, hidden, intermediate_hidden]`。
- `recipe` 固定 `(1, 1, 32)`；`check_sf_layout(..., gran_mn=1, gran_k=32, type_check=torch.int)` 要求 **SF 为已打包的 `int32` UE8M0**，不接受 FP32，也没有自动变换。布局见 [scaling-factor-format.md](./scaling-factor-format.md) 6.3。
- routed L1 SF：`[num_experts_per_rank, 2*intermediate_hidden, ceil_div(hidden, 128)]`；routed L2 SF：`[num_experts_per_rank, hidden, ceil_div(intermediate_hidden, 128)]`。
- shared expert SF：同契约，二维（无 group 维）。
- shared 权重与 routed 权重的 `is_localized` 状态必须一致。

**行为说明**

- `hidden % 128 == 0`、`intermediate_hidden % 128 == 0`、`shared_intermediate_hidden % 128 == 0`（FP8 路径）。
- `num_tokens <= num_max_tokens_per_rank`；`num_experts == num_experts_per_rank * num_ranks`。
- 若设置了 `DG_COMM_KERNEL_DEBUG=1`，内核会在每次调用后清零整个对称缓冲区（调试用），因此调用方必须在每次调用前重新拷贝输入。
- 仅 SM100。

---

## bf16_mega_moe

```python
bf16_mega_moe(
    y,
    l1_weights, l2_weights,
    sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    activation='swiglu',
    activation_clamp=None,
    fast_math=True,
)
```

BF16 版本，`l1_weights`、`l2_weights` 及 shared 权重均为纯 BF16 张量（不传 SF）。其余语义与 `fp8_fp4_mega_moe` 相同。

---

## get_block_m_for_mega_moe

```python
get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_tokens, num_topk, mma_type) -> int
```

根据配置返回内核使用的 `block_m`（用于准备 shared SF 的循环布局等）。`num_tokens >= 0`。

## get_token_alignment_for_mega_moe

```python
get_token_alignment_for_mega_moe() -> int
```

返回 `num_max_tokens_per_rank` 的对齐要求（`kLCMCandidateBlockM`）。

---

## 完整示例（多进程）

```python
import torch, torch.distributed as dist
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

# 1. 初始化分布式
rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
num_experts, num_topk = 384, 6
num_experts_per_rank = num_experts // num_ranks
hidden, intermediate_hidden = 7168, 3072
num_max_tokens_per_rank, num_shared_experts = 8192, 1

# 2. 分配对称缓冲区
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    num_shared_experts=num_shared_experts, mma_type='fp8xfp4')

# 3. 量化权重（FP4 + UE8M0 SF），并变换布局
def cast_to_fp4(w):  # w: [g, n, k] bf16
    g, n, k = w.shape
    out = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    sf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(g):
        out[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), num_groups=g)
    return out, sf

l1 = cast_to_fp4(torch.randn((num_experts_per_rank, 2*intermediate_hidden, hidden), device='cuda'))
l2 = cast_to_fp4(torch.randn((num_experts_per_rank, hidden, intermediate_hidden), device='cuda'))
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

# 4. 每次调用前把输入拷进 buffer
num_tokens = 4096
x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

# 这里用随机 top-k 演示
scores = torch.randn((num_tokens, num_experts), device='cuda')
topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)

buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 5. 运行融合 mega MoE
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)

# 6. 清理
buffer.destroy()
dist.destroy_process_group()
```

## 相关

- 权重 SF 变换与布局：[scaling-factor-format.md](./scaling-factor-format.md) 6.3
- Locality domain（可选权重本地化）：[api-locality-domain.md](./api-locality-domain.md)
- 测试参考：`tests/test_mega_moe.py`

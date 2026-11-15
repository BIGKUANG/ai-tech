# 工具、测试与 Legacy API

- 源码：`deep_gemm/utils/`、`deep_gemm/testing/`、`deep_gemm/legacy/`
- Python 导入：
  - `from deep_gemm.utils import ...`（`deep_gemm.utils` 的子模块 `math`、`layout`、`dist` 全部导出）
  - `from deep_gemm.testing import ...`
  - `from deep_gemm.legacy import ...`

---

## 数学与量化工具（`deep_gemm.utils` / `deep_gemm.utils.math`）

### 基础数学

| 函数 | 说明 |
| --- | --- |
| `ceil_div(x, y)` | `(x + y - 1) // y` |
| `align(x, y)` | `ceil_div(x, y) * y` |

### UE8M0

```python
ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor
```

把 FP32 张量向 UE8M0 取整（只保留指数，尾数置零，指数 clamp 到 `[1, 254]`）。

```python
pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor
```

把 UE8M0 FP32 值按 4 个一组打包为 `int32`（低字节在前）。要求 `x.dtype == float`、最后一维 % 4 == 0、所有值已是指数形式（`x_int & 0x7fffff == 0` 且非负）。

```python
unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor
```

把打包 `int32` 还原为 FP32 UE8M0 值。

### 量化 cast

所有 cast 函数返回 `(data, sf)`：

```python
per_token_cast_to_fp8(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False)
```
按行（token）量化 2D 张量到 FP8 E4M3。SF 为 `[M, ceil_div(N, gran_k)]`。`use_packed_ue8m0=True` 时返回打包 `int32`（此时要求 SF 个数为 4 的倍数，不足则以 `1.0` 补齐）。amax 下限 `1e-4`。

```python
per_channel_cast_to_fp8(x, use_ue8m0, gran_k=128)
```
按列（channel）量化：`x` 视为 `[-1, gran_k, N]`，对 `gran_k` 维求 amax。要求 `x.size(0) % gran_k == 0`。

```python
per_block_cast_to_fp8(x, use_ue8m0, gran_k=128)
```
按 `gran_k x gran_k` 块量化，输出与输入同形。

```python
per_custom_dims_cast_to_fp8(x, dims, use_ue8m0)
```
在指定的 `dims` 上求 amax，其余维广播；返回 `(x_scaled, sf.squeeze())`。

```python
per_token_cast_to_fp4(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False)
```
按行量化到 FP4 E2M1（打包为 `int8`，每字节 2 个 FP4）。SF 上限 `6.0`。`use_packed_ue8m0=True` 时返回打包 UE8M0。

```python
transpose_packed_fp4(a: torch.Tensor) -> torch.Tensor
```
转置打包 FP4 张量（`[M, N/2]` → `[N/2, M]` 形式）。要求 `a.dtype == torch.int8`、`M % 2 == 0`。

```python
cast_back_from_fp4(packed, sf, gran_k=128, use_packed_ue8m0=False) -> torch.Tensor
cast_back_from_fp8(x_fp8, sf, gran_k=128, use_packed_ue8m0=False) -> torch.Tensor
```
反量化工具，用于构造参考结果。`use_packed_ue8m0=True` 时先 `unpack_ue8m0_from_int`。

**示例**

```python
import torch
from deep_gemm.utils import per_token_cast_to_fp8, cast_back_from_fp8, align

x = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
x_fp8, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)
x_restored = cast_back_from_fp8(x_fp8, sf, gran_k=128, use_packed_ue8m0=True)
```

### 布局工具（`deep_gemm.utils.layout`）

见 [api-layout.md](./api-layout.md)。`deep_gemm.utils.layout` 还导出别名 `get_m_alignment_for_contiguous_layout`、`get_k_alignment_for_contiguous_layout`（均等价于 `get_mk_alignment_for_contiguous_layout`）。

---

## 分布式工具（`deep_gemm.utils.dist`）

```python
init_dist(local_rank: int, num_local_ranks: int) -> (rank, world_size, group)
```

初始化 NCCL 进程组。通过环境变量 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK` 读取集群配置，返回全局 rank、world size 与新的进程组。

```python
uneven_all_gather(tensor, dim=0, group=None) -> torch.Tensor
```

变长 all-gather：先交换各 rank 在 `dim` 上的尺寸，pad 到最大值后 all-gather，再去掉 padding 拼接。

```python
dist_print(s='', once_in_node=False)
```

带 barrier 的打印；`once_in_node=True` 时仅本地 rank 0 打印。

---

## 测试与基准工具（`deep_gemm.testing`）

### bench

```python
bench(fn, num_warmups=5, num_tests=10, high_precision=False) -> float
```

简单 CUDA event 计时（返回秒）。会先 flush 256 MB L2 缓存；`high_precision=True` 时插入一个大 kernel 消除 CPU 启动开销。

### bench_kineto

```python
bench_kineto(fn, kernel_names, num_tests=30, suppress_kineto_output=False,
             trace_path=None, flush_l2=True, with_multiple_kernels=False, barrier=None)
```

用 `torch.profiler` 按 kernel 名过滤并统计平均耗时（秒）。`kernel_names` 为字符串或元组；元组时返回对应元组。

- `DG_USE_NVIDIA_TOOLS=1` 时直接返回占位值（避免与 Nsight 冲突）。
- `with_multiple_kernels=True` 允许某个名字对应多个 kernel。
- `barrier` 用于多进程基准中消除不均衡的 CPU 启动开销。
- `trace_path` 可导出 Chrome trace。

### 输出抑制

- `empty_suppress()`：上下文管理器，空操作。
- `suppress_stdout_stderr()`：临时把 stdout/stderr 重定向到 `/dev/null`。

### 数值工具

```python
calc_diff(x, y) -> float
count_bytes(*tensors) -> int
assert_bitwise_equal(x, y, label='')
```

- `calc_diff`：`1 - 2<x,y> / (||x||^2 + ||y||^2)`；全零返回 0。
- `count_bytes`：统计张量（可嵌套 tuple/list，忽略 `None`）的字节数。
- `assert_bitwise_equal`：按字节比较，不等时报告首个不匹配的字节 / 元素 / 坐标。

### 环境工具

```python
get_arch_major() -> int
test_filter(condition)
ignore_env(name, condition)
```

- `get_arch_major()`：返回 CUDA 设备主版本号（9 = Hopper，10 = Blackwell）。
- `test_filter(condition)`：装饰器，条件为假时跳过测试并打印过滤信息。
- `ignore_env(name, condition)`：装饰器，条件为真时临时移除环境变量 `name`。

**示例**

```python
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major

if get_arch_major() == 10:
    t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt(a, b, d), 'gemm_', suppress_kineto_output=True)
```

---

## Legacy Triton 内核（`deep_gemm.legacy`）

以下内核为 **A100 专用、基于 Triton 的旧实现**，可能在未来被废弃或改写为 TileLang。它们的接口与对应 C++ 内核不同：使用 `m_indices` / `k_indices` 句柄，且仅支持 BF16。对应实现：

| 函数 | 说明 |
| --- | --- |
| `m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d, m_indices)` | M 分组连续 NT，`m_indices` 为每个 M 行的 expert 标签 |
| `m_grouped_bf16_gemm_nn_contiguous_tl(a, b, d, m_indices)` | 同上，NN（内部 `b.mT`） |
| `a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d, mappings)` | A 融合（带 `m_row_indices`），`mappings = (m_indices, m_row_indices)` |
| `a_fused_m_grouped_bf16_gemm_nn_contiguous_tl(a, b, d, mappings)` | 同上，NN |
| `a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a, b, d, handle, acc)` | K 分组 TN，`handle = (k_indices, k_start, k_end)`，`acc` 控制累加 |
| `b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a, b, d, handle, acc)` | 同上（B 变体） |

**共同约束**

- `a`、`b` 为 `torch.bfloat16`，`d` 为 `torch.bfloat16`；索引为 `torch.int32`。
- 连续张量（`b` 允许 `b.mT` 连续）。
- M/K 分组要求 group 维度按 `get_mk_alignment_for_contiguous_layout()` 对齐。
- 非 fused 的 `m_grouped` 变体使用 `m_indices` 标记空 token（`-1` 时写零）。
- `deep_gemm.legacy` 加载失败时 `deep_gemm/__init__.py` 会打印 `Failed to load legacy DeepGEMM A100 Triton kernels: ...` 并继续。

> 新代码请优先使用 C++ 内核（[api-gemm.md](./api-gemm.md)、[api-gemm-grouped.md](./api-gemm-grouped.md)）。

---

## 相关

- 布局：[api-layout.md](./api-layout.md)
- 量化 / SF 格式：[scaling-factor-format.md](./scaling-factor-format.md)

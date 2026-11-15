# 测试与基准工具（`flashinfer.testing`）

> 源码：`flashinfer/testing/utils.py`（`flashinfer/testing/__init__.py` 负责导出）
> 上游文档：`docs/api/testing.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.testing` 提供测试与性能分析工具：随机种子设置、Attention 的 FLOPs / 吞吐量估算，
以及三种 GPU 计时后端（CUDA Events、CUDA Graphs、CUPTI）。这些工具主要用于基准测试与
单元测试，不参与推理路径。

## 导入

```python
from flashinfer.testing import (
    set_seed,
    sleep_after_kernel_run,
    attention_flops,
    attention_flops_with_actual_seq_lens,
    attention_tflops_per_sec,
    attention_tflops_per_sec_with_actual_seq_lens,
    attention_tb_per_sec,
    attention_tb_per_sec_with_actual_seq_lens,
    bench_gpu_time,
    bench_gpu_time_with_cuda_event,
    bench_gpu_time_with_cudagraph,
    bench_gpu_time_with_cupti,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `set_seed` | 函数 | 设置 Python / NumPy / PyTorch / CUDA 的随机种子 |
| `sleep_after_kernel_run` | 函数 | 按内核执行时长动态睡眠，缓解热降频 |
| `attention_flops` | 函数 | 按均匀序列长度估算 Attention FLOPs |
| `attention_flops_with_actual_seq_lens` | 函数 | 按真实逐序列长度估算 Attention FLOPs |
| `attention_tflops_per_sec` | 函数 | 按均匀序列长度估算 TFLOPS/s |
| `attention_tflops_per_sec_with_actual_seq_lens` | 函数 | 按真实序列长度估算 TFLOPS/s |
| `attention_tb_per_sec` | 函数 | 按均匀序列长度估算 TB/s（含 Q/K/V/O 读写） |
| `attention_tb_per_sec_with_actual_seq_lens` | 函数 | 按真实序列长度估算 TB/s |
| `bench_gpu_time` | 函数 | 统一基准入口，按配置分派计时后端 |
| `bench_gpu_time_with_cuda_event` | 函数 | 使用 CUDA Events 的计时 |
| `bench_gpu_time_with_cudagraph` | 函数 | 使用 CUDA Graph 摊薄启动开销的计时 |
| `bench_gpu_time_with_cupti` | 函数 | 使用 CUPTI 硬件剖析的精确内核计时 |

## API 详情

### `set_seed`

```python
set_seed(random_seed)
```

**功能**：设置随机种子以保证测试可复现。内部同时对 `torch.manual_seed`、`random.seed`、
`np.random.seed` 以及（当 CUDA 可用时）`torch.cuda.manual_seed` / `torch.cuda.manual_seed_all`
进行设置。

**参数**

- `random_seed`：要设置的随机种子。

**返回**

- `None`。

**示例**

```python
from flashinfer.testing import set_seed

set_seed(42)
```

---

### `sleep_after_kernel_run`

```python
sleep_after_kernel_run(execution_time)
```

**功能**：在一次内核运行后动态睡眠，最长 1 秒，睡眠时长按执行时间估算
（`min(execution_time / 200, 1.0)`），用于缓解持续压测导致的热降频。当执行时间为
`inf` 时使用固定 0.01 秒。

**参数**

- `execution_time`（`float`）：内核执行时间，单位毫秒。

**返回**

- `None`。

**示例**

```python
from flashinfer.testing import sleep_after_kernel_run

sleep_after_kernel_run(0.35)  # 约睡眠 1.75 ms
```

---

### `attention_flops`

```python
attention_flops(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
)
```

**功能**：估算一层 Attention 的 FLOPs，假设 batch 内所有序列长度相同。分别计算
QK^T（bmm1）与 PV（bmm2）两次矩阵乘的 FLOPs 并求和。causal 时按三角掩码折减。

**参数**

- `batch_size`（`int`）：batch 大小。
- `qo_seqlen`（`int`）：query 序列长度（batch 内一致）。
- `kv_seqlen`（`int`）：key/value 序列长度（batch 内一致）。
- `head_dim_qk`（`int`）：query 与 key 的 head 维度。
- `head_dim_vo`（`int`）：value 的 head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `causal`（`bool`）：是否使用 causal 掩码，causal 时 FLOPs 折减。

**返回**

- `total_flops`（`int`）：该层的总 FLOPs。

**注意事项**

- causal 分支要求 `qo_seqlen <= kv_seqlen`，否则抛出 `ValueError`。

**示例**

```python
from flashinfer.testing import attention_flops

flops = attention_flops(1, 4096, 4096, 128, 128, 32, causal=True)
```

---

### `attention_flops_with_actual_seq_lens`

```python
attention_flops_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
)
```

**功能**：使用真实逐序列长度（一维张量）估算一层 Attention 的 FLOPs，不假设 batch 内长度一致。

**参数**

- `actual_seq_lens_q`（`torch.Tensor`）：每个请求的 query 实际长度数组。
- `actual_seq_lens_kv`（`torch.Tensor`）：每个请求的 key/value 实际长度数组。
- `head_dim_qk`（`int`）：query 与 key 的 head 维度。
- `head_dim_vo`（`int`）：value 的 head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `causal`（`bool`）：是否使用 causal 掩码。

**返回**

- `total_flops`（`int`）：该层的总 FLOPs（`torch.Tensor` 标量）。

**注意事项**

- decode 时 causal 必须为 `False`，因为该函数假设 `qo_seqlen == kv_seqlen`。
- 当 `causal=True` 且存在 `actual_seq_lens_q > actual_seq_lens_kv` 时抛出 `ValueError`。

**示例**

```python
import torch
from flashinfer.testing import attention_flops_with_actual_seq_lens

lens_q = torch.tensor([4096, 2048], dtype=torch.int32)
lens_kv = lens_q.clone()
flops = attention_flops_with_actual_seq_lens(lens_q, lens_kv, 128, 128, 32, False)
```

---

### `attention_tflops_per_sec`

```python
attention_tflops_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    time,
)
```

**功能**：基于均匀序列长度估算一层 Attention 的每秒 TFLOPS，内部调用
`attention_flops` 并除以时间。

**参数**

- `batch_size`、（`int`）batch 大小。
- `qo_seqlen`（`int`）：query 序列长度。
- `kv_seqlen`（`int`）：key/value 序列长度。
- `head_dim_qk`（`int`）：query/key head 维度。
- `head_dim_vo`（`int`）：value head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `causal`（`bool`）：是否 causal。
- `time`（`float`）：执行时间，单位毫秒。

**返回**

- `tflops_per_sec`（`float`）：每秒 TFLOPS；当 `time` 为 `NaN` 时返回 `0.0`。

**示例**

```python
from flashinfer.testing import attention_tflops_per_sec

t = attention_tflops_per_sec(1, 4096, 4096, 128, 128, 32, True, 1.2)
```

---

### `attention_tflops_per_sec_with_actual_seq_lens`

```python
attention_tflops_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    ms,
)
```

**功能**：基于真实逐序列长度估算每秒 TFLOPS，不假设 batch 内长度一致。

**参数**

- `actual_seq_lens_q`（`torch.Tensor`）：query 实际长度数组。
- `actual_seq_lens_kv`（`torch.Tensor`）：key/value 实际长度数组。
- `head_dim_qk`（`int`）：query/key head 维度。
- `head_dim_vo`（`int`）：value head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `causal`（`bool`）：是否 causal。
- `ms`（`float`）：执行时间，单位毫秒。

**返回**

- `tflops_per_sec`（`float`）：每秒 TFLOPS；当 `ms` 为 `NaN` 时返回 `0.0`。

**示例**

```python
import torch
from flashinfer.testing import attention_tflops_per_sec_with_actual_seq_lens

lens = torch.tensor([4096, 2048], dtype=torch.int32)
t = attention_tflops_per_sec_with_actual_seq_lens(lens, lens, 128, 128, 32, False, 0.9)
```

---

### `attention_tb_per_sec`

```python
attention_tb_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    k_dtype=torch.bfloat16,
    v_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
)
```

**功能**：基于均匀序列长度估算一层 Attention 的每秒 TB（`1e12` 字节，非 TiB）吞吐，
统计 Q/K/V/O 的读写字节数。

**参数**

- `batch_size`（`int`）：batch 大小。
- `qo_seqlen`（`int`）：query 序列长度。
- `kv_seqlen`（`int`）：key/value 序列长度。
- `head_dim_qk`（`int`）：query/key head 维度。
- `head_dim_vo`（`int`）：value head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `num_kv_heads`（`int`）：key/value head 数。
- `time`（`float`）：执行时间，单位毫秒。
- `q_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：query 数据类型。
- `k_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：key 数据类型。
- `v_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：value 数据类型。
- `o_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出数据类型。

**返回**

- `tb_per_sec`（`float`）：每秒 TB；当 `time` 为 `NaN` 时返回 `0.0`。

**示例**

```python
from flashinfer.testing import attention_tb_per_sec

bw = attention_tb_per_sec(32, 1, 8192, 128, 128, 32, 8, 0.5)
```

---

### `attention_tb_per_sec_with_actual_seq_lens`

```python
attention_tb_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    k_dtype=torch.bfloat16,
    v_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
)
```

**功能**：基于真实逐序列长度估算每秒 TB 吞吐，不假设 batch 内长度一致。

**参数**

- `actual_seq_lens_q`（`torch.Tensor`）：query 实际长度数组。
- `actual_seq_lens_kv`（`torch.Tensor`）：key/value 实际长度数组。
- `head_dim_qk`（`int`）：query/key head 维度。
- `head_dim_vo`（`int`）：value head 维度。
- `num_qo_heads`（`int`）：query head 数。
- `num_kv_heads`（`int`）：key/value head 数。
- `time`（`float`）：执行时间，单位毫秒。
- `q_dtype`/`k_dtype`/`v_dtype`/`o_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：各张量数据类型。

**返回**

- `tb_per_sec`（`float`）：每秒 TB；当 `time` 为 `NaN` 时返回 `0.0`。

**示例**

```python
import torch
from flashinfer.testing import attention_tb_per_sec_with_actual_seq_lens

lens = torch.tensor([4096, 2048], dtype=torch.int32)
bw = attention_tb_per_sec_with_actual_seq_lens(lens, lens, 128, 128, 32, 8, 0.5)
```

---

### `bench_gpu_time`

```python
bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,
    l2_flush_size_mb: Optional[int] = None,
    l2_flush_device: Optional[str] = None,
    sleep_after_run: bool = False,
    enable_cupti: bool = False,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
)
```

**功能**：推荐的统一 GPU 内核基准入口，按配置分派到不同计时实现。

- `enable_cupti=True`：使用 CUPTI（需 cupti-python >= 13，即 CUDA 13+），最精确，测量纯内核时间。
- `use_cuda_graph=True`：使用 CUDA Graph 摊薄启动开销。
- 默认：使用 CUDA Events，最简单但包含启动开销。

冷 L2 策略随后端自动选择：CUPTI 与纯 CUDA Events 使用 L2 flush；CUDA Events + CUDA Graphs
使用轮转缓冲区（rotating buffers）。

**参数**

- `fn`（`Callable`）：待基准的内核函数。
- `dry_run_iters`（`int`，可选）：预热迭代次数（不计时）；为 `None` 时由 `dry_run_time_ms` 推导。
- `repeat_iters`（`int`，可选）：测量迭代次数；为 `None` 时由 `repeat_time_ms` 推导。
- `dry_run_time_ms`（`int`，默认 `25`）：预热目标时长（毫秒）。
- `repeat_time_ms`（`int`，默认 `100`）：测量目标时长（毫秒）。
- `l2_flush`、`l2_flush_size_mb`、`l2_flush_device`：**已废弃**，请改用 `cold_l2_cache`。
- `sleep_after_run`（`bool`，默认 `False`）：每次迭代后短暂睡眠以降低热降频。
- `enable_cupti`（`bool`，默认 `False`）：是否使用 CUPTI 计时。
- `use_cuda_graph`（`bool`，默认 `False`）：是否使用 CUDA Graph 计时。
- `num_iters_within_graph`（`int`，默认 `10`）：每个 graph 中捕获的内核调用数（仅 CUDA Graph 模式）。
- `input_args`（`tuple`）：传给 `fn` 的位置参数。
- `input_kwargs`（`dict`，可选）：传给 `fn` 的关键字参数。
- `cold_l2_cache`（`bool`，默认 `True`）：是否为每次迭代保证冷 L2；具体策略随计时后端自动选择。
- `aggregate_op`（`Callable`，默认 `max`）：跨 rank 的聚合操作。

**返回**

- `List[float]`：每次迭代的执行时间，单位毫秒。

**注意事项**

- 传入已废弃的 `l2_flush*` 参数会发出 `DeprecationWarning`，并在可用时映射到 `cold_l2_cache`。

**示例**

```python
import numpy as np
import flashinfer
from flashinfer.testing import bench_gpu_time

def run_kernel(x, y, out):
    flashinfer.single_prefill_with_kv_cache(x, y, y, out)

x = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
y = torch.randn_like(x)
out = torch.empty_like(x)
times = bench_gpu_time(fn=run_kernel, input_args=(x, y, out), use_cuda_graph=True)
print(np.median(times))
```

---

### `bench_gpu_time_with_cuda_event`

```python
bench_gpu_time_with_cuda_event(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,
    l2_flush_size_mb: Optional[int] = None,
    l2_flush_device: Optional[str] = None,
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
)
```

**功能**：使用 CUDA Events 测量内核执行时间（不使用 CUDA Graph），是最简单的方法，
适合启动开销相对执行时间可忽略的内核。流程为：5 次快速估算迭代 → 预热（不计时）→
带 CUDA Event 的测量迭代。迭代次数可直接给出，也可由目标时长推导。

**参数**

- `fn`（`Callable`）：待基准的内核函数。
- `dry_run_iters`（`int`，可选）：预热迭代次数；为 `None` 时由 `dry_run_time_ms` 推导。
- `repeat_iters`（`int`，可选）：测量迭代次数；为 `None` 时由 `repeat_time_ms` 推导。
- `dry_run_time_ms`（`int`，默认 `25`）：预热目标时长（毫秒）。
- `repeat_time_ms`（`int`，默认 `100`）：测量目标时长（毫秒）。
- `l2_flush`、`l2_flush_size_mb`、`l2_flush_device`：**已废弃**，改用 `cold_l2_cache`。
- `sleep_after_run`（`bool`，默认 `False`）：是否在每次迭代后睡眠。
- `input_args`（`tuple`）：传给 `fn` 的位置参数。
- `input_kwargs`（`dict`，可选）：传给 `fn` 的关键字参数。
- `cold_l2_cache`（`bool`，默认 `True`）：是否每次迭代前 flush L2 缓存。
- `aggregate_op`（`Callable`，默认 `max`）：跨 rank 聚合操作。

**返回**

- `List[float]`：每次迭代执行时间，单位毫秒。

**注意事项**

- 不使用 CUDA Graph，因此每次迭代都包含内核启动开销；对启动延迟敏感的场景请改用
  `bench_gpu_time_with_cudagraph`。

**示例**

```python
import torch
from flashinfer.testing import bench_gpu_time_with_cuda_event

def my_kernel(a, b):
    return torch.matmul(a, b.T)

q = torch.randn(1024, 128, device="cuda")
k = torch.randn(1024, 128, device="cuda")
times = bench_gpu_time_with_cuda_event(fn=my_kernel, input_args=(q, k))
```

---

### `bench_gpu_time_with_cudagraph`

```python
bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    l2_flush: Optional[bool] = None,
    l2_flush_size_mb: Optional[int] = None,
    l2_flush_device: Optional[str] = None,
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
)
```

**功能**：使用 CUDA Graph 摊薄内核启动开销。在单个 graph 内运行多次迭代，使启动延迟被摊销，
测量结果更接近纯 GPU 时间。当 `cold_l2_cache=True` 时使用**轮转缓冲区**保证每次内核调用
都处于冷 L2：会根据设备 L2 大小自动计算需要复制的缓冲区份数。

**参数**

- `fn`（`Callable`）：待基准的内核函数。
- `dry_run_iters`（`int`，可选）：预热迭代次数；为 `None` 时由 `dry_run_time_ms` 推导。
- `repeat_iters`（`int`，可选）：测量次数（graph 重放次数）；为 `None` 时由 `repeat_time_ms` 推导。
- `dry_run_time_ms`（`int`，默认 `25`）：预热目标时长（毫秒）。
- `repeat_time_ms`（`int`，默认 `100`）：测量目标时长（毫秒）。
- `num_iters_within_graph`（`int`，默认 `10`）：graph 内捕获的内核调用数；越大越能摊薄启动开销，但轮转缓冲区占用更多显存。
- `l2_flush`、`l2_flush_size_mb`、`l2_flush_device`：**已废弃**，改用 `cold_l2_cache`。
- `sleep_after_run`（`bool`，默认 `False`）：是否在每次迭代后睡眠。
- `input_args`（`tuple`）：传给 `fn` 的位置参数；`cold_l2_cache=True` 时其中的 GPU 张量会被克隆。
- `input_kwargs`（`dict`，可选）：传给 `fn` 的关键字参数；同上会被克隆。
- `cold_l2_cache`（`bool`，默认 `True`）：使用轮转缓冲区保证冷 L2。
- `aggregate_op`（`Callable`，默认 `max`）：跨 rank 聚合操作。

**返回**

- `List[float]`：每次迭代执行时间，单位毫秒；每次时间等于 graph 重放时长除以 `num_iters_within_graph`。

**注意事项**

- 使用 `input_args`/`input_kwargs` 时，函数必须把张量作为实参接收（不能从闭包中捕获）。
- GPU 张量会被自动检测并克隆；非张量参数在所有副本中保持不变。
- 显存占用随超过 L2 所需的轮转份数增加。

**示例**

```python
import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph

def run_attention(q, k, v, o):
    flashinfer.single_prefill_with_kv_cache(q, k, v, o)

q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q); o = torch.empty_like(q)
times = bench_gpu_time_with_cudagraph(fn=run_attention, input_args=(q, k, v, o))
```

---

### `bench_gpu_time_with_cupti`

```python
bench_gpu_time_with_cupti(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,
    l2_flush_size_mb: Optional[int] = None,
    l2_flush_device: Optional[str] = None,
    sleep_after_run: bool = False,
    use_cuda_graph: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
)
```

**功能**：使用 CUPTI（CUDA Profiling Tools Interface）活动追踪进行精确内核计时，测量真实
GPU 内核执行时间、排除 CPU 侧启动开销。冷 L2 通过迭代间 L2 flush 实现（CUPTI 按迭代测量，
因此无论是否 `use_cuda_graph` 都有效）。若 CUPTI 不可用则会回退：`use_cuda_graph=True`
时回退到 `bench_gpu_time_with_cudagraph`（轮转缓冲区），否则回退到
`bench_gpu_time_with_cuda_event`（L2 flush），并发出警告。

**参数**

- `fn`（`Callable`）：待基准的内核函数。
- `dry_run_iters`（`int`，可选）：预热迭代次数；为 `None` 时由 `dry_run_time_ms` 推导。
- `repeat_iters`（`int`，可选）：测量迭代次数；为 `None` 时由 `repeat_time_ms` 推导。
- `dry_run_time_ms`（`int`，默认 `25`）：预热目标时长（毫秒）。
- `repeat_time_ms`（`int`，默认 `100`）：测量目标时长（毫秒）。
- `l2_flush`、`l2_flush_size_mb`、`l2_flush_device`：**已废弃**，改用 `cold_l2_cache`。
- `sleep_after_run`（`bool`，默认 `False`）：是否在每次迭代后睡眠。
- `use_cuda_graph`（`bool`，默认 `False`）：是否在 CUDA Graph 中捕获操作以减少启动开销。
- `input_args`（`tuple`）：传给 `fn` 的位置参数。
- `input_kwargs`（`dict`，可选）：传给 `fn` 的关键字参数。
- `cold_l2_cache`（`bool`，默认 `True`）：是否每次迭代前 flush L2 缓存。
- `aggregate_op`（`Callable`，默认 `max`）：跨 rank 聚合操作。

**返回**

- `List[float]`：每次迭代的 GPU 内核执行时间，单位毫秒。

**注意事项**

- 需要 `cupti-python` 包版本 >= 13.0.0（`pip install -U cupti-python`）。
- 若 CUPTI 不可用会发出警告并自动回退到 CUDA Event 或 CUDA Graph 计时。

**示例**

```python
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

def my_kernel(a, b):
    return torch.matmul(a, b.T)

q = torch.randn(1024, 128, device="cuda")
k = torch.randn(1024, 128, device="cuda")
times = bench_gpu_time_with_cupti(fn=my_kernel, input_args=(q, k))
```

## 相关链接 / 参考

- `bench_gpu_time` 是统一入口，其余三个 `bench_gpu_time_with_*` 为可直接调用的具体实现。
- 冷 L2 策略与选择逻辑详见源码中各函数的 `Note` 与 `__init__` 文档。


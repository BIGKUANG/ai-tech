# 绿色上下文的创建（`flashinfer.green_ctx`）

> 源码：`flashinfer/green_ctx.py`
> 上游文档：`docs/api/green_ctx.rst`
> 适用版本：FlashInfer 0.7.2

本模块提供创建 CUDA <a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html">green context</a>
的实用工具。green context 允许把一个 GPU 的 SM 划分为多个独立的资源分区，在各分区上并发执行不同的
kernel（例如把 prefetch / attention / GEMM 放到不同 SM 组）。本模块基于 cuda-python 的
`cuda.bindings.driver` 驱动 API 实现，会为每个分区创建一个 `torch.cuda.Stream`，并返回对应的
`CUdevResource` 句柄，便于调用方查询实际分配的 SM 数量。

## 导入

```python
from flashinfer.green_ctx import (
    split_device_green_ctx,
    split_device_green_ctx_by_sm_count,
)
```

## 依赖

- 需要安装与 CUDA 版本匹配的 cuda-python（`import cuda.bindings.driver` / `cuda.bindings.runtime`）。
  导入本模块时若缺失会抛出 `ImportError`。
- 设备需要支持 green context（一般需要较新的驱动与 CUDA 版本）。

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `split_device_green_ctx` | 函数 | 按组数与每组最小 SM 数划分设备，返回 streams 与 resources。 |
| `split_device_green_ctx_by_sm_count` | 函数 | 按每组指定 SM 数划分设备，返回 streams 与 resources。 |

## API 详情

### `split_device_green_ctx`

```python
split_device_green_ctx(
    dev: torch.device,
    num_groups: int,
    min_count: int,
) -> Tuple[List[torch.Stream], List[CUdevResource]]
```

**功能**：把设备划分为 `num_groups` 个 green context，返回每个分组对应的 `torch.Stream` 与
`CUdevResource`，以及剩余 SM 对应的最后一组。green context 允许在不同 SM 分区上并发执行多个 kernel。

**参数**

- `dev`（`torch.device`）：要划分的设备。
- `num_groups`（`int`）：划分的组数。
- `min_count`（`int`）：每组要求的最小 SM 数。实际值会被向上取整以满足对齐与粒度要求（见“注意事项”）。

**返回**

- `streams`（`List[torch.Stream]`）：与各 green context 对应的 `torch.Stream` 列表。
- `resources`（`List[CUdevResource]`）：与各 green context 对应的 `CUdevResource` 列表。

返回的 `streams` 与 `resources` 长度为 `num_groups + 1`，最后一个为剩余 SM。

**注意事项**

- SM 数会向上取整以满足对齐与粒度要求，例如：
  - 请求 7 个 SM → 分配 8 个（向上取整到最小值）；
  - 请求 10 个 SM → 分配 16 个（向上取整到 8 的倍数）；
  - 请求 16 个 SM → 分配 16 个（无需取整）；
  - 请求 17 个 SM → 分配 24 个（向上取整到 8 的倍数）。
- 当请求的 SM 总量超过设备容量（`num_groups * rounded_min_count > total_device_sms`）时抛出
  `RuntimeError`。底层 CUDA 错误码 914（`CUDA_ERROR_INVALID_RESOURCE_TYPE`）与 915
  （`CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`）会被转换为更易读的 `RuntimeError`。
- 实际 SM 数可通过返回资源的 `.sm.smCount` 字段获取。

**示例**

```python
from flashinfer.green_ctx import split_device_green_ctx
import torch

dev = torch.device("cuda:0")
streams, resources = split_device_green_ctx(dev, 2, 16)
print([r.sm.smCount for r in resources])
# [16, 16, 100]
with torch.cuda.stream(streams[0]):
    x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    z = x @ y
    print(z.shape)  # torch.Size([8192, 8192])
```

---

### `split_device_green_ctx_by_sm_count`

```python
split_device_green_ctx_by_sm_count(
    dev: torch.device,
    sm_counts: List[int],
) -> Tuple[List[torch.Stream], List[CUdevResource]]
```

**功能**：把设备划分为多个 green context，每个分组使用指定的 SM 数，返回对应的 `torch.Stream` 与
`CUdevResource`，以及剩余 SM 对应的最后一组。与 `split_device_green_ctx` 的区别在于：这里显式给定
每个分区的 SM 数，而非组数加统一最小值。

**参数**

- `dev`（`torch.device`）：要划分的设备。
- `sm_counts`（`List[int]`）：每个分区期望的 SM 数列表。每个值会被向上取整以满足最小值与对齐要求。

**返回**

- `streams`（`List[torch.Stream]`）：各分区的 `torch.Stream` 列表。
- `resources`（`List[CUdevResource]`）：各分区的 `CUdevResource` 列表。

返回的 `streams` 与 `resources` 长度为 `len(sm_counts) + 1`，最后一个为未被分配的剩余 SM。

**注意事项**

- 当 `sum(rounded_sm_counts) > total_device_sms`、CUDA 资源类型非法（错误码 914）、或资源
  配置非法（错误码 915，例如设备未正确初始化）时抛出 `RuntimeError`。
- 当 `sm_counts` 为空或包含非法值（如负值）时抛出 `ValueError`（`SM count` 必须为正）。
- 实际分配的 SM 数可通过返回资源的 `.sm.smCount` 字段获取。
- Compute Capability 9.0+ 的对齐示例：请求 7 → 8；请求 10 → 16；请求 16 → 16；请求 17 → 24。

**示例**

```python
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import torch

dev = torch.device("cuda:0")
streams, resources = split_device_green_ctx_by_sm_count(dev, [8, 16, 24])
print([r.sm.smCount for r in resources])
# [8, 16, 24, 84]  # 最后一个为剩余 SM

with torch.cuda.stream(streams[0]):
    x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    y = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    z = x @ y
    print(f"Partition 0 result: {z.shape}")

with torch.cuda.stream(streams[1]):
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    c = a @ b
    print(f"Partition 1 result: {c.shape}")
```

## 相关链接 / 参考

- CUDA Driver API：CUDA Green Contexts
  （`https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html`）。
- `flashinfer.green_ctx` 中还有若干未在上游 Sphinx 文档中公开列出的内部辅助函数
  （`get_sm_count_constraint`、`get_cudevice`、`get_device_resource`、`split_resource`、
  `split_resource_by_sm_count`、`create_green_ctx_streams`），本文件不作为公开 API 记录。

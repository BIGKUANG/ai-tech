# Cake FMHA（`flashinfer.cake_fmha`）

> 源码：`flashinfer/cake_fmha.py`、`flashinfer/cake_fmha_request_ordered.py`
> 上游文档：`docs/api/cake_fmha.rst`
> 适用版本：FlashInfer 0.7.2

`cake_fmha` 是 FlashInfer 常规 TensorRT-LLM 分页 FMHA decode / context 契约的 Cake 实现。
它是一个显式的 Blackwell 后端：导入 FlashInfer 或在不指定 backend 的情况下调用既有 API
时，仍会选择既有的 FlashInfer 实现。公开函数支持 B200/GB200（SM100a，CUDA 12.8+）与
B300/GB300（SM103a，CUDA 12.9+），参数与返回值与
`flashinfer.decode.trtllm_batch_decode_with_kv_cache` /
`flashinfer.prefill.trtllm_batch_context_with_kv_cache` 保持一致。

已签入的源码产物包含优化过的 Cake 路由组合（portfolio）、完整域兼容组件，以及 DCP
投机解码扩展。注册表（`csrc/cake_fmha/registry.json`）列出每个组件的源码、公开 C ABI、
启动绑定与基础能力矩阵；DCP 组件在 `csrc/cake_fmha/cuda/dcp_spec/` 下维护自己的注册表。

## 导入

```python
from flashinfer.cake_fmha import (
    cake_batch_decode_with_kv_cache,
    cake_batch_context_with_kv_cache,
    cake_fmha_manifest,
    get_cake_fmha_module,
)
```

也可在既有 API 上以 `backend="cake"` 选择同一实现：

```python
output = flashinfer.trtllm_batch_decode_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_kv_len,
    backend="cake",
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `cake_batch_decode_with_kv_cache` | 函数 | 通过 Cake FMHA 运行 TRTLLM 分页 decode ABI |
| `cake_batch_context_with_kv_cache` | 函数 | 通过 Cake FMHA 运行 TRTLLM 分页 context ABI |
| `cake_fmha_manifest` | 函数 | 返回已认证产物/能力清单的防御性拷贝 |
| `get_cake_fmha_module` | 函数 | 加载已认证的完整域兼容模块 |

## API 详情

### `cake_batch_decode_with_kv_cache`

```python
cake_batch_decode_with_kv_cache(*args, **kwargs)
```

**功能**：通过 Cake FMHA 运行 FlashInfer TRTLLM 分页 decode ABI。参数与返回值与
`flashinfer.trtllm_batch_decode_with_kv_cache` 一致；选择是显式的，永远不会替换
FlashInfer 的默认后端选择。

**参数**

- `*args`：透传给 `flashinfer.decode.trtllm_batch_decode_with_kv_cache` 的位置参数。
- `**kwargs`：透传给同一函数的关键字参数。可包含 `backend`，其值必须为 `"cake"`
  （缺省即 `"cake"`）。

**返回**

- 与 `flashinfer.trtllm_batch_decode_with_kv_cache` 相同（输出张量，或在请求时附带 LSE）。

**注意事项**

- 若显式传入 `backend` 且其值不等于 `"cake"`，抛出
  `ValueError("cake_batch_decode_with_kv_cache requires backend='cake'")`。
- 若向该函数提供 `causal_seqlens_kv_global`，则选择 `cake_fmha_dcp_spec`（分布式上下文并行
  投机解码）profile；普通调用仍选择常规 FMHA。DCP JIT 模块使用精确的 SM100a 或 SM103a
  目标，并与基础包使用相同的构建标签。
- 优化路由是 fail-closed 的：例如优化过的 FP8 decode 仅针对 HND pages、共享 K/V page table
  与 GQA group size 为 8 的情况获得认可；其他合法 FP8 decode 形状仍由 Cake 拥有并使用完整域
  兼容组件。selector miss、路由 workspace 不足、NVFP4 adapter 加载失败都会 fail-closed 回到
  `compat_v1`。

**示例**

```python
import flashinfer
from flashinfer.cake_fmha import cake_batch_decode_with_kv_cache

output = cake_batch_decode_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_kv_len,
)
```

---

### `cake_batch_context_with_kv_cache`

```python
cake_batch_context_with_kv_cache(*args, **kwargs)
```

**功能**：通过 Cake FMHA 运行 FlashInfer TRTLLM 分页 context（prefill）ABI。参数与返回值与
`flashinfer.trtllm_batch_context_with_kv_cache` 一致；选择是显式的，永远不会替换
FlashInfer 的常规默认实现。

**参数**

- `*args`：透传给 `flashinfer.prefill.trtllm_batch_context_with_kv_cache` 的位置参数。
- `**kwargs`：透传给同一函数的关键字参数。可包含 `backend`，其值必须为 `"cake"`
  （缺省即 `"cake"`）。

**返回**

- 与 `flashinfer.trtllm_batch_context_with_kv_cache` 相同（输出张量，或在请求时附带 LSE）。

**注意事项**

- 若显式传入 `backend` 且其值不等于 `"cake"`，抛出
  `ValueError("cake_batch_context_with_kv_cache requires backend='cake'")`。
- BF16 context adapter 还包含两个经过实测的单掩码 profile，用于 batch-four HND 调用：
  uniform q511/KV2047 配 P32 共享 page table，以及 uniform q257/KV1024 配 P1024 分离
  K/V page table。只有在所有语义与形状守卫都匹配，且四个 KV 长度加五个 Q-indptr 值确认了
  精确的 uniform 长度后，FlashInfer 才会选择这些 body；非均匀长度、近似形状或 CUDA Graph 捕获
  都会保留通用 context body，且绝不会复用定长特化。

**示例**

```python
from flashinfer.cake_fmha import cake_batch_context_with_kv_cache

output = cake_batch_context_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens,
    max_q_len, max_kv_len, bmm1_scale, bmm2_scale,
)
```

---

### `cake_fmha_manifest`

```python
cake_fmha_manifest() -> dict[str, Any]
```

**功能**：返回已认证的产物/能力清单（manifest）的一份防御性拷贝（`copy.deepcopy`）。
清单由 `csrc/cake_fmha/registry.json` 提供。

**参数**

- 无。

**返回**

- `dict[str, Any]`：核心注册表内容的深拷贝。

**注意事项**

- 清单中的 per-route 计数是库存元数据，并不证明某个特定的高层 selector 版本能重现被钉定的
  （pinned）能力矩阵。已签入的无分配重放会枚举独立的能力语料（80,768 个原始单元、57,280 个
  有效单元），调用真实的高层 selector，并用规范的 SHA-256
  `d47bf01c2d27409c6a39759d02e30bb9df65e98c353f53d7335081dd26b3f3a8` 认证每个
  case/route 对，要求恰好 1,798 个优化单元与 55,482 个 `compat_v1` 单元。

**示例**

```python
from flashinfer.cake_fmha import cake_fmha_manifest

manifest = cake_fmha_manifest()
print(manifest.keys())
```

---

### `get_cake_fmha_module`

```python
get_cake_fmha_module(device: torch.device)
```

**功能**：加载已认证的完整域兼容模块。

**参数**

- `device`（`torch.device`）：目标 CUDA 设备，用于解析精确的 Blackwell cubin 目标。

**返回**

- 已认证的完整域兼容模块

**注意事项**

- 通过 `_cake_fmha_target` 解析精确的 Blackwell cubin 目标，不做跨架构回退：
  - compute capability `(10, 0)`（B200/GB200）要求 CUDA 12.8 或更新，目标 `sm100a`；
  - compute capability `(10, 3)`（B300/GB300）要求 CUDA 12.9 或更新，目标 `sm103a`；
  - 其他 capability 抛出 `RuntimeError`。

**示例**

```python
import torch
from flashinfer.cake_fmha import get_cake_fmha_module

module = get_cake_fmha_module(torch.device("cuda"))
```

## 相关链接 / 参考

- 路由类与请求排序计划（`CakeFmhaDecodeRoute`、`CakeFmhaContextRoute`、
  `CakeFmhaRequestOrderedDecodePlan`）以及 `cake_fmha_route_is_optimized` 定义在
  `flashinfer/cake_fmha.py` 的模块级 `__all__` 中，但未列于 `docs/api/cake_fmha.rst` 的
  autosummary，故不在此展开。
- 请求排序主机侧辅助函数
  `plan_cake_fmha_request_ordered_paged_decode` 位于
  `flashinfer/cake_fmha_request_ordered.py`，同样未列于该 rst。


# Locality Domain API

- 源码：`csrc/apis/locality_domain.hpp`（C++），`deep_gemm/locality_domain.py`（Python 封装）
- Python 导入：`import deep_gemm`（顶层）或 `import deep_gemm.locality_domain`
- 目标架构：SM100（需 MLOPart 支持）

Locality Domain 把权重按 GPU 的物理内存域（locality domain）切分并存放在就近的内存中，让 SM 从本地域读取数据，降低跨域访问延迟。该特性依赖 MLOPart；不可用时 `is_localization_available()` 返回 `False`，程序仍能正确运行（只是不做本地化）。

## 函数总览

| 函数 | 说明 |
| --- | --- |
| `get_num_locality_domains()` | 本地域数量 |
| `get_granularity()` | 分配粒度（字节） |
| `is_localization_available()` | 本地化是否可用（MLOPart） |
| `empty(sizes, dtype, domain_idx)` | 在指定域内分配张量 |
| `empty_per_domain(sizes, dtype)` | 每个域分配一份，返回 `(num_domains, *sizes)` |
| `is_localized(t)` | 判断张量是否本地化 |
| `get_sm_locality_domains()` | 每个 SM 所属的域 |
| `get_balanced_sm_locality_domains()` | 均衡后的 SM→域 映射 |
| `release_mlopart()` | 释放 MLOPart 分配器 |
| `localize(t, dim=-2)` | 把权重分布到各本地域 |
| `destroy_localizer()` | 退出 MLOPart 分配器 |

---

## get_num_locality_domains

```python
get_num_locality_domains() -> int
```

返回 GPU 的本地域数量。

## get_granularity

```python
get_granularity() -> int
```

返回分配器的字节粒度。

## is_localization_available

```python
is_localization_available() -> bool
```

返回 MLOPart 是否可用（`mlopart::is_available()`）。

---

## empty

```python
empty(sizes, dtype, domain_idx) -> torch.Tensor
```

等价于 `torch.empty(sizes, dtype=dtype)`，但把内存映射到本地域 `domain_idx`。返回的张量通过 `torch.from_blob` 构造，销毁时回调 `release`。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `sizes` | `IntArrayRef` | 形状 |
| `dtype` | `torch.dtype` | 元素类型 |
| `domain_idx` | int | 本地域索引 |

## empty_per_domain

```python
empty_per_domain(sizes, dtype) -> torch.Tensor
```

等价于 `torch.empty((num_domains, *sizes), dtype=dtype)`，切片 `d` 位于域 `d` 中（各切片 stride 已对齐到粒度）。

---

## is_localized

```python
is_localized(t) -> bool
```

判断张量 `t` 是否已本地化（即形状首维等于 `num_domains`，且每片确实映射到对应域）。若未本地化，则断言 `t` 连续并返回 `False`。

## get_sm_locality_domains

```python
get_sm_locality_domains() -> torch.Tensor
```

返回长度为 SM 数、dtype `uint8` 的张量，给出每个 SM 所属的域。结果缓存于 runtime。本地化不可用时返回均匀分配的映射。

实现细节：通过微基准探测（`probe`）每个域的访问延迟，重复最多 5 次，要求近/远域延迟比达到阈值且同一 TPC 内的两个 SM 域一致；否则回退到均匀映射并打印 warning。

## get_balanced_sm_locality_domains

```python
get_balanced_sm_locality_domains() -> torch.Tensor
```

在 `get_sm_locality_domains()` 基础上做域间负载均衡（把多余的一侧 TPC 翻转到另一域，使两域 TPC 数相等），结果缓存。Mega MoE 内核使用该映射。

## release_mlopart

```python
release_mlopart() -> None
```

释放 MLOPart。实现上会先强制解析一次 `get_balanced_sm_locality_domains()`。

---

## localize

```python
localize(t, dim=-2) -> torch.Tensor
```

把权重分布到各本地域：

```
(*B, N, K) -> (num_domains, *B, N / num_domains, K)
```

沿 `dim`（默认 `-2`）把张量切成 `num_domains` 份，分别分配到对应域，并在最前面新增一个域维。要求 `t.size(dim) % num_domains == 0`。

**典型用法**（Mega MoE 权重本地化）：

```python
transformed_l1 = (deep_gemm.localize(transformed_l1[0]), transformed_l1[1])
transformed_l2 = (deep_gemm.localize(transformed_l2[0]), transformed_l2[1])
deep_gemm.destroy_localizer()
```

> `localize` 只移动权重数据；对应的缩放因子（SF）张量不参与本地化（保持原布局）。`is_localized(weight) == is_localized(weight)` 在 Mega MoE 中用于校验 routed 与 shared 权重的本地化状态一致。

## destroy_localizer

```python
destroy_localizer() -> None
```

退出由 MLOPart 支撑的分配器。已分配的张量保持有效。

## 相关

- Mega MoE 权重本地化：[api-mega-moe.md](./api-mega-moe.md)
- 测试参考：`tests/test_locality_domain.py`

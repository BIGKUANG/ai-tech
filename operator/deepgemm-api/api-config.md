# 运行时配置 API

- 源码：`csrc/apis/config.hpp`
- Python 导入：`import deep_gemm`
- 适用范围：全局

本组函数配置 DeepGEMM 的运行时行为（SM 数、TC 利用率、PDL、确定性算法、JIT 编译维度等）。

## 函数总览

| 函数 | 说明 |
| --- | --- |
| `set_num_sms(new_num_sms)` / `get_num_sms()` | 设置 / 获取使用的最大 SM 数 |
| `set_tc_util(new_tc_util)` / `get_tc_util()` | 设置 / 获取近似的张量核心利用率（百分比） |
| `set_pdl(new_enable_pdl)` / `get_pdl()` | 启用 / 禁用 Programmatic Dependent Launch |
| `use_deterministic_algorithms(enabled)` | 启用 / 禁用确定性算法 |
| `set_ignore_compile_dims(new_value)` | 配置 JIT 编译时忽略维度 |
| `set_block_size_multiple_of(new_value)` | 约束 block size 为给定值的倍数 |
| `shutdown()` | 释放运行时资源（atexit 注册） |

---

## set_num_sms / get_num_sms

```python
set_num_sms(new_num_sms: int)
get_num_sms() -> int
```

设置 / 获取内核网格使用的最大 SM 数。默认使用设备全部 SM。常用于多进程 / MPS 场景下限制并发。

```python
deep_gemm.set_num_sms(64)
print(deep_gemm.get_num_sms())
```

## set_tc_util / get_tc_util

```python
set_tc_util(new_tc_util: int)
get_tc_util() -> int
```

设置 / 获取近似的张量核心利用率（百分比，整数）。用于 heuristic 估算。

## set_pdl / get_pdl

```python
set_pdl(new_enable_pdl: bool)
get_pdl() -> bool
```

启用 / 禁用 Programmatic Dependent Launch（PDL）。PDL 允许后续内核在前序内核尚未完全结束时开始启动，降低内核间空隙。

## use_deterministic_algorithms

```python
use_deterministic_algorithms(enabled: bool)
```

启用 / 禁用确定性算法。启用后：

- BF16 GEMM 不走 cuBLASLt（cuBLAS 的 split-K 归约顺序不保证确定）。
- Mega mHC 的 Split-K 固定为 16。
- Mega Gate 不做 K 切分（batch invariance）。
- 多次调用同一输入的输出逐位一致。

注意：确定性通常更慢。

```python
deep_gemm.use_deterministic_algorithms(True)
deep_gemm.bf16_gemm_nt(a, b, d)
deep_gemm.use_deterministic_algorithms(False)
```

## set_ignore_compile_dims

```python
set_ignore_compile_dims(new_value: bool)
```

配置 JIT 编译时是否忽略维度（不把维度作为编译期常量）。可减少编译出的内核数量、加快编译，代价是运行时性能可能下降。

## set_block_size_multiple_of

```python
set_block_size_multiple_of(new_value)
```

约束 tile 的 block size 为给定值的倍数。`new_value` 可以是：

- `int x`：`block_m` 与 `block_n` 都必须是 `x` 的倍数；
- `(x, y)` 元组：`block_m` 是 `x` 的倍数、`block_n` 是 `y` 的倍数。

用于对齐到特定硬件 / 内存布局要求。

## shutdown

```python
shutdown()
```

释放运行时资源（重置 runtime 与 cuBLASLt 句柄）。DeepGEMM 在 import 时通过 `atexit.register(shutdown)` 注册，一般无需手动调用。Python 侧封装为 `deep_gemm.shutdown`。

## 内部：init

```python
deep_gemm._C.init(library_root_path: str)
```

初始化 JIT（传入库根目录）。由 `deep_gemm/__init__.py` 在导入时自动调用，用户一般无需关心。

## 相关

- 连续布局 M/K 对齐（另一组全局配置）：[api-layout.md](./api-layout.md)
- 环境变量列表：[README.md](./README.md#环境变量)

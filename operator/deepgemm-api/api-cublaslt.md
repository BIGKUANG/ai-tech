# cuBLASLt 算子 API

- 源码：`csrc/apis/gemm.hpp`（`cublaslt_gemm_*`、`cublaslt_nvfp4_gemm_nt`、`batched_syrk`、`batched_symm`），底层 `csrc/jit_kernels/impls/smxx_cublaslt.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM90 / SM100

DeepGEMM 在内部式中会优先复用 cuBLASLt（尤其 BF16 GEMM）。这些函数是对 cuBLASLt 的薄封装，供用户显式调用或与 DeepGEMM 内核对比基准。

## 函数总览

| 函数 | 计算 | 说明 |
| --- | --- | --- |
| `cublaslt_gemm_nt` | `D = C + A @ B.T` | 通用 GEMM（支持 FP32/BF16/FP16） |
| `cublaslt_gemm_nn` | `D = C + A @ B` | — |
| `cublaslt_gemm_tn` | `D = C + A.T @ B` | — |
| `cublaslt_gemm_tt` | `D = C + A.T @ B.T` | — |
| `cublaslt_nvfp4_gemm_nt` | `D = C + A @ B.T` | NVFP4 打包 FP4 + UE4M3 SF |
| `batched_syrk` | `D = A @ A.T` | 对称秩-k 更新（批量/非批量） |
| `batched_symm` | `D = A @ B` | A 按契约对称（批量/非批量） |

---

## cublaslt_gemm_nt / nn / tn / tt

```python
cublaslt_gemm_nt(a, b, d, c=None)
cublaslt_gemm_nn(a, b, d, c=None)
cublaslt_gemm_tn(a, b, d, c=None)
cublaslt_gemm_tt(a, b, d, c=None)
```

形状遵循 `D = C + A @ B` 约定：`nt` 为 `A[M, K] @ B[N, K].T`。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `[M, K]` 或按后缀转置 | 输入 |
| `b` | `[N, K]` 或按后缀转置 | 输入 |
| `d` | `[M, N]` | 输出 |
| `c` | `[M, N]`，可选 | 累加输入 |

**返回**：无返回值，结果写入 `d`。

**说明**

- `tn` 会传入 `a.transpose(0, 1)`；`nn` 传入 `b.transpose(0, 1)`；`tt` 同时转置 A、B；`nn`/`tn` 的 `BIAS` 语义由 cuBLASLt 处理。
- 当 `m`、`n` 或 `k` 为 0 时直接返回（`early_return`）。
- 一般不建议直接调用，除非需要与 DeepGEMM 内核做基准对比；普通场景请用 [api-gemm.md](./api-gemm.md)。

---

## cublaslt_nvfp4_gemm_nt

```python
cublaslt_nvfp4_gemm_nt(a, b, d, c=None)
```

`D = C + A @ B.T`，A、B 均为打包 FP4（K-major）并配 NVFP4 recipe 的缩放因子。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `(a_data, sfa)` | `a_data` 形状 `[M, K/2]`（逻辑 FP4 元素数 `K`，每字节 2 个），`torch.int8`，连续 |
| `b` | `(b_data, sfb)` | `b_data` 形状 `[N, K/2]`，`torch.int8`，连续 |
| `d` | `[M, N]` | 输出 |
| `c` | `[M, N]`，可选 | 累加输入 |

**SF 契约**（与 DeepGEMM 自己的 FP8/FP4 GEMM 不同）

- 缩放因子必须是 **UE4M3 字节**（NVFP4 recipe，16 元素块）。
- 布局为 cuBLASLt tiled layout：`[ceil_div(mn, 128), ceil_div(k, 64), 32, 4, 4]`（以字节为单位），其中 `mn` 对 A 为 `M`、对 B 为 `N`。
- must be contiguous；`sf.numel() == ceil_div(mn, 128) * ceil_div(k, 64) * 512`。
- dtype 为 `torch.uint8` 或 `torch.int8`；`k % 32 == 0`。

**示例**

```python
# a_packed: [M, K//2] int8; sfa: cuBLASLt tiled UE4M3 字节布局
# deep_gemm.cublaslt_nvfp4_gemm_nt((a_packed, sfa), (b_packed, sfb), d)
```

---

## batched_syrk

```python
batched_syrk(a, d)
```

计算 `D = A @ A.T`。A、D 可为二维张量，也可为三维批量张量（批量在 `dim 0`）。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `[M, K]` 或 `[B, M, K]` | 输入 |
| `d` | `[M, M]` 或 `[B, M, M]` | 输出（对称） |

**约束**

- `a` 与 `d` 的 dtype 相同。
- 批量维度、`m` 必须一致，`d` 为方阵 `M x M`。
- `num_batches == 0` 或 `m == 0` 时直接返回。

**示例**

```python
import torch, deep_gemm

a = torch.randn(2, 512, 256, dtype=torch.bfloat16, device='cuda')
d = torch.empty(2, 512, 512, dtype=torch.bfloat16, device='cuda')
deep_gemm.batched_syrk(a, d)
```

---

## batched_symm

```python
batched_symm(a, b, d)
```

计算 `D = A @ B`，其中 A 按契约对称（cuBLASLt 以 strided-batched GEMM 执行）。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `[M, M]` 或 `[B, M, M]` | 对称输入 |
| `b` | `[M, K]` 或 `[B, M, K]` | 输入 |
| `d` | `[M, K]` 或 `[B, M, K]` | 输出 |

**约束**：批量维度一致；`a` 为 `M x M`；dtype 三者相同。

**示例**

```python
import torch, deep_gemm

a = torch.randn(512, 512, dtype=torch.bfloat16, device='cuda')
a = a + a.mT          # 保证对称（契约）
b = torch.randn(512, 1024, dtype=torch.bfloat16, device='cuda')
d = torch.empty(512, 1024, dtype=torch.bfloat16, device='cuda')
deep_gemm.batched_symm(a, b, d)
```

---

## 相关

- 稠密 GEMM：[api-gemm.md](./api-gemm.md)
- 分组 GEMM：[api-gemm-grouped.md](./api-gemm-grouped.md)

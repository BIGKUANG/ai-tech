# 稠密 GEMM API

- 源码：`csrc/apis/gemm.hpp`
- 注册：`csrc/python_api.cpp` → `deep_gemm::gemm::register_apis`
- Python 导入：`import deep_gemm`（直接访问顶层函数）
- 目标架构：SM90 / SM100

本文档介绍 DeepGEMM 的稠密（非分组）GEMM 算子。命名约定为 `D = C + A @ B`，后缀 `nt/nn/tn/tt` 表示 A、B 的内存布局。所有算子仅支持 `D` 的最后一维连续（N-major）。

## 函数总览

| 函数 | 计算 | A 布局 | B 布局 | 架构 |
| --- | --- | --- | --- | --- |
| `fp8_fp4_gemm_nt` | `D = C + A @ B.T` | `[M, K]` | `[N, K]` | SM90/SM100 |
| `fp8_fp4_gemm_nn` | `D = C + A @ B` | `[M, K]` | `[K, N]` | SM100 |
| `fp8_fp4_gemm_tn` | `D = C + A.T @ B` | `[K, M]` | `[K, N]` | SM100 |
| `fp8_fp4_gemm_tt` | `D = C + A.T @ B.T` | `[K, M]` | `[N, K]` | SM100 |
| `fp8_gemm_nt/nn/tn/tt` | 同 `fp8_fp4_gemm_*` | — | — | 别名 |
| `fp4_gemm_nt` | `D = C + A @ B.T` | `[M, K]` | `[N, K]` | 别名 |
| `bf16_gemm_nt` | `D = C + A @ B.T` | `[M, K]` | `[N, K]` | SM90/SM100 |
| `bf16_gemm_nn` | `D = C + A @ B` | `[M, K]` | `[K, N]` | SM100 |
| `bf16_gemm_tn` | `D = C + A.T @ B` | `[K, M]` | `[K, N]` | SM100 |
| `bf16_gemm_tt` | `D = C + A.T @ B.T` | `[K, M]` | `[N, K]` | SM100 |

> `fp8_fp4_gemm_*` 是统一的 FP8/FP4 入口，可接受任意 FP8/FP4 操作数组合。别名函数（`fp8_gemm_nt`、`fp4_gemm_nt` 等）与对应统一函数是同一个对象（`m.attr(...)` 绑定）。例如 `fp4_gemm_nt` ≡ `fp8_fp4_gemm_nt`，`fp8_gemm_nt` ≡ `fp8_fp4_gemm_nt`。

常见公共参数：

- `recipe` / `recipe_a` / `recipe_b`：量化 recipe，描述缩放因子的存储粒度。`recipe` 为三元组 `(gran_m, gran_n, gran_k)`，`recipe_a` / `recipe_b` 为二元组 `(gran_mn, gran_k)`。三者互斥：要么传 `recipe`，要么同时传 `recipe_a` + `recipe_b`。省略时按数据类型推导默认 recipe（SM100 的 FP8 默认 `(1, 1, 128)`，SM90 默认 `(1, 128, 128)`）。
- `compiled_dims`：字符串，控制 JIT 编译期把哪些维度视为常量。取值 `"nk"`（默认）、`"mn"` 或 `""`。
- `disable_ue8m0_cast`：SM100 上把 FP32 缩放因子直接做 MN-major 转置而不转 UE8M0（用于 1D2D/legacy 格式测试）。SM90 忽略。
- `alpha`：`float`，epilogue 缩放系数（仅 SM100，且与 `epilogue` 互斥）。
- `epilogue`：`deep_gemm.epilogue.*` 实例（仅 SM100，见 [api-epilogue.md](./api-epilogue.md)）。
- `c`：可选累加输入。若省略，输出从 0 开始累加；若与 `d` 为同一张量，内核原地累加；否则先 `d.copy_(c)`。

---

## FP8/FP4 GEMM

### fp8_fp4_gemm_nt

```python
fp8_fp4_gemm_nt(
    a, b, d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    alpha=None,
    epilogue=None,
)
```

`D = C + A @ B.T`。A、B 均为 `(data, scale_factor)` 元组。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `(a_data, sfa)` | `a_data` 形状 `[M, K]`（逻辑元素数；FP4 时底层按字节存储 `[M, K/2]`，K-major），FP8 = `torch.float8_e4m3fn`，FP4 = 打包字节（`torch.int8`，每字节 2 个 FP4） |
| `b` | `(b_data, sfb)` | `b_data` 形状 `[N, K]`（同上） |
| `d` | `[M, N]` | 输出，`torch.bfloat16` 或 `torch.float`（也支持 `float8_e4m3fn`，此时必须配 `epilogue=FP8Quantization` 或 `(d, sfd)` 输出对） |
| `c` | `[M, N]`，可选 | 累加输入，须与 `d` 同 dtype、同布局 |
| `recipe` | `(gran_m, gran_n, gran_k)`，可选 | 量化 recipe |
| `recipe_a` / `recipe_b` | `(gran_mn, gran_k)`，可选 | A/B 各自的 recipe（必须成对提供） |
| `disable_ue8m0_cast` | bool | 见上 |
| `alpha` | float，可选 | epilogue 缩放系数（SM100） |
| `epilogue` | `EpilogueClass`，可选 | 自定义 epilogue（SM100） |

**布局/类型检查**

- SM90 仅支持 K-major A 和 B（`fp8_fp4_requires_k_major`）。SM100 上若 A、B 均为打包 FP4，则要求 K-major。
- `d` 必须 N-major（`stride(-1) == 1`）。
- `d` 为 FP8 时必须同时提供 `SFD`（`epilogue` 或 `(d, sfd)` 输出对）；无 SFD 的内核不接受 FP8 `D`。

**返回**

无返回值，结果写入 `d`（以及可能的 `sfd`）。`k == 0` 时执行退化路径：`d = c`（或清零），`sfd` 清零。

**示例**

```python
import torch, deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

m, n, k = 4096, 4096, 4096
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')

a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=True, use_packed_ue8m0=True)
b_fp8, sfb = per_token_cast_to_fp8(b, use_ue8m0=True, use_packed_ue8m0=True)
sfa = deep_gemm.transform_sf_into_required_layout(sfa, m, k, (1, 128))
sfb = deep_gemm.transform_sf_into_required_layout(sfb, n, k, (1, 128))

d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d, recipe=(1, 1, 128))
```

### fp8_fp4_gemm_nn / fp8_fp4_gemm_tn / fp8_fp4_gemm_tt

```python
fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                compiled_dims="nk", disable_ue8m0_cast=False, alpha=None, epilogue=None)
fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                compiled_dims="mn", disable_ue8m0_cast=False, alpha=None, epilogue=None)
fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                compiled_dims="mn", disable_ue8m0_cast=False, alpha=None, epilogue=None)
```

三者都是 `fp8_fp4_gemm_nt` 的转置包装：

- `nn`：`nt(a, b.transpose(0, 1), ...)`
- `tn`：`nt(a.transpose(0, 1), b.transpose(0, 1), ...)`
- `tt`：`nt(a.transpose(0, 1), b, ...)`

仅 SM100 支持 `nn/tn/tt`；SM90 只支持 NT。

---

## BF16 GEMM

### bf16_gemm_nt

```python
bf16_gemm_nt(
    a, b, d,
    c=None,
    compiled_dims="nk",
    alpha=None,
    epilogue=None,
)
```

`D = C + A @ B.T`，A、B 为纯 BF16 张量（无缩放因子）。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `[M, K]`，`torch.bfloat16` | 查询矩阵 |
| `b` | `[N, K]`，`torch.bfloat16` | 权重矩阵 |
| `d` | `[M, N]`，`torch.bfloat16` 或 `torch.float` | 输出 |
| `c` | `[M, N]`，可选 | 累加输入，dtype 须与 `d` 一致 |
| `compiled_dims` | str | 默认 `"nk"` |
| `alpha` | float，可选 | epilogue 缩放（SM100） |
| `epilogue` | `EpilogueClass`，可选 | 自定义 epilogue（SM100） |

**行为说明**

- 关闭确定性算法且 cuBLASLt 可用、且只涉及普通 store（可选 alpha 缩放）时，内部会走 cuBLASLt 路径；否则走 DeepGEMM 内核。
- SM90：不支持 `alpha` 与自定义 `epilogue`。
- `epilogue` 为 `BF16StochasticRounding` 时要求 `d` 为 `torch.bfloat16` 且无 `c`。

**示例**

```python
import torch, deep_gemm

a = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
b = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
d = torch.empty((4096, 4096), dtype=torch.bfloat16, device='cuda')

deep_gemm.bf16_gemm_nt(a, b, d)
```

### bf16_gemm_nn / bf16_gemm_tn / bf16_gemm_tt

```python
bf16_gemm_nn(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None)
bf16_gemm_tn(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None)
bf16_gemm_tt(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None)
```

转置包装，均为 `bf16_gemm_nt` 的变形：

- `nn`：`bf16_gemm_nt(a, b.transpose(0, 1), ...)`
- `tn`：`bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), ...)`
- `tt`：`bf16_gemm_nt(a.transpose(0, 1), b, ...)`

仅 SM100 支持 `nn/tn/tt`。

---

## Epilogue 与输出量化

SM100 的 GEMM 支持单个 epilogue 类（不可叠加），见 [api-epilogue.md](./api-epilogue.md)：

- `deep_gemm.epilogue.Alpha(alpha)`：`D = alpha * (C + A@B)`。
- `deep_gemm.epilogue.FP8Quantization(sfd)`：把累加器量化为 FP8 E4M3，`sfd` 为动态 per-32 打包 UE8M0 缩放因子。
- `deep_gemm.epilogue.BF16StochasticRounding()`：用随机舍入把累加器写入 BF16（要求直接 BF16 输出）。
- `deep_gemm.epilogue.Identity()`：普通存储。

ISO FP8 输出也可写作 `d` 传 `(d_fp8, sfd)` 对（`fp8_einsum`）。

## 相关

- 分组 GEMM：[api-gemm-grouped.md](./api-gemm-grouped.md)
- 缩放因子布局规范：[scaling-factor-format.md](./scaling-factor-format.md)
- 布局变换 helper：[api-layout.md](./api-layout.md)
- cuBLASLt 包装：[api-cublaslt.md](./api-cublaslt.md)

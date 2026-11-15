# Epilogue API

- 源码：`csrc/apis/epilogue_class.hpp`（用户可见类），`csrc/jit_kernels/impls/epilogue_class.hpp`（实现）
- Python 导入：`deep_gemm.epilogue`
- 目标架构：SM100

GEMM 的 epilogue 类用于选择设备端 `epilogue::operators` 的实现，携带其运行时状态，并校验 GEMM 输出。**每个 SM100 GEMM 只使用一个 epilogue 类，不能叠加。**

## 类总览

| 类 | 构造 | 语义 | 约束 |
| --- | --- | --- | --- |
| `epilogue.Identity` | `Identity()` | 普通存储 `D = C + A@B` | — |
| `epilogue.Alpha` | `Alpha(alpha)` | `D = alpha * (C + A@B)` | 与 cuBLASLt 兼容（`get_alpha`） |
| `epilogue.FP8Quantization` | `FP8Quantization(sfd)` | 量化累加器为 FP8 E4M3，写 `sfd` | 要求直接 FP8 `d`、无 `c` |
| `epilogue.BF16StochasticRounding` | `BF16StochasticRounding()` | 随机舍入写入 BF16 | 要求直接 BF16 `d`、无 `c`；编译目标为 `sm_100a` |

## 参数互斥关系

GEMM 的 `epilogue` 参数、独立 `alpha` 参数、以及 FP8 `(d, sfd)` 输出对三者互斥（最多提供一个）。内部 `resolve_epilogue_class` 会：

1. 若提供 `epilogue`，使用之；
2. 否则若提供 `alpha`，构造 `AlphaEpilogue`；
3. 否则若提供 `sfd`，构造 `FP8QuantizationEpilogue`；
4. 都没有则使用 `IdentityEpilogue`。
5. 最后调用 `check(c, d)` 校验 GEMM 输出。

---

## epilogue.Identity

```python
deep_gemm.epilogue.Identity()
```

普通存储。`get_alpha()` 返回 `1.0`，因此内部可在 cuBLASLt 可用时走 cuBLASLt 直算。

---

## epilogue.Alpha

```python
deep_gemm.epilogue.Alpha(alpha: float)
```

`D = alpha * (C + A@B)`。对应 `epilogue::operators::ScaleByAlpha`，`get_alpha()` 返回 `alpha`。

GEMM 函数也提供等价的顶层 `alpha` 参数（与 `epilogue` 互斥），两者语义相同：

```python
deep_gemm.bf16_gemm_nt(a, b, d, alpha=0.5)
# 等价于
deep_gemm.bf16_gemm_nt(a, b, d, epilogue=deep_gemm.epilogue.Alpha(0.5))
```

---

## epilogue.FP8Quantization

```python
deep_gemm.epilogue.FP8Quantization(sfd)
```

把累加器（FP32）量化为 FP8 E4M3 输出，并将动态 per-32 打包 UE8M0 缩放因子写入 `sfd`。对应 `epilogue::operators::QuantizeToFP8`。

**参数**：`sfd`（`torch.int32`，打包 UE8M0）。

**约束**：`check` 要求无 `c`，且 `d.scalar_type() == torch.float8_e4m3fn`。

**示例**

```python
import torch, deep_gemm
from deep_gemm.utils import align, ceil_div, cast_back_from_fp8

m, n = 4096, 4096
d_fp8 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device='cuda')
sfd = torch.empty_strided((m, ceil_div(n, 32 * 4)), (1, align(m, 4)),
                          dtype=torch.int32, device='cuda')
deep_gemm.bf16_gemm_nt(a, b, d_fp8, epilogue=deep_gemm.epilogue.FP8Quantization(sfd))
```

FP8 输出对是同一 epilogue 的另一种写法（目前仅 `fp8_einsum` 的 `'bhr,hdr->bhd'` 接受 `(d, sfd)` 输出对）。

---

## epilogue.BF16StochasticRounding

```python
deep_gemm.epilogue.BF16StochasticRounding()
```

用随机舍入把 FP32 累加器写入 BF16 输出，避免确定性的舍入偏差。对应 `epilogue::operators::StochasticRoundToBF16`。

**约束**：`check` 要求无 `c`，且 `d.scalar_type() == torch.bfloat16`。编译目标是架构特定的 `sm_100a`（`cvt.rs` 要求具体架构，而非 family）。

**示例**

```python
import torch, deep_gemm

d_sr = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
deep_gemm.bf16_gemm_nt(a, b, d_sr, epilogue=deep_gemm.epilogue.BF16StochasticRounding())
```

## 相关

- 稠密 GEMM 的 `alpha` / `epilogue` 参数：[api-gemm.md](./api-gemm.md)
- FP8 输出量化：[scaling-factor-format.md](./scaling-factor-format.md)
- 测试参考：`tests/test_fp8_fp4.py`、`tests/test_bf16.py`

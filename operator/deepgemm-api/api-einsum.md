# Einsum API

- 源码：`csrc/apis/einsum.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM90 / SM100

DeepGEMM 提供少量**硬编码**的 Einstein 求和表达式（不支持任意表达式，也不做表达式规范化）。每个表达式对应一个针对注意力 / 融合场景优化的专用内核。

## 函数总览

| 函数 | 数据类型 | 说明 |
| --- | --- | --- |
| `einsum(expr, a, b, d, c=None, epilogue=None)` | BF16 | BF16 专用表达式 |
| `fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None)` | FP8 | FP8 专用表达式，支持 FP8 输出 |

支持的表达式：

| 表达式 | 计算 | 架构 | `d` 类型 | 支持 epilogue |
| --- | --- | --- | --- | --- |
| `"bmk,bnk->mn"` | `D += sum_b A[b,m,k] @ B[b,n,k]` | SM90/SM100 | FP32 或 BF16 | 否 |
| `"bhr,hdr->bhd"` | `D[b,h,d] = sum_r A[b,h,r] @ B[h,d,r]` | SM90/SM100 | BF16/FP32 | 是（SM100） |
| `"bhd,hdr->bhr"` | `D[b,h,r] = sum_d A[b,h,d] @ B[h,d,r]` | SM90/SM100 | BF16/FP32 | 是（SM100） |
| `"bhd,bhr->hdr"` | `D[h,d,r] += sum_b A[b,h,d] @ B[b,h,r]` | SM90/SM100 | FP32 | 否 |

`fp8_einsum` 额外支持（SM100）：

| 表达式 | `d` 类型 | 说明 |
| --- | --- | --- |
| `"bhr,hdr->bhd"` | `(d_fp8, sfd)` 对或 BF16 | 唯一支持 FP8 输出对的表达式 |
| `"bhd,hdr->bhr"` | BF16 | 仅 SM100 |
| `"bhd,bhr->hdr"` | FP32 | 仅 SM100 |

> SF 布局与 permute 细节：`fp8_einsum` 会**在内部同时 permute 操作数与其缩放因子**，见 [scaling-factor-format.md](./scaling-factor-format.md) 6.2。

---

## einsum

```python
einsum(expr, a, b, d, c=None, epilogue=None)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `expr` | str | 上面列出的表达式之一，其它值会触发 `DG_HOST_UNREACHABLE` |
| `a` | `torch.bfloat16` | 输入，形状随表达式 |
| `b` | `torch.bfloat16` | 输入，形状随表达式 |
| `d` | 见上表 | 输出 |
| `c` | 可选 | 累加输入 |
| `epilogue` | `EpilogueClass`，可选 | 仅 `"bhr,hdr->bhd"` 与 `"bhd,hdr->bhr"` 支持（SM100） |

**各表达式约束**

- `"bmk,bnk->mn"`：
  - `a`：`[s, m, k]`，`b`：`[s, n, k]`。
  - `d` 为 FP32 时要求 `c`（FP32）非空且与 `d` 为同一张量（累加表达式）。
  - `d` 为 BF16 时要求 `c` 为空；内部用 FP32 workspace 累加后拷贝回 BF16。
  - `a`、`b`、`d` 必须连续；不支持 epilogue。
- `"bhr,hdr->bhd"`：`a`：`[b, h, r]`，`b`：`[h, d, r]`，`d`：`[b, h, d]`。要求 `a.stride(2) == 1`、`b.stride(2) == 1`、`d.stride(2) == 1`。
- `"bhd,hdr->bhr"`：`a`：`[b, h, d]`，`b`：`[h, d, r]`，`d`：`[b, h, r]`。同样要求最后一维连续。
- `"bhd,bhr->hdr"`：`a`：`[b, h, d]`，`b`：`[b, h, r]`，`d`：`[h, d, r]`（FP32）。`c` 可选，须与 `d` 同 dtype、同形状、`stride(2) == 1`。内部走 cuBLASLt。

**示例**

```python
import torch, deep_gemm

# bhr,hdr->bhd
b, h, r, d = 32, 128, 512, 128
x = torch.randn((b, h, r), dtype=torch.bfloat16, device='cuda')
y = torch.randn((h, d, r), dtype=torch.bfloat16, device='cuda')
z = torch.empty((b, h, d), dtype=torch.bfloat16, device='cuda')
deep_gemm.einsum('bhr,hdr->bhd', x, y, z)

# bmk,bnk->mn，累加式
s, m, n, k = 4096, 256, 256, 256
a = torch.randn((s, m, k), dtype=torch.bfloat16, device='cuda')
bb = torch.randn((s, n, k), dtype=torch.bfloat16, device='cuda')
dd = torch.randn((m, n), dtype=torch.float, device='cuda')
deep_gemm.einsum('bmk,bnk->mn', a, bb, dd, c=dd)
```

---

## fp8_einsum

```python
fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `expr` | str | `"bhr,hdr->bhd"`、`"bhd,hdr->bhr"`（SM100）、`"bhd,bhr->hdr"`（SM100） |
| `a` | `(a_data, sfa)` | FP8 数据 + 缩放因子 |
| `b` | `(b_data, sfb)` | FP8 数据 + 缩放因子 |
| `d` | `torch.Tensor`，或对 `"bhr,hdr->bhd"` 为 `(d, sfd)` 对 | 输出。FP8 输出对表示动态 per-32 打包 UE8M0 量化的 FP8 输出 |
| `c` | 可选 | 累加输入 |
| `recipe` | `(gran_m, gran_n, gran_k)` | 默认 `(1, 128, 128)` |
| `epilogue` | `EpilogueClass`，可选 | FP8 输出对与其它 epilogue 互斥 |

**FP8 输出对**

仅 `"bhr,hdr->bhd"` 接受 `d = (d_fp8, sfd)`：

- `d_fp8`：`torch.float8_e4m3fn`。
- `sfd`：`int32` 打包 UE8M0，其 SF 列在 D 的最后两维上展平（等价于对 `d.flatten(1, 2)` 做 per-token cast）。
- 要求 `arch_major == 10`、`d_fp8.stride(0) == n`、`n % 128 == 0`。
- 与生产管线（先 BF16 D 再 per-token cast）逐位对齐。

**Permute 说明**

`fp8_einsum` 内部把操作数与 SF 一同 permute 到 `(batch, m, n, k)`：

| 表达式 | `(batch, m, n, k)` | SFA permute | SFB permute |
| --- | --- | --- | --- |
| `'bhr,hdr->bhd'` | `(h, b, d, r)` | `(1, 0, 2)` | 无 |
| `'bhd,hdr->bhr'` | `(h, b, r, d)` | `(1, 0, 2)` | `(0, 2, 1)` |
| `'bhd,bhr->hdr'` | `(h, d, r, b)` | `(1, 2, 0)` | `(1, 2, 0)` |

预转换的 `int32` SF 必须按 **post-permute** 坐标构造，详见 [scaling-factor-format.md](./scaling-factor-format.md) 6.2。

**示例**

```python
import torch, deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8, align, ceil_div

b, h, r, d = 128, 8, 4096, 1024
x = torch.randn((b, h, r), dtype=torch.bfloat16, device='cuda')
y = torch.randn((h, d, r), dtype=torch.bfloat16, device='cuda')

x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True)
x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
         torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), dtype=torch.float, device='cuda'))
for i in range(h):
    y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

z = torch.empty((b, h, d), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z)
```

## 相关

- Epilogue：[api-epilogue.md](./api-epilogue.md)
- 缩放因子布局：[scaling-factor-format.md](./scaling-factor-format.md)

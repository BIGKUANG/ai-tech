# HyperConnection 预归一化 GEMM API

- 源码：`csrc/apis/hyperconnection.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM90 / SM100

## tf32_hc_prenorm_gemm

```python
tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None)
```

HyperConnection（HC）预归一化 GEMM：`D = A @ B.T`，同时输出 A 每行的平方和 `sqr_sum`，用于后续的归一化。

**输入 / 输出**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `[m, k]`，`torch.bfloat16` | 输入，必须 K-major（`stride(-1) == 1`） |
| `b` | `[n, k]`，`torch.float` | 输入，必须 K-major |
| `d` | `[m, n]`，`torch.float`；或 `[num_splits, m, n]` | 输出，必须 N-major |
| `sqr_sum` | `[m]`，`torch.float`；或 `[num_splits, m]` | A 每行的平方和，连续 |
| `num_splits` | int，可选 | 为 `None` 时不切分；给出时要求 `>= 1`，`d` 与 `sqr_sum` 增加首维 |

**行为说明**

- `a`、`b` 必须 K-major；`d` 必须 N-major；`sqr_sum` 必须连续。
- `n > 0`、`k > 0`；`m == 0` 时直接返回。
- `num_splits` 给定时，`d` 形状须为 `[num_splits, m, n]`、`sqr_sum` 形状须为 `[num_splits, m]`。内核沿 K 轴切分累加；调用方需对 `dim 0` 求和得到最终 `d` / `sqr_sum`。
- 该 GEMM 使用 TF32 精度，PyTorch 侧做基准对比时需打开 `torch.backends.cuda.matmul.allow_tf32 = True`。

**示例**

```python
import torch, deep_gemm

m, n, k = 4096, 24, 7168
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.float, device='cuda')

# 不切分
d = torch.empty((m, n), dtype=torch.float, device='cuda')
s = torch.empty((m,), dtype=torch.float, device='cuda')
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s)

# Split-K 模式
num_splits = 16
d = torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
s = torch.empty((num_splits, m), dtype=torch.float, device='cuda')
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
d = d.sum(0)
s = s.sum(0)

ref_d = a.float() @ b.T
ref_s = a.float().square().sum(-1)
```

## 相关

- Mega mHC 内的混合路线也使用类似的 TF32 预处理：`mega_mhc`（[api-mega-mhc.md](./api-mega-mhc.md)）。
- 测试参考：`tests/test_hyperconnection.py`

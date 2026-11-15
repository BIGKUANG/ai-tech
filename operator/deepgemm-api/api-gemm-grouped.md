# 分组 GEMM API

- 源码：`csrc/apis/gemm.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM90 / SM100

DeepGEMM 的分组 GEMM 与传统 CUTLASS 不同：**只沿 M 轴或 K 轴分组**，N 与 K（M 分组）或 M 与 N（K 分组）保持固定。这针对 MoE 中每个 expert 形状相同、但 token 数不同的场景。

分三类：

1. **M 分组，contiguous 布局**：每个 expert 的 token 拼接成一个连续张量，训练前向 / 推理 prefill 使用。
2. **M 分组，masked 布局**：推理 decode 阶段配合 CUDA Graph 使用，用掩码标出每个 expert 实际有效的行，避免 CPU 同步。
3. **K 分组，contiguous 布局**：MoE 权重反向（wgrad）使用，M 与 N 固定、K 变化。

## 函数总览

| 函数 | 数据类型 | 布局 | 架构 |
| --- | --- | --- | --- |
| `m_grouped_fp8_fp4_gemm_nt_contiguous` | FP8/FP4 | NT | SM90/SM100 |
| `m_grouped_fp8_fp4_gemm_nn_contiguous` | FP8/FP4 | NN | SM100 |
| `m_grouped_fp8_fp4_gemm_nt_masked` | FP8/FP4 | NT | SM90/SM100 |
| `m_grouped_bf16_gemm_nt_contiguous` | BF16 | NT | SM90/SM100 |
| `m_grouped_bf16_gemm_nn_contiguous` | BF16 | NN | SM100 |
| `m_grouped_bf16_gemm_nt_masked` | BF16 | NT | SM90/SM100 |
| `k_grouped_fp8_gemm_tn_contiguous` | FP8 | TN | SM100（SM90 用 `nt`） |
| `k_grouped_fp8_gemm_nt_contiguous` | FP8 | NT | SM90 |
| `k_grouped_fp4_gemm_nt_contiguous` | FP4 | NT | SM100 |
| `k_grouped_bf16_gemm_tn_contiguous` | BF16 | TN | SM90/SM100 |

**别名**：`m_grouped_fp4_gemm_nt_contiguous` ≡ `m_grouped_fp8_gemm_nt_contiguous` ≡ `m_grouped_fp8_fp4_gemm_nt_contiguous`；`m_grouped_fp8_gemm_nn_contiguous` ≡ `m_grouped_fp8_fp4_gemm_nn_contiguous`；`m_grouped_fp4_gemm_nt_masked` ≡ `m_grouped_fp8_gemm_nt_masked` ≡ `m_grouped_fp8_fp4_gemm_nt_masked`。

分组 GEMM 的 `grouped_layout` 与 PSUM 布局语义、缩放因子契约较复杂，详见 [scaling-factor-format.md](./scaling-factor-format.md) 第 5 节。

---

## M 分组（contiguous）

### m_grouped_fp8_fp4_gemm_nt_contiguous

```python
m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    use_psum_layout=False,
    ensure_zero_padding=True,
    expected_m_for_psum_layout=None,
    epilogue=None,
)
```

形状为 `[M, K] @ [G, N, K].mT`：A 沿 M 轴按 expert 拼接，B 为每个 expert 的权重 `[num_groups, N, K]`。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `(a_data, sfa)` | `a_data` 形状 `[M, K]`（逻辑），K-major |
| `b` | `(b_data, sfb)` | `b_data` 形状 `[num_groups, N, K]`（逻辑），K-major |
| `d` | `[M, N]`，`torch.bfloat16` | 输出，N-major |
| `grouped_layout` | `[?]`，`torch.int`，连续 | 语义取决于 `use_psum_layout`，见下 |
| `use_psum_layout` | bool | 是否使用 PSUM（前缀和）布局 |
| `ensure_zero_padding` | bool | PSUM 布局下是否保证组间 gap 行写零 |
| `expected_m_for_psum_layout` | int，可选 | PSUM 布局下每组期望的有效行数（用于 heuristic） |
| `recipe` / `recipe_a` / `recipe_b` | — | 同稠密 GEMM |

**`grouped_layout` 语义**

| `use_psum_layout` | `grouped_layout[i]` 含义 | 张量长度 |
| --- | --- | --- |
| `False` | 第 `i` 组的 M 大小 | `num_grouped = M`（每个 token 一个组标签，取值 `-1` 表示空 token） |
| `True` | 第 `i` 组的前缀和结束位置（组 `i` 占据 `[align(layout[i-1], alignment), layout[i])`） | `num_groups` |

注意：**非 PSUM 时** `grouped_layout` 是长度为 `M` 的逐行 expert 标签张量（内核据此在可选 tile 调度中跳过空组）；**PSUM 时**它才是长度为 `num_groups` 的组边界。

**约束**

- 每个 expert 段的起始行必须按 GEMM M block 对齐。对齐值由 `deep_gemm.set_mk_alignment_for_contiguous_layout()` 设置，可用 `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()` 获取理论最小值。
- `m == 0` 时直接返回。
- `use_psum_layout=True` 且 `ensure_zero_padding=True` 时，gap 行会被写零（安全有限值）。
- SM90 不支持自定义 epilogue。

**示例（PSUM 布局）**

```python
import torch, deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

num_groups, n, k = 8, 4096, 4096
deep_gemm.set_mk_alignment_for_contiguous_layout(
    deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())

# 每组有效 token 数（无需提前对齐）
m_per_group = [200, 1000, 4096, 33, 800, 128, 640, 512]
alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
psum, prev = [], 0
for m in m_per_group:
    end = (prev + alignment - 1) // alignment * alignment + m
    psum.append(end)
    prev = end
grouped_layout = torch.tensor(psum, dtype=torch.int, device='cuda')
m = psum[-1]

a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=True, use_packed_ue8m0=True)
b_fp8, sfb = per_token_cast_to_fp8(b.view(-1, k), use_ue8m0=True, use_packed_ue8m0=True)
sfb = sfb.view(num_groups, n, k // 128)
sfa = deep_gemm.transform_sf_into_required_layout(
    sfa, m, k, (1, 128), psum_layout=grouped_layout)
sfb = deep_gemm.transform_sf_into_required_layout(sfb, n, k, (1, 128), num_groups=num_groups)

d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a_fp8, sfa), (b_fp8, sfb), d, grouped_layout,
    recipe=(1, 1, 128), use_psum_layout=True,
)
```

### m_grouped_fp8_fp4_gemm_nn_contiguous

```python
m_grouped_fp8_fp4_gemm_nn_contiguous(
    a, b, d, grouped_layout,
    recipe=None, recipe_a=None, recipe_b=None,
    compiled_dims="nk", disable_ue8m0_cast=False,
    use_psum_layout=False, ensure_zero_padding=True, epilogue=None)
```

与 `nt` 版本相同，但 B 为 `[num_groups, K, N]`（`nn` 布局）。实现上把 B 的后两维转置后调用 `nt`。

---

## M 分组（masked）

### m_grouped_fp8_fp4_gemm_nt_masked

```python
m_grouped_fp8_fp4_gemm_nt_masked(
    a, b, d, masked_m, expected_m,
    recipe=None, recipe_a=None, recipe_b=None,
    compiled_dims="nk", disable_ue8m0_cast=False, epilogue=None)
```

形状为 `[G, M, K] @ [G, N, K].mT`。与 contiguous 版本不同，A 与 D 是规则的 `[G, M, K]` / `[G, M, N]` 三维张量，每个组的有效行数由 `masked_m[g]` 给出。

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `(a_data, sfa)` | `a_data` 形状 `[num_groups, M, K]`，K-major |
| `b` | `(b_data, sfb)` | `b_data` 形状 `[num_groups, N, K]`，K-major |
| `d` | `[num_groups, M, N]`，`torch.bfloat16` | 输出，N-major |
| `masked_m` | `[num_groups]`，`torch.int`，连续 | 每个 expert 实际有效的行数 |
| `expected_m` | int | 预期每组平均有效行数（`> 0`），用于 heuristic |

**用途**：推理 decode 阶段，CUDA Graph 下 CPU 不知道每个 expert 收到多少 token，用 `masked_m` 在 device 上标记有效行。典型输入来自 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟内核。

**约束**：A、B 均为 K-major；`masked_m` 连续；`expected_m > 0`；SM90 不支持自定义 epilogue。

**示例**

```python
import torch, deep_gemm

num_groups, max_m, n, k = 8, 512, 4096, 4096
expected_m = int(0.8 * max_m)
alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

a = torch.randn((num_groups, max_m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device='cuda')
d = torch.empty((num_groups, max_m, n), dtype=torch.bfloat16, device='cuda')
masked_m = torch.randint(1, max_m, (num_groups,), dtype=torch.int, device='cuda')

# 假设 a/b 已量化并转换 SF 布局
# deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_q, b_q, d, masked_m, expected_m, recipe=(1, 1, 128))
```

### m_grouped_bf16_gemm_nt_masked

```python
m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims="nk", epilogue=None)
```

BF16 版本，语义同上。`a`、`b` 为纯 BF16 张量，`d` 为 BF16。

---

## K 分组（contiguous）

K 分组 GEMM 沿 K 轴拼接多个组，`D = [num_groups, m, n]`。SM100 的 FP8 TN API 使用逻辑操作数 `A = [sum_k, m]`、`B = [sum_k, n]`；SM100 的 FP4 NT API 使用 `A = [m, sum_k]`、`B = [n, sum_k]`（底层按字节存储 `[m, sum_k/2]`、`[n, sum_k/2]`）。

### k_grouped_fp8_gemm_tn_contiguous

```python
k_grouped_fp8_gemm_tn_contiguous(
    a, b, d,
    ks_cpu,
    grouped_layout,
    c=None,
    recipe=(1, 1, 128),
    compiled_dims="mn",
    use_psum_layout=False,
    epilogue=None,
)
```

**参数**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `a` | `(a_data, sfa)` | `a_data` 形状 `[sum_k, m]` |
| `b` | `(b_data, sfb)` | `b_data` 形状 `[sum_k, n]` |
| `d` | `[num_groups, m, n]`，连续 | 输出 |
| `ks_cpu` | `List[int]` 或 `None` | 每组 K 大小（CPU 上）；PSUM 布局下可为 `None`/`[]` |
| `grouped_layout` | `[num_groups]`，`torch.int` | 见 [scaling-factor-format.md](./scaling-factor-format.md) 5.1 |
| `c` | `[num_groups, m, n]`，FP32/FP16 | **虽然默认 `None`，但 k-grouped 断言 `c` 非空**：需传与 `d` 同形状的累加器。推荐传与 `d` 相同的张量以原地累加 |
| `recipe` | `(1, 1, gran_k)` | `gran_m`、`gran_n` 必须为 1；`gran_k ∈ {32, 128}`（SM100） |
| `use_psum_layout` | bool | PSUM 布局 |

**约束**

- `recipe` 必须是 1D1D：`std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1`。
- `k_alignment`（由 `set_mk_alignment_for_contiguous_layout` 设置）必须是 `BLOCK_K=128` 的倍数。
- A、B、D 均需连续。
- SM100 支持 PSUM 与自定义 epilogue；SM90（`k_grouped_fp8_gemm_nt_contiguous`）不支持 PSUM，且要求 `c` 非空、`ks_cpu` 非空。

**示例（非 PSUM，FP32 缩放因子）**

```python
import torch, deep_gemm

gran_k, k_alignment = 32, 128
deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)
ks = [2048, 4096, 1024]
sum_k = sum(ks)
grouped_layout = torch.tensor(ks, dtype=torch.int, device='cuda')

# a_fp8: [sum_k, m] float8_e4m3fn;  sfa: [sum_k // gran_k, m] float32, contiguous
# b_fp8: [sum_k, n] float8_e4m3fn;  sfb: [sum_k // gran_k, n] float32, contiguous
d = torch.zeros((len(ks), m, n), dtype=torch.float, device='cuda')
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    (a_fp8, sfa), (b_fp8, sfb), d,
    ks_cpu=ks, grouped_layout=grouped_layout,
    c=d, recipe=(1, 1, gran_k),
)
```

### k_grouped_fp4_gemm_nt_contiguous

```python
k_grouped_fp4_gemm_nt_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None,
    recipe=(1, 1, 32), compiled_dims="mn",
    use_psum_layout=False, epilogue=None)
```

FP4 版本，仅 SM100。要求 A、B 均为打包 FP4 且 K-major；`recipe` 固定为 `(1, 1, 32)`；`k_alignment` 必须是 `BLOCK_K=256` 的倍数。

### k_grouped_bf16_gemm_tn_contiguous

```python
k_grouped_bf16_gemm_tn_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None,
    compiled_dims="mn", use_psum_layout=False, epilogue=None)
```

BF16 版本。A、B 为 BF16，SM90 要求 `c` 非空。

---

## 相关

- 稠密 GEMM：[api-gemm.md](./api-gemm.md)
- 缩放因子 / PSUM / K 分组 SF 契约：[scaling-factor-format.md](./scaling-factor-format.md)
- 对齐设置：[api-layout.md](./api-layout.md)
- Epilogue：[api-epilogue.md](./api-epilogue.md)

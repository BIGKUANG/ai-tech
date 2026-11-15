# 布局与对齐 API

- 源码：`csrc/apis/layout.hpp`、`csrc/jit_kernels/impls/smxx_layout.hpp`、`csrc/utils/layout.hpp`、`deep_gemm/utils/layout.py`
- Python 导入：`import deep_gemm`（顶层）或 `from deep_gemm.utils import ...`
- 目标架构：SM90 / SM100

本组 API 负责 Scaling Factor（SF）的布局变换、TMA 对齐，以及 M/K 对齐配置。SF 的完整格式规范见 [scaling-factor-format.md](./scaling-factor-format.md)。

## 函数总览

| 函数 | 说明 |
| --- | --- |
| `transform_sf_into_required_layout` | 把 SF 变换到内核要求的布局 |
| `get_tma_aligned_size` | 计算 TMA 对齐后的尺寸 |
| `get_mn_major_tma_aligned_tensor` | FP32 SF → MN-major、TMA 对齐（不打包） |
| `get_mn_major_tma_aligned_packed_ue8m0_tensor` | FP32 SF → 打包 UE8M0、MN-major、TMA 对齐 |
| `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor` | K 分组打包 UE8M0 |
| `set_mk_alignment_for_contiguous_layout` | 设置连续布局的 M/K 对齐 |
| `get_mk_alignment_for_contiguous_layout` | 读取连续布局的 M/K 对齐 |
| `get_theoretical_mk_alignment_for_contiguous_layout` | 理论最小 M/K 对齐 |

别名：`get_m_alignment_for_contiguous_layout` ≡ `get_k_alignment_for_contiguous_layout` ≡ `get_mk_alignment_for_contiguous_layout`（定义于 `deep_gemm/utils/layout.py`）。

---

## transform_sf_into_required_layout

```python
transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe,
    num_groups=None,
    is_sfa=None,
    disable_ue8m0_cast=False,
    psum_layout=None,
) -> torch.Tensor
```

把缩放因子变换到内核要求的布局。这是用户最常用的布局函数。

**参数**

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `sf` | `torch.float32` 或 `torch.int32` | 输入 SF |
| `mn` | int | `M`（`is_sfa=True`）或 `N` |
| `k` | int | K |
| `recipe` | `(gran_m, gran_n, gran_k)` 或 `(gran_mn, gran_k)` | 量化 recipe |
| `num_groups` | int，可选 | SF 带前导 group 维时设置 |
| `is_sfa` | bool，可选 | 3 元组 recipe 时**必需**（`True` 取 `gran_m`，`False` 取 `gran_n`）；2 元组 recipe 时**禁止** |
| `disable_ue8m0_cast` | bool | SM90 / legacy 路径下只做 MN-major 转置，不转 UE8M0 |
| `psum_layout` | `torch.Tensor`，可选 | 仅 PSUM 布局下的 SFA 使用（跳过 gap 行） |

**Dispatch 表**（详见规范文档第 4 节）

| 输入 dtype | `gran_mn` | `gran_k` | 架构 | 动作 |
| --- | --- | --- | --- | --- |
| `float32` | 1 | 128 | SM90（或 `disable_ue8m0_cast`） | 转置为 MN-major、TMA 对齐 `float32`（不打包） |
| `float32` | 128 | 128 | SM90（或 `disable_ue8m0_cast`） | 仅校验 |
| `float32` | 任意 | 32 或 128 | SM100 | 沿 MN 广播到 `gran_mn=1`，打包为 UE8M0 `int32`，MN-major、TMA 对齐 |
| `int32` | 1 | 32 或 128 | SM100 | 仅校验（已是预转换格式） |

**返回**：变换后的 SF 张量。SM100 的 `float32` 行返回的正是规范文档第 3 节的预转换格式（`int32`，MN-major，TMA 对齐），可缓存复用。

**示例**

```python
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8

# 权重 SF（一次性变换后缓存）
sfw = deep_gemm.transform_sf_into_required_layout(
    sf=weight_sf, mn=n, k=k,
    recipe=(1, 128, 128), is_sfa=False, num_groups=num_groups)

# 激活 SF
x_fp8, sfx = per_token_cast_to_fp8(x, use_ue8m0=True, use_packed_ue8m0=True)
sfx = deep_gemm.transform_sf_into_required_layout(sfx, m, k, (1, 128))
```

---

## get_tma_aligned_size

```python
get_tma_aligned_size(x, element_size) -> int
```

计算 TMA 对齐后的尺寸：把 `x` 上调到 `16 / element_size` 的整数倍（即保证行 stride 至少 16 字节）。

**参数**：`x`（int，原始尺寸），`element_size`（int，元素字节数，必须整除 16）。

**返回**：对齐后的整数。

```python
deep_gemm.get_tma_aligned_size(mn, 4)   # int32 SF 的 TMA 对齐 MN
```

---

## get_mn_major_tma_aligned_tensor

```python
get_mn_major_tma_aligned_tensor(sf) -> torch.Tensor
```

把 FP32 SF 转置为 **MN-major 且 TMA 对齐** 的布局（不做 UE8M0 打包）。用于 SM90 的 `float32` SF 或 FP32 兼容路径。

**输入**：`sf` 为 `float32`，2D 或 3D（3D 时首维为 batch/group）。

**返回**：形状 `[..., mn, sf_k]`、stride `[..., 1, align(mn, 4)]` 的张量，其中 `sf_k = ceil_div(k, gran_k)`。若输入已是该布局则直接返回。

---

## get_mn_major_tma_aligned_packed_ue8m0_tensor

```python
get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum_layout=None) -> torch.Tensor
```

把 FP32 SF 先转 UE8M0，再按 4 个一组打包为 `int32`，输出 MN-major、TMA 对齐布局。SM100 的 1D1D 路径使用此函数（`transform_sf_into_required_layout` 的 `float32` 分支内部即调用它）。

**参数**：`sf`（`float32`，2D/3D），`psum_layout`（可选，PSUM 布局张量，允许内核跳过未初始化的 gap 行；要求 `sf` 2D 且连续）。

**返回**：`int32` 张量，形状 `[..., mn, ceil_div(sf_k, 4)]`，stride `[..., 1, align(mn, 4)]`。

```python
packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfx_float32)
```

---

## get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor

```python
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf, grouped_layout, ks_cpu, gran_k, k_alignment,
    use_psum_layout=False) -> torch.Tensor
```

K 分组 GEMM 的 SF 打包内核。每个 group 独立打包后在 K 轴拼接。

**参数**

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `sf` | `float32`，`[sum_sf_k, mn]`，连续 | 输入 SF |
| `grouped_layout` | `torch.int`，`[num_groups]`，连续 | PSUM / 非 PSUM 语义见规范文档 5.1 |
| `ks_cpu` | `List[int]` 或 `None`/`[]` | 每组 K（须为 `k_alignment` 与 `gran_k` 的倍数）。`None`/`[]` 时要求 `use_psum_layout=True` |
| `gran_k` | int | 32 或 128（SM100） |
| `k_alignment` | int | 必须是 128 的倍数 |
| `use_psum_layout` | bool | — |

**返回**：`int32` 张量，形状 `[sum(ceil_div(k_i, gran_k * 4)), mn]`（连续，`mn % 4 == 0`）。当 `ks_cpu` 缺失时，按 `(sf_k + num_groups * 3) / 4` 上界分配。`gran_k == 128` 时必须传 `float32`（不能预打包 `int32`）。

**示例**

```python
sfa_packed = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sfa, grouped_layout, ks_cpu=ks, gran_k=32, k_alignment=128)
```

---

## 对齐配置

### set_mk_alignment_for_contiguous_layout

```python
set_mk_alignment_for_contiguous_layout(new_value: int)
```

设置连续布局（M 分组 / K 分组）的组级 M/K 对齐。每个 expert 段必须按该值对齐。

### get_mk_alignment_for_contiguous_layout

```python
get_mk_alignment_for_contiguous_layout() -> int
```

读取当前对齐值。默认 `128`。

### get_theoretical_mk_alignment_for_contiguous_layout

```python
get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int
```

返回理论最小对齐：

- 非 SM10（SM90）：固定返回 128。
- SM100：固定返回 256（新增的 UMMA_N=256 支持；`expected_m` 当前被忽略，因此小 M 场景可能有性能损失，这类负载建议直接用 Mega MoE）。

**示例**

```python
import deep_gemm

alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
```

## 相关

- 完整 SF 格式规范：[scaling-factor-format.md](./scaling-factor-format.md)
- 分组 GEMM：[api-gemm-grouped.md](./api-gemm-grouped.md)
- 量化 / cast 工具：[api-utils.md](./api-utils.md)

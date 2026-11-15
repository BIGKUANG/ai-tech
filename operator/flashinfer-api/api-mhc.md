# 多头残差组合 mHC（`flashinfer.mhc`）

> 源码：`flashinfer/mhc.py`
> 上游文档：`docs/api/mhc.rst`
> 适用版本：FlashInfer 0.7.2

Multi-head residual Combination（mHC）kernel，用于 Agentic 风格的残差混合，固定 ``HC=4`` 的子头
（sub-head）布局。mHC 把 Transformer 的残差流从单一向量扩展为 4 个子头，并在每层通过
``post_layer_mix`` / ``comb_res_mix`` 系数把层输出与历史多头残差重新组合。本模块提供：

- `mhc_post`：后映射（把层输出与多头残差组合成新的多头残差）；
- `mhc_pre_big_fuse`：前映射大融合（由外部投影 logits 与残差平方和得到 pre/post/comb 因子与
  层输入），带 Sinkhorn 迭代与 RMS 归一化；
- `mhc_pre_big_fuse_with_prenorm`：与前者相同，但在 kernel 内由 `residual` 计算 RMS 平方和，
  不要求外部提供 `sqrsum`。

模块内常量：``HC = 4``、``MIX = HC * (2 + HC) = 24``（24 个投影槽位 =
4 pre + 4 post + 16 comb）。

## 导入

```python
from flashinfer.mhc import mhc_post, mhc_pre_big_fuse, mhc_pre_big_fuse_with_prenorm
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mhc_post` | 函数 | mHC 后映射（HC=4）：`x` 按 post 系数与多头残差按 comb 矩阵组合。 |
| `mhc_pre_big_fuse` | 函数 | mHC 前映射大融合：由外部 `dot_mix` / `sqrsum` 生成 pre/post/comb 因子与层输入。 |
| `mhc_pre_big_fuse_with_prenorm` | 函数 | 同上，但由 `residual` 内部计算 RMS 平方和。 |

## API 详情

### `mhc_post`

```python
mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor
```

**功能**：对 HC=4 应用 mHC 后映射：

```text
out[..., new_hc, h] = x[..., h] * post_layer_mix[..., new_hc]
                    + sum_old residual[..., old_hc, h] * comb_res_mix[..., old_hc, new_hc]
```

**参数**

- `x`（`torch.Tensor`）：层输出张量，形状 ``[..., H]``（外维与 `residual` 一致，
  ``H = residual.shape[-1]``）。
- `residual`（`torch.Tensor`）：多头残差张量，形状 ``[..., HC=4, H]``。``HC=4`` 轴必须等于 4
  （当前 mHC 硬编码为 4 个子头）。
- `post_layer_mix`（`torch.Tensor`）：层输出混合权重，形状 ``[..., HC=4]`` 或 ``[..., HC=4, 1]``
  （两者皆可）。提供施加到 `x` 上的每个新头系数。
- `comb_res_mix`（`torch.Tensor`）：残差组合矩阵，形状 ``[..., HC=4, HC=4]``。每个
  ``[old_hc, new_hc]`` 项是把 ``residual[..., old_hc, :]`` 混合到新头 ``new_hc`` 的系数。

**返回**

- `torch.Tensor`：与 `residual` 形状相同的输出 ``[..., HC=4, H]``。

**注意事项**

- 形状校验：`residual.ndim >= 3`；`x` 的外维与 `residual` 一致、最后一维为 `H`；
  `post_layer_mix` 形状须为 ``outer + (4,)`` 或 ``outer + (4, 1)``；`comb_res_mix` 形状须为
  ``outer + (4, 4)``。不满足抛 `ValueError`。
- 输入在内部会被 reshape 为连续张量后调用 CUDA kernel。

**示例**

```python
import torch
from flashinfer.mhc import mhc_post

T, H = 1024, 4096
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
residual = torch.randn(T, 4, H, device="cuda", dtype=torch.bfloat16)
post = torch.randn(T, 4, device="cuda", dtype=torch.float32)
comb = torch.randn(T, 4, 4, device="cuda", dtype=torch.float32)
out = mhc_post(x, residual, post, comb)
```

---

### `mhc_pre_big_fuse`

```python
mhc_pre_big_fuse(
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    num_splits: int = 1,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：使用外部投影与平方和计算 mHC 前映射大融合。`dot_mix` 含按
``[pre(4), post(4), comb(16)]`` 排列的原始投影 logits；`sqrsum` 含用于 RMS 归一化的残差平方和。
当 `num_splits > 1` 时，两个输入都带一个前导 split 维，并在 CUDA kernel 内部归约。

**参数**

- `dot_mix`（`torch.Tensor`）：原始投影 logits，`num_splits=1` 时形状 ``[..., 24]``，否则
  ``[num_splits, ..., 24]``。末尾 24 个槽位 = 4 (pre) + 4 (post) + 16 (comb)，在 kernel 内被切分
  并 reshape 为 per-head 的 ``pre / post / comb`` 因子。
- `sqrsum`（`torch.Tensor`）：预先计算的 per-token 残差平方和，用于 RMS 归一化。形状 ``[...]``
  与 `dot_mix` 外维匹配，`num_splits > 1` 时为 ``[num_splits, ...]``（kernel 内部归约 split 轴）。
  dtype 由实现定义。
- `residual`（`torch.Tensor`）：多头残差张量，形状 ``[..., HC=4, H]``（bfloat16），布局与
  `mhc_post` 的 `residual` 相同。
- `mhc_scale`（`torch.Tensor`）：Sinkhorn / mix 步骤消费的 mHC 缩放张量。形状与 dtype 由实现定义。
- `mhc_base`（`torch.Tensor`）：Sinkhorn / mix 步骤消费的 mHC 偏置 / base 张量。形状与 dtype
  由实现定义。
- `k`（`int`）：传给 CUDA kernel 的 mHC 算法参数 `k`。
- `rms_eps`（`float`，默认 `1e-6`）：RMSNorm epsilon，必须严格为正。
- `mhc_pre_eps`（`float`，默认 `1e-6`）：mHC pre-map 步骤的数值稳定 epsilon，必须严格为正。
- `mhc_sinkhorn_eps`（`float`，默认 `1e-6`）：Sinkhorn 迭代步骤的数值稳定 epsilon，必须严格为正。
- `mhc_post_mult_value`（`float`，默认 `1.0`）：施加到 mHC post 输出的乘法因子。
- `sinkhorn_repeat`（`int`，默认 `20`）：Sinkhorn 迭代次数。
- `num_splits`（`int`，默认 `1`）：前导维度的 split 因子，须为 ``{1, 2, 4, 8, 16}``。大于 1 时
  `dot_mix` 与 `sqrsum` 带一个前导 split 轴并在 kernel 内归约。
- `block_size`（`int`，默认 `0`）：转发给 kernel 的 CUDA block-size 提示（``0`` 选择 kernel
  默认值）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：``(post_mix, comb_mix, layer_input)``，
  形状分别为 ``[..., HC=4, 1]``、``[..., HC=4, HC=4]``、``[..., H]``。

**注意事项**

- `num_splits` 必须为 ``{1, 2, 4, 8, 16}`` 之一，否则抛 `ValueError`。
- `rms_eps` / `mhc_pre_eps` / `mhc_sinkhorn_eps` 必须严格为正。
- `dot_mix` / `sqrsum` 的形状必须与 `num_splits` 及 `residual` 外维一致；不匹配抛 `ValueError`。
- 输出 dtype：`post_mix` 与 `comb_mix` 为 float32，`layer_input` 为 bfloat16。

---

### `mhc_pre_big_fuse_with_prenorm`

```python
mhc_pre_big_fuse_with_prenorm(
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：应用 mHC 前映射大融合，并由 `residual` 计算 RMS 平方和。当没有预计算的 `sqrsum` 时，
这与 Agentic 的 ``mhc_pre_finalize`` 边界匹配。`dot_mix` 可为 ``[..., 24]`` 或 ``[1, ..., 24]``。

**参数**

- `dot_mix`（`torch.Tensor`）：原始投影 logits，形状 ``[..., 24]`` 或 ``[1, ..., 24]``。末尾 24 个
  槽位 = 4 (pre) + 4 (post) + 16 (comb)，在 kernel 内被切分并 reshape 为 per-head 因子。
- `residual`（`torch.Tensor`）：多头残差张量，形状 ``[..., HC=4, H]``（bfloat16）；`sqrsum` 在
  kernel 内由该张量计算，布局与 `mhc_post` 的 `residual` 相同。
- `mhc_scale`（`torch.Tensor`）：Sinkhorn / mix 步骤消费的 mHC 缩放张量。形状与 dtype 由实现定义。
- `mhc_base`（`torch.Tensor`）：Sinkhorn / mix 步骤消费的 mHC 偏置 / base 张量。形状与 dtype
  由实现定义。
- `rms_eps`（`float`，默认 `1e-6`）：RMSNorm epsilon，必须严格为正。
- `mhc_pre_eps`（`float`，默认 `1e-6`）：mHC pre-map 步骤的数值稳定 epsilon，必须严格为正。
- `mhc_sinkhorn_eps`（`float`，默认 `1e-6`）：Sinkhorn 迭代步骤的数值稳定 epsilon，必须严格为正。
- `mhc_post_mult_value`（`float`，默认 `1.0`）：施加到 mHC post 输出的乘法因子。
- `sinkhorn_repeat`（`int`，默认 `20`）：Sinkhorn 迭代次数。
- `block_size`（`int`，默认 `0`）：转发给 kernel 的 CUDA block-size 提示（``0`` 选择默认值）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：``(post_mix, comb_mix, layer_input)``，
  形状分别为 ``[..., HC=4, 1]``、``[..., HC=4, HC=4]``、``[..., H]``。

**注意事项**

- 与 `mhc_pre_big_fuse` 不同，本函数没有 `sqrsum` 与 `num_splits` 参数。
- `dot_mix` 形状须为 ``[..., 24]`` 或 ``[1, ..., 24]``，否则抛 `ValueError`。
- 三个 epsilon 参数必须严格为正。输出 dtype 同 `mhc_pre_big_fuse`。

## 相关链接 / 参考

- 上游 `docs/api/mhc.rst`：仅列出 `mhc_post`、`mhc_pre_big_fuse`、`mhc_pre_big_fuse_with_prenorm`
  三个公开 API。
- 具体系数语义（`pre / post / comb` 的投影推导、Sinkhorn 归一化细节）请参见 mHC 论文 /
  参考实现。

# Cake MoE 分组 GEMM（`flashinfer.experimental.cake_moe_grouped_gemm`）

> 源码：`flashinfer/experimental/cake_moe_grouped_gemm/`（`__init__.py`、`cake_backend.py`、`cake_autograd.py`、`cake_jit.py`）
> 上游文档：`docs/api/cake_moe_grouped_gemm.rst`
> 适用版本：FlashInfer 0.7.2

``cake_moe_grouped_gemm`` 是一个实验性的 Cake 后端：为 Mixture-of-Experts 层的专家投影生成
ragged BF16 分组 GEMM 程序，目标架构为 SM100（B200 / GB200）、SM103（B300 / GB300）与
SM107（R200）。API 与后端均为实验性；调用下述任一函数，或在稳定 API 中命名该后端，即为显式
opt-in，并会发出一次 ``ExperimentalWarning``。

分组由 int32 **设备** 张量描述，有两种约定：

- ``offs[E]``：累积结束偏移；
- ``m_indptr[E + 1]``：带前导 0（与 ``torch._grouped_mm``、`flashinfer.grouped_mm.grouped_mm_bf16`
  一致）。

kernel 在设备端读取偏移量，主机端从不同步；不需要 padding，组大小可失衡、可为空、可为一行，
也可为非 tile 整数倍。所有操作数为 bfloat16 且 fp32 tensor-core 累加；归约按位确定（无 atomics），
空组产生精确的零权重梯度。

| 操作 | 计算 | 输出 |
| --- | --- | --- |
| `fwd` | ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` | ``[sum_m, N]`` bfloat16 |
| `dgrad` | ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]``（W 原地读取） | ``[sum_m, K]`` bfloat16 |
| `wgrad` | ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` | ``[E, N, K]`` bfloat16 或 float32 |

## 导入

```python
from flashinfer.experimental.cake_moe_grouped_gemm import (
    grouped_gemm_fwd,
    grouped_gemm_dgrad,
    grouped_gemm_wgrad,
    cake_grouped_mm,
    prepare_grouped_gemm_fwd,
    prepare_grouped_gemm_dgrad,
    prepare_grouped_gemm_wgrad,
    GroupedGemmLaunch,
)
```

稳定 API 的 opt-in 形式：

```python
out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `grouped_gemm_fwd` | 函数 | 分组前向投影 ``Y = X @ W[e].T``（一次性调用）。 |
| `grouped_gemm_dgrad` | 函数 | 激活梯度 ``dX = G @ W[e]``（一次性调用）。 |
| `grouped_gemm_wgrad` | 函数 | 权重梯度 ``dW[e] = G[e].T @ X[e]``（一次性调用，bf16/fp32 输出）。 |
| `cake_grouped_mm` | 函数 | 可微分组 GEMM 的 `torch.autograd.Function` 包装。 |
| `CakeGroupedMm` | 类 | `cake_grouped_mm` 使用的 `torch.autograd.Function`。 |
| `prepare_grouped_gemm_fwd` | 函数 | 校验参数、绑定程序，返回可复用的 `GroupedGemmLaunch`（前向）。 |
| `prepare_grouped_gemm_dgrad` | 函数 | 同上（dgrad）。 |
| `prepare_grouped_gemm_wgrad` | 函数 | 同上（wgrad）。 |
| `GroupedGemmLaunch` | 类 | 一次已准备的分组 GEMM：`launch()` 无分配、可 CUDA Graph 捕获。 |

## API 详情

### `grouped_gemm_fwd`

```python
grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：对每个分组 ``e`` 计算 ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``。

**参数**

- `x`（`torch.Tensor`）：``[sum_m, K]`` bfloat16，所有分组的行按组依次排列。
- `w`（`torch.Tensor`）：``[E, N, K]`` bfloat16 权重，每组一个 ``[N, K]`` 矩阵。
- `offs`（`torch.Tensor`）：``[E]`` int32 设备端累积结束偏移（``offs[-1] == sum_m``），或带前导 0
  的 ``m_indptr`` ``[E + 1]``；由 kernel 读取，从不在主机读取。
- `out`（`Optional[torch.Tensor]`）：``[sum_m, N]`` bfloat16 输出（省略时分配）；允许行方向 padding。

**返回**

- `torch.Tensor`：`out`。要求 ``N % 256 == 0`` 且 ``K % 64 == 0``。

**注意事项**

- 本函数一次完成“准备 + 启动”。对同一批张量重复启动（或 CUDA Graph 捕获）时应先用
  `prepare_grouped_gemm_fwd` 准备一次，再对该对象调用 `launch()`。

---

### `grouped_gemm_dgrad`

```python
grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：对每个分组 ``e`` 计算 ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]``。

**参数**

- `g`（`torch.Tensor`）：``[sum_m, N]`` bfloat16。
- `w`（`torch.Tensor`）：``[E, N, K]`` bfloat16，原地读取（无转置拷贝）。
- `offs`（`torch.Tensor`）：``[E]`` int32 设备端结束偏移（或 ``m_indptr`` ``[E + 1]``）。
- `out`（`Optional[torch.Tensor]`）：``[sum_m, K]`` bfloat16 输出。

**返回**

- `torch.Tensor`：`out`。要求 ``K % 256 == 0`` 且 ``N % 64 == 0``。

**注意事项**

- 准备形式参见 `grouped_gemm_fwd`（使用 `prepare_grouped_gemm_dgrad`）。

---

### `grouped_gemm_wgrad`

```python
grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
) -> torch.Tensor
```

**功能**：对每个分组 ``e`` 计算 ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]``。

**参数**

- `g`（`torch.Tensor`）：``[sum_m, N]`` bfloat16。
- `x`（`torch.Tensor`）：``[sum_m, K]`` bfloat16。
- `offs`（`torch.Tensor`）：``[E]`` int32 设备端结束偏移（``E = offs.numel()``；传入
  ``num_groups=E`` 时也可用 ``m_indptr`` ``[E + 1]``）。
- `out`（`Optional[torch.Tensor]`）：``[E, N, K]``，dtype 为 `out_dtype`。
- `out_dtype`（`Optional[torch.dtype]`，可选）：bfloat16 或 float32；默认取 `out` 的 dtype，
  否则为 bfloat16。
- `num_groups`（`Optional[int]`，可选）：分组数；给定后可接受 ``m_indptr`` ``[E + 1]`` 形式。

**返回**

- `torch.Tensor`：`out`。fp32 累加、按位确定、空组为精确零。要求 ``N % 256 == 0`` 且 ``K`` 为所选
  k tile（256，或选定时的 512）的倍数。

---

### `cake_grouped_mm`

```python
cake_grouped_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    deterministic: bool = True,
) -> torch.Tensor
```

**功能**：可微 ragged 分组 GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``，
是 `CakeGroupedMm`（`torch.autograd.Function`）的包装。其 `backward` 用 dgrad 程序产生
``x.grad``，用 wgrad 程序产生 ``w.grad``（dtype 与 `w` 一致）。

**参数**

- `x`（`torch.Tensor`）：``[sum_m, K]`` bfloat16。
- `w`（`torch.Tensor`）：``[E, N, K]`` bfloat16。
- `offs`（`torch.Tensor`）：int32 设备端 ``[E]`` 累积结束偏移或 ``m_indptr`` ``[E + 1]``；由 kernel
  读取，从不在主机读取。
- `deterministic`（`bool`，默认 `True`）：梯度的归约模式。Cake 程序始终按固定顺序归约（无 atomics）；
  ``False`` 仅为 API 兼容而接受，当前仍运行相同的确定性程序。

**返回**

- `torch.Tensor`：``[sum_m, N]`` bfloat16。`backward` 产生 ``x.grad`` ``[sum_m, K]`` bfloat16
  与 ``w.grad`` ``[E, N, K]``（``w`` 的 dtype，bfloat16）；如需 fp32 权重梯度，请直接使用
  `grouped_gemm_wgrad`。

**注意事项**

- `x` 与 `w` 必须均为 bfloat16，否则抛 `ValueError`。
- Cake 程序固定顺序归约，故 `deterministic=False` 目前运行相同程序。

---

### `CakeGroupedMm`

```python
class CakeGroupedMm(torch.autograd.Function)
```

**功能**：``Y = grouped X @ W[e].T`` 的 autograd 实现，`dX` 与 `dW` 由 Cake 程序计算。

- `forward(ctx, x, w, offs, deterministic)`：``x`` ``[sum_m, K]`` bf16，``w`` ``[E, N, K]`` bf16，
  ``offs`` int32 设备端结束偏移 ``[E]``（或 ``m_indptr`` ``[E + 1]``）；返回 ``[sum_m, N]`` bf16。
- `backward(ctx, grad_y)`：返回 ``dX`` ``[sum_m, K]`` bf16 与 ``dW`` ``[E, N, K]``（``w`` 的 dtype）
  以及 ``None, None``。两种归约均按位确定，空组为精确零。

**注意事项**

- 反向会在尚未触碰 CUDA runtime 的 worker 线程上运行；程序通过驱动 API 编码 TMA 描述符，需要主
  上下文在当前线程上，故 `backward` 会先做一次 runtime 调用以绑定上下文。
- 只有实际需要梯度的输入才会被计算（`ctx.needs_input_grad`）。

---

### `prepare_grouped_gemm_fwd` / `prepare_grouped_gemm_dgrad` / `prepare_grouped_gemm_wgrad`

```python
prepare_grouped_gemm_fwd(x, w, offs, out=None) -> GroupedGemmLaunch
prepare_grouped_gemm_dgrad(g, w, offs, out=None) -> GroupedGemmLaunch
prepare_grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None) -> GroupedGemmLaunch
```

**功能**：校验参数、绑定一个生成程序，并返回 `GroupedGemmLaunch`。其 `launch()` 不进行分配，也
不与主机同步，且可被 CUDA Graph 捕获；当调用方向绑定的张量写入新的偏移量或数值时，同一个已准备的
launch 仍然有效。

**参数/约束**

- `prepare_grouped_gemm_fwd`：``x`` ``[sum_m, K]`` bf16；``w`` ``[E, N, K]`` bf16；``offs`` int32
  ``[E]`` 设备端结束偏移；``N % 256 == 0``、``K % 64 == 0``。`out` 允许行方向 padding（任意行
  stride，列 stride 为 1）。
- `prepare_grouped_gemm_dgrad`：``g`` ``[sum_m, N]`` bf16；``w`` ``[E, N, K]`` bf16（原地读取）；
  ``K % 256 == 0``、``N % 64 == 0``。
- `prepare_grouped_gemm_wgrad`：``g`` ``[sum_m, N]`` bf16；``x`` ``[sum_m, K]`` bf16；要求两者
  最后一维 stride 为 1；``N % 256 == 0``，``K`` 为所选 k tile 的倍数；空组产生精确零；`out_dtype`
  默认取 `out` 的 dtype（两者皆无时为 bf16）。

**返回**

- `GroupedGemmLaunch`：见下。

**注意事项**

- 当形状、dtype 或设备不匹配时抛 `ValueError`；设备不是 compute capability 10.0 / 10.3 / 10.7
  时抛 `ValueError`。

---

### `GroupedGemmLaunch`

```python
@dataclass(frozen=True)
class GroupedGemmLaunch
```

**功能**：一次已准备的 ragged 分组 GEMM。`launch()` 在当前 torch stream 上运行绑定的程序，写入
调用方拥有的 `out`，无 CUDA 分配、无主机同步，并返回 `out`。每次启动都以值传递操作数的 TMA 描述符
（由绑定根据被绑定张量的元数据编码），且 kernel 每次启动都在设备端读取 `offs`，因此同一已准备 launch
（或被 CUDA Graph 捕获的 launch）在调用方向 `offs` 写入新分组边界、或向操作数写入新数值时仍然有效。
形状、dtype 或张量绑定变化时需重新准备。

**字段 / 属性**

- `op`：操作名（``"fwd"`` / ``"dgrad"`` / ``"wgrad"``）。
- `record_name`、`stages`：绑定的路由（route）的舞台集合。
- `plan`（`dict`）：主机端决策（grid、k tile、tail 拆分、raster、启动次数）。
- `out`（`torch.Tensor`）：输出张量。
- `zero_fill`（`bool`）：是否在启动前清零 `out`。
- `workspaces`（`tuple`）：该 launch 拥有的调用方缓冲区（wgrad 的 per-CTA 描述符槽与 fp32 tail
  partials），与 launch 一同保活。

**方法**

- `launch() -> torch.Tensor`：执行启动并返回 `out`（`__call__` 的别名）。
- `stage_launchers`（属性）：每个实际启动的舞台对应一个零参启动函数（用于测试与计时）。
- `stage_modules`（属性）：舞台到模块名的映射。

**注意事项**

- 形状、dtype 或张量绑定变化时需重新准备新 launch。
- launch 通过 `tvm_ffi.use_torch_stream()` 运行，因此在 torch stream 语义下可被捕获。

## 说明 / 注意事项

- **稳定 API opt-in**：前向投影可从稳定分组 GEMM API 命名后端访问：``flashinfer.grouped_mm.
  grouped_mm_bf16(a, b, m_indptr, backend="cake")``。它接受 cuDNN 后端的 ``[cum_m, K]`` /
  ``[E, N, K]`` / ``[E + 1]`` 参数，仅产生 bfloat16 输出，无 tactic 索引，并要求 ``N % 256 == 0``
  且 ``K % 64 == 0``。`m_indptr[-1]` 之后的行在 `out` 中保持不动。
- **平台**：需要 SM100 / SM103 / SM107 设备。SM107 等架构的生成程序随本 checkout 提供；若当前
  checkout 未为该设备注册程序，则相应调用不可用。
- **确定性**：Cake 程序始终按固定顺序归约，`deterministic=False` 当前运行相同程序；wgrad 需要
  fp32 权重梯度时请直接使用 `out_dtype=torch.float32`。
- 本模块的 ``cake_jit``（生成程序注册表）与 ``cake_backend`` 中的计划辅助函数为内部实现细节，
  不作为公开 API 记录。

## 相关链接 / 参考

- 上游文档 `docs/api/cake_moe_grouped_gemm.rst`。
- `flashinfer.grouped_mm.grouped_mm_bf16`：稳定分组 GEMM API（`backend="cake"` 可路由到此后端）。

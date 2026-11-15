# Cake MXFP4 × MXFP8 SiTU 路由 MoE（`flashinfer.fused_moe.cake` / `cake_cute`）

> 源码：`flashinfer/fused_moe/cake/` 与 `flashinfer/fused_moe/cake_cute/`
> （`__init__.py`、`cake_mxfp4_situ_moe_plan.py`、`cake_mxfp4_situ_moe_kernels.py`）
> 上游文档：`docs/api/cake_mxfp4_situ_moe.rst`
> 适用版本：FlashInfer 0.7.2

Cake MXFP4 × MXFP8 SiTU 路由 MoE 是 Kimi K3 W4A8 专家层的实现（MXFP4 权重、MXFP8 激活、
带 per-expert ``beta`` / ``linear_beta`` 的 SiTU 激活、top-k finalize 到 BF16 输出），被复现为
一组 Cake 程序并降级到两个相互独立的后端。两个包来自同一个 pin 的 Cake revision，各自带一个
内容寻址的 manifest（schema ``cake.library_export.v5``），其 ``producer_revision`` 即该 revision。

- `backend="cake"` → `flashinfer.fused_moe.cake`：CUDA C++ 翻译单元，使用 ``tvm_ffi_utils``
  绑定，位于 ``csrc/fused_moe/cake_mxfp4_situ_moe/cuda/sm_103a/``，经 FlashInfer 的 TVM-FFI JIT
  构建。保留已验证生产构建的指针 TMA ABI（每次启动由 plan 拥有的调用方描述符工作区）。
- `backend="cake_cute"` → `flashinfer.fused_moe.cake_cute`：CuTe DSL 设备模块，使用
  ``cutedsl_tvm_ffi`` 绑定，位于 ``csrc/fused_moe/cake_mxfp4_situ_moe/cute/sm_103a/``，首次使用时
  经 DSL 的 ``cute.compile`` 编译（需 ``pip install nvidia-cutlass-dsl``）。采用按值的
  grid-constant TMA ABI。

两者都面向 ``sm_103a``（B300 / GB300）。kernel 是手写 CuTe DSL swap-AB MoE 链的精确复现：
相同的 warp 角色、barrier 与 pipeline 结构、tile 形状、TMEM 布局、epilogue 指令类别、PDL 放置、
persistent scheduler 协议与工作区布局；数值已冻结（FP32 累加、SiTU FP32 序列、UE8M0 向上取整
requantization、BF16 路由权重缩放后 reduce-add）。

## 导入

```python
from flashinfer.fused_moe.cake import CakeMxfp4MoEWrapper, prepare_cake_mxfp4_weights
# 或
from flashinfer.fused_moe.cake_cute import CakeMxfp4MoEWrapper, prepare_cake_mxfp4_weights
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `CakeMxfp4MoEWrapper` | 类 | 仅元数据的 runner：`decide` / `get_workspace_size` / `plan`。 |
| `prepare_cake_mxfp4_weights` | 函数 | 把规范 rank-local ``[up, gate]`` MXFP4 bank 转换为 tile-major kernel 操作数。 |
| `resolve_layout` | 函数 | 由显式元数据解析 rank-local 几何，返回 `RankLayout`。 |
| `RankLayout` | 类 | rank-local 几何（mode / 专家区间 / shard）。 |
| `PlanDecision` | 类 | 由 ``(layout, T)`` 决定的路径、子形式标志与启动列表。 |
| `CakeSwapAbPolicy` | 类 | 手写 wrapper 的决策函数集合。 |
| `CakeSwapAbPlan` | 类 | plain 链的可执行计划（routing → GEMM1 → GEMM2）。 |
| `CakeDensePlan` | 类 | dense 链的可执行计划。 |
| `CakeSplitPlan` | 类 | split two-stage 链的可执行计划。 |
| `CakeHybridPlan` | 类 | hybrid 链的可执行计划。 |
| `CakeMixed192Plan` | 类 | mixed192 链的可执行计划。 |
| `GemmForm` / `Launch` / `WorkspaceField` / `RoutingConfig` / `PlanDecision` | 类 | 计划与工作区的数据结构。 |
| `plan_decision` / `routing_config` / `kernel_stages` | 函数 | 计划决策与路由配置辅助。 |
| `is_available` / `load` / `compile_all` / `manifest` / `modules` / `cake_revision` | 函数/对象 | kernel 加载器与 manifest 接口。 |
| `select_gemm1` / `select_gemm2` / `select_routing` / `unsupported_forms` | 函数 | kernel 选择与不支持形式查询。 |

## API 详情

### `CakeMxfp4MoEWrapper`

```python
CakeMxfp4MoEWrapper(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    num_local_experts: int | None = None,
    local_expert_offset: int | None = None,
    intermediate_shard: int | None = None,
    backend: str = PACKAGE_BACKEND,
    enable_pdl: bool = False,
    swapab_n_tile: int = SWAP_ROW_TILE,
    swapab_max_tokens: int | None = None,
    swapab_tile_policy=None,
    sm_count: int | None = None,
)
```

**功能**：单后端的仅元数据 runner（对应手写 ``CuteDslMxfp4MoEWrapper``）。``num_experts`` /
``intermediate_size`` 为全局模型值；rank 布局显式给出（专家并行用 ``num_local_experts`` /
``local_expert_offset``，MoE-TP shard 用 ``intermediate_shard``）。``get_workspace_size`` 不需要
CUDA；``plan`` 会编译各形式并运行一次 warmup。

**参数**

- `num_experts`（`int`）：全局专家数。
- `top_k`（`int`）：每 token 选择的路由数，须满足 ``1 <= top_k <= min(32, num_experts)``。
- `hidden_size`（`int`）：隐藏维度，须为正的 128 倍数。
- `intermediate_size`（`int`）：中间维度，须为正的 128 倍数。
- `num_local_experts`（`int | None`，可选）：本 rank 的专家数（专家并行）。
- `local_expert_offset`（`int | None`，可选）：本 rank 专家区间的起始全局索引。
- `intermediate_shard`（`int | None`，可选）：MoE-TP 的中间维度 shard。
- `backend`（`str`，默认 `PACKAGE_BACKEND`）：必须等于包自身的后端（``"cake"`` 或 ``"cake_cute"``），
  否则抛 `ValueError`。
- `enable_pdl`（`bool`，默认 `False`）：是否启用 PDL。
- `swapab_n_tile`（`int`，默认 `SWAP_ROW_TILE`）：swap-AB 的 n tile，须为 8/16/32/64/128 之一。
- `swapab_max_tokens`（`int | None`，可选）：swap-AB 的最大 token 数（``>= 0``）。
- `swapab_tile_policy`（可选）：升序的 ``(max_tokens, tile)`` 列表，tile ∈ {8,16,32,64,128}。
- `sm_count`（`int | None`，可选）：计划所针对的 SM 数。为 `None` 时读取当前 CUDA 设备的 SM 数
  （``sm_count_source == "device"``）；无 CUDA 时使用 contract 的 148。显式传入是调用方的决定
  （针对另一部件的计划，runner 随后要求 cooperative grid 保持 co-resident）。

**属性**

- `parallel_mode -> str`：布局模式（``single`` / ``expert_parallel`` / ``moe_tensor_parallel``）。
- `sm_count` / `sm_count_source` / `num_local_experts` / `local_expert_offset` / `intermediate_shard`。

**方法**

- `decide(num_tokens: int) -> PlanDecision`：仅主机端决策（不编译、不启动）。
- `get_workspace_size(num_tokens: int) -> int`：返回工作区字节数（不需要 CUDA）。
- `plan(x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf, *, beta, linear_beta, workspace, output)`
  → 五类 Plan 之一：绑定缓冲区、编译各形式并运行一次 warmup。必须在 CUDA Graph 捕获之前调用；
  捕获中调用抛 `RuntimeError`。

**`plan` 输入约束**

- `x`：``[T, H]`` ``float8_e4m3fn``，连续。
- `x_sf`：``[T, H/32]`` ``uint8``（UE8M0 码），连续。
- `topk_ids`：``[T, top_k]`` ``int32`` 全局专家 id，连续。
- `topk_weights`：``[T, top_k]`` BF16 或 FP32，连续；``None`` 选择 packed BF16-in-``int32`` 路由
  布局。
- `w1` / `w1_sf` / `w2` / `w2_sf`：`prepare_cake_mxfp4_weights` 的操作数（tile-major 权重与
  512 字节 scale atoms），形状分别为
  ``[L*(2*I/128), H/128, 128, 64]``、``[L*(2*I/128), H/128, 4, 128]``、
  ``[L*(H/128), I/128, 128, 64]``、``[L*(H/128), I/128, 4, 128]``，均为 ``uint8``。
- `beta` / `linear_beta`：FP32 ``[num_local_experts]``，连续；`beta` 不能为 `None`（SiTU 需要
  运行期 beta）。
- `workspace`：连续 ``uint8`` 1-D，至少 ``get_workspace_size(T)`` 字节、256 字节对齐，不得与
  任何输入或 `output` 重叠。
- `output`：``[T, H]`` bfloat16，连续；不得与工作区重叠。

**返回**

- `CakeSwapAbPlan` / `CakeDensePlan` / `CakeSplitPlan` / `CakeHybridPlan` / `CakeMixed192Plan`，
  取决于 `decision.path`。对未由本链入队（``EXECUTABLE_PATHS`` 之外）或 IR 形式未追踪的行抛
  `NotImplementedError`。

**示例**

```python
import torch
from flashinfer.fused_moe.cake import CakeMxfp4MoEWrapper, prepare_cake_mxfp4_weights

weights = prepare_cake_mxfp4_weights(w1, w1_scale, w2, w2_scale)

runner = CakeMxfp4MoEWrapper(
    num_experts, top_k, hidden_size, intermediate_size,
    num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
    backend="cake",
)
decision = runner.decide(num_tokens)
workspace = torch.empty(runner.get_workspace_size(num_tokens), dtype=torch.uint8, device="cuda")

plan = runner.plan(x, x_sf, topk_ids, topk_weights,
                   weights["w1"], weights["w1_sf"], weights["w2"], weights["w2_sf"],
                   beta=beta, linear_beta=linear_beta, workspace=workspace, output=out)
plan.run()  # 可 CUDA Graph 捕获；无分配、无主机同步
```

---

### `prepare_cake_mxfp4_weights`

```python
prepare_cake_mxfp4_weights(w1, w1_scale, w2, w2_scale) -> dict[str, Any]
```

**功能**：把规范 rank-local ``[up, gate]`` MXFP4 bank 转换为 Cake 链的操作数。``w1``
``[L, 2*I_shard, H/2]``（up 行在前、gate 行在后）、``w1_scale`` ``[L, 2*I_shard, H/32]``、
``w2`` ``[L, H, I_shard/2]``、``w2_scale`` ``[L, H, I_shard/32]``（contract 的 ``physical_abi``）。
W1 的行与 scale 按 64 行的 up/gate 组交错，与 ``prepare_cute_dsl_mxfp4_weights`` 完全一致；两者随后
都采用 tile-major 布局与 512 字节 scale atom。所有 payload 字节均保留。

**参数**

- `w1, w1_scale, w2, w2_scale`（`torch.Tensor`）：3-D 连续 ``uint8`` 张量，形状如上。

**返回**

- `dict[str, Any]`：``{"w1": ..., "w1_sf": ..., "w2": ..., "w2_sf": ..., "num_local_experts": L,
  "intermediate_shard": rows_w1 // 2, "hidden_size": hk * 2}``。

**注意事项**

- 四个输入须为连续的 3-D ``uint8``；形状不一致时抛 `ValueError`。
- 为纯 torch 操作，无需 CUDA；每个 rank 只做一次。

---

### `resolve_layout` 与 `RankLayout`

```python
resolve_layout(
    num_experts: int,
    intermediate_size: int,
    *,
    num_local_experts: int | None = None,
    local_expert_offset: int | None = None,
    intermediate_shard: int | None = None,
) -> RankLayout
```

**功能**：显式的专家区间 / shard 形式（对应 ``resolve_mxfp4_moe_layout``）。``intermediate_shard``
小于 ``intermediate_size`` 时选择 MoE-TP shard（每个专家都本地）；合适的专家区间选择专家并行；
全部本地则为 ``single``。

`RankLayout` 为 ``@dataclass(frozen=True)``，字段：`mode`、`num_experts`、
`intermediate_size`、`num_local_experts`、`local_expert_offset`、`intermediate_shard`；
属性 ``gemm1_n = 2 * intermediate_shard``、``gemm2_k = intermediate_shard``。

**注意事项**

- `num_experts` 与 `intermediate_size` 须为正。
- 本地专家须构成非空的连续全局专家区间。
- shard 须满足 ``shard > 0``、``shard <= intermediate_size``、``intermediate_size % shard == 0``
  且 ``shard % 128 == 0``。
- 混合专家 / 张量并行不受支持，会抛 `ValueError`。

---

### `PlanDecision`

```python
@dataclass(frozen=True)
class PlanDecision
```

**功能**：手写计划由 ``(layout, T)`` 决定的全部内容：路径、子形式标志、按入队顺序的启动列表及每次
启动对应的 Cake 形式，且 ``supported`` = 每次启动都有已追踪的 Cake 形式（``missing_forms`` 命名
合并树缺失的手写 kernel 配置；任何启动都不会被替换为不同形式）。``mixed_backend`` 表示在
``cutedsl`` 后端下计划会启动一个 ``cuda_cpp``-only 模块（K6）。

**关键字段**

- `num_tokens`、`use_swapab`、`tile`、`group_rows`、`hybrid`、`mixed`、`split`、`wide192`、
  `two_stage`、`fused_route_cap`、`fused_routing`、`single_tile_per_expert`、`clear_output`、
  `pdl`、`dep_prefetch`、`token_index`、`tiles`、`rows`、`launches`。
- `gemm1` / `gemm2`（`GemmForm | None`）、`supported`（`bool`）、`reason`（`str`）。
- `path`：``plain`` / ``split_two_stage`` / ``hybrid`` / ``mixed192`` / ``dense``。
- `launch_plan`（`tuple[Launch, ...]`）、`missing_forms`、`mixed_backend`、`mixed192`、
  `split_dense`、`dense`（`DenseSelection | None`）。

**注意事项**

- `decision.supported` 为 ``False`` 仅当手写选择需要一个本链尚未追踪的 IR 形式；
  `decision.reason` 命名缺失形式。
- 其它行报告 ``supported`` 为 ``True``，但 `plan` 还会拒绝 ``decision.path`` 不在
  ``EXECUTABLE_PATHS`` 中的行（例如 MoE-TP 17 到 127 token 上的 ``two_stage`` finalize 被追踪
  但本链不入队）。需要硬性答案的调用方应同时检查 ``decision.path in EXECUTABLE_PATHS`` 与
  ``decision.supported``。

---

### 可执行计划类

下列计划都共享同一契约：调用方拥有输出与工作区，在 CUDA Graph 捕获之外准备；构造函数会运行一次链
作为 warmup / 有效性后置条件；`run()` 不分配、不 JIT、不同步主机，并且可被捕获。

#### `CakeSwapAbPlan`

plain 链：``run()`` 在调用方当前 stream 上入队 routing → GEMM1 → GEMM2。

- 属性 `executed_chain -> dict`：链记录。
- `launch_sequence() -> tuple[tuple[str, tuple[int, int, int]], ...]`：``("routing_fused", ...)``、
  ``("gemm1_swapab_situ", ...)``、``("gemm2_swapab_finalize", ...)`` 及其 grid。
- `run()`：入队三次启动并返回绑定的输出。

#### `CakeDensePlan`

dense 链（K1 ``_RoutePreprocess`` + K6 ``moe_sort`` init + cooperative kernel + W1 dense gather
GEMM1（SiTU，blocked output scales）+ W11 dense finalize GEMM2）。dual-tile 行在 ``out_alt_*``
列表上追加 K6 256 行 alternate padding 及 2-CTA alternate GEMM1 / GEMM2 启动。

- 属性 `dual -> bool`：是否为 dual-tile 对（已入队 alternate 启动）。
- 属性 `executed_chain -> dict`。
- `launch_sequence()` / `run()`：同上语义（dense 链）。

#### `CakeSplitPlan`

split two-stage 链：split-layout 融合 routing（清零输出）、W1 row-group gather GEMM1、W11
row-group finalize GEMM2（对 wide 专家的 ``SWAP_SPLIT_WIDE_TILE`` 行组做 reduce-add）、swap-AB
SiTU GEMM1、swap-AB partial GEMM2（对 narrow 组）、K7 ``finalize_rows`` 累加。

- `executed_chain` / `launch_sequence()` / `run()`。

#### `CakeHybridPlan`

hybrid 链的可执行计划（需 ``HYBRID_CHAIN_EXECUTABLE`` 打开，否则 `plan` 抛
`NotImplementedError`）。

- `executed_chain` / `launch_sequence()` / `run()`。

#### `CakeMixed192Plan`

mixed192 链（双流，服务 8192 及以上 token 的 MoE-TP 行）的可执行计划。

- `executed_chain` / `launch_sequence()` / `run()`。

**注意事项**

- runner 会保留一个按执行 stream 与几何为键的、有界的工作区 / 内核缓存；不同 token 数的形状与
  路由张量在捕获前需各自 warm up；未见过的工作区形状或未验证的路由张量代次在捕获期间会被拒绝，
  而非隐式初始化。
- 所有 kernel 都是几何特化的；准备与 warmup 应在 CUDA Graph 捕获之前完成。

---

### Kernel 加载器与 manifest 接口

以下函数 / 对象由生成包提供（两个后端同名）：

| 名称 | 说明 |
| --- | --- |
| `ARCH` | 目标架构（``sm_103a``）。 |
| `BACKEND` / `PACKAGE_BACKEND` | 后端名（``"cake"`` / ``"cake_cute"``）。 |
| `is_available()` | 当前 checkout 是否已注册/可构建所需程序。 |
| `load(...)` | 加载（必要时编译）指定生成 source。 |
| `compile_all(...)` | 编译该包全部 kernel。 |
| `manifest()` / `cake_revision` | 内容寻址 manifest 与 pin 的 Cake revision。 |
| `modules()` | 模块清单。 |
| `select_gemm1` / `select_gemm2` / `select_routing` | 形式选择函数。 |
| `unsupported_forms()` | 该包声明的不支持形式（见下）。 |
| `kernel_stages(plan)` | 计划用到的 kernel 舞台元组。 |

kernels 通过每个模块 ``route.form`` 中记录的 trace-time 常量选择（`kind`、`n_tile`、`kbps`、
`m_group` 等）；模块文件名是内容指纹。``gate`` 命名每个形式在 Cake 中经过的验证：
``e2e_registry``（编译、静态分析与两后端 GPU 正确性）、``plan_gate``（由计划在 contract 行上
选择并端到端验证）、``plan_variant``（plan-gated 形式的路由模式变体，可通过公开 API 到达但不在
contract 行上）。

## 说明 / 注意事项

- **Host plan**：两个包附带同一 host plan（``cake_mxfp4_situ_moe_plan.py``），按后端渲染；其决策
  表逐条复现手写 ``CuteDslMxfp4MoEWrapper`` / ``Mxfp4MoESwapAbPlan``（``swapab_max_tokens``、
  tile 策略、group rows、stage 深度、weight 分组、fused-routing cap、PDL 链、工作区布局、
  启动序列）。计划按设备决定：`CakeMxfp4MoEWrapper` 在构造时读取一次 CUDA 设备的 SM 数，所有依赖
  SM 的选择（cooperative ``moe_sort`` grid 为 ``SMs - 8`` CTA；per-thread 有界 / 通用状态
  ``T * top_k <= 4 * (SMs - 8) * 896``；persistent GEMM grid ``min(tiles, SMs)``）都由这一个计数
  推出。`plan()` 会拒绝 SM 数与计划所决定的不一致的设备。
- **EXECUTABLE_PATHS**：`plan` 会以 `NotImplementedError` 拒绝两类行（IR 形式未追踪；
  或 ``decision.path`` 不在 ``EXECUTABLE_PATHS`` 内，例如 MoE-TP 17 到 127 token 上的
  ``two_stage`` finalize）。两流 ``mixed192`` 链服务 8192 token 及以上的 MoE-TP 行。
- **不支持的生成形式**：某后端不构建的形式在该包 manifest 的 ``contract.unsupported_forms``
  中声明原因。``backend="cake_cute"`` 缺少 cooperative ``moe_sort`` 对（其寄存器状态跨越
  ``grid.sync``，CuTe DSL 后端拒绝）；CUDA 包携带两个 kernel。
- **Manifest**：每个 manifest 的 ``contract`` 记录 operator、backend、Cake revision、架构、
  PDL 启动状态、形式表（每形式的 ``gate`` 与 ``backends``）、服务的几何、决策表服务的 contract
  行（含每个的手写路径、每包以每种启动形式携带哪些、以及 ``EXECUTABLE_PATHS``）、不支持形式，
  以及包内每个集成文件的 SHA-256。kernel 加载器在编译前用 manifest 认证每个生成 source。
- **权重准备**：`prepare_cake_mxfp4_weights` 只做纯 torch 的字节重排；在最终 device-local 权重
  载入之后、warmup 或 graph capture 之前准备一次并复用。

## 相关链接 / 参考

- 上游文档 `docs/api/cake_mxfp4_situ_moe.rst`。
- `flashinfer.fused_moe.cake_mxfp4_situ_moe` 的生成 csrc 与 manifest 位于
  ``csrc/fused_moe/cake_mxfp4_situ_moe/``。

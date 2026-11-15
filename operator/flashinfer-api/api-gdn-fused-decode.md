# 融合 GDN 解码步（`flashinfer.gdn_fused_decode_step`）

> 源码：`flashinfer/gdn_kernels/experimental/gdn_fused_decode.py`
> 上游文档：`docs/api/gdn_fused_decode.rst`
> 适用版本：FlashInfer 0.7.2

单个 token 的融合 Gated Delta-Rule 解码步，作用于分页（paged）的 conv / ssm 状态池。它把整条
per-layer 服务链——b/a 投影 GEMV、depthwise causal conv1d 状态更新、q/k/v head 拆分，以及
`gated_delta_rule_decode_pretranspose` 的门控 delta-rule 解码——折叠为一次操作，并原地更新两个
状态池。

本操作**没有 backend 参数、也没有环境变量开关**：某次调用运行 SM120 专用 kernel 还是可组合的
torch 路径，由库根据 workload registry 与设备自行决定。`gdn_fused_decode_step_supported` 以廉价、
capture-safe 的方式预先给出该决策，便于框架对无法加速的形状保留自己的组合实现。实现（dispatch、
workload registry 与 kernel）位于 `flashinfer/gdn_kernels/experimental/` 下。*Experimental* 描述
的是代码位置，而非该操作的调用方式。

## 导入

```python
from flashinfer import gdn_fused_decode_step, gdn_fused_decode_step_supported
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `gdn_fused_decode_step_supported` | 函数 | 廉价的 routing 探测：判断给定几何形状在该设备上是否会走专用 kernel。 |
| `gdn_fused_decode_step` | 函数 | 执行融合的单 token GDN 解码步，原地更新两个状态池。 |

## API 详情

### `gdn_fused_decode_step_supported`

```python
gdn_fused_decode_step_supported(
    batch_size: int,
    hidden_size: int = 5120,
    n_ba: int = 96,
    qkv_dim: int = 10240,
    num_qk_heads: int = 16,
    num_v_heads: int = 48,
    head_dim: int = 128,
    conv_width: int = 4,
    conv_state_len: int = 3,
    device: Optional[torch.device] = None,
    conv_state_layout: str = "SD",
) -> bool
```

**功能**：面向框架使用者的廉价 routing 探测。当 `gdn_fused_decode_step` 会在该设备上用专用 kernel
服务该几何形状时返回 `True`：即该几何形状与 conv-state 池布局在该设备计算能力对应的
`gdn_fused_decode_registry.json` 中已注册、且注册的实现可导入、也未被此前的 kernel 失败锁死。
探测为纯主机侧（capture-safe）。

**参数**

- `batch_size`（`int`）：正在考虑的调用中的解码行数。唯一没有默认值的参数，因为它是 shipped
  registry 行中唯一变化的量（B ∈ {1, 2, 4, 8}）。
- `hidden_size`（`int`，默认 `5120`）：模型 hidden size；`[B, hidden_size]` 输入行宽。
- `n_ba`（`int`，默认 `96`）：b/a 投影的组合宽度，即 `w_ba.shape[1]`。
- `qkv_dim`（`int`，默认 `10240`）：head 拆分前打包 q/k/v 投影的宽度。
- `num_qk_heads`（`int`，默认 `16`）：query/key head 数（`h_q`）。它不是独立轴：dispatch 时由
  `(qkv_dim - num_v_heads * head_dim) / (2 * head_dim)` 还原；此处接受它是让调用方能直接声明，
  而不依赖该算式。
- `num_v_heads`（`int`，默认 `48`）：value head 数（`hv`）。
- `head_dim`（`int`，默认 `128`）：每 head 维度（`d`）；循环状态为 `[hv, d, d]`。
- `conv_width`（`int`，默认 `4`）：depthwise causal 卷积宽度。
- `conv_state_len`（`int`，默认 `3`）：保留的 conv-state 长度，通常为 `conv_width - 1`。
- `device`（`torch.device`，可选）：决定答案的设备计算能力。`None` 与无 index 的 `"cuda"` 均表示
  *当前设备*，每次调用都会重新解析。正因如此该公开入口不适合缓存。
- `conv_state_layout`（`str`，默认 `"SD"`）：分页 conv-state 池行的物理布局：`"SD"` 表示
  `(state_len, dim)`（vLLM 分配的形式），`"DS"` 表示 `(dim, state_len)`。只有 registry 中存在的
  布局会被服务；实际路由依据*由真实张量 stride 推断*的布局，因此该参数只是一个用于路由的声明，
  dispatch 时从不采信。

**返回**

- `bool`：当 `gdn_fused_decode_step` 会在该设备上用专用 kernel 服务该几何形状时返回 `True`。
  返回 `False` 不是错误：该操作仍可调用且正确，只是走可组合路径，因此调用方应保留自己优化的
  组合实现。

**注意事项**

- 几何参数默认到某个已发布的 layer 形状，仅为测试与文档方便；registry 中注册了多种几何形状，因此
  询问不同模型的调用方必须传入全部参数。
- 本函数只回答*是否支持*，从不回答*策略*：决定不使用该操作的框架不应调用它，而非指望它返回
  `False`。
- 服务侧每层每解码步调用一次，因此重复答案必须廉价：几何 → 答案的映射被 memoize，只有某个几何
  形状的首次调用会触碰 registry 或设备。该 memo 位于内层，且故意不放在本函数上：`device=None`
  表示“当前此刻的设备”，若以本签名为键缓存会在 `None` 下钉住答案，并在随后
  `torch.cuda.set_device` 到不同计算能力设备时继续返回旧答案。内层 memo 以*解析后的*计算能力为键。

**示例**

```python
import torch
import flashinfer

if flashinfer.gdn_fused_decode_step_supported(batch_size=4):
    out = flashinfer.gdn_fused_decode_step(
        hidden_states, w_ba, mixed_qkv, conv_weight, conv_bias,
        conv_state, A_log, dt_bias, scale, ssm_state, state_indices,
    )
else:
    # 保留框架自己的组合实现
    ...
```

---

### `gdn_fused_decode_step`

```python
gdn_fused_decode_step(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：融合的单 token GDN 解码步，作用于分页 conv / ssm 状态池。把 per-layer 服务链（b/a 投影
GEMV、causal conv1d 更新、q/k/v 拆分、带 qk-L2-norm 的门控 delta-rule 解码）折叠为一次操作。
两个状态池均**原地**更新并返回。

**参数**

- `hidden_states`（`torch.Tensor`）：层输入，形状 `[B, hidden]`，bfloat16。
- `w_ba`（`torch.Tensor`）：融合 b/a 投影权重，形状 `[hidden, 2*HV]`，bfloat16（列 `[:HV]` 产生
  beta gate 输入，`[HV:]` 产生 decay 输入）。
- `mixed_qkv`（`torch.Tensor`）：原始（conv 前）融合 q/k/v 通道，形状 `[B, qkv_dim]`，bfloat16，
  其中 `qkv_dim = (2*H_q + HV) * D`。
- `conv_weight`（`torch.Tensor`）：depthwise conv 权重，形状 `[qkv_dim, width]`，bfloat16。
- `conv_bias`（`torch.Tensor`）：conv 偏置，形状 `[qkv_dim]`，bfloat16。
- `conv_state`（`torch.Tensor`）：分页 conv-state 池，作为逻辑视图 `[P, qkv_dim, width-1]`，保存
  最近 `width-1` 个原始通道输入，bfloat16，原地更新。支持两种物理布局：SD 池
  （`(width-1, qkv_dim)` 行，vLLM 默认——传入 `pool.transpose(-1, -2)`）或 DS-dense 池
  （`(qkv_dim, width-1)` 行，连续）；page stride 可带 padding。
- `A_log`（`torch.Tensor`）：对数衰减参数，形状 `[HV]`，float32。
- `dt_bias`（`torch.Tensor`）：衰减偏置，形状 `[HV]`，bfloat16。
- `scale`（`float`，可选）：query 缩放。`None` **和** `0.0` 都选择默认 `1/sqrt(D)`：零 scale 会使
  整个 attention 输出为零，故被视为“未设置”而非被采信。要覆盖请传入显式非零值。
- `ssm_state`（`torch.Tensor`）：分页 fp32 循环状态池，形状 `[P, HV, V, K]`（V-major / K-last），
  行 stride 可带 padding（`stride(0) >= HV*V*K`），原地更新。
- `state_indices`（`torch.Tensor`）：每 batch 池 slot 索引，形状 `[B]`，int32。两个池用同一个值
  索引，因此每个 batch 项读写同一 slot。
- `use_qk_l2norm`（`bool`，默认 `True`）：是否对 q 和 k 做 L2 归一化。
- `out`（`torch.Tensor`，可选）：预分配 attention 输出，形状 `[B, 1, HV, V]`，bfloat16，稠密
  （连续）。提供时原地写入并返回（省去一次向框架自有输出缓冲的拷贝）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：`(output, conv_state, ssm_state)`，`output` 形状
  `[B, 1, HV, V]`（bfloat16），两个池均原地修改。

**注意事项**

- 没有 backend 选择器、也没有环境变量开关：库根据 registry 与设备在专用 kernel 与可组合路径间
  选择。该选择可观测——`gdn_fused_decode_step_supported` 在调用前即可回答，且不运行任何东西——
  但不可按调用覆盖。不想要该操作的框架直接不调用它。
- 专用 kernel 在 SM120 上服务已注册的 traced workload 签名；在其它任何设备上，或对 registry 未
  列出的几何形状，本函数就是精确的可组合 torch 实现。
- **Padding / inactive 行**：负索引（vLLM 的 `PAD_SLOT_ID` 为 `-1`）标记不拥有池 slot 的 batch
  项——即 CUDA-graph replay 在活跃请求数与捕获 batch size 之间携带的行。此类项被**完全跳过**：
  既不读也不写 `conv_state` 与 `ssm_state`，其 `output` 行写为**零**。这与
  `gated_delta_rule_decode_pretranspose` 的 float32 路径契约一致。
  索引 `>= P`（池的 leading extent）是**调用方 bug，而非 padding 约定**：它们既不被 clamp 也不被
  跳过，导致的访问越界、效果未定义。只有负值表示 padding。
- 专用 kernel 失败不会破坏本操作：它会警告一次，把该实现锁死到进程余下部分，并由可组合路径服务
  本次调用。
- CUDA graphs：每个专用实现在其 `(layer geometry, batch, scale, conv-state layout)` 变体的首次
  eager dispatch 时惰性编译；capture 期间只有当该变体已 warm 才录制，否则为该形状烘焙
  （capture-safe 的）可组合路径。

**示例**

```python
import torch
import flashinfer

B, hidden, HV, H_q, D, W = 4, 5120, 48, 16, 128, 4
qkv_dim = (2 * H_q + HV) * D
P = 256

hidden_states = torch.randn(B, hidden, device="cuda", dtype=torch.bfloat16)
w_ba = torch.randn(hidden, 2 * HV, device="cuda", dtype=torch.bfloat16)
mixed_qkv = torch.randn(B, qkv_dim, device="cuda", dtype=torch.bfloat16)
conv_weight = torch.randn(qkv_dim, W, device="cuda", dtype=torch.bfloat16)
conv_bias = torch.zeros(qkv_dim, device="cuda", dtype=torch.bfloat16)
conv_state = torch.zeros(P, qkv_dim, W - 1, device="cuda", dtype=torch.bfloat16)
A_log = torch.zeros(HV, device="cuda", dtype=torch.float32)
dt_bias = torch.zeros(HV, device="cuda", dtype=torch.bfloat16)
ssm_state = torch.zeros(P, HV, D, D, device="cuda", dtype=torch.float32)
state_indices = torch.arange(B, device="cuda", dtype=torch.int32)

out, conv_state, ssm_state = flashinfer.gdn_fused_decode_step(
    hidden_states, w_ba, mixed_qkv, conv_weight, conv_bias,
    conv_state, A_log, dt_bias, None, ssm_state, state_indices,
)
```

## 相关链接 / 参考

- `docs/api/gdn_decode.rst`：`gated_delta_rule_decode_pretranspose`，本操作第 4 步所复用的解码。
- `flashinfer/gdn_kernels/experimental/README.md`：registry schema 与 impl-module 接口说明。

# 扩散算子（`flashinfer.diffusion_ops`）

> 源码：`flashinfer/diffusion_ops/`（及重新导出的 `flashinfer/norm/__init__.py`）
> 上游文档：`docs/api/diffusion_ops.rst`
> 适用版本：FlashInfer 0.7.2

面向 diffusion-transformer 推理的融合算子。本模块以 MiniMax-H3（视频 DiT）与 WAN 2.2 5B
（DIT 自注意力）为例，提供预注意力投影、输出投影、MLP 块、变长注意力，以及残差 / LayerNorm /
scale-shift / RoPE 等融合算子，覆盖 BF16、FP8（W8A8）与 NVFP4（W4A4）等精度路径。

## 导入

```python
from flashinfer.diffusion_ops import (
    minimax_h3_bf16_pre_attention,
    minimax_h3_fp8_pre_attention,
    minimax_h3_nvfp4_pre_attention,
    # ...
    fused_qk_rmsnorm_rope,
    fused_dit_residual_layernorm_scale_shift,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `minimax_h3_bf16_pre_attention` | 函数 | MiniMax-H3 BF16 预注意力投影阶段。 |
| `minimax_h3_fp8_pre_attention` | 函数 | MiniMax-H3 FP8（W8A8）融合预注意力（SM120）。 |
| `minimax_h3_nvfp4_pre_attention` | 函数 | MiniMax-H3 NVFP4 融合预注意力（SM120）。 |
| `quantize_minimax_h3_qkv_weight_fp8` | 函数 | QKV 权重逐输出通道 E4M3 量化。 |
| `quantize_minimax_h3_qkv_weight_nvfp4` | 函数 | QKV 权重 FlashInfer NVFP4 量化。 |
| `minimax_h3_fp8_out_proj` | 函数 | MiniMax-H3 FP8 输出投影 + 门控残差（SM120）。 |
| `minimax_h3_nvfp4_out_proj` | 函数 | MiniMax-H3 NVFP4 输出投影 + 门控残差（SM120）。 |
| `quantize_minimax_h3_o_weight_fp8` | 函数 | 输出权重逐输出通道 E4M3 量化。 |
| `quantize_minimax_h3_o_weight_nvfp4` | 函数 | 输出权重 FlashInfer NVFP4 量化。 |
| `minimax_h3_sm120_varlen_attention_fp8` | 函数 | FP8 非因果 packed-varlen 自注意力（SM120）。 |
| `minimax_h3_sm120_varlen_attention_nvfp4` | 函数 | NVFP4 非因果 packed-varlen 自注意力（SM120）。 |
| `minimax_h3_sm120_varlen_attention_nvfp4_nodelta` | 函数 | 上述 NVFP4 路由的未补偿变体。 |
| `minimax_h3_fc1_swiglu_fp8` | 函数 | FP8 RMSNorm + AdaLN + FC1 GEMM + SwiGLU（SM120）。 |
| `prepare_minimax_h3_fc1_weight_fp8` | 函数 | 为 `minimax_h3_fc1_swiglu_fp8` 量化 FC1 权重。 |
| `minimax_h3_mlp_fp8_sm120` | 函数 | FP8 MiniMax-H3 MLP 块（SM120）。 |
| `minimax_h3_mlp_nvfp4_sm120` | 函数 | NVFP4 MiniMax-H3 MLP 块（SM120）。 |
| `prepare_minimax_h3_fc2_weight_fp8` | 函数 | 为 `minimax_h3_mlp_fp8_sm120` 量化 FC2 权重。 |
| `prepare_minimax_h3_fc2_weight_nvfp4_sm120` | 函数 | 为 `minimax_h3_mlp_nvfp4_sm120` 量化 FC2 权重。 |
| `fused_qk_rmsnorm_rope` | 函数 | 融合 QK RMSNorm + 3D RoPE + V 拷贝。 |
| `fused_dit_residual_layernorm_scale_shift` | 函数 | 融合残差 + LayerNorm + scale/shift。 |
| `fused_dit_gate_residual_layernorm_scale_shift` | 函数 | 融合 gate + 残差 + LayerNorm + scale/shift。 |
| `fused_dit_gate_residual_layernorm_gamma_beta` | 函数 | 融合 gate + 残差 + LayerNorm(gamma, beta)。 |

## API 详情

### `minimax_h3_bf16_pre_attention`

```python
minimax_h3_bf16_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    ulysses_degree: int,
    out: torch.Tensor,
    eps: float = _EPS,
    qk_eps: Optional[float] = None,
    rope_positions: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：运行 MiniMax-H3 的 BF16 预注意力投影阶段。该操作应用输入 RMSNorm、索引化
AdaLN、BF16 QKV 投影、逐 head Q/K RMSNorm、部分 3-D split-half NeoX RoPE，以及
destination-major 输出打包，作为当前流上的两次内核启动：归一化与调制后的激活写入 BF16
`[M, 5376]` 工作区，随后持久化 QKV GEMM 通过 TMA 流式读取它，其 epilogue 应用 Q/K
RMSNorm、RoPE 与打包。消费 `out` 的集合通信在此操作之外。

**参数**

- `x`（`torch.Tensor`）：连续 BF16 输入，形状 `[M, 5376]`。
- `x_norm_weight`（`torch.Tensor`）：连续 BF16 输入 RMSNorm 权重，形状 `[5376]`。
- `adaln_scale`、`adaln_shift`（`torch.Tensor`）：BF16 AdaLN 表，形状 `[rows, 5376]`，
  `rows >= 1`。`stride(1)` 必须为 `1`；`stride(0)` 可超过 `5376`（例如引擎
  `[rows, 6 * 5376]` 调制投影的列块，为 `6 * 5376`），且必须是 8 个元素（16 字节）的倍数；
  数据指针须 16 字节对齐。
- `adaln_index`（`torch.Tensor`）：连续 int64 行索引，形状 `[M]`。`[0, rows)` 内的值选择
  表行；其它 int64 值在 CUDA 内核中被守护，使输出行为全零而非越界寻址。
- `qkv_weight`（`torch.Tensor`）：连续 BF16 融合 QKV 投影权重，物理形状 `[21504, 5376]`，
  采用引擎驻留行序 `[qkv_kind, head, head_dim]`（`[q_all | k_all | v_all]`）。
- `q_norm_weight`、`k_norm_weight`（`torch.Tensor`）：连续 BF16 逐 head RMSNorm 权重，形状 `[128]`。
- `rope_cos_sin`（`torch.Tensor`）：连续 BF16 缓存，形状 `[S, 96]`，`S >= 1`。列 `[0, 48)`
  为 frame/height/width 余弦，列 `[48, 96)` 为对应正弦。投影的第 `m` 行使用缓存行
  `rope_positions[m]`。RoPE 变换 Q/K 的维度 `[0, 96)`；维度 `[96, 128)` 直通。
- `ulysses_degree`（`int`，关键字）：目的地数量，取 `1`、`2`、`4` 或 `8`。
- `out`（`torch.Tensor`，关键字）：调用方拥有的连续 BF16 目的地，形状 `[P, M, 56 // P, 3, 128]`。
- `eps`（`float`，关键字，默认 `_EPS`）：输入 RMSNorm 的 epsilon（运行时参数）。
- `qk_eps`（`float`，可选）：逐 head Q/K RMSNorm 的 epsilon；`None` 使用 `eps`。
- `rope_positions`（`torch.Tensor`，可选）：连续 int64 `[M]`，每 token 的缓存行。超出
  `[0, S)` 的位置在设备上钳位。`None` 表示恒等（第 `m` 行使用缓存行 `m`，故要求 `S >= M`）；
  恒等以按 `(M, device)` 缓存的 `arange` 张量传入，首次该 `M` 调用时分配并复用，因此恒等
  路径在同一 `M` 的一次 eager 调用预热后即 CUDA-graph 安全。
- `workspace`（`torch.Tensor`，可选）：连续 BF16 `[M, 5376]` 暂存（会被覆盖）。`None` 时每次
  调用从缓存分配器分配；固定内存或捕获 CUDA graph 的调用方可传入自己的缓冲。

**返回**

- `torch.Tensor`：与 `out` 相同的张量。

**注意事项**

- CUDA 内核独立守护其 AdaLN 表加载，因此畸形索引不会形成越界地址。有效索引保持
  MiniMax-H3 检查点语义，且无需同步的主机归约。
- 这是所有受支持目的地数量（`P in {1, 2, 4, 8}`）的直接入口；在 SM100a 与 SM103a 的每个
  生产中心上，两次启动阶段都比分段 norm + cuBLAS + 融合后处理链更快，包括 `P=1`。

---

### `minimax_h3_fp8_pre_attention`

```python
minimax_h3_fp8_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = MINIMAX_H3_DEFAULT_EPS,
    out_mode: str = 'bf16',
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> MiniMaxH3PreAttentionOutput
```

**功能**：SM120（RTX 5090 / RTX PRO 6000 Blackwell）上的融合 FP8（W8A8）MiniMax-H3
预注意力。对 `M` 个 token 计算：

```text
n   = bf16(rmsnorm(x, eps) * x_norm_weight)
a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))   # i = adaln_index[t]
a_q = e4m3(a / s_t),  s_t = amax_t(|a|) / 448               # per-token 激活缩放
y   = bf16(a_q @ qkv_weight_qᵀ * s_t * qkv_weight_scale)    # [M, 56 heads x (Q,K,V) x 128]
q   = rope(bf16(rmsnorm(y_q, eps) * q_norm_weight)),  k 同理用 k_norm_weight，v = y_v
```

RoPE 以前 96 通道的通道对 `(d, d + 48)` 旋转，`rope_cos_sin[t] = [cos(48), sin(48)]`，
通道 96..127 直通。两次启动（norm/AdaLN/量化与融合持久 GEMM）都在当前流上。

**参数**

- `x`（`torch.Tensor`）：BF16 `[M, 5376]` 隐状态。
- `x_norm_weight`（`torch.Tensor`）：BF16 `[5376]` pre-norm 权重。
- `adaln_scale`、`adaln_shift`（`torch.Tensor`）：BF16 `[rows, 5376]` AdaLN 调制表。
- `adaln_index`（`torch.Tensor`）：Int32 `[M]`，为每 token 选择一行 AdaLN 调制。
- `qkv_weight_q`、`qkv_weight_scale`（`torch.Tensor`）：由
  `quantize_minimax_h3_qkv_weight_fp8` 产生的 `float8_e4m3fn [21504, 5376]` 与 FP32 `[21504]`。
  输出列 `h * 384 + kind * 128 + d` 是 head `h` 的通道 `d`，`kind` 0=Q、1=K、2=V。
- `q_norm_weight`、`k_norm_weight`（`torch.Tensor`）：BF16 `[128]` 逐 head RMSNorm 权重。
- `rope_cos_sin`（`torch.Tensor`）：BF16 `[M, 96]` = 每 token `[cos(48), sin(48)]`。
- `eps`（`float`，关键字，默认 `MINIMAX_H3_DEFAULT_EPS`）：pre-norm 与 Q/K norm 的 RMSNorm epsilon。
- `out_mode`（`str`，关键字，默认 `"bf16"`）：
  - `"bf16"`：Q/K/V 为 BF16 `[M, 56, 128]`。
  - `"e4m3"`：`float8_e4m3fn [M, 56, 128]`，按调用方逐张量 `q_descale`/`k_descale`/`v_descale`
    存储 `RN(value / descale)`。
  - `"nvfp4"`：打包 E2M1 `uint8 [M, 56, 64]` 加 UE4M3 block-16 缩放 `uint8 [M, 56, 8]`
    （行主序），使用调用方逐张量 `*_global_scale`（FlashInfer `fp4_quantize` 语义）。
- `q, k, v, q_sf, k_sf, v_sf`（`torch.Tensor`，可选）：可选预分配输出；省略时分配。
- `q_descale, k_descale, v_descale`（`Scalar`，可选）：`out_mode == "e4m3"` 时必需的逐张量
  E4M3 反量化缩放。
- `q_global_scale, k_global_scale, v_global_scale`（`Scalar`，可选）：`out_mode == "nvfp4"` 时
  必需的 NVFP4 全局缩放。
- `act_q, act_scale`（`torch.Tensor`，可选）：可选 stage-1 工作区
  （`float8_e4m3fn [M, 5376]`、FP32 `[M]`）；省略时分配。

**返回**

- `MiniMaxH3PreAttentionOutput`：`(q, k, v, q_sf, k_sf, v_sf)`；除非 `out_mode == "nvfp4"`，
  缩放项为 `None`。

---

### `minimax_h3_nvfp4_pre_attention`

```python
minimax_h3_nvfp4_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    qkv_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = MINIMAX_H3_DEFAULT_EPS,
    out_mode: str = 'bf16',
    alpha: Optional[float] = None,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> MiniMaxH3PreAttentionOutput
```

**功能**：SM120（RTX 5090 / RTX PRO 6000 Blackwell）上的融合 NVFP4 MiniMax-H3 预注意力。
pre-norm、AdaLN、GEMM epilogue 与输出格式与 `minimax_h3_fp8_pre_attention` 相同，但使用
FlashInfer 约定的 NVFP4 操作数：归一化激活 `a` 逐 token 量化，如同
`fp4_quantize(a, act_global_scale, sf_vec_size=16, is_sf_swizzled_layout=False)`
（E2M1 码 `uint8 [M, 2688]` + UE4M3 块缩放 `uint8 [M, 336]`），权重来自
`quantize_minimax_h3_qkv_weight_nvfp4`（128x4 swizzled 缩放），累加器按
`alpha = 1 / (act_global_scale * qkv_weight_global_scale)` 重缩放。

**参数**

- `x`、`x_norm_weight`、`adaln_scale`、`adaln_shift`、`adaln_index`、`q_norm_weight`、
  `k_norm_weight`、`rope_cos_sin`、`eps`：同 `minimax_h3_fp8_pre_attention`。
- `qkv_weight_q`、`qkv_weight_sf`（`torch.Tensor`）：`quantize_minimax_h3_qkv_weight_nvfp4` 的输出。
- `qkv_weight_global_scale`（`Scalar`）：NVFP4 权重全局缩放，浮点或单元素张量。
- `act_global_scale`（`torch.Tensor`）：FP32 `[1]` CUDA 张量，校准后归一化激活的
  `448 * 6 / amax`；在设备上读取，无主机同步。
- `out_mode`（`str`，关键字，默认 `"bf16"`）：输出格式 `"bf16"`、`"e4m3"` 或 `"nvfp4"`；
  形状与量化约定同 `minimax_h3_fp8_pre_attention`。
- `alpha`（`float`，可选）：`1 / (act_global_scale * qkv_weight_global_scale)`；显式传入可
  避免读取全局缩放所需的主机同步；省略时由它们导出。
- `q, k, v`（`torch.Tensor`，可选）：可选预分配输出。
- `q_sf, k_sf, v_sf`（`torch.Tensor`，可选）：可选预分配 NVFP4 缩放输出；仅 `out_mode ==
  "nvfp4"` 时使用，其它输出模式下忽略。
- `q_descale, k_descale, v_descale`、`q_global_scale, k_global_scale, v_global_scale`、
  `act_q, act_sf`：语义同 FP8 变体（`act_q`/`act_sf` 为打包 E2M1 `uint8 [M, 2688]` 与
  UE4M3 缩放 `uint8 [M, 336]`）。

**返回**

- `MiniMaxH3PreAttentionOutput`：`(q, k, v, q_sf, k_sf, v_sf)`；除非 `out_mode == "nvfp4"`，
  缩放项为 `None`。

---

### `quantize_minimax_h3_qkv_weight_fp8`

```python
quantize_minimax_h3_qkv_weight_fp8(
    qkv_weight: torch.Tensor,
    chunk_rows: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：对 BF16 `[21504, 5376]` 融合 QKV 权重做逐输出通道 E4M3 量化。返回
`(qkv_weight_q float8_e4m3fn [21504, 5376], qkv_weight_scale float32 [21504])`，其中
`qkv_weight_scale[n] = RN(amax(row n) / 448)`，`qkv_weight_q = RN(row / scale)`。

**参数**

- `qkv_weight`（`torch.Tensor`）：BF16 `[21504, 5376]` 融合 QKV 权重。
- `chunk_rows`（`int`，默认 `2048`）：每次量化的块大小。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：量化权重与 FP32 逐行缩放。

---

### `quantize_minimax_h3_qkv_weight_nvfp4`

```python
quantize_minimax_h3_qkv_weight_nvfp4(
    qkv_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：对 BF16 `[21504, 5376]` 融合 QKV 权重做 FlashInfer NVFP4 量化。返回
`(qkv_weight_q uint8 [21504, 2688], qkv_weight_sf uint8 (128x4 swizzled 布局),
qkv_weight_global_scale float32 [1])`，与 `flashinfer.fp4_quantize` 在 `sf_vec_size=16`、
`is_sf_swizzled_layout=True` 下的产出完全一致。

**参数**

- `qkv_weight`（`torch.Tensor`）：BF16 `[21504, 5376]` 融合 QKV 权重。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：量化权重、128x4 swizzled 块缩放、
  FP32 `[1]` 全局缩放。

---

### `minimax_h3_fp8_out_proj`

```python
minimax_h3_fp8_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：SM120（GB202：RTX 5090 / RTX PRO 6000 Blackwell）上的 MiniMax-H3 FP8（W8A8）
注意力输出投影，融合门控残差。一次启动：`attn_out` 的逐 token E4M3 量化
（`scale = RN(amax / 448)`）由持久内核 producer warpgroup 的空闲 warp 执行，并通过逐 tile
ready 标志与 `[M, 7168] x [7168, 5376]` 的 `mma.sync` GEMM 重叠，其 epilogue 应用逐 token x
逐通道反量化缩放，将投影 `o` 一次性舍入为 BF16 并写出
`out = BF16(residual + BF16(gate[gate_index[m]] * o))`。这是 SGLang 的 `indexed_gate_bf16`
round-point 约定。`gate_index` 落在 `[0, 9)` 之外的行贡献 `gate = 0`（`out = residual`）。

**参数**

- `attn_out`（`torch.Tensor`）：BF16 `[M, 7168]` 打包注意力输出（`[M, 56, 128]` NHD 输出按
  行主序视图）。
- `o_weight_q`（`torch.Tensor`）：E4M3 `[5376, 7168]` 逐输出通道量化权重（见
  `quantize_minimax_h3_o_weight_fp8`）。
- `o_weight_scale`（`torch.Tensor`）：FP32 `[5376]` 反量化乘子。
- `gate`（`torch.Tensor`）：BF16 `[9, 5376]` 逐索引 `gate_msa` 行。
- `gate_index`（`torch.Tensor`）：int32 `[M]` 逐行表索引。
- `residual`（`torch.Tensor`）：BF16 `[M, 5376]` 残差流。
- `out`（`torch.Tensor`，可选）：BF16 `[M, 5376]` 输出；`None` 时分配。
- `act_q`（`torch.Tensor`，可选）：调用方拥有的 E4M3 激活工作区 `[M, 7168]`。
- `act_scale`（`torch.Tensor`，可选）：调用方拥有的 FP32 逐行激活缩放工作区 `[M]`。

**返回**

- `torch.Tensor`：BF16 `[M, 5376]` 注意力后隐状态。

---

### `minimax_h3_nvfp4_out_proj`

```python
minimax_h3_nvfp4_out_proj(
    attn_out: torch.Tensor,
    o_weight_q: torch.Tensor,
    o_weight_sf: torch.Tensor,
    o_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：SM120（GB202）上的 MiniMax-H3 NVFP4（W4A4）注意力输出投影，融合门控残差。一次
启动：`attn_out` 的 FlashInfer 约定 block-16 NVFP4 量化
（`sf = UE4M3(block_amax * act_global_scale / 6)`，`code = RN(x * act_global_scale / sf)`）由
持久内核 producer warpgroup 的空闲 warp 执行，并通过逐 tile ready 标志与
`kind::mxf4nvf4` 块缩放 `mma.sync` GEMM 重叠，其 epilogue 为
`out = BF16(residual + BF16(gate[gate_index[m]] * o))`。`gate_index` 落在 `[0, 9)` 之外的行
贡献 `gate = 0`。

**参数**

- `attn_out`（`torch.Tensor`）：BF16 `[M, 7168]` 打包注意力输出。
- `o_weight_q`（`torch.Tensor`）：uint8 `[5376, 3584]` E2M1x2 权重（见
  `quantize_minimax_h3_o_weight_nvfp4`）。
- `o_weight_sf`（`torch.Tensor`）：uint8 UE4M3 block-16 权重缩放，FlashInfer 128x4 swizzled 布局。
- `o_weight_global_scale`（`float` 或 `torch.Tensor`）：FP32 逐张量权重全局缩放 `448 * 6 / amax(W)`。
- `act_global_scale`（`torch.Tensor`）：FP32 `[1]` CUDA 张量，校准后的激活全局缩放 `448 * 6 / amax`。
- `gate`、`gate_index`、`residual`：同 FP8 变体。
- `out`（`torch.Tensor`，可选）：BF16 `[M, 5376]` 输出。
- `act_q`（`torch.Tensor`，可选）：调用方拥有的 E2M1x2 激活工作区（uint8 `[M, 3584]`）。
- `act_sf`（`torch.Tensor`，可选）：调用方拥有的激活块缩放工作区（uint8 `[M, 448]`）。

**返回**

- `torch.Tensor`：BF16 `[M, 5376]` 注意力后隐状态。

---

### `quantize_minimax_h3_o_weight_fp8`

```python
quantize_minimax_h3_o_weight_fp8(
    o_weight: torch.Tensor,
    chunk_rows: int = 1792,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：对 BF16 `[5376, 7168]` 注意力输出权重做逐输出通道 E4M3 量化。

**参数**

- `o_weight`（`torch.Tensor`）：BF16（或任意浮点）`[5376, 7168]` CUDA 张量。
- `chunk_rows`（`int`，默认 `1792`）：每块量化的行数（限定 FP32 临时缓冲；结果不依赖它）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(o_weight_q, o_weight_scale)`，分别为
  float8_e4m3fn `[5376, 7168]` 码与 FP32 `[5376]` 反量化乘子，其中
  `o_weight_scale[n] = RN(amax(row n) / 448)`、`o_weight_q = RN(row / scale)`。

---

### `quantize_minimax_h3_o_weight_nvfp4`

```python
quantize_minimax_h3_o_weight_nvfp4(
    o_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：对 BF16 `[5376, 7168]` 注意力输出权重做 FlashInfer NVFP4 量化。

**参数**

- `o_weight`（`torch.Tensor`）：BF16（或任意浮点）`[5376, 7168]` CUDA 张量。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：`(o_weight_q, o_weight_sf,
  o_weight_global_scale)`，分别为 uint8 `[5376, 3584]` E2M1x2 码、uint8 UE4M3 block-16
  缩放（128x4 swizzled 布局）、FP32 `[1]` 全局缩放 `448 * 6 / amax(W)`，与
  `flashinfer.fp4_quantize`（`sf_vec_size=16`、`is_sf_swizzled_layout=True`）产出完全一致。

---

### `minimax_h3_sm120_varlen_attention_fp8`

```python
minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor
```

**功能**：SM120（GB202）上的 FP8（E4M3）非因果 packed-varlen 自注意力。对 `cu_seqlens` 的
每个分段 `[a, b)` 与每个 head `h` 计算
`out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]ᵀ * softmax_scale) @ v[a:b, h]`，使用 FP8 E4M3
张量核操作数与 FP32 softmax：Q 逐 token 量化，K 在减去其分段的逐通道均值后按 128-key 块
量化（精确的 softmax 平移），概率以 E4M3 存储（2^8 指数偏置），V 按 (segment, channel)
量化。QK^T 与 PV 都运行 `mma.sync.m16n8k32 kind::f8f6f4`；输出一次性舍入为 BF16。一套运行时
可变内核服务任意 `tokens` 与任意分段长度；量化操作数位于按 (device, stream) 的只增工作区
（`workspace_bytes`）；保留最近 8 个 stream 集与 256 个分段计划。

**参数**

- `q, k, v`（`torch.Tensor`）：连续 `bfloat16` CUDA 张量，形状 `[tokens, heads, 128]`
  （packed THD 布局；MiniMax-H3 使用 56 heads）。
- `cu_seqlens`（`torch.Tensor`）：`int32`，形状 `[segments + 1]`，同设备，`cu_seqlens[0] = 0`，
  非降序，`cu_seqlens[-1] = tokens`。
- `out`（`torch.Tensor`，可选）：与 `q` 同形状 / dtype 的预分配输出；省略时分配。
- `cu_seqlens_host`（`Sequence[int]`，关键字，可选）：`cu_seqlens` 的主机副本（避免同步的
  设备到主机拷贝）。分段计划按 `(cu_seqlens, heads, device)` 缓存（最近 256）。
- `softmax_scale`（`float`，关键字，可选）：softmax 缩放；默认 `1 / sqrt(128)`。

**返回**

- `torch.Tensor`：`bfloat16` `[tokens, heads, 128]` 注意力输出。

---

### `minimax_h3_sm120_varlen_attention_nvfp4`

```python
minimax_h3_sm120_varlen_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor
```

**功能**：SM120（GB202）上的实验性 NVFP4 非因果 packed-varlen 自注意力，遵循
SageAttention3 FP4 配方。对 `cu_seqlens` 的每个分段 `[a, b)` 与 head `h` 计算
`out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]ᵀ * softmax_scale) @ v[a:b, h]`，scores 与 value
乘积都使用块缩放 NVFP4 张量核操作数与 FP32 softmax：K 行以分段逐通道均值居中，Q 行以其
128 行块的均值居中（块均值项 `qm Kᵀ` 在内核内加回，故 softmax 不变）；Q/K/V 以 E2M1 码
存储，每 16 个元素一个 UE4M3 缩放（块 amax / 6；V 转置且每 16 个 key 一个缩放）；QK^T 与
PV 都运行 `mma.sync.m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X`；概率使用 Sage3 的
两级方案；输出一次性舍入为 BF16。

E2M1 scores 与概率的量化误差大于 FP8 算子（高斯输入上相对 L2 误差约 4 倍，与
SageAttention3 自身相同）；该路由是实验性的精度 / 延迟权衡，用 FP4 块缩放容差
`atol = 1.0, rtol = 0.1` 对 FP32 oracle 验证，而非 FP8 算子的 `0.1`。

**参数**

- `q, k, v`（`torch.Tensor`）：连续 `bfloat16` CUDA 张量，形状 `[tokens, heads, 128]`。
- `cu_seqlens`（`torch.Tensor`）：`int32`，形状 `[segments + 1]`，约束同上。
- `out`（`torch.Tensor`，可选）：预分配输出，省略时分配。
- `cu_seqlens_host`（`Sequence[int]`，关键字，可选）：主机副本；分段计划按
  `(cu_seqlens, heads, device)` 缓存。
- `softmax_scale`（`float`，关键字，可选）：默认 `1 / sqrt(128)`。

**返回**

- `torch.Tensor`：`bfloat16` `[tokens, heads, 128]` 注意力输出。

**注意事项**

- 内核每线程需 246 个寄存器，每 SM 常驻一个 8-warp CTA；张量流水线活跃约 58% 的时间。
  量化操作数位于按设备的只增工作区（`workspace_bytes_nvfp4`），与
  `minimax_h3_sm120_varlen_attention_fp8` 共享。

---

### `minimax_h3_sm120_varlen_attention_nvfp4_nodelta`

```python
minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor
```

**功能**：`minimax_h3_sm120_varlen_attention_nvfp4` 的可选**未补偿**变体（相同契约、
签名、工作区与预处理启动；仅注意力内核不同）。默认路由在移除 128 行 Q 块的块均值后量化
每行，并在内核内把被移除的项 `qm Kᵀ` 加回 scores，故其 softmax 与未居中的 softmax 在 FP4
舍入范围内相等。本路由保留了居中但**不加回该项**：每个 Q 块的 scores 带有一个逐 key 偏移
`-(qm . k_j)`，softmax 不会抵消它（不同于 K 均值移除的逐行常量）。这是刻意的精度 / 延迟
权衡，不属于 SageAttention3 配方：

- 相对 FP32 oracle：高斯输入上相对 L2 约 0.209（默认路由 0.191，+9%），max-abs 约 0.064
  （默认 0.060）；FP4 容差 `atol = 1.0, rtol = 0.1` 仍在每个元素上成立。
- 短分段上偏差增大：被丢弃项是 128 行 Q 块的均值，`n` 行的部分块其块均值约大
  `sqrt(128 / n)` 倍。**少于 8 个 token 的分段在该路由上超出 FP4 容差**，故不要用于含此类
  分段的流——默认路由在那里仍在容差内。
- 延迟：在 RTX 5090 与 RTX PRO 6000 Blackwell 上注意力启动约快 9-15%（寄存器 220 而非 246）。

除非已针对具体模型验证了额外的误差，否则请使用默认路由。输入、输出、`cu_seqlens` 处理、
工作区与缓存与默认路由完全相同。

**参数**

- `q, k, v`、`cu_seqlens`、`out`、`cu_seqlens_host`、`softmax_scale`：同
  `minimax_h3_sm120_varlen_attention_nvfp4`。`cu_seqlens` 短于 8 个 token 的分段超出本路由
  FP4 容差。

**返回**

- `torch.Tensor`：未补偿路由的 `bfloat16` `[tokens, heads, 128]` 注意力输出。

---

### `minimax_h3_fc1_swiglu_fp8`

```python
minimax_h3_fc1_swiglu_fp8(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_q: Optional[torch.Tensor] = None,
    workspace_scale: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor
```

**功能**：SM120（RTX 5090 / RTX PRO 6000 Blackwell）上的 MiniMax-H3 视频 DiT 块融合
FP8（W8A8）RMSNorm + 索引化 AdaLN + FC1 GEMM + SwiGLU。对 `M` 行计算：

```text
n   = bf16(rmsnorm(x, eps) * x_norm_weight)                 # FP32 平方和，rsqrt
a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))   # i = adaln_index[row]
a[i outside [0, 9)] = 0                                      # 设备侧守护
a_q = e4m3(a / s_row),  s_row = RN(amax_row(|a|) / 448)     # per-token 激活缩放
h   = bf16(a_q @ fc1_weight_qᵀ * s_row * fc1_weight_scale)  # FP32 累加，[M, 28672]
y   = bf16(bf16(silu(h_gate)) * h_up)                        # [M, 14336]
```

当前流上两个内核：每行一个 CTA 的 norm/AdaLN/量化内核将 `a_q` 与 `s_row` 写入工作区，
以及一个持久 `mma.sync` GEMM，其 epilogue 应用反量化缩放与 SwiGLU。

**参数**

- `x`（`torch.Tensor`）：连续 `bfloat16` `[M, 5376]` 隐状态，`1 <= M <= 2**24`。
- `x_norm_weight`（`torch.Tensor`）：`bfloat16` `[5376]` RMSNorm 权重。
- `adaln_scale`、`adaln_shift`（`torch.Tensor`）：`bfloat16` `[9, 5376]` AdaLN 表。
- `adaln_index`（`torch.Tensor`）：`int32` `[M]` 每激活行的表行。
- `fc1_weight_q`、`fc1_weight_scale`（`torch.Tensor`）：来自
  `prepare_minimax_h3_fc1_weight_fp8` 的 `float8_e4m3fn [28672, 5376]` 与 `float32 [28672]`
  （SM120 预打包行序）。
- `out`（`torch.Tensor`，可选）：可选 `bfloat16` `[M, 14336]` 输出（省略时分配）。
- `workspace_q`（`torch.Tensor`，可选）：调用方拥有的 `float8_e4m3fn [M, 5376]` 缓冲，接收 `a_q`。
- `workspace_scale`（`torch.Tensor`，可选）：调用方拥有的 `float32 [M]` 缓冲，接收逐 token 缩放。
- `eps`（`float`，关键字，默认 `MINIMAX_H3_EPS`）：RMSNorm epsilon（正；在 `1e-5` 处验证）。

**返回**

- `torch.Tensor`：`bfloat16` `[M, 14336]`。

---

### `prepare_minimax_h3_fc1_weight_fp8`

```python
prepare_minimax_h3_fc1_weight_fp8(
    fc1_weight: torch.Tensor,
    chunk_rows: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：为 `minimax_h3_fc1_swiglu_fp8`（SM120）量化融合 FC1 权重。`fc1_weight` 为 BF16
`[28672, 5376]`，gate 行 `[0, 14336)` 之后跟 up 行 `[14336, 28672)`。每个输出通道（行）
量化为 E4M3，`scale = RN(max(amax(row), 1e-12) / 448)`（真 IEEE 除法）、
`q = RN_sat(row / scale)`，随后行被置换为 SM120 预打包序（八行 gate，然后是相同输出列的八行
up）。返回按预打包行序的 `(fc1_weight_q float8_e4m3fn [28672, 5376], fc1_weight_scale
float32 [28672])`。该布局专属 SM120 算子，不可与 SM100/SM103 的
`prepare_minimax_h3_fc1_weight_*` 输出互换。

**参数**

- `fc1_weight`（`torch.Tensor`）：BF16 CUDA 张量，形状 `[28672, 5376]`，gate 行在前、up 行在后。
- `chunk_rows`（`int`，默认 `2048`）：每个临时 FP32 块量化的权重行数。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：预打包 E4M3 权重及其 FP32 逐行缩放。

---

### `minimax_h3_mlp_fp8_sm120`

```python
minimax_h3_mlp_fp8_sm120(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_scale: Optional[torch.Tensor] = None,
    workspace_y: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_scale: Optional[torch.Tensor] = None,
    workspace_flags: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
    fp8_mma_form: int = FP8_MMA_FORM_AUTO,
) -> torch.Tensor
```

**功能**：SM120 上的融合 FP8（W8A8）MiniMax-H3 MLP 块：RMSNorm + 索引化 AdaLN + FC1 +
SwiGLU + FC2 + 门控残差。对 `M` 行计算：

```text
n   = bf16(rmsnorm(x, eps) * x_norm_weight)
a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))    # i = adaln_index[row]
a[i outside [0, rows)] = 0                                    # 设备侧守护
a_q = e4m3(a / s_a),  s_a = RN(amax_row(|a|) / 448)
h   = bf16(a_q @ fc1_weight_qᵀ * s_a * fc1_weight_scale)     # [M, 28672]
y   = bf16(bf16(silu(h_gate)) * h_up)                         # [M, 14336]
y_q = e4m3(y / s_y),  s_y = RN(amax_row(|y|) / 448)
o   = bf16(y_q @ fc2_weight_qᵀ * s_y * fc2_weight_scale)      # [M, 5376]
out = bf16(residual + bf16(gate[i] * o))
```

当前流上三个内核（外加一次 FC2 tile 标志的 memset）：每行一个 CTA 的
norm/AdaLN/量化内核、带 SwiGLU epilogue 的持久 FC1 GEMM，以及持久 FC2 GEMM（其 producer
warp 逐 tile 量化 `y`，epilogue 应用 gate 与残差）。在 GeForce GB202 板上，FP8 `mma.sync`
以 `kind::mxf8f6f4` 形式发出并使用单位 UE8M0 缩放（是 dense 形式 FP32 累加发射率的两倍）；
结果与 RTX PRO 6000 使用的 legacy 形式逐位相同。

**参数**

- `x`（`torch.Tensor`）：连续 `bfloat16` `[M, 5376]` 隐状态，`1 <= M <= 149796`。
- `x_norm_weight`（`torch.Tensor`）：`bfloat16` `[5376]` RMSNorm 权重。
- `adaln_scale`、`adaln_shift`、`gate`（`torch.Tensor`）：`bfloat16` `[rows, 5376]` 表视图，
  共享一个行数（单位列 stride、16 字节对齐基址与行距；`adaln_scale` 与 `adaln_shift` 共享
  一个行 stride）。更宽 `[rows, k * 5376]` 调制投影的列块无需拷贝即合格。
- `adaln_index`（`torch.Tensor`）：`int64` `[M]` 每激活行的表行；索引超出 `[0, rows)` 的行得到
  零激活与零 gate。
- `residual`（`torch.Tensor`）：连续 `bfloat16` `[M, 5376]` 残差流。
- `fc1_weight_q`、`fc1_weight_scale`（`torch.Tensor`）：来自 `prepare_minimax_h3_fc1_weight_fp8`
  的 `float8_e4m3fn [28672, 5376]` 与 `float32 [28672]`。
- `fc2_weight_q`、`fc2_weight_scale`（`torch.Tensor`）：来自 `prepare_minimax_h3_fc2_weight_fp8`
  的 `float8_e4m3fn [5376, 14336]` 与 `float32 [5376]`。
- `out`（`torch.Tensor`，可选）：可选 `bfloat16` `[M, 5376]` 输出；可以是 `residual` 或 `x`。
- `workspace_a_q`、`workspace_a_scale`（`torch.Tensor`，可选）：调用方拥有的
  `float8_e4m3fn [M, 5376]` / `float32 [M]` 缓冲，接收量化激活。
- `workspace_y`、`workspace_y_q`、`workspace_y_scale`（`torch.Tensor`，可选）：调用方拥有的
  `bfloat16 [M, 14336]` / `float8_e4m3fn [M, 14336]` / `float32 [M]` 缓冲。
- `workspace_flags`（`torch.Tensor`，可选）：调用方拥有的 `int32` 缓冲，
  `minimax_h3_mlp_fc2_flags_sm120` 的 `(M)` 个元素（FC2 tile ready 标志；算子会将其置零）。
- `eps`（`float`，关键字，默认 `MINIMAX_H3_EPS`）：RMSNorm epsilon（正；在 `1e-5` 处验证）。
- `fp8_mma_form`（`int`，关键字，默认 `FP8_MMA_FORM_AUTO`）：`-1` 选择板卡的 FP8 指令形式，
  `0` 强制 legacy 形式，`2` 使用带单位缩放的 `kind::mxf8f6f4` 形式。

**返回**

- `torch.Tensor`：`bfloat16` `[M, 5376]`（`out`）。

---

### `minimax_h3_mlp_nvfp4_sm120`

```python
minimax_h3_mlp_nvfp4_sm120(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    a_global_scale: torch.Tensor,
    fc1_weight_q: torch.Tensor,
    fc1_scale_tiles: torch.Tensor,
    fc1_alpha: Scalar,
    y_global_scale: torch.Tensor,
    fc2_weight_q: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    fc2_alpha: Scalar,
    *,
    out: Optional[torch.Tensor] = None,
    workspace_a_q: Optional[torch.Tensor] = None,
    workspace_a_sf: Optional[torch.Tensor] = None,
    workspace_y_q: Optional[torch.Tensor] = None,
    workspace_y_sf: Optional[torch.Tensor] = None,
    workspace_flags: Optional[torch.Tensor] = None,
    eps: float = MINIMAX_H3_EPS,
) -> torch.Tensor
```

**功能**：SM120 上的融合 NVFP4（W4A4）MiniMax-H3 MLP 块。norm 内核计算 BF16 调制激活 `a`
（同 `minimax_h3_mlp_fp8_sm120`）并用 FlashInfer 的 `nvfp4_quantize` 配方量化它（每 16 个连续
K 元素 `sf = E4M3_RN(g * absmax * rcp(6))` 在 448 饱和，`g = a_global_scale`，码
`E2M1_RN_saturate(a * rcp(sf * rcp(g)))`；全零块写 `sf = 0` 与码 0）。FC1 GEMM 以 FP32
累加 `(a_q * a_sf) . (w_q * w_sf)`，在 `h` 的 BF16 舍入前应用 `fc1_alpha`，在寄存器中形成
`y = bf16(bf16(silu(h_gate)) * h_up)` 并在同一 epilogue 用同一配方（`g = y_global_scale`）
量化 `y`；FC2 GEMM 直接消费这些操作数：

```text
h   = bf16(fc1_alpha * ((a_q * a_sf) @ (fc1_weight_q * fc1_scale_tiles)ᵀ))
y   = bf16(bf16(silu(h_gate)) * h_up)
o   = bf16(fc2_alpha * ((y_q * y_sf) @ (fc2_weight_q * fc2_weight_sf)ᵀ))
out = bf16(residual + bf16(gate[i] * o))
```

**参数**

- `x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, eps`：同
  `minimax_h3_mlp_fp8_sm120`。
- `a_global_scale`、`y_global_scale`（`torch.Tensor`）：`float32 [1]` CUDA 张量，`a` 与 `y`
  的激活全局缩放（`448 * 6 / absmax` 约定，通常校准）；在设备上读取。
- `fc1_weight_q`、`fc1_scale_tiles`（`torch.Tensor`）：打包 E2M1 `uint8 [28672, 2688]` 权重与
  扁平 `uint8 [28672 * 336]` swizzled 块缩放，来自
  `prepare_minimax_h3_fc1_weight_nvfp4_sm120`。
- `fc1_alpha`、`fc2_alpha`（`float` 或 `torch.Tensor`）：`1 / (a_global_scale *
  w1_global_scale)` 与 `1 / (y_global_scale * w2_global_scale)`（Python float 或单元素张量；
  设备张量读取时带主机同步）。
- `fc2_weight_q`、`fc2_weight_sf`（`torch.Tensor`）：打包 E2M1 `uint8 [5376, 7168]` 权重与
  扁平 `uint8 [5376 * 896]` swizzled 块缩放，来自 `prepare_minimax_h3_fc2_weight_nvfp4_sm120`。
- `out`（`torch.Tensor`，可选）：可选 `bfloat16` `[M, 5376]` 输出；可为 `residual` 或 `x`。
- `workspace_a_q`、`workspace_a_sf`（`torch.Tensor`，可选）：调用方拥有的 `uint8 [M, 2688]` /
  `uint8 [M, 336]` 缓冲（`a` 的打包码与 dense 行主序块缩放）。
- `workspace_y_q`、`workspace_y_sf`（`torch.Tensor`，可选）：调用方拥有的 `uint8 [M, 7168]` /
  `uint8 [M, 896]` 缓冲（`y` 的打包码与 dense 行主序块缩放）。
- `workspace_flags`（`torch.Tensor`，可选）：调用方拥有的 `int32` 缓冲，
  `minimax_h3_mlp_fc2_flags_sm120` 的 `(M)` 个元素。

**返回**

- `torch.Tensor`：`bfloat16` `[M, 5376]`（`out`）。

---

### `prepare_minimax_h3_fc2_weight_fp8`

```python
prepare_minimax_h3_fc2_weight_fp8(
    fc2_weight: torch.Tensor,
    chunk_rows: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：为 `minimax_h3_mlp_fp8_sm120` 量化 FC2 权重。`fc2_weight` 为 BF16 `nn.Linear` 矩阵
`[5376, 14336]`（输出通道 x FFN）。每个输出通道（行）量化为 E4M3，
`scale = RN(max(amax(row), 1e-12) / 448)`（真 IEEE 除法）、`q = RN_sat(row / scale)`；行序不变。
返回 `(fc2_weight_q float8_e4m3fn [5376, 14336], fc2_weight_scale float32 [5376])`。

**参数**

- `fc2_weight`（`torch.Tensor`）：BF16 `[5376, 14336]` FC2 权重。
- `chunk_rows`（`int`，默认 `1024`）：每个块的行数。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：量化权重与 FP32 逐行缩放。

---

### `prepare_minimax_h3_fc2_weight_nvfp4_sm120`

```python
prepare_minimax_h3_fc2_weight_nvfp4_sm120(
    fc2_weight: torch.Tensor,
    w_global_scale: Scalar,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：为 `minimax_h3_mlp_nvfp4_sm120` 量化 FC2 权重。`fc2_weight` 为 BF16
`[5376, 14336]` 矩阵，`w_global_scale` 为其 float32 全局缩放（`448 * 6 / absmax`）。行用
FlashInfer 的 `nvfp4_quantize`（`sfLayout=SfLayout.layout_128x4, do_shuffle=False`；每 16 个
连续 K 元素：UE4M3 缩放 = `E4M3_RN(g * absmax / 6)` 在 448 饱和，E2M1 对 `w * g / scale`
做 round-to-nearest 并饱和）量化。

**参数**

- `fc2_weight`（`torch.Tensor`）：BF16 `[5376, 14336]` FC2 权重。
- `w_global_scale`（`Scalar`）：float32 全局缩放 `448 * 6 / absmax`。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(fc2_weight_q, fc2_weight_sf)`——打包 E2M1
  `uint8 [5376, 7168]` 权重（偶元素在低半字节）与一个扁平 `uint8` 张量（`5376 * 896` 字节），
  块缩放为 FlashInfer 128x4 swizzled 布局（GEMM 以 42 个 448 行 x 256 字节的输出 tile
  流式读取）。

---

### `fused_qk_rmsnorm_rope`

```python
fused_qk_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float = 1e-06,
    base: float = 10000.0,
    interleave: bool = True,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    output_fp8: bool = False,
    output_quant_scale: float = 1.0,
    v_quant_scale: float = 1.0,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：面向视频生成 DIT 自注意力的融合 QK RMSNorm + 3D RoPE + V 拷贝。对 Q、K 应用
跨 head RMSNorm，随后以 3D 空间分解（frame/height/width）做旋转位置编码，并将 V 拷贝到连续
输出缓冲。可选将所有输出量化为 FP8 E4M3。

**参数**

- `qkv`（`torch.Tensor`）：组合 QKV 输入，BF16，连续。接受的形状：3D
  `[batch, seq_len, (num_heads_q+num_heads_k+num_heads_v)*head_dim]`；2D
  `[num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]`，其中 `num_tokens` 须能被
  `ppf*pph*ppw` 整除。
- `q_weight`（`torch.Tensor`）：Q 的 RMSNorm 权重 `[num_heads_q * head_dim]`，BF16。
- `k_weight`（`torch.Tensor`）：K 的 RMSNorm 权重 `[num_heads_k * head_dim]`，BF16。
- `ppf`、`pph`、`ppw`（`int`，关键字）：frame / height / width 维度的 patch 数；
  `seq_len = ppf * pph * ppw`。
- `num_frame_channels`、`num_height_channels`、`num_width_channels`（`int`，关键字）：
  各维度的 RoPE 频率通道数（须为偶数），三者之和等于 `head_dim`。
- `num_heads_q`、`num_heads_k`、`num_heads_v`（`int`，关键字）：查询 / 键 / 值 head 数。
- `head_dim`（`int`，关键字）：每 head 维度（须为 64、128 或 256）。
- `eps`（`float`，关键字，默认 `1e-06`）：RMSNorm epsilon。
- `base`（`float`，关键字，默认 `10000.0`）：RoPE 基频。
- `interleave`（`bool`，关键字，默认 `True`）：`True` 为交错 RoPE（非 NeoX 风格），`False` 为
  NeoX 风格。
- `factor`（`float`，关键字，默认 `1.0`）：YARN RoPE 缩放因子，1.0 禁用 YARN。
- `low`、`high`（`float`，关键字，默认 `0.0`）：YARN 低频 / 高频阈值。
- `attention_factor`（`float`，关键字，默认 `1.0`）：作用于 cos/sin 的 YARN 注意力因子；
  `factor` 为 1.0 时须为 1.0。
- `is_qk_norm`（`bool`，关键字，默认 `True`）：是否应用 RMSNorm（`False` = 仅 RoPE，跳过归一化）。
- `output_fp8`（`bool`，关键字，默认 `False`）：将 Q、K、V 输出量化为 FP8 E4M3。
- `output_quant_scale`（`float`，关键字，默认 `1.0`）：Q、K 输出的 FP8 量化缩放。
- `v_quant_scale`（`float`，关键字，默认 `1.0`）：V 输出的 FP8 量化缩放。
- `q_out`、`k_out`、`v_out`（`torch.Tensor`，可选）：预分配 Q / K / V 输出（destination-passing）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：`(q_out, k_out, v_out)`。输入为 3D 时
  各形状 `[batch, seq_len, num_heads_x, head_dim]`；输入为 2D 时各形状
  `[num_tokens, num_heads_x, head_dim]`。

---

### `fused_dit_residual_layernorm_scale_shift`

```python
fused_dit_residual_layernorm_scale_shift(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    residual: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-06,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：面向 DIT 自注意力的融合残差 + LayerNorm + scale/shift。单个内核内计算：

```python
residual_out = residual + input  # residual 为 None 时即 input
norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)
```

**参数**

- `input`（`torch.Tensor`）：`[batch, num_rows, 3072]`，BF16，连续。
- `scale`（`torch.Tensor`）：BF16，来自 `temb.chunk(6, dim=2)`，stride 为 `6 * hidden_dim`。
- `shift`（`torch.Tensor`）：BF16，stride 约定同 `scale`。
- `residual`（`torch.Tensor`，可选）：BF16，连续；`None` 时不加残差。
- `scale_bias`、`shift_bias`（`torch.Tensor`，可选）：偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，关键字，默认 `1e-06`）：LayerNorm epsilon。
- `use_nvfp4`（`bool`，关键字，默认 `False`）：将 norm 输出量化为 NVFP4（仅 SM100+）。
- `use_mxfp8`（`bool`，关键字，默认 `False`）：将 norm 输出量化为 MXFP8（仅 SM100+）。
- `global_scaling_factor`（`torch.Tensor`，可选）：NVFP4 输出的全局缩放 `[1]`，FP32。
- `input_global_scaling_factor`（`torch.Tensor`，可选）：预量化 NVFP4 输入的缩放 `[1]`，FP32。
- `residual_out`（`torch.Tensor`，可选）：预分配残差输出 `[batch, num_rows, 3072]`，BF16。
- `norm_out`（`torch.Tensor`，可选）：预分配 norm 输出，形状取决于输出格式。
- `sf_out`（`torch.Tensor`，可选）：NVFP4/MXFP8 的预分配缩放因子输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。

**注意事项**

- 该内核面向 WAN 2.2 5B（仅 `hidden_dim=3072`）。主要目标：SM90（Hopper）、SM100/SM103
  （Blackwell）。BF16 输出兼容 SM80+；NVFP4/MXFP8 需要 SM100+。

---

### `fused_dit_gate_residual_layernorm_scale_shift`

```python
fused_dit_gate_residual_layernorm_scale_shift(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-06,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：面向 DIT 自注意力的融合 gate + 残差 + LayerNorm + scale/shift。单个内核内计算：

```python
residual_out = residual + input * (gate + gate_bias)
norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)
```

**参数**

- `input`（`torch.Tensor`）：`[batch, num_rows, 3072]`，BF16，连续。
- `residual`（`torch.Tensor`）：与 `input` 同形状，BF16，连续。
- `gate`（`torch.Tensor`）：BF16，来自 `temb.chunk(6, dim=2)`，stride 为 `6 * hidden_dim`。
- `scale`、`shift`（`torch.Tensor`）：BF16，stride 约定同 `gate`。
- `gate_bias`、`scale_bias`、`shift_bias`（`torch.Tensor`，可选）：偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，关键字，默认 `1e-06`）：LayerNorm epsilon。
- `use_nvfp4`、`use_mxfp8`（`bool`，关键字，默认 `False`）：将 norm 输出量化为 NVFP4 / MXFP8
  （仅 SM100+）。
- `global_scaling_factor`（`torch.Tensor`，可选）：NVFP4 输出的全局缩放 `[1]`，FP32。
- `input_global_scaling_factor`（`torch.Tensor`，可选）：预量化 NVFP4 输入的缩放 `[1]`，FP32。
- `residual_out`、`norm_out`、`sf_out`（`torch.Tensor`，可选）：预分配输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。

**注意事项**

- 该内核面向 WAN 2.2 5B（仅 `hidden_dim=3072`）。主要目标：SM90、SM100/SM103。
  BF16 输出兼容 SM80+；NVFP4/MXFP8 需要 SM100+。

---

### `fused_dit_gate_residual_layernorm_gamma_beta`

```python
fused_dit_gate_residual_layernorm_gamma_beta(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-06,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：面向 DIT 自注意力的融合 gate + 残差 + LayerNorm(gamma, beta)。单个内核内计算：

```python
residual_out = residual + input * (gate + gate_bias)
norm_out = LayerNorm(residual_out, gamma, beta)
```

**参数**

- `input`（`torch.Tensor`）：`[batch, num_rows, 3072]`，BF16，连续。
- `residual`（`torch.Tensor`）：与 `input` 同形状，BF16，连续。
- `gate`（`torch.Tensor`）：`[batch, num_rows, 3072]`，BF16。可在 dim 1 上非连续、stride 为
  `6 * hidden_dim`（来自 WAN 的 `temb.chunk(6, dim=2)` 模式）。
- `gamma`（`torch.Tensor`）：LayerNorm 权重 `[3072]`，FP32。
- `beta`（`torch.Tensor`）：LayerNorm 偏置 `[3072]`，FP32。
- `gate_bias`（`torch.Tensor`，可选）：gate 的偏置 `[3072]` 或 `[1, 3072]`，FP32。
- `epsilon`（`float`，关键字，默认 `1e-06`）：LayerNorm epsilon。
- `use_nvfp4`、`use_mxfp8`（`bool`，关键字，默认 `False`）：将 norm 输出量化为 NVFP4 / MXFP8
  （仅 SM100+）。
- `global_scaling_factor`（`torch.Tensor`，可选）：NVFP4 输出的全局缩放 `[1]`，FP32；
  `use_nvfp4=True` 时必需。
- `input_global_scaling_factor`（`torch.Tensor`，可选）：gating 前作用于 input 的缩放
  （用于预量化 NVFP4 输入）`[1]`，FP32。
- `residual_out`、`norm_out`、`sf_out`（`torch.Tensor`，可选）：预分配输出。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(residual_out, norm_out)`。
  BF16 时两者均为 `[batch, num_rows, 3072]` BF16；NVFP4/MXFP8 时 `residual_out` 为 BF16，
  `norm_out` 为打包 int32。

**注意事项**

- 该内核面向 WAN 2.2 5B（仅 `hidden_dim=3072`）。主要目标：SM90、SM100/SM103。
  BF16 输出兼容 SM80+；NVFP4/MXFP8 需要 SM100+。

## 相关链接 / 参考

- `flashinfer.quantization.fp4_quantize` / `nvfp4_quantize`：MiniMax-H3 NVFP4 权重与激活
  量化使用的底层算子。
- `flashinfer.norm`：`fused_qk_rmsnorm_rope` 与 `fused_dit_*` 系列的定义模块，本模块将其
  重新导出。

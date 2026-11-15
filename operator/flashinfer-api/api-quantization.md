# 量化（`flashinfer.quantization`）

> 源码：`flashinfer/quantization/__init__.py`、`quantization/fp4_quantization.py`、`quantization/fp8_quantization.py`、`quantization/nvfp4_quantization_utils.py`、`quantization/packbits.py`、`quantization/kernels/*`、`flashinfer/gated_act_mxfp8.py`
> 上游文档：`docs/api/quantization.rst`
> 适用版本：FlashInfer 0.7.2

本模块提供 FP4（NVFP4 / MXFP4）、FP8（MXFP8）量化与反量化内核，以及 packbits 工具。
它是 FP4/MXFP4 GEMM 与 MoE 流水线在 Python 侧的公共入口，涵盖类型枚举、NVFP4 4over6
配方、缩放因子布局辅助函数、KV cache 量化，以及条件可用的 CuTe-DSL 实验内核。

## 导入

```python
from flashinfer.quantization import (
    SfLayout,
    NVFP44Over6Config,
    NVFP44Over6ErrMode,
    packbits,
    segment_packbits,
    fp4_quantize,
    nvfp4_quantize,
    mxfp4_quantize,
    mxfp8_quantize,
    # ...
)
# FP8 也可从子模块导入
from flashinfer.quantization.fp8_quantization import mxfp8_quantize
```

## API 列表

### 类型与枚举

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `SfLayout` | 枚举（IntEnum） | 缩放因子（scale factor）的内存布局。 |
| `NVFP44Over6Config` | 数据类（frozen dataclass） | NVFP4 4over6 配方配置。 |
| `NVFP44Over6ErrMode` | 枚举（IntEnum） | 4over6 两个候选块缩放之间的误差度量。 |

### Packbits 工具

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `packbits` | 函数 | 将二值数组按位打包为 uint8。 |
| `segment_packbits` | 函数 | 将一批二值分段打包为 uint8。 |

### NVFP4 4over6 配方

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `resolve_nvfp4_4over6` | 函数 | 将公共 `nvfp4_4over6=` 参数解析为内核使用配方。 |
| `nvfp4_4over6_code` | 函数 | 将已解析配方打包为自定义算子接受的 `int64`。 |
| `nvfp4_4over6_from_code` | 函数 | `nvfp4_4over6_code` 的逆运算。 |
| `make_nvfp4_global_scale` | 函数 | 依据已解析 4over6 配方构造 NVFP4 全局缩放。 |
| `nvfp4_e4m3_max` | 函数 | 返回配方隐含的 E4M3 块缩放上界。 |

### FP4 量化

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `fp4_quantize` | 函数 | 通用 FP4 量化（NVFP4 / MXFP4）。 |
| `nvfp4_quantize` | 函数 | NVFP4 量化（含可选 per-token 激活缩放）。 |
| `nvfp4_batched_quantize` | 函数 | 批处理 NVFP4 量化。 |
| `mxfp4_quantize` | 函数 | MXFP4 量化（UE8M0 缩放）。 |
| `mxfp4_dequantize` | 函数 | MXFP4 反量化。 |
| `mxfp4_dequantize_host` | 函数 | 主机端 MXFP4 反量化。 |
| `block_scale_interleave` | 函数 | 将块缩放张量 swizzle 为 FP4 布局。 |
| `e2m1_and_ufp8sf_scale_to_float` | 函数 | 将 E2M1 + UFP8 缩放反量化为 float32。 |
| `scaled_fp4_grouped_quantize` | 函数 | 带逐行掩码的批处理 NVFP4 量化。 |
| `silu_and_mul_nvfp4_quantize` | 函数 | 融合 SwiGLU 与 NVFP4 量化。 |
| `shuffle_matrix_a` | 函数 | 矩阵 A 行洗牌（TRT-LLM-gen 等价实现）。 |
| `shuffle_matrix_sf_a` | 函数 | 线性布局缩放因子的行洗牌并重排到 128x4。 |

### FP4 KV Cache 量化

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `nvfp4_kv_quantize` | 函数 | KV cache NVFP4 量化（线性块缩放布局，SM100+）。 |
| `nvfp4_kv_dequantize` | 函数 | KV cache NVFP4 反量化（SM80+）。 |
| `nvfp4_kv_dequantize_paged` | 函数 | 分页 KV cache NVFP4 反量化（SM80+）。 |
| `nvfp4_quantize_paged_kv_cache` | 函数 | 分页 KV cache NVFP4 量化（trtllm-gen MHA 布局）。 |

### FP8 量化

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `mxfp8_quantize` | 函数 | MXFP8 量化。 |
| `mxfp8_grouped_quantize` | 函数 | 分组 MXFP8 量化（cuTile 后端）。 |
| `mxfp8_dequantize_host` | 函数 | 主机端 MXFP8 反量化。 |
| `per_token_group_quant_8bit` | 函数 | 逐 token 分组 8-bit 量化（FP8/INT8）。 |

### 融合门控激活 MXFP8 量化（`flashinfer.gated_act_mxfp8`）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `silu_and_mul_mxfp8_quantize` | 函数 | SwiGLU 与 MXFP8 量化融合（前向）。 |
| `silu_and_mul_mxfp8_quantize_backward` | 函数 | SwiGLU 反向变换与 MXFP8 量化融合。 |

### CuTe-DSL 量化内核（实验性）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `nvfp4_quantize_cute_dsl` | 函数 | CuTe-DSL NVFP4 量化。 |
| `nvfp4_quantize_per_token_cute_dsl` | 函数 | CuTe-DSL per-token NVFP4 量化。 |
| `mxfp4_quantize_cute_dsl` | 函数 | CuTe-DSL MXFP4 量化。 |
| `mxfp8_quantize_cute_dsl` | 函数 | CuTe-DSL MXFP8 量化。 |

## API 详情

### `SfLayout`

```python
class SfLayout(IntEnum):
    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2
```

**功能**：描述量化缩放因子（scale factor）张量的内存布局。定义于
`flashinfer/tllm_enums.py`，并由 `flashinfer.quantization` 重新导出。

**字段**

- `layout_128x4 = 0`：128x4 swizzled 布局。FP4 GEMM/MoE 内核默认消费的布局。
- `layout_8x4 = 1`：8x4 swizzled 布局。用于部分后端（如 trtllm）的 A 缩放。
- `layout_linear = 2`：线性（非 swizzled）布局，无 padding。

---

### `NVFP44Over6ErrMode`

```python
class NVFP44Over6ErrMode(IntEnum):
    MAE = 0
    MSE = 1
```

**功能**：选择 4over6 两个块缩放候选时的误差度量。每个 16 元素块会分别用标准的
`amax / 6` 缩放和更紧的 `amax / 4` 缩放量化、反量化并与源比较。`MAE` 累加绝对误差，
`MSE` 累加平方误差。更紧的候选会裁剪块内的最大幅值，MSE 对其惩罚更重，因此 MSE 选择
该候选的频率更低。

**字段**

- `MAE = 0`：累加绝对误差（默认）。
- `MSE = 1`：累加平方误差。

**注意事项**

- 其 `__repr__` 返回限定名（如 `NVFP44Over6ErrMode.MAE`），以保证 MoE 配置树的
  `eval(repr(cfg))` 往返可用。

---

### `NVFP44Over6Config`

```python
@dataclass(frozen=True)
class NVFP44Over6Config:
    e4m3_max: int = 448
    err_mode: Union[NVFP44Over6ErrMode, str] = NVFP44Over6ErrMode.MAE
    err_use_fast_math: bool = False
```

**功能**：Python 驱动与内核共享的 NVFP4 4over6 配置。定义于
`flashinfer.quantization.nvfp4_quantization_utils`，并由 `flashinfer.quantization` 重新导出。

**字段**

- `e4m3_max`（`int`，默认 `448`）：E4M3 块缩放的上界，只能取 `448`（满量程）或 `256`。
  它同时把逐张量全局缩放固定为 `1 / (e4m3_max * 6)`。`256` 用于让 `amax / 4` 候选保持
  可表示。
- `err_mode`（`NVFP44Over6ErrMode` 或 `str`，默认 `NVFP44Over6ErrMode.MAE`）：两候选
  之间的选择度量。字符串（`"MAE"` / `"MSE"`）会被归一化为枚举成员。
- `err_use_fast_math`（`bool`，默认 `False`）：以 fp16 而非精确 fp32 评估候选误差。
  更便宜，但在接近平局时可能选择另一个候选，因此应视为一种不同的配方。

**属性**

- `err_mode_name -> str`：返回误差模式的大写名称。

**注意事项**

- `__post_init__` 会校验 `e4m3_max ∈ {256, 448}`，否则抛出 `ValueError`；误差模式非
  MAE/MSE 时同样抛出 `ValueError`。
- `__repr__` 只打印非默认字段，例如 `NVFP44Over6Config(e4m3_max=256, err_mode=NVFP44Over6ErrMode.MSE)`。

---

### `packbits`

```python
packbits(x: torch.Tensor, bitorder: str = "big") -> torch.Tensor
```

**功能**：将二值数组的元素按位打包进 uint8 数组，语义与
[`numpy.packbits`](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html) 相同。

**参数**

- `x`（`torch.Tensor`）：待打包的 1D 二值数组。
- `bitorder`（`str`，默认 `"big"`）：输出的位序（`"big"` / `"little"`）。

**返回**

- `torch.Tensor`：uint8 打包数组，形状 `((x.size(0) + 7) // 8,)`。

**示例**

```python
>>> import torch
>>> from flashinfer import packbits
>>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool, device="cuda")
>>> x_packed = packbits(x)
>>> list(map(bin, x_packed.tolist()))
['0b10110011']
```

---

### `segment_packbits`

```python
segment_packbits(
    x: torch.Tensor, indptr: torch.Tensor, bitorder: str = "big"
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：将一批二值分段分别打包进 uint8 数组。对每个分段，语义与 `numpy.packbits`
相同。

**参数**

- `x`（`torch.Tensor`）：待打包的 1D 二值数组，形状 `(indptr[-1],)`。
- `indptr`（`torch.Tensor`）：每个分段在 `x` 中的索引指针，形状 `(batch_size + 1,)`。
  `x` 中第 i 个分段为 `x[indptr[i]:indptr[i+1]]`。
- `bitorder`（`str`，默认 `"big"`）：输出的位序（`"big"` / `"little"`）。

**返回**

- `torch.Tensor`：uint8 打包数组，形状 `(new_indptr[-1],)`。`y[new_indptr[i]:new_indptr[i+1]]`
  包含 `x[indptr[i]:indptr[i+1]]` 的打包位。
- `torch.Tensor`：打包后各分段的新索引指针，形状 `(batch_size + 1,)`。保证
  `new_indptr[i+1] - new_indptr[i] == (indptr[i+1] - indptr[i] + 7) // 8`。

**注意事项**

- `torch.compile` 不支持该函数，因为其结果是数据相关的。

**示例**

```python
>>> import torch
>>> from flashinfer import segment_packbits
>>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=torch.bool, device="cuda")
>>> x_packed, new_indptr = segment_packbits(x, torch.tensor([0, 4, 7, 11], device="cuda"), bitorder="big")
>>> list(map(bin, x_packed.tolist()))
['0b10110000', '0b100000', '0b11010000']
```

## NVFP4 4over6 配方

标准 NVFP4 为每个 16 元素块派生单个 E4M3 块缩放，取自该块的绝对最大值 `amax / 6`，
因为 `6` 是单个 E2M1 元素能表示的最大幅值。“4over6” 是一次两候选块缩放搜索——量化器
还构造出更紧 1.5x 的 `amax / 4` 缩放，用两个缩放分别量化该块、反量化，并保留重建误差
更小的那个。`amax / 4` 候选通过裁剪块内最大幅值，为块中大部分元素换取分辨率，因此在
没有主导离群值的块上胜出。额外开销完全在量化器内部完成：结果是普通 NVFP4，下游无需
任何改动。

4over6 要求 fp16/bf16 输入以及 E4M3（而非 UE8M0）块缩放。

实践中两个旋钮最重要：

- **`e4m3_max=256`**：把 `256`（而不是满量程 `448`）视为 E4M3 块缩放范围的顶部。这是
  让 `amax / 4` 候选保持可表示的关键：`256 * 1.5 = 384` 仍能放进 E4M3，而基于 448 的
  全局缩放需要 `672`，在接近张量 amax 的块上会饱和。它同时改变全局缩放——见下方警告。
- **`err_mode="MSE"`**：用平方误差之和而非绝对误差之和为两个候选打分。MSE 对 `amax / 4`
  裁剪产生的少量大残差惩罚远重于 MAE，因此更少选择 `amax / 4`。当少量离群激活主导下游
  精度时优先使用。
- `err_use_fast_math=True` 会以 fp16 而非精确 fp32 评估该误差。它更便宜，并且在接近平局
  时可能选择另一个候选，因此应视为一种不同的配方而非纯粹的速度开关。

> **警告**：全局缩放必须与量化器调用使用**相同**的配方构建。`e4m3_max` 同时出现在
> `make_nvfp4_global_scale` 和内核的候选搜索中；不一致会静默地重新缩放整个张量，而不会
> 报错。

```python
import torch
from flashinfer.quantization import (
    NVFP44Over6Config,
    make_nvfp4_global_scale,
    nvfp4_quantize,
)

x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
recipe = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")

# 两次调用使用同一个配方对象——见上方警告。
global_sf = make_nvfp4_global_scale(
    x, per_token_activation=False, nvfp4_4over6_config=recipe
)
x_q, sf = nvfp4_quantize(x, global_sf, nvfp4_4over6=recipe)
```

### Precedence（优先级）

所有曾经读取环境变量的 NVFP4 入口现在都接受一个 `nvfp4_4over6=` 关键字，类型为
`Optional[NVFP44Over6Config]`。共有三种情况，但只有一个公共类型：

| `nvfp4_4over6=` | 含义 |
| --- | --- |
| 省略（默认） | 读取 `FLASHINFER_NVFP4_4OVER6`；当其值为 `"1"` 时，其余三个 `FLASHINFER_NVFP4_4OVER6_*` 变量提供配方。每次调用都读取。行为与此参数出现前逐字节相同，另在环境变量开启 4over6 时发出 `FutureWarning`。 |
| `None` | 关闭 4over6。`FLASHINFER_NVFP4_4OVER6=1` 无法再打开它。这与 `None` 在内核驱动中一直以来的含义相同。 |
| `NVFP44Over6Config(...)` | 以该精确配方开启。环境变量被忽略，且**没有**逐字段合并：留在数据类默认值上的字段保持默认值，而不会拾取环境变量的值。 |

四个 `FLASHINFER_NVFP4_4OVER6*` 变量记录于仓库的 `CLAUDE.md`。它们仍作为兼容垫片
受支持但已废弃：由于是进程级的，无法在同一进程内为不同配方服务的两个模型表达不同
配方。垫片退役时 `nvfp4_4over6=` 的默认值将变为 `None`，签名不变。

> **提示**：`make_nvfp4_global_scale` 和 `nvfp4_e4m3_max` 接收**已解析**配方作为
> `nvfp4_4over6_config=`，且从不读取环境变量（`None`，其默认值，一直表示标准 NVFP4）。
> 当量化调用未设置 `nvfp4_4over6` 时，向缩放辅助函数传入 `resolve_nvfp4_4over6()`，
> 使两侧一致。

解析只发生在一个地方：`resolve_nvfp4_4over6`。在其之下，每个内核驱动看到的是两态的
`Optional[NVFP44Over6Config]`（其中 `None` 表示关闭），而 `nvfp4_4over6_code` 将解析值
打包成自定义算子接受的 `int64`。`nvfp4_4over6_from_code` 反转该打包，日志与复现工具
据此恢复某次捕获调用实际运行的配方。

---

### `resolve_nvfp4_4over6`

```python
resolve_nvfp4_4over6(
    setting: Optional[NVFP44Over6Config] = _UNSET,
) -> Optional[NVFP44Over6Config]
```

**功能**：将公共 `nvfp4_4over6=` 参数解析为内核使用的配方。这是 Python 侧**唯一**
咨询 `FLASHINFER_NVFP4_4OVER6*` 环境变量之处，因此优先级只决定一次。

**参数**

- `setting`（`NVFP44Over6Config` 或 `None`，默认 `_UNSET`）：
  - 省略（默认）：读取环境变量。当其开启 4over6 时发出 `FutureWarning`（因为这些变量
    是兼容垫片）。
  - `None`：关闭 4over6，环境变量被忽略。
  - `NVFP44Over6Config`：以该精确配方开启，环境变量被忽略，且无逐字段合并。

**返回**

- `Optional[NVFP44Over6Config]`：解析后的配方，`None` 表示 4over6 关闭。这是公共边界
  之下每个内核驱动期望的值。

**注意事项**

- 传入非 `NVFP44Over6Config`/`None`/省略的其它类型会抛出 `TypeError`。
- 子类会被规范化为基础 `NVFP44Over6Config`，以保持缓存键和磁盘产物一致。

---

### `nvfp4_4over6_code`

```python
nvfp4_4over6_code(config: Optional[NVFP44Over6Config]) -> int
```

**功能**：将**已解析**配方打包成自定义算子 / TVM-FFI 接受的 `int64`。因为它接收已解析
值，FlashInfer 自身的 Python 永远不会发出 `NVFP4_4OVER6_CODE_FROM_ENV`。

**参数**

- `config`（`Optional[NVFP44Over6Config]`）：已解析配方，`None` 表示关闭。

**返回**

- `int`：位布局（解码器为 `nvfp4Recipe.h` 中的 `resolveNVFP4Recipe`）：
  - `None` → `0`（4over6 关闭）
  - bit 0：恒为 1（4over6 开启）
  - bit 1：`e4m3_max == 256`
  - bits 2..3：`err_mode`（0 = MAE，1 = MSE）
  - bit 4：`err_use_fast_math`

---

### `nvfp4_4over6_from_code`

```python
nvfp4_4over6_from_code(code: int) -> Optional[NVFP44Over6Config]
```

**功能**：`nvfp4_4over6_code` 的逆运算，用于日志、复现工具与测试。`-1`
（`NVFP4_4OVER6_CODE_FROM_ENV`）解码为“参数省略”哨兵。与 `nvfp4Recipe.h` 中的
`resolveNVFP4Recipe` 镜像（包括其拒绝逻辑），因此 Python 侧日志不会与运行的内核不一致。

**参数**

- `code`（`int`）：打包后的配方编码。

**返回**

- `Optional[NVFP44Over6Config]`：解码后的配方，或在“参数省略”时返回 `_UNSET`。

**注意事项**

- 若 `code` 既不是 `NVFP4_4OVER6_CODE_FROM_ENV`，也不是 `nvfp4_4over6_code` 能产生的
  值，则抛出 `ValueError`。

---

### `make_nvfp4_global_scale`

```python
make_nvfp4_global_scale(
    input_tensor: torch.Tensor,
    per_token_activation: bool,
    global_scale: float | None = None,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> torch.Tensor
```

**功能**：依据**已解析** 4over6 配方构造 NVFP4 全局缩放。配方必须与量化调用一致：E4M3
上界同时出现在该缩放和内核的候选搜索中，不一致会静默地重新缩放整个张量。请传入与量化
调用上 `nvfp4_4over6=` 相同的值；或当该调用未设置时传入 `resolve_nvfp4_4over6()`。

**参数**

- `input_tensor`（`torch.Tensor`）：其 amax 用于派生缩放的张量（仅逐张量模式）。
- `per_token_activation`（`bool`）：per-token 激活模式，此时缩放为纯函数
  `1 / (e4m3_max * 6)`。
- `global_scale`（`float`，可选）：显式逐张量缩放，绕过 amax 归约。
- `nvfp4_4over6_config`（`NVFP44Over6Config` 或 `None`，默认 `None`）：已解析配方，
  `None` 表示 4over6 关闭。与 `nvfp4_e4m3_max` 一样，省略它不会读取环境变量。

**返回**

- `torch.Tensor`：`float32` 缩放张量。per-token 模式或显式 `global_scale` 返回形状
  `(1,)` 的张量；逐张量模式返回 `(e4m3_max * 6 / amax)` 的 `(1,)` 重塑结果。当 amax 为
  0 时返回 `float32` 最大值。

---

### `nvfp4_e4m3_max`

```python
nvfp4_e4m3_max(nvfp4_4over6_config: Optional[NVFP44Over6Config] = None) -> float
```

**功能**：返回**已解析**配方隐含的 E4M3 块缩放钳位。`None`（默认）表示标准 NVFP4（448）。

**参数**

- `nvfp4_4over6_config`（`NVFP44Over6Config` 或 `None`，默认 `None`）：已解析配方。

**返回**

- `float`：`448.0`（标准）或配方中的 `e4m3_max`（256）。

**注意事项**

- 与量化器不同，此处省略配方不会读取环境变量：请传入与对应量化调用匹配的已解析配方。

## FP4 / FP8 量化

### `fp4_quantize`

```python
fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    is_global_scale_inversed: bool = False,
    enable_pdl: Optional[bool] = None,
    backend: str = "cuda",
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：将输入张量量化为 FP4 格式。实现将输入转换为压缩 FP4 并附带缩放因子的
FP4 量化，支持多种输入数据类型与缩放因子布局（覆盖 NVFP4 与 MXFP4 两种配方）。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`，dtype 为 fp16/bf16/fp8_quantized。
- `global_scale`（`torch.Tensor`，可选）：形状 `[1]`、dtype `float32` 的全局缩放。
- `sf_vec_size`（`int`，默认 `16`）：缩放因子向量大小。
- `sf_use_ue8m0`（`bool`，默认 `False`）：是否使用 UE8M0 格式缩放因子。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：是否使用 swizzled 缩放布局。
- `is_sf_8x4_layout`（`bool`，默认 `False`）：使用 8x4 而非 128x4 swizzled 布局。
- `is_global_scale_inversed`（`bool`，默认 `False`）：为 `True` 时 `global_scale`
  被解释为逆缩放。
- `enable_pdl`（`bool`，可选）：是否启用 Programmatic Dependent Launch；`None` 时按设备
  能力自动检测。
- `backend`（`str`，默认 `"cuda"`）：
  - `"cuda"`：稳定 CUDA 内核（默认）。
  - `"cute-dsl"`：CuTe-DSL 内核（SM100+，**实验性**）。支持组合：
    `sf_vec_size=16, sf_use_ue8m0=False`（NVFP4，所有布局，fp16/bf16/fp8）；
    `sf_vec_size=32, sf_use_ue8m0=True`（MXFP4，所有布局，fp16/bf16）。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：NVFP4 “4over6” 缩放候选
  搜索，要求 fp16/bf16 输入。省略时读取 `FLASHINFER_NVFP4_4OVER6*`；`None` 关闭；
  传配置则按该配方开启。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`，`x_q` 形状 `[M, K/2]`、dtype
  `FLOAT4_E2M1X2`；`sf` 形状取决于布局与 `sf_vec_size`。

**注意事项**

- `sf_vec_size` 只能是 16 或 32，否则抛 `NotImplementedError`。
- 对 `"cute-dsl"` 后端不支持的参数组合、或把显式 4over6 配方与 MXFP4 参数/ fp8 输入
  结合时抛 `ValueError`。
- `"cute-dsl"` 后端为**实验性**，不属于稳定 API，可能变更或移除。
- 请用相同配方构造全局缩放：`make_nvfp4_global_scale(..., nvfp4_4over6_config=<同一值>)`。

---

### `nvfp4_quantize`

```python
nvfp4_quantize(
    a,
    a_global_sf,
    sfLayout=SfLayout.layout_128x4,
    do_shuffle=False,
    sf_vec_size=16,
    enable_pdl=None,
    backend: str = "cuda",
    per_token_activation: bool = False,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：将输入张量量化为 NVFP4 格式。

**参数**

- `a`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16/float8_e4m3fn。
- `a_global_sf`（`float` 或 `torch.Tensor`）：全局缩放。CuTe-DSL 后端接受主机侧 float，
  或为向后兼容接受单元素张量。
- `sfLayout`（`SfLayout`，默认 `SfLayout.layout_128x4`）：缩放因子布局。
- `do_shuffle`（`bool`，默认 `False`）：是否洗牌缩放因子。仅 TRT-LLM 后端需要洗牌
  张量 B 的缩放因子。
- `sf_vec_size`（`int`，默认 `16`）：缩放因子向量大小。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时按设备能力自动检测。
- `backend`（`str`，默认 `"cuda"`）：
  - `"cuda"`：稳定 CUDA 内核（默认）。
  - `"cute-dsl"`：CuTe-DSL 内核（SM100+，**实验性**）；支持所有 `sfLayout` 值、
    fp16/bf16/float8_e4m3fn 输入，但仅 `sf_vec_size == 16`。
  - `"cake"`：生成的 Cake per-token 内核（SM100/SM103，**实验性**，需显式选择）；
    `per_token_activation=True` 且 `layout_128x4`、不洗牌、仅 fp16/bf16 输入，输出与
    `"cute-dsl"` 逐位相等。
- `per_token_activation`（`bool`，默认 `False`）：是否使用 per-token NVFP4 激活缩放。
  此模式下 `a_global_sf` 为逆基础缩放乘子（通常 `1 / (448 * 6)`），函数还会返回
  per-token FP32 缩放。
- `expanded_idx_to_permuted_idx`（`torch.Tensor`，可选）：per-token 激活量化的可选行重映射。
- `out_scale`（`torch.Tensor`，可选）：返回的 per-token 缩放所乘的标量。仅用于
  `per_token_activation=True` 且 `backend="cute-dsl"` / `"cake"`。不改变量化值。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：同 `fp4_quantize` 的
  优先级语义。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`，`x_q` 形状 `[M, K/2]`、dtype
  `FLOAT4_E2M1X2`；`sf` 形状取决于布局与 `sf_vec_size`。当 `per_token_activation=True`
  时还会返回第三个张量（per-token FP32 缩放）。

**注意事项**

- `"cute-dsl"` 后端为**实验性**。
- 用相同配方构建全局缩放。
- `per_token_activation=True` 时要求 `sf_vec_size=16`；当 `nvfp4_4over6` 显式给出且
  `a_global_sf` 无需设备同步即可读取时，会检查两者配对。

---

### `nvfp4_batched_quantize`

```python
nvfp4_batched_quantize(
    a,
    a_global_sf,
    sf_vec_size=16,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：将批处理输入张量量化为 NVFP4 格式。

**参数**

- `a`（`torch.Tensor`）：形状 `[B, M, K]`，dtype fp16/bf16。
- `a_global_sf`（`torch.Tensor`）：形状 `[1]`、dtype `float32` 的全局缩放。
- `sf_vec_size`（`int`，默认 `16`）：缩放因子向量大小。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：4over6 配方，语义同上。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`，`x_q` 形状 `[B, M, K/2]`、dtype
  `FLOAT4_E2M1X2`；`sf` 为逐 batch swizzled 缩放，形状
  `[B, ceil(M / 128) * 128 * ceil(K / sf_vec_size / 4) * 4]`（M padding 至 128 的倍数，
  `K / sf_vec_size` 上取整至 4 的倍数）。

---

### `mxfp4_quantize`

```python
mxfp4_quantize(
    a: torch.Tensor,
    backend: str = "cuda",
    enable_pdl: Optional[bool] = None,
    sfLayout: SfLayout = SfLayout.layout_128x4,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：将输入张量量化为 MXFP4 格式。

**参数**

- `a`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16。
- `backend`（`str`，默认 `"cuda"`）：`"cuda"`（默认）或 `"cute-dsl"`（SM100+，**实验性**）。
- `enable_pdl`（`bool`，可选）：仅在 `backend == "cute-dsl"` 时使用。
- `sfLayout`（`SfLayout`，默认 `SfLayout.layout_128x4`）：支持 `layout_128x4`、
  `layout_8x4` 与 `layout_linear`。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`，`x_q` 形状 `[M, K/2]`、dtype `uint8`
  （`FLOAT4_E2M1X2`）；`sf` 为 UE8M0 缩放（`uint8`），形状取决于布局（`sf_vec_size` 固定为 32）。

**注意事项**

- `"cute-dsl"` 后端为**实验性**，生产使用风险自负。

---

### `mxfp4_dequantize`

```python
mxfp4_dequantize(
    a_fp4,
    a_sf,
    sfLayout: SfLayout = SfLayout.layout_128x4,
)
```

**功能**：将 MXFP4 打包权重反量化为 float32。

**参数**

- `a_fp4`（`torch.Tensor`）：形状 `[M, K/2]`、dtype `uint8`（`FLOAT4_E2M1X2`）的量化张量。
- `a_sf`（`torch.Tensor`）：UE8M0 缩放（`uint8`），形状取决于 `sfLayout`。
- `sfLayout`（`SfLayout`，默认 `SfLayout.layout_128x4`）：`a_sf` 使用的布局。

**返回**

- `torch.Tensor`：形状 `[M, K]`、dtype `float32` 的反量化张量。

---

### `mxfp4_dequantize_host`

```python
mxfp4_dequantize_host(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor
```

**功能**：主机端 MXFP4 反量化。

**参数**

- `weight`（`torch.Tensor`）：形状 `[M, K/2]`、dtype `uint8`（`FLOAT4_E2M1X2`）。
- `scale`（`torch.Tensor`）：UE8M0 缩放（`uint8`），形状取决于布局与 `group_size`。
- `group_size`（`int`，默认 `32`）：反量化分组大小。

**返回**

- `torch.Tensor`：形状 `[M, K]`、dtype `float32`。

---

### `block_scale_interleave`

```python
block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor
```

**功能**：将 FP4 块缩放张量重排（swizzle），以优化 FP4 GEMM/MoE 内核的内存访问模式。
输出在 `m` 维度上 padding 到 128 的倍数。

**参数**

- `unswizzled_sf`（`torch.Tensor`）：dtype 为 `uint8` 或 `bfloat16` 的缩放因子张量。

**返回**

- `torch.Tensor`：1D 扁平 swizzled 缩放缓冲区，形状 `(num_experts * expert_out_size,)`；
  3D 输入时 `num_experts` 为 `unswizzled_sf.shape[0]`（否则为 1）。这与 `unswizzled_sf`
  的逻辑形状不同；下游 FP4 GEMM/MoE 内核直接消费该扁平缓冲区。

**注意事项**

- dtype 非 `uint8` / `bfloat16` 时抛出 `AssertionError`。
- `flashinfer.quantization.nvfp4_block_scale_interleave` 是它的别名（同一 Python 对象）。

---

### `e2m1_and_ufp8sf_scale_to_float`

```python
e2m1_and_ufp8sf_scale_to_float(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
) -> torch.Tensor
```

**功能**：将 E2M1 张量结合 UFP8 缩放反量化为 float32。

**参数**

- `e2m1_tensor`（`torch.Tensor`）：E2M1 打包 FP4 张量，形状 `[M, K/2]`、dtype `uint8`。
- `ufp8_scale_tensor`（`torch.Tensor`）：UFP8 缩放，dtype `uint8`。
- `global_scale_tensor`（`torch.Tensor`，可选）：形状 `[1]`、dtype `float32` 的全局缩放。
- `sf_vec_size`（`int`，默认 `16`）：缩放向量大小。
- `ufp8_type`（`int`，默认 `1`）：UFP8 类型（`0` = UE8M0，`1` = E4M3）。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：缩放是否 swizzled 存储。
- `is_sf_8x4_layout`（`bool`，默认 `False`）：swizzled 时是否使用 8x4 布局；
  `is_sf_swizzled_layout=False` 时必须为 `False`。

**返回**

- `torch.Tensor`：形状 `[M, K]`、dtype `float32`。

**注意事项**

- 在低于 SM90 的设备上会退化到纯 PyTorch CPU 回退实现。

---

### `scaled_fp4_grouped_quantize`

```python
scaled_fp4_grouped_quantize(
    a,
    mask,
    a_global_sf,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
)
```

**功能**：使用逐行掩码将批处理输入张量量化为 NVFP4。

**参数**

- `a`（`torch.Tensor`）：形状 `[B, M, K]`，dtype fp16/bf16。
- `mask`（`torch.Tensor`）：量化前应用的掩码张量。
- `a_global_sf`（`torch.Tensor`）：形状 `[1]`、dtype `float32` 的全局缩放。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：4over6 配方。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`。`x_q` 逻辑形状 `[M, K/2, B]`、dtype
  `FLOAT4_E2M1X2`（实现将物理 `[B, M, K/2]` 置换为 batch 维在最后，符合 FlashInfer masked
  grouped GEMM 要求）；`sf` 为 6D swizzled 缩放，逻辑形状
  `[32, 4, padded_M // 128, 4, padded_K // 64, B]`，视为 `float8_e4m3fn`。

---

### `silu_and_mul_nvfp4_quantize`

```python
silu_and_mul_nvfp4_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    enable_pdl: Optional[bool] = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：在单个 CuTe-DSL 内核中融合 SwiGLU 与 NVFP4 量化。量化前计算
`silu(input[..., :K]) * input[..., K:]`。需要 CuTe-DSL 与 SM100+ GPU。

**参数**

- `input`（`torch.Tensor`）：连续 fp16/bf16 张量，形状 `[..., 2K]`，前导维度展平为 M。
- `global_scale`（`torch.Tensor`）：`float32` 形状 `[1]`，通常为
  `(448 * 6) / silu_and_mul(input).abs().max()`。
- `sf_vec_size`（`int`，默认 `16`）：仅支持 16。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：使用 swizzled 缩放布局。
- `is_sf_8x4_layout`（`bool`，默认 `False`）：使用 8x4 而非 128x4 swizzled 布局。
- `enable_pdl`（`bool`，可选）：启用 PDL；`None` 时自动检测。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：4over6 配方，要求 fp16/bf16 输入。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：打包 FP4 值（形状 `[M, K/2]`）及其缩放因子。

---

### `shuffle_matrix_a`

```python
shuffle_matrix_a(input_tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor
```

**功能**：TRT-LLM-gen `shuffleMatrixA` 的 PyTorch 等价实现。

**参数**

- `input_tensor`（`torch.Tensor`）：待洗牌的行主序矩阵。
- `epilogue_tile_m`（`int`）：沿 M 方向的 epilogue tile 大小，决定洗牌置换。

**返回**

- `torch.Tensor`：`input_tensor` 的行洗牌副本。

---

### `shuffle_matrix_sf_a`

```python
shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
)
```

**功能**：面向线性布局缩放因子的 TRT-LLM-gen `shuffleMatrixSfA` CUDA 实现。与上游
`shuffleMatrixSfA`（读写 128x4 布局）不同，本函数期望 `input_tensor` 处于量化 NVFP4
检查点使用的*线性*布局，不添加 padding。

**参数**

- `input_tensor`（`torch.Tensor`）：线性布局的缩放因子张量。
- `epilogue_tile_m`（`int`）：沿 M 方向的 epilogue tile 大小，决定行置换。
- `num_elts_per_sf`（`int`，默认 `16`）：每个缩放向量包含的元素数。

**返回**

- `torch.Tensor`：行洗牌后的缩放因子张量，重排进入 128x4 布局。

---

### `nvfp4_quantize_paged_kv_cache`

```python
nvfp4_quantize_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_layout: str = "HND",
    k_global_sf: Optional[torch.Tensor] = None,
    v_global_sf: Optional[torch.Tensor] = None,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    float,
    float,
]
```

**功能**：将分页 KV cache 量化为 NVFP4，供 trtllm-gen MHA 内核使用。采用两级缩放
（全局 FP32 + 逐块 FP8），并为 SM100 trtllm-gen MHA 内核布局 swizzle 缩放因子。

**参数**

- `k_cache`（`torch.Tensor`）：Key cache。`HND` 布局为
  `[num_pages, num_kv_heads, page_size, head_dim]`；`NHD` 布局为
  `[num_pages, page_size, num_kv_heads, head_dim]`。
- `v_cache`（`torch.Tensor`）：Value cache，与 `k_cache` 同布局。
- `kv_layout`（`str`，默认 `"HND"`）：输入 KV cache 布局，`"HND"` 或 `"NHD"`。
- `k_global_sf`（`torch.Tensor`，可选）：K 的全局缩放（float32 标量张量）；`None` 时
  自动计算为 `FLOAT8_E4M3_MAX / k_amax`。
- `v_global_sf`（`torch.Tensor`，可选）：V 的全局缩放；`None` 时自动计算。

**返回**

- `(kv_cache_fp4, kv_cache_sf, k_global_scale, v_global_scale)`：
  - `kv_cache_fp4` 为 `(k_fp4, v_fp4)`，与输入同布局但 `head_dim` 替换为 `head_dim // 2`，
    dtype `uint8`；
  - `kv_cache_sf` 为 `(k_scales, v_scales)`（`k_scales` 保持线性输入布局，`v_scales`
    使用 TRT-LLM 的 4-token 交错布局，二者 `head_dim` 替换为 `head_dim // 16`，
    dtype `float8_e4m3fn`）；
  - 末尾两个 float 为 `1 / k_global_sf` 与 `1 / v_global_sf`。

**注意事项**

- V 缩放 swizzle 要求 `page_size % 4 == 0` 且 `head_dim % 64 == 0`，否则抛 `ValueError`。

---

### `nvfp4_kv_dequantize`

```python
nvfp4_kv_dequantize(
    fp4_data: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor
```

**功能**：使用线性块缩放布局对 NVFP4 KV cache 做 GPU 反量化。要求 SM80+。

**参数**

- `fp4_data`（`torch.Tensor`）：形状 `[M, K/2]`、dtype `uint8` 的打包 FP4 数据。
- `block_scales`（`torch.Tensor`）：形状 `[M, K/16]`、dtype `uint8` 的逐块 FP8 E4M3 缩放。
- `global_scale`（`torch.Tensor`）：形状 `[1]`、dtype `float32` 的全局缩放，与 `fp4_data`
  同 CUDA 设备。
- `output_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：输出 dtype，`torch.bfloat16`
  或 `torch.float16`。

**返回**

- `torch.Tensor`：形状 `[M, K]`、指定输出 dtype 的反量化张量。

**注意事项**

- `K` 必须能被 16 整除，否则抛 `ValueError`。

---

### `nvfp4_kv_dequantize_paged`

```python
nvfp4_kv_dequantize_paged(
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_sf: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    kv_layout: str = "NHD",
) -> None
```

**功能**：将分页 NVFP4 KV cache 反量化到调用方拥有的连续输出。要求 SM80+。通过
`block_tables` gather 页，并以 `[batch, max_seq_len, num_heads, head_dim]` 布局写出
K/V。位置 `>= seq_lens[batch]` 的 token 保持不变。

**参数**

- `paged_kv_cache`（`torch.Tensor` 或其 tuple）：打包的 NVFP4 分页 KV cache，接受与分页
  注意力 API 相同的 tuple 或堆叠格式。tuple 输入时每个张量在 `kv_layout="NHD"` 下形状
  `[num_pages, page_size, num_kv_heads, head_dim // 2]`，`"HND"` 下为
  `[num_pages, num_kv_heads, page_size, head_dim // 2]`。
- `kv_cache_sf`（`torch.Tensor` 或其 tuple）：逐块 FP8 E4M3 缩放，格式同上，但不含
  `head_dim // 16` 而非 `head_dim // 2`。
- `block_tables`（`torch.Tensor`）：物理页表，形状 `[batch, max_pages_per_request]`，
  dtype `int32` 或 `int64`。
- `seq_lens`（`torch.Tensor`）：形状 `[batch]`、dtype `int32` 的序列长度。
- `k_scale`、`v_scale`（`torch.Tensor`）：dtype `float32`、与 cache 同 CUDA 设备的全局
  反量化缩放。
- `output_k`、`output_v`（`torch.Tensor`）：调用方拥有的输出，`[batch, max_seq_len,
  num_heads, head_dim]`，必须连续且 dtype 为 `torch.float16` 或 `torch.bfloat16`。
- `kv_layout`（`str`，默认 `"NHD"`）：分页输入布局，`"NHD"` 或 `"HND"`。

**返回**

- `None`：就地写入 `output_k` 与 `output_v`。

---

### `nvfp4_kv_quantize`

```python
nvfp4_kv_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用线性块缩放布局将数据 GPU 量化为 NVFP4 KV-cache 格式。要求 SM100+
（Blackwell），因为需要 `cvt.rn.satfinite.e2m1x2.f32` PTX 指令。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`、dtype bf16 或 fp16；`K` 必须能被 16 整除。
- `global_scale`（`torch.Tensor`）：形状 `[1]`、dtype `float32` 的全局缩放，与 `input`
  同 CUDA 设备。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(fp4_output, block_scales)`，`fp4_output` 为
  形状 `[M, K/2]`、dtype `uint8` 的打包 FP4；`block_scales` 为形状 `[M, K/16]`、
  dtype `uint8` 的逐块 FP8 E4M3 缩放。

**注意事项**

- `K` 必须能被 16 整除，否则抛 `ValueError`。

## FP8 量化

### `mxfp8_quantize`

```python
mxfp8_quantize(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
    sf_swizzle_layout: Optional[SfLayout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：将输入张量量化为 MxFP8 格式，附带缩放因子，支持多种输入数据类型与缩放布局。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16/fp8_quantized。`"cute-dsl"`
  后端额外支持 fp32 输入；`"cuda"` 不支持。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：是否使用 swizzled 缩放布局。
- `alignment`（`int`，默认 `32`）：`sfVecSize`。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时按设备能力（SM >= 9.0）自动检测。
- `backend`（`{"cuda", "cute-dsl"}`，默认 `"cuda"`）：`"cuda"` 为稳定 JIT CUDA 内核；
  `"cute-dsl"` 为 CuTe-DSL 内核（SM100+，**实验性**）。
- `sf_swizzle_layout`（`SfLayout`，可选）：缩放 swizzle 布局；提供时覆盖
  `is_sf_swizzled_layout`。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`，`x_q` 形状 `[M, K]`、dtype
  `FLOAT8_E4M3`；`sf` 形状取决于布局（`sf_vec_size` 固定为 32）。

**注意事项**

- `"cute-dsl"` 后端为**实验性**。

---

### `mxfp8_grouped_quantize`

```python
mxfp8_grouped_quantize(
    a: torch.Tensor,
    mask: torch.Tensor,
    backend: Literal["cutile", "cake"] = "cutile",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：以 UE8M0 块缩放将分组输入量化为 MXFP8。

**参数**

- `a`（`torch.Tensor`）：形状 `[B, M, K]`，dtype `float16` 或 `bfloat16`。
- `mask`（`torch.Tensor`）：int32 CUDA 张量，形状 `[B]`。每个值给出对应组要量化的有效
  行数，必须满足 `0 <= mask[i] <= M`。该前置条件由调用方负责：不会在运行时校验，因为
  读取设备侧 `mask` 会强制主机同步并破坏 CUDA-graph 捕获。越界值为未定义行为。内核写出
  缩放时不启用边界检查，因此 `mask[i] > M` 会破坏相邻组或越界写。
- `backend`（`{"cutile", "cake"}`，默认 `"cutile"`）：`"cutile"` 保持默认且为回退路径；
  `"cake"` 显式选择生成的、精确架构的 Blackwell profile，未安装该 profile 时给出清晰
  错误；运行时启动失败绝不会静默改用其它实现重试。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, sf)`。`x_q` 逻辑形状
  `[M, padded_K, B]`、dtype `float8_e4m3fn`；`sf` 逻辑形状
  `[32, 4, padded_M // 128, 4, padded_K // 128, B]`、dtype `uint8`。`padded_K` 将 K 上取整
  到 128 的倍数。物理布局先按 `B` 分组再置换，以匹配 FlashInfer masked grouped GEMM
  约定。仅写入每组前 `mask[i]` 行；`>= mask[i]` 的行（及其缩放）未定义，消费方必须使用
  相同 `mask` 且只读有效行。

**注意事项**

- 使用 cuTile 后端并要求 SM100+ 与 `cuda.tile`；`K` 必须能被 32 整除，内部 padding 到
  128 列的 tile。

---

### `mxfp8_dequantize_host`

```python
mxfp8_dequantize_host(
    input: torch.Tensor,
    scale_tensor: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    sf_swizzle_layout: Optional[SfLayout] = None,
) -> torch.Tensor
```

**功能**：主机端将 MxFP8 张量反量化为 float32。

**参数**

- `input`（`torch.Tensor`）：MxFP8 打包 FP8 张量，形状 `[M, K]`、dtype `FLOAT8_E4M3`。
- `scale_tensor`（`torch.Tensor`）：缩放因子张量（形状取决于布局与 `sf_vec_size`）。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：缩放是否 swizzled 存储。
- `sf_swizzle_layout`（`SfLayout`，可选）：显式 swizzle 布局，提供时覆盖
  `is_sf_swizzled_layout`；可选 `SfLayout.layout_128x4` 与 `SfLayout.layout_linear`。

**返回**

- `torch.Tensor`：形状 `[M, K]`、dtype `float32`。

---

### `per_token_group_quant_8bit`

```python
per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    backend: Literal["cutile"] = "cutile",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：逐 token 分组 8-bit 量化（FP8 或 INT8）。沿最后一维以连续 `group_size` 元素
为一组量化 `x`，产生量化张量与逐组缩放张量。这是块缩放 FP8/INT8 GEMM 路径使用的
per-token-group 量化（如 DeepGEMM 风格分组量化）。

**参数**

- `x`（`torch.Tensor`）：待量化输入；最后一维必须是 `group_size` 的倍数。
- `group_size`（`int`）：每个量化组的元素数（沿最后一维）。
- `eps`（`float`，默认 `1e-10`）：计算逐组缩放时的数值稳定 epsilon。
- `dst_dtype`（`torch.dtype`，可选）：量化输出 dtype（`torch.float8_e4m3fn` 或
  `torch.int8`），默认 `torch.float8_e4m3fn`。
- `column_major_scales`（`bool`，默认 `False`）：为 `True` 时返回列主序（Fortran）内存
  布局的缩放张量，逻辑形状不变。
- `scale_tma_aligned`（`bool`，默认 `False`）：为 `True` 时为 TMA 对齐 padding 缩放张量的
  前导维。
- `scale_ue8m0`（`bool`，默认 `False`）：为 `True` 时以 UE8M0 格式编码缩放
  （Blackwell / sm100+）。
- `backend`（`str`，默认 `"cutile"`）：实现后端，目前仅支持 `"cutile"`（cuda.tile Python 后端）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(x_q, x_s)`，量化张量（与 `x` 同形状）与逐组缩放
  张量，形状 `(*x.shape[:-1], x.shape[-1] // group_size)`。

## 融合门控激活 MXFP8 量化（`flashinfer.gated_act_mxfp8`）

### `silu_and_mul_mxfp8_quantize`

```python
silu_and_mul_mxfp8_quantize(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs
```

**功能**：应用 SwiGLU 并输出 RCEIL MXFP8 的 rowwise 和/或 colwise 结果。`gated_input`
形状 `[M, 2K]`，先存 gate 值后存 up 值。返回元组顺序为
`(row_output, col_output, row_scales, col_scales)`。被禁用的方向以输入设备上的零尺寸
张量表示。

**参数**

- `gated_input`（`torch.Tensor`）：连续 BF16 CUDA 张量，形状 `[M, 2K]`。
- `rowwise`（`bool`，默认 `True`）：是否输出 rowwise MXFP8 值与缩放。
- `colwise`（`bool`，默认 `False`）：是否输出 colwise MXFP8 值与缩放。

**返回**

- 张量元组：rowwise 值、colwise 值、rowwise 缩放、colwise 缩放。禁用方向输出为零尺寸
  张量。

**注意事项**

- 要求 SM100、SM103 或 SM107；`M`、`K` 必须为正的 128 的倍数，且 `gated_input` 需
  32 字节对齐。

---

### `silu_and_mul_mxfp8_quantize_backward`

```python
silu_and_mul_mxfp8_quantize_backward(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> GatedActMxfp8Outputs
```

**功能**：应用 SwiGLU 反向变换并输出 RCEIL MXFP8。逻辑结果拼接 gate 输入与 up 输入的
梯度，量化前形状为 `[M, 2K]`，不物化中间 BF16 结果到全局内存。

**参数**

- `gated_input`（`torch.Tensor`）：连续 BF16 CUDA 张量，形状 `[M, 2K]`。
- `grad_output`（`torch.Tensor`）：连续 BF16 CUDA 张量，形状 `[M, K]`。
- `rowwise`（`bool`，默认 `True`）：是否输出 rowwise MXFP8 值与缩放。
- `colwise`（`bool`，默认 `False`）：是否输出 colwise MXFP8 值与缩放。

**返回**

- 张量元组：拼接后 gate 输入与 up 输入梯度的 rowwise 值、colwise 值、rowwise 缩放、
  colwise 缩放。禁用方向输出为零尺寸张量。

## CuTe-DSL 量化内核（实验性）

CuTe-DSL 后端在安装了 `nvidia-cutlass-dsl` 包时条件可用。运行时它们也会在可用时以
`flashinfer.quantization.{nvfp4,mxfp4,mxfp8}_quantize_cute_dsl` 重新导出。

### `nvfp4_quantize_cute_dsl`

```python
nvfp4_quantize_cute_dsl(
    input: torch.Tensor,
    global_scale: float | torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用 CuTe-DSL 内核将输入张量量化为 NVFP4，与
`flashinfer.quantization.nvfp4_quantize` 匹配：E4M3 缩放（FP8）、E2M1 输出格式
（4-bit，2 值/字节）、支持 128x4 / 8x4 / 线性布局、`sf_vec_size = 16`。内核分别针对
主机标量与设备张量全局缩放编译，并在运行时处理变化的 `M` 而无需重编译。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16/float8_e4m3fn。
- `global_scale`（`float` 或 `torch.Tensor`）：主机侧 NVFP4 全局缩放；为向后兼容也接受
  单元素张量。
- `sf_layout`（`int`，默认 `SF_LAYOUT_128x4`）：缩放布局（`0=128x4`、`1=8x4`、`2=linear`）。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时按设备能力（SM >= 9.0）自动检测。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：4over6 配方，要求 fp16/bf16 输入。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(fp4_tensor, scale_tensor)`，`fp4_tensor` 形状
  `[M, K/2]`、dtype `uint8`；`scale_tensor` 为 E4M3 缩放（`uint8`），重塑为
  `[padded_rows, K/16]`。

---

### `nvfp4_quantize_per_token_cute_dsl`

```python
nvfp4_quantize_per_token_cute_dsl(
    input: torch.Tensor,
    global_scale_inv: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
    out_scale: torch.Tensor | None = None,
    nvfp4_4over6: Optional[NVFP44Over6Config] = _UNSET,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：使用 CuTe-DSL 内核做 per-token NVFP4 激活量化。与
`nvfp4_quantize_cute_dsl`（单一全局缩放）不同，本变体对激活的**每一行（token）**计算
一个量化缩放，每行独立缩放以使其最大幅值映射到 NVFP4 动态范围，并将得到的 per-token
缩放与打包 FP4 输出、E4M3 块缩放一同返回。

**参数**

- `input`（`torch.Tensor`）：2-D 激活张量，形状 `[M, K]`，dtype fp16/bf16；`K` 必须能被
  `NVFP4_SF_VEC_SIZE`（16）整除。
- `global_scale_inv`（`torch.Tensor`）：标量张量（`float32`），叠加在 per-token 缩放上的
  逆全局缩放；也可接受 Python `float` 并在内部包装为张量。
- `sf_layout`（`int`，默认 `SF_LAYOUT_128x4`）：缩放布局（`0=128x4`、`1=8x4`、`2=linear`）。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时自动检测；传 `False` 强制关闭。
- `out_scale`（`torch.Tensor`，可选）：同设备标量张量（`float32`），返回的
  `per_token_scale` 乘以其值，由内核折叠，使下游输出标量无需单独处理。不改变
  `fp4_output` 或 `scale_output`。
- `nvfp4_4over6`（`NVFP44Over6Config` 或 `None`，默认省略）：4over6 配方。

**返回**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`：`(fp4_output, scale_output,
  per_token_scale)`：
  - `fp4_output`：形状 `[M, K/2]`、dtype `uint8`（每字节两个 E2M1 值）。
  - `scale_output`：E4M3 块缩放（`uint8`），重塑为 `[padded_rows, padded_sf_cols]`；
    padding 取决于 `sf_layout`（`linear` 保持 `M` 行，`128x4` / `8x4` padding 至 tile）。
  - `per_token_scale`：per-row 量化缩放，形状 `[M]`、dtype `float32`，给定 `out_scale`
    时已乘以其值。

---

### `mxfp4_quantize_cute_dsl`

```python
mxfp4_quantize_cute_dsl(
    input: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用 CuTe-DSL 内核将输入张量量化为 MXFP4，采用双路径优化：**LINEAR** 布局为
flat SF-block 迭代并自适应 1T/4T per SF block（低 SM 数 GPU（<=80 SM）用 4T/SF 以获得
合并访问，高 SM 数 GPU 用 1T/SF）；**SWIZZLED** 布局为基于行的迭代并带 padding 快速路径。
内核按 `(K, dtype, sf_layout, pdl, use_4t)` 元组编译一次，运行时处理变化的 `M`。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16。
- `sf_layout`（`int`，默认 `SF_LAYOUT_128x4`）：缩放布局（`0=128x4`、`1=8x4`、`2=linear`）。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时自动检测。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(fp4_tensor, scale_tensor)`，`fp4_tensor` 形状
  `[M, K/2]`、dtype `uint8`；`scale_tensor` 为 UE8M0 缩放（`uint8`）。线性输出形状
  `[M, K/32]`；swizzled 输出形状 `[padded_rows, padded_scale_columns]`。

---

### `mxfp8_quantize_cute_dsl`

```python
mxfp8_quantize_cute_dsl(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: bool | None = None,
    is_sf_8x4_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：使用 CuTe-DSL 内核将输入张量量化为 MXFP8，采用双路径优化：**LINEAR** 布局为
SF-block 迭代（快速）；**SWIZZLED** 布局为基于行的迭代并带 padding 快速路径。内核按
`(K, dtype, pdl, sf_layout)` 元组编译一次，运行时处理变化的 `M`。

**参数**

- `input`（`torch.Tensor`）：形状 `[M, K]`，dtype fp16/bf16/fp32。
- `is_sf_swizzled_layout`（`bool`，默认 `True`）：是否使用 swizzled 布局；`True` 时默认
  128x4，传 `is_sf_8x4_layout=True` 使用 8x4。
- `alignment`（`int`，默认 `32`）：K 维对齐（必须是 `SF_VEC_SIZE` 的倍数）。
- `enable_pdl`（`bool`，可选）：是否启用 PDL；`None` 时自动检测。
- `is_sf_8x4_layout`（`bool`，默认 `False`）：`is_sf_swizzled_layout=True` 时选择 8x4
  而非默认 128x4；线性布局下忽略。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(fp8_tensor, scale_tensor)`，`fp8_tensor` 形状
  `[M, padded_K]`、dtype `float8_e4m3fn`；`scale_tensor` 为 UE8M0 缩放（`uint8`）。

## 相关链接 / 参考

- `flashinfer.fp4_quantization`：向后兼容垫片，重新导出同一批 FP4 符号（见
  `api-fp4-quantization.md`）。
- `flashinfer.gemm.mm_fp4` / `mm_mxfp8`：消费本模块产出的量化张量与缩放布局。

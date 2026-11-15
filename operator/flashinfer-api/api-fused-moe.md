# 融合 MoE（`flashinfer.fused_moe`）

> 源码：`flashinfer/fused_moe/`（`__init__.py`、`api.py`、`layer.py`、`core.py`、`runners.py`、`prepare.py`、`bgmv_moe.py`、`monomoe.py`、`cute_dsl/`、`cutile/`、`backends/` 等）
> 上游文档：`docs/api/fused_moe.rst`
> 适用版本：FlashInfer 0.7.2

本模块提供针对不同后端与数据类型优化的融合 Mixture-of-Experts（MoE）算子。它包含三层 API：

1. **统一 MoE API**：后端无关的配置（`MoEConfig` / `QuantConfig` / `RoutingConfig` /
   `ExpertConfig` / `ExecutionConfig` / `MoEFinalizeConfig` / `BackendOptions`）与 `MoELayer` 类。
   `MoELayer` 是本 API 的正式入口：其构造函数与调用运算符都带 ``@flashinfer_api``，参与
   ``FLASHINFER_LOGLEVEL`` 日志与 ``FLASHINFER_DUMP_*`` 捕获。
2. **各后端 prepare 辅助**：`TrtllmFp4Config`、`CudnnFrost*Config`、`CuteDslConfig`、
   `Cutlass*Config`、`CuTile*Config`、`CudnnGroupedGemm*Config`、`CakeWarpDecodeConfig`、
   `PrimsTsConfig` 等，每个提供 ``prepare_weights`` / ``prepare_activations`` 与 ``supported``。
3. **传统扁平参数 API**：`cutlass_fused_moe`、`trtllm_*_moe`、`prims_ts_*_moe`、
   `cute_dsl_fused_moe*`、`b12x_fused_moe`、`mono_moe` 等。这些与其自身姿态同样是正式的；两层
   API 设计为共存，互不取代。

## 导入

```python
from flashinfer.fused_moe import (
    MoELayer, MoEConfig, RoutingConfig, QuantConfig, QuantFormat,
    ExpertConfig, ExecutionConfig, MoEFinalizeConfig, BackendOptions,
    MoEActivationPack, MoEWeightPack, SwiGLU, SiTU,
    BackendOptions, TrtllmFp4Config, CudnnFrostNvfp4Config,
    cutlass_fused_moe, trtllm_bf16_moe, cute_dsl_fused_moe,
    reorder_rows_for_gated_act_gemm,
)
from flashinfer.tllm_enums import (
    ActivationType, RoutingMethodType, WeightLayout, SfLayout, is_gated_activation,
)
```

## API 列表

### 类型与枚举

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `RoutingMethodType` | 枚举（IntEnum） | top-K 路由方法（Default / Renormalize / DeepSeekV3 / …）。 |
| `WeightLayout` | 枚举（IntEnum） | 权重布局（MajorK / MajorMn / BlockMajorK）。 |
| `ActivationType` | 枚举（IntEnum） | 激活类型（Gelu / Silu / Swiglu / Situ / …），传统 API 使用。 |
| `SfLayout` | 枚举（IntEnum） | block scale 的布局（128x4 / 8x4 / linear）。 |
| `is_gated_activation` | 函数 | 判断激活是否为 gated（SwiGLU 家族）。 |

### 统一 MoE API

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `QuantFormat` | 枚举 | 一个 MMA 操作数或层输出的数值格式。 |
| `RoutingConfig` | dataclass | 专家路由参数。 |
| `QuantConfig` | dataclass | 权 / 激活格式与输出格式。 |
| `ExpertConfig` | dataclass | 专家几何。 |
| `ExecutionConfig` | dataclass | 运行时执行参数。 |
| `MoEFinalizeConfig` | dataclass | finalize（combine）行为。 |
| `BackendOptions` | dataclass | 后端候选的有序列表。 |
| `MoEConfig` | dataclass | 顶层 MoE 配置。 |
| `MoEActivationPack` | dataclass | 每次调用的后端原生激活 + 路由输入。 |
| `MoEWeightPack` | dataclass | 长期权重容器（含各后端原生 materialization）。 |
| `MoELayer` | 类 | 带跨后端 autotune 的有状态 MoE 层。 |
| `CudnnFrostBf16Config` / `CudnnFrostMxfp8Config` / `CudnnFrostNvfp4Config` / `CudnnFrostMxfp8Mxfp4Config` | 类 | cuDNN Frost 后端配置。 |
| `SwiGLU` / `SiTU` / `GELU` / `GeGLU` / `GeGLUTanh` / `SwiGLUStep` / `ReLU` / `ReLU2` / `SiLU` / `Identity` | dataclass | 带类型的融合激活值。 |

### 工具函数

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `convert_to_block_layout` | 函数 | 把 2-D 张量 reshape 为 3-D block 布局。 |
| `reorder_rows_for_gated_act_gemm` | 函数 | 重排 gated-activation GEMM 权重行。 |
| `interleave_moe_weights_for_sm90_mixed_gemm` | 函数 | 为 SM90 mixed-input GEMM 交错 4-bit 权重。 |
| `interleave_moe_scales_for_sm90_mixed_gemm` | 函数 | 折叠 SM90 mixed-input GEMM 的权重 scale。 |
| `preprocess_moe_weights_for_sm90_mixed_gemm_humming` | 函数 | Humming 风格 MXFP4 权重预处理。 |
| `fused_topk_deepseek` | 函数 | DeepSeek-V3 融合专家路由（top-k 选择）。 |
| `hash_topk` | 函数 | DeepSeek-V4 hash-MoE 表查找路由。 |

### AlphaMoE Router / FP8 / NVFP4

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `AlphaMoERoutePlan` | NamedTuple | 设备端路由输出与可复用私有工作区。 |
| `allocate_alphamoe_route_plan` | 函数 | 为 logits 分配可复用的路由计划。 |
| `alphamoe_fused_router` | 函数 | FP32 logits → top-k 权重与专家分组对齐计划。 |
| `alphamoe_interleave_gated_weights` | 函数 | 交错 gate/up 权重到 alphamoe_sm100 设备布局。 |
| `alphamoe_fp8_block_scale_aligned_moe` | 函数 | 预对齐计划上的 W8A8 block-scale MoE。 |
| `alphamoe_nvfp4_aligned_moe` | 函数 | AlphaMoE NVFP4 对齐入口。 |
| `alphamoe_nvfp4_routed_moe` | 函数 | AlphaMoE NVFP4 路由入口。 |
| `prepare_nvfp4_w1_scales` / `prepare_nvfp4_w1_data` / `prepare_nvfp4_w2_data` / `prepare_nvfp4_w2_data_k256` / `prepare_nvfp4_w2_scales` / `prepare_nvfp4_w2_scales_k256` / `prepare_nvfp4_w1_gate_up_data` / `prepare_nvfp4_w1_gate_up_scales` | 函数 | 一次性准备 NVFP4 权重 / scale 面板。 |

### Multi-LoRA MoE（BGMV）

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `bgmv_moe` | 函数 | 高层次多 LoRA MoE BGMV：shrink + expand。 |
| `prepare_bgmv_moe` | 函数 | 准备可图重放的 shrink+expand 流水线。 |
| `BGMVMoECakePlan` / `BGMVMoEPortablePlan` | 类 | Cake / portable 执行计划。 |
| `bgmv_moe_shrink` / `bgmv_moe_expand` | 函数 | 底层 shrink / expand。 |
| `bgmv_moe_gemm1_lora_delta` / `bgmv_moe_gemm2_lora_delta` | 函数 | FC1 / FC2 LoRA delta。 |

### 后端 Fused MoE

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `cutlass_fused_moe` / `cutlass_fused_moe_workspace_size` | 函数 | CUTLASS 融合 MoE 与其工作区查询。 |
| `trtllm_bf16_moe` / `trtllm_bf16_routed_moe` | 函数 | TRT-LLM BF16 MoE（logits / 预路由）。 |
| `trtllm_fp8_per_tensor_scale_moe` / `trtllm_fp8_per_tensor_scale_routed_moe` | 函数 | FP8 per-tensor-scale。 |
| `trtllm_fp8_per_channel_scale_moe` / `trtllm_fp8_per_channel_scale_routed_moe` | 函数 | FP8 per-channel-scale。 |
| `trtllm_fp8_block_scale_moe` / `trtllm_fp8_block_scale_routed_moe` | 函数 | FP8 block-scale。 |
| `trtllm_fp4_block_scale_moe` / `trtllm_fp4_block_scale_routed_moe` | 函数 | FP4 block-scale。 |
| `trtllm_mxint4_block_scale_moe` / `trtllm_mxint4_block_scale_routed_moe` | 函数 | MXINT4 block-scale。 |
| `prims_ts_bf16_moe` / `prims_ts_bf16_routed_moe` | 函数 | Prims-TS BF16。 |
| `prims_ts_fp4_block_scale_moe` / `prims_ts_fp4_block_scale_routed_moe` | 函数 | Prims-TS FP4。 |
| `prims_ts_fp8_block_scale_moe` / `prims_ts_fp8_block_scale_routed_moe` / `prims_ts_fp8_per_tensor_scale_moe` | 函数 | Prims-TS FP8。 |
| `trtllm_gen_routing` / `TrtllmGenRoutingResult` | 函数 / NamedTuple | 独立的 TRT-LLM Gen 路由阶段。 |
| `cute_dsl_fused_moe` / `cute_dsl_fused_moe_bf16` / `cute_dsl_fused_moe_nvfp4` / `cute_dsl_fused_moe_mxfp8_mxfp4` | 函数 | CuteDSL 融合 MoE。 |
| `b12x_fused_moe` | 函数 | SM120/SM121 b12x 融合 MoE。 |
| `CuteDslMoEWrapper` / `CuteDslBf16MoEWrapper` / `CuteDslMxfp8Mxfp4MoEWrapper` / `B12xMoEWrapper` | 类 | 各 CuteDSL / b12x 包装类。 |
| `mono_moe` / `has_monomoe` / `get_scratchpad_size_bytes` / `alloc_scratchpad` / `interleave_for_tma_wgmma_up` | 函数 | MonoMoE（单 kernel block-FP8，SM90a）。 |
| `CakeWarpDecodeConfig` / `CakeWarpDecodeRunner` | 类 | Cake NVFP4 warp-decode（SM100/SM103）。 |
| `PrimsTsConfig` / `PrimsTsRunner` | 类 | Prims-TS 统一 MoE（SM100/SM103）。 |

### cuTile Fused MoE

CuTile 后端以 Config / Runner 成对提供：`CuTileBf16Config`/`Runner`、`CuTileFp8PerTensorConfig`/
`Runner`、`CuTileFp8PerTensorBf16Config`/`Runner`、`CuTileMxfp4Config`/`Runner`、
`CuTileMxfp4Bf16Config`/`Runner`、`CuTileMxfp8Config`/`Runner`、`CuTileMxfp8Bf16Config`/`Runner`、
`CuTileNvfp4Config`/`Runner`、`CuTileNvfp4Bf16Config`/`Runner`、`CuTileMxfp4Mxfp8Config`/`Runner`。

### cuDNN Grouped-GEMM MoE

`CudnnGroupedGemmBf16Config`/`Runner`、`CudnnGroupedGemmFp8PerTensorConfig`/`Runner`、
`CudnnGroupedGemmMxfp8Config`/`Runner`、`CudnnGroupedGemmNvfp4Config`/`Runner`。

---

## API 详情

### 类型与枚举

#### `RoutingMethodType`

```python
class RoutingMethodType(IntEnum):
    Default = 0          # Softmax → TopK
    Renormalize = 1      # TopK → Softmax
    DeepSeekV3 = 2       # Sigmoid → RoutingBiasAdd → Top-2 in group → Top4 groups → Top8 experts
    Llama4 = 3           # Top1 → Sigmoid
    RenormalizeNaive = 4 # Softmax → TopK → Renormalize（Qwen3 风格）
    TopK = 5             # 仅 TopK（无 softmax）
    SigmoidRenorm = 6    # Sigmoid → TopK → Renormalize
    MiniMax2 = 7         # Sigmoid + Bias → TopK → ScaledSumNormalize
    Sigmoid = 8          # Sigmoid → TopK（不重归一化）
    TopKSigmoid = 9      # TopK → Sigmoid（不重归一化）
    Unspecified = 10
```

**注意事项**：其实现在 ``flashinfer.tllm_enums``，与 ``include/flashinfer/trtllm/fused_moe/runner.h``
保持同步。成员定义了评估安全的 ``__repr__``（``RoutingMethodType.Default``），供统一 MoE 配置的
``eval(repr(cfg))`` 往返使用。

#### `WeightLayout`

```python
class WeightLayout(IntEnum):
    MajorK = 0        # K-major（默认）。[Mn, K]
    MajorMn = 1       # A 为 M-major、B 为 N-major。[K, Mn]
    BlockMajorK = 2   # K 维分块。[K / blockK, Mn, blockK]，blockK 固定 128B
```

#### `ActivationType`

```python
class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    SwigluStep = 7
    GegluTanh = 8
    Identity = 9
    Situ = 10
    InvalidType = 11
    ClampedRelu2 = 12   # 仅 CUTLASS 后端
```

**属性/方法**：`is_gated`（属性）返回该激活是否消费 gate 分支（SwiGLU 家族）。

#### `SfLayout`

```python
class SfLayout(IntEnum):
    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2
```

#### `is_gated_activation`

```python
is_gated_activation(activation_type: Union[int, ActivationType]) -> bool
```

**功能**：返回给定激活是否为 gated（例如 SwiGLU 家族）。镜像 C++ 的 ``isGatedActivation()``。

**参数**：`activation_type`（`Union[int, ActivationType]`）——激活类型或其整数值。

**返回**：`bool`——属于 ``Swiglu`` / ``Geglu`` / ``SwigluBias`` / ``SwigluStep`` / ``GegluTanh`` /
``Situ`` 时为 `True`，否则 `False`。

**示例**

```python
>>> from flashinfer.tllm_enums import ActivationType, is_gated_activation
>>> is_gated_activation(ActivationType.Swiglu)
True
>>> is_gated_activation(ActivationType.Relu)
False
```

---

### 统一 MoE API

#### `QuantFormat`

```python
class QuantFormat(Enum):
    BF16 = 0
    FP16 = 1
    FP8PerTensor = 2
    DeepSeekFp8 = 3
    MXFP8 = 4
    NVFP4 = 5
    MXFP4 = 6
    MXINT4 = 7
    INT4 = 8
```

**功能**：某个 MMA 操作数或层输出的数值格式。轴描述 MMA 消费的格式，而非跨越 Python API 的张量
dtype。一个接收 BF16 激活、在 kernel 内量化的 b12x / cuTile NVFP4 runner 仍是 ``(NVFP4, NVFP4)``。

#### `RoutingConfig`

```python
@dataclass
class RoutingConfig:
    num_experts: int
    top_k: int
    method: RoutingMethodType = RoutingMethodType.Default
    n_group: Optional[int] = None          # keyword-only
    topk_group: Optional[int] = None       # keyword-only
    routed_scaling_factor: Optional[float] = None  # keyword-only
```

**字段**

- `num_experts`（`int`）：全局专家总数（EP 分片前）。
- `top_k`（`int`）：每 token 选择的专家数。
- `method`（`RoutingMethodType`，默认 `Default`）：路由策略。
- `n_group`（`Optional[int]`，默认 `None`）：DeepSeekV3 路由的专家组数。
- `topk_group`（`Optional[int]`，默认 `None`）：DeepSeekV3 中选择的组数。
- `routed_scaling_factor`（`Optional[float]`，默认 `None`）：固定路由权重缩放（DeepSeekV3）。

#### `QuantConfig`

```python
@dataclass
class QuantConfig:
    weight: QuantFormat = QuantFormat.BF16
    activation: QuantFormat = QuantFormat.BF16
    output: QuantFormat = QuantFormat.BF16
    # keyword-only：
    swizzled_scale_factors: Optional[bool] = None
    per_token_scale: Optional[bool] = None
    nvfp4_4over6: Optional[NVFP44Over6Config] = UNSPECIFIED
```

**字段**

- `weight, activation`（`QuantFormat`）：MMA 权 / 激活格式。省略的轴为 BF16（未量化）：即
  ``QuantConfig()`` 为 BF16×BF16，``QuantConfig(weight=MXFP4)`` 为 MXFP4 权重 + BF16 激活（W4A16）。
  MXFP4×MXFP8 必须同时拼写两个轴。
- `output`（`QuantFormat`，默认 BF16）：层输出格式。传 `QuantFormat.FP16` 而非 ``torch.float16``。
- `swizzled_scale_factors`（`Optional[bool]`，默认 `None`）：block scale 是否使用 swizzled（相对
  linear）布局；`None` 表示后端默认。镜像 core 的 ``swizzled_input_sf``。
- `per_token_scale`（`Optional[bool]`，默认 `None`）：激活是否带 per-token scale。
- `nvfp4_4over6`（`NVFP44Over6Config | None`）：NVFP4 激活量化的 “4over6” scale 候选搜索；
  未设置时读取 ``FLASHINFER_NVFP4_4OVER6*`` 环境变量，``None`` 表示关闭，给 config 表示启用并
  忽略环境变量。仅 NVFP4 激活接受显式值。

**注意事项**：本 dataclass 只检查每个轴是 `QuantFormat`；合法组合属于 runner 能力
（``supported_quant_variants`` / ``supported_output_formats``），而非这里的 allow-list。
``nvfp4_4over6`` 在 `activation` 非 NVFP4 时设置会抛 `ValueError`。

#### 激活值类

```python
@dataclass(frozen=True)
class SwiGLU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Swiglu
    alpha: float = DEFAULT_SWIGLU_ALPHA   # 1.0
    beta: float = DEFAULT_SWIGLU_BETA     # 0.0
    limit: float = DEFAULT_SWIGLU_LIMIT   # float32 max

@dataclass(frozen=True)
class SiTU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Situ
    gate_scale: float = DEFAULT_SITU_BETA        # 4.0
    linear_scale: Optional[float] = DEFAULT_SITU_LINEAR_BETA  # 25.0
    clamp_limit: Optional[float] = None
```

`SwiGLU`：线性分支被夹到 ``[-limit, limit]``，gate 分支被夹到 ``(-inf, limit]``，然后计算
``gate * sigmoid(alpha * gate) * (linear + beta)``。

`SiTU`：``linear_scale`` 施加 ``linear_scale * tanh(linear / linear_scale)``；``None`` 选择不夹的
线性分支（目前仅 CuTe-DSL 可表达）。默认值是规范 SiTU（Kimi-K3）scale。后端适配器必须降级这些
精确值，或拒绝不支持语义，而不是依赖不同的原生默认值。

其余带类型激活：`GELU`、`GeGLU`、`GeGLUTanh`、`SwiGLUStep`、`ReLU`、`ReLU2`、`SiLU`、`Identity`
（均为 `ActivationConfig` 的子类，携带各自的标量语义）。`ActivationConfig` 是 `MoEConfig` 接受的
带类型融合激活值；per-expert 张量覆盖仍是后端原生 weight view 的属性。

#### `ExpertConfig`

```python
@dataclass(frozen=True)
class ExpertConfig:
    intermediate_size: int
    local_expert_offset: int = 0
    local_num_experts: Optional[int] = None
    num_fused_shared_experts: int = 0
```

**字段**

- `intermediate_size`（`int`）：专家 FFN 的隐藏维度（gemm1 的 MxK → MxN 中的 N）。
- `local_expert_offset`（`int`，默认 `0`）：专家并行分片的起始索引。
- `local_num_experts`（`Optional[int]`，默认 `None`）：本 rank 的专家数；`None` → 运行期用
  ``num_experts``。
- `num_fused_shared_experts`（`int`，默认 `0`）：每个 token 都运行的共享专家数。其行跟在路由专家
  之后，因此权重有 ``E + S`` 行，而路由字段仍仅路由。跨字段约束由 `MoEConfig` 检查。

#### `MoEFinalizeConfig`

```python
@dataclass(frozen=True)
class MoEFinalizeConfig:
    do_finalize: bool = True
    use_fused_finalize: bool = True
```

**字段**

- `do_finalize`（`bool`，默认 `True`）：是否应用路由权重缩放并把 per-expert 部分结果累加进输出。
  ``False`` 则以 ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]`` 返回未归约中间
  结果，把 combine 留给调用方（``expanded_idx_to_permuted_idx[token * top_k + slot]`` 是
  ``gemm2_output`` 中持有该分配的行，专家不在本地时为 ``-1``）。仅广告未 finalize 输出的后端支持
  该模式。
- `use_fused_finalize`（`bool`，默认 `True`）：支持的后端是否在 GEMM2 epilogue（原子累加）中归约，
  而非单独运行归约 kernel；不支持的后端忽略该标志。

#### `ExecutionConfig`

```python
@dataclass(frozen=True)
class ExecutionConfig:
    enable_pdl: Optional[bool] = None
    tune_max_num_tokens: int = 8192
```

**字段**

- `enable_pdl`（`Optional[bool]`，默认 `None`）：持续设备启动；`None` 表示自动（sm90+ 为 True）。
- `tune_max_num_tokens`（`int`，默认 `8192`）：autotuner / CUDA Graph 捕获的 token 预算提示。

#### `BackendOptions`

```python
@dataclass(frozen=True)
class BackendOptions:
    candidates: Tuple[BackendConfigType, ...] = ()
```

**功能**：dispatch 与 autotune 用的后端候选有序列表。

**方法/属性**

- `valid_for(arch)`：返回适合该架构的候选项。
- `__len__` / `__iter__`：候选数量与迭代。
- 每个后端 config 实现 ``supported(arch)``，``arch`` 使用与 `valid_for` 记录的 CUDA compute
  capability 编码。

#### `MoEConfig`

```python
@dataclass(frozen=True)
class MoEConfig:
    routing: RoutingConfig
    quant: QuantConfig
    experts: ExpertConfig
    activation: ActivationConfig = field(default_factory=SwiGLU)
    backend: BackendOptions = field(default_factory=lambda: _DEFAULT_BACKEND)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    finalize: MoEFinalizeConfig = field(default_factory=MoEFinalizeConfig)
```

**功能**：顶层 MoE 配置，把所有子配置组合为单个可哈希、可序列化对象，支持 ``**config`` 解包
（dict 协议）。

**方法**：`keys()` 与 `__getitem__` 提供 dict 协议，使 ``fused_moe(tensors, **config)`` 可用。

**示例**

```python
>>> config = MoEConfig(
...     routing=RoutingConfig(num_experts=64, top_k=8,
...                           method=RoutingMethodType.DeepSeekV3),
...     quant=QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
...     experts=ExpertConfig(intermediate_size=2048),
... )
>>> output = fused_moe(tensors, **config)
```

**注意事项**：跨字段约束（例如 fused shared experts 与 EP 的组合）由 `MoEConfig.__post_init__` /
`_validate_fused_shared_experts` 校验。

#### `MoEActivationPack`

```python
@dataclass(frozen=True)
class MoEActivationPack:
    hidden_states_q: Tensor
    hidden_states_scale: Optional[Tensor]
    topk_ids: Optional[Tensor] = None
    topk_weights: Optional[Tensor] = None
    # keyword-only：
    per_token_scale: Optional[Tensor] = None
    routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed
    routing_logits: Optional[Tensor] = None
    routing_bias: Optional[Tensor] = None
```

**功能**：每次调用的后端原生激活 + 路由输入。

**字段**：`hidden_states_q`（量化激活）、`hidden_states_scale`（block scale，可选）、`topk_ids` /
`topk_weights`（预计算路由）、`per_token_scale`、`routing_input_mode`
（`FromLogits` / `PackedPrecomputed` / `UnpackedPrecomputed`）、`routing_logits`、`routing_bias`。

**属性**：`num_tokens`。

**注意事项**：具体字段组合由 `routing_input_mode` 决定；`__post_init__` 会校验一致性。

#### `MoEWeightPack`

```python
@dataclass
class MoEWeightPack:
    native_views: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)

    def prepare_for(self, backend_key: str, view: Dict[str, Tensor]) -> None: ...
    def get_view(self, backend_key: str) -> Dict[str, Tensor]: ...
```

**功能**：长期权重容器，按后端 ``backend_key`` 存各后端原生权重布局（量化、swizzle、MMA 排序等）。
在模型载入 / 层初始化时经 ``prepare_for(key, view)`` 填充一次，每次调用经 ``get_view(key)`` 读取。

**方法**

- `prepare_for(backend_key, view)`：注册一个后端原生 weight view。调用方负责量化 / swizzle /
  布局转换，本方法只存储结果。
- `get_view(backend_key)`：返回该后端 view；未准备时抛 `KeyError`，消息中列出可用 key。

**注意事项**：持有多个 materialization 是有意的——这是跨后端 autotune 的内存代价。每个 view 正是
对应 runner ``forward`` 期望的 weight-side kwargs dict。所有 ``XxxConfig.prepare_weights`` 都采用
同一规范源布局：`w1` 为 gated 激活的 ``[E, 2I, H]``（行序 ``[up, gate]``）或非 gated 的
``[E, I, H]``；`w2` 为 ``[E, H, I]``。

常用 ``backend_key``：`trtllm_fp4_routed`、`trtllm_fp8_block`、`trtllm_fp8_per_tensor`、
`trtllm_bf16_routed`、`trtllm_mxint4_routed`、`prims_ts`、`cutlass_bf16`、`cutlass_w4a16`、
`cutlass_nvfp4`、`cutlass_fp8_per_tensor`、`cutlass_fp8_block`、`cutlass_mxfp8_mxfp4`、
`cutlass_mxfp8`、`cutlass_w4a8`、`cutlass_humming`、`cutile_*`、`cute_dsl`、`b12x_nvfp4`、
`b12x_w4a16`、`sm12x_fp8` / `sm12x_mxfp8_mxfp4` / `sm12x_nvfp4_bf16`、
`cudnn_frost_bf16` / `cudnn_frost_mxfp8` / `cudnn_frost_nvfp4` / `cudnn_frost_mxfp8_mxfp4`、
`cudnn_grouped_gemm_bf16` / `cudnn_grouped_gemm_fp8_per_tensor` / `cudnn_grouped_gemm_mxfp8` /
`cudnn_grouped_gemm_nvfp4`、`cake`。

#### `MoELayer`

```python
class MoELayer:
    def __init__(self, config: MoEConfig, device: Optional[torch.device] = None): ...
    def __call__(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> Union[torch.Tensor, List[torch.Tensor]]: ...
```

**功能**：带跨后端 autotune 的有状态 MoE 层。TRTLLM runner 把不可变的启动元数据绑定到每个 packed
输入列表，因此交错的 ``pack_inputs -> forward`` 对不能互换权重或路由配置；其它后端适配器可能仍保留
per-call 工作区或 prepared-weight 状态。请在每个线程 / stream 使用一个 `MoELayer`，直到这些适配器
采用相同约定。

**`__init__` 参数**

- `config`（`MoEConfig`）：路由、量化、专家几何、激活、后端候选与执行参数。跨字段约束由
  `MoEConfig` 自身校验。
- `device`（`torch.device | None`）：用其 compute capability 选择可用后端的设备；`None` → 当前
  CUDA 设备。

**`__init__` 抛出**：`RuntimeError`——若该设备架构下没有任何已配置后端可用。

**`__call__` 参数**

- `act_pack`（`MoEActivationPack`）：本次调用的后端原生激活 + 路由输入。
- `weight_pack`（`MoEWeightPack`）：为所选后端原生布局准备好的专家权重。

**`__call__` 返回**

- `torch.Tensor | list[torch.Tensor]`：层输出。``config.finalize.do_finalize=False`` 时返回未归约
  中间结果 ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``。

**`__call__` 抛出**：`ValueError`——若 ``act_pack.num_tokens`` 超过层构建时的
``execution.tune_max_num_tokens`` 上限；`NotImplementedError`——若没有可用后端支持该 pack 的
``routing_input_mode``。

**其它方法/属性**：`winner_backend`（最近一次调用选中的后端 key，首次调用前为 `None`）、
`reset_winner()`（清除所有 per-bucket 缓存 winner，下次调用重新调优）、`_select_winner`
（对各 runner 调优后按 packing + forward GPU 延迟选择）。

**注意事项**：只有支持该 pack ``routing_input_mode`` 的 runner 才参与竞争（并非每个后端都有
in-kernel router），winner 按 ``(token-count bucket, mode)`` 缓存，重复调用跳过重新选择。

**示例**

```python
>>> layer = MoELayer(config)
>>> out = layer(act_pack, weight_pack)
```

#### `CudnnFrostBf16Config` / `CudnnFrostMxfp8Config` / `CudnnFrostNvfp4Config` / `CudnnFrostMxfp8Mxfp4Config`

```python
@dataclass(frozen=True)
class CudnnFrostBf16Config:        # supported: SM107, SM120
@dataclass(frozen=True)
class CudnnFrostMxfp8Config:       # supported: SM107
@dataclass(frozen=True)
class CudnnFrostNvfp4Config:       # supported: SM107
@dataclass(frozen=True)
class CudnnFrostMxfp8Mxfp4Config:  # supported: SM107
```

**功能**：cuDNN Frost 后端配置，带 packed precomputed routing。每个 config 暴露 ``prepare_weights``
与 ``prepare_activations``（复用对应的 Cutlass 配置实现），可经 `BackendOptions` 显式选择，或在
autotune 中与合格后端一起被接纳；不需要 ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS``。

**注册 key / 支持的配置**

| Config | 激活 / 权重格式 | 原生 weight-view key |
| --- | --- | --- |
| `CudnnFrostBf16Config` | BF16 / BF16 | `cudnn_frost_bf16` |
| `CudnnFrostMxfp8Config` | MXFP8 / MXFP8 | `cudnn_frost_mxfp8` |
| `CudnnFrostNvfp4Config` | NVFP4 / NVFP4 | `cudnn_frost_nvfp4` |
| `CudnnFrostMxfp8Mxfp4Config` | MXFP8 / MXFP4 | `cudnn_frost_mxfp8_mxfp4` |

**注意事项**（转述上游设计说明）

- Frost 也接受对应的规范 ``cutlass_*`` weight view 而不转换；两者都存在时，Frost 命名的 view 优先。
- 后端要求预计算路由（连续 int32 专家 id 与 FP32 路由权重），返回 finalize 后的 BF16 输出；支持
  默认激活契约；bias、专家并行、共享专家、自定义激活参数与从 logits 路由不受支持。量化 hidden /
  intermediate 维度须能被 128 整除。
- SM107 依赖基线：PyTorch 2.14.1、CuTe DSL 4.8.0、TVM-FFI 0.1.14.post1、外部 PTXAS 13.4.92。
  SM107 上当 DSL 捆绑较旧汇编器时设置 ``FLASHINFER_CUDNN_FROST_PTXAS=/path/to/ptxas``；仅设
  ``CUDA_HOME`` 不能替代。外部 PTXAS 与捆绑汇编器都必须报告 CUDA 13.4 或更新。
- 在 CUDA Graph 捕获前完成 prepare 与可选的 autotune。
- 旧的 ``flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm`` 导入是毕业后保留
  一个版本的兼容别名；单独的 ``cudnn_frost_grouped_gemm1_swiglu`` API 仍为实验性。

**示例**

```python
from flashinfer.fused_moe import (
    BackendOptions, CudnnFrostNvfp4Config, ExpertConfig, MoEActivationPack,
    MoEConfig, MoELayer, MoEWeightPack, QuantConfig, QuantFormat, RoutingConfig,
)

backend = CudnnFrostNvfp4Config
config = MoEConfig(
    routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
    experts=ExpertConfig(intermediate_size=intermediate_size),
    quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
    backend=BackendOptions((backend(),)),
)
view = backend.prepare_weights(
    w1_bf16, w2_bf16, num_local_experts=num_experts,
    hidden_size=hidden_size, intermediate_size=intermediate_size,
    activation=config.activation,
)
weights = MoEWeightPack({"cudnn_frost_nvfp4": view})
xq, xsf = backend.prepare_activations(x_bf16, quant=config.quant)
activations = MoEActivationPack(xq, xsf, topk_ids, topk_weights)
layer = MoELayer(config)
output = layer(activations, weights)
```

---

### 工具函数

#### `convert_to_block_layout`

```python
convert_to_block_layout(input_tensor: torch.Tensor, blockK: int) -> torch.Tensor
```

**功能**：把 2-D 张量 reshape 为 3-D block 布局。把内层 ``K`` 维拆成 ``K // blockK`` 个大小
``blockK`` 的块并转置，使块维在最外。这是 TensorRT-LLM block-scaled MoE kernel 消费的规范布局。

**参数**

- `input_tensor`（`torch.Tensor`）：形状 ``(M, K)``。
- `blockK`（`int`）：沿 ``K`` 的块大小；``K`` 必须能被 ``blockK`` 整除。

**返回**：`torch.Tensor`——形状 ``(K // blockK, M, blockK)`` 的连续张量。

#### `reorder_rows_for_gated_act_gemm`

```python
reorder_rows_for_gated_act_gemm(x: torch.Tensor) -> torch.Tensor
```

**功能**：重排权重张量的行以适配 TensorRT-LLM gated-activation GEMM 布局。是 TRT-LLM
``reorderRowsForGatedActGemm`` 的纯 PyTorch 重实现，用于预置换 up/gate 权重矩阵，使融合
gated-activation kernel 能以单次连续载入访问两半。

**参数**：`x`（`torch.Tensor`）——待置换行的权重张量，任意 dtype，仅行维被重排。

**返回**：`torch.Tensor`——行置换后的副本（物化为新的连续张量；PyTorch 高级索引总是拷贝，不别名）。

#### `interleave_moe_weights_for_sm90_mixed_gemm`

```python
interleave_moe_weights_for_sm90_mixed_gemm(weight: torch.Tensor, quant_type: str = "fp4") -> torch.Tensor
```

**功能**：为 SM90 mixed-input GEMM 交错 4-bit packed MoE 权重。``cutlass_fused_moe`` 在
``use_w4_group_scaling=True`` 下使用该布局；不预处理时基于 LUT 的 FP4→BF16 转换会读错字节位置，
任何 ``K > 128`` 都会与反量化参考发散。

**参数**

- `weight`（`torch.Tensor`）：``[num_experts, n, k // 2]`` uint8 CUDA 张量（4-bit 值两两打包）。
- `quant_type`（`str`，默认 `"fp4"`）：``"fp4"``（MXFP4，W4A16）、``"fp4_fp8"``（FP8/Humming
  pre-MMA-scale 路径消费的 MXFP4）或 ``"int4"``（W4A8）。

**返回**：`torch.Tensor`——与 `weight` 同形状的新 uint8 交错布局张量，直接作为
`fc1_expert_weights` / `fc2_expert_weights` 传给 `cutlass_fused_moe`。

#### `interleave_moe_scales_for_sm90_mixed_gemm`

```python
interleave_moe_scales_for_sm90_mixed_gemm(scales: torch.Tensor, group_size: int = 32) -> torch.Tensor
```

**功能**：折叠 SM90 mixed-input MoE GEMM 的权重 scale。

**参数**

- `scales`（`torch.Tensor`）：``[num_experts, rows, K // group_size]`` 标量权重 scale。MXFP4 用
  uint8 E8M0 scale 且 ``group_size=32``；W4A8 用 bf16 bit-pattern scale 且 ``group_size=128``。
- `group_size`（`int`，默认 `32`）：权重量化组大小。

**返回**：连续张量，形状 ``[num_experts, rows // 64, K // 128, folded_m, physical_cols]``。

#### `preprocess_moe_weights_for_sm90_mixed_gemm_humming`

```python
preprocess_moe_weights_for_sm90_mixed_gemm_humming(
    weight: torch.Tensor,
    raw_scale: torch.Tensor,
    max_range: int = 11,
    *,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：为 SM90 Humming 风格 FP8 激活路径准备 MXFP4 权重。

**参数**

- `weight`（`torch.Tensor`）：``[num_experts, rows, K // 2]`` uint8，含 packed MXFP4 payload。
- `raw_scale`（`torch.Tensor`）：``[num_experts, rows, K // 32]`` uint8，含原始 E8M0 MXFP4 权重 scale。
- `max_range`（`int`，默认 `11`）：pre-MMA FP4→E4M3 offset 中保留的 per-expert E8M0 指数范围上限。
- `interleave`（`bool`，默认 `True`）：为 true 时返回可直接用于 `cutlass_fused_moe` 的张量；为
  false 时返回逻辑处理后的 packed 权重与 offset scale（便于验证）。

**返回**：`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`——``(weight_out, scale_out, residual)``。
`residual` 为每个本地专家一个 FP32 值；用于 `cutlass_fused_moe` 时需乘以 Humming 固定的 ``2^6``
补偿，并作为 ``[num_local_experts]`` 张量传入 quant-scale 槽 1（FC1）或 4（FC2）。

**注意事项**：E8M0 范围夹取、residual-scale 分解与 FP4 payload 重写方案改编自 Humming
（``https://github.com/inclusionAI/humming``）。

#### `fused_topk_deepseek`

```python
fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    launch_with_pdl: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    *,
    backend: Literal["default", "cake"] = "default",
) -> None
```

**功能**：DeepSeek-V3 的融合专家路由与 top-k 选择。算法：① 计算 biased scores
``sigmoid(scores) + bias``；② 分组并按组内 top-2 之和计算组分数；③ 选 top ``topk_group`` 组；
④ 从所选组选 top ``topk`` 专家；⑤ 归一化 ``sigmoid_scores / sum(sigmoid_scores) * routed_scaling_factor``。

**参数**

- `scores`（`torch.Tensor`）：``(num_tokens, num_experts)`` 路由 logits，bf16/fp16/fp32，须连续。
- `bias`（`torch.Tensor`）：``(num_experts,)`` per-expert 路由 bias，dtype 与 `scores` 相同。
- `n_group`（`int`）：专家组数，须 ``n_group <= 32`` 且 ``num_experts % n_group == 0``。
- `topk_group`（`int`）：选择的 top 组数，须 ``topk_group <= n_group`` 且 ``topk_group * n_group >= topk``。
- `topk`（`int`）：每 token 选择的专家数，``<= num_experts``，硬上限 ``topk <= 32``，且两个分支都要求
  ``topk <= 8``。``n_group > 1`` 时还要求 ``num_experts / n_group <= 32`` 且
  ``(num_experts / n_group) * topk_group <= 128``；``n_group == 1`` 时 ``num_experts <= 384``。
- `routed_scaling_factor`（`float`）：施加到归一化专家权重的缩放。
- `topk_values`（`torch.Tensor`）：预分配输出 ``(num_tokens, topk)``，dtype 与 `scores` 相同。
- `topk_indices`（`torch.Tensor`）：预分配输出 ``(num_tokens, topk)``，int32。
- `launch_with_pdl`（`bool`，默认 `True`）：是否用 PDL 启动。
- `routing_replay_out`（`Optional[torch.Tensor]`）：预分配 int16 张量记录所选专家 id，须
  ``shape[0] >= num_tokens`` 且 ``shape[1] == topk``；`None` 时跳过该写入。
- `backend`（`Literal["default", "cake"]`，默认 `"default"`）：实现后端；``"cake"`` 显式选择 Cake
  kernel，超出其支持契约时抛错。

**返回**：`None`——结果原地写入 `topk_values` 与 `topk_indices`（以及可选的 `routing_replay_out`）。

**注意事项**：kernel 内部使用 float32 保证数值精度（无论输入 dtype）。支持 Ada（SM89）、Hopper
（SM90）与 Blackwell（SM100/SM103/SM120/SM121）。

#### `hash_topk`

```python
hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    launch_with_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：DeepSeek-V4 的 hash 型 MoE 专家路由。DSv4-Pro hash-MoE 层从预计算 token-to-expert
表（`tid2eid`）选择专家，而非动态 top-k，因此路由是 ``O(1)`` 表查找后跟 ``sqrt(softplus(.))`` 分数
归一化。一个 warp 处理一个 token。选中专家的路由权重为
``sqrt(softplus(router_logits[token, expert])) / sum_over_routed``；``num_fused_shared_experts == 1``
时额外追加一个 id 为 ``num_routed_experts``、权重为 ``1 / routed_scaling_factor`` 的共享专家槽。

**参数**

- `router_logits`（`torch.Tensor`）：``(num_tokens, num_routed_experts)`` float32。
- `input_ids`（`torch.Tensor`）：``(num_tokens,)`` int64 词表 token id，用于索引 `tid2eid`。
- `tid2eid`（`torch.Tensor`）：``(vocab, topk)`` int32 预计算 token-to-expert 表。
- `num_fused_shared_experts`（`int`，默认 `0`）：追加的共享专家数（0 或 1）。
- `routed_scaling_factor`（`float`，默认 `1.0`）：共享专家权重的缩放。
- `launch_with_pdl`（`bool`，默认 `True`）：是否用 PDL 启动（SM90+）。

**返回**

- `topk_weights`（`torch.Tensor`）：``(num_tokens, topk + num_fused_shared_experts)`` float32。
- `topk_ids`（`torch.Tensor`）：``(num_tokens, topk + num_fused_shared_experts)`` int32。

**注意事项**：签名与 SGLang 的 ``sglang.jit_kernel.deepseek_v4.hash_topk`` 匹配，可作 drop-in 替代；
实现 DSv4 tracker 的 ``MOE-01-HASH``。

---

### AlphaMoE Router / FP8 / NVFP4

#### `AlphaMoERoutePlan`

```python
class AlphaMoERoutePlan(NamedTuple)
```

**功能**：设备端路由输出与可复用的私有工作区。前五个字段是 AlphaMoE 计算 kernel 消费的 route-plan
ABI；`expert_counts`、`expert_offsets`、`expert_scatter_offsets` 是实现拥有的工作区，保留在元组中
以便重复调用避免分配。消费者只能使用 ``num_tokens_post_padded[0]`` 所指的 `sorted_token_ids` 前缀
及对应的 `expert_ids` 前缀，不需要在主机读取该设备标量。

#### `allocate_alphamoe_route_plan`

```python
allocate_alphamoe_route_plan(
    logits: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    has_shared_expert: bool = False,
) -> AlphaMoERoutePlan
```

**功能**：为 `logits` 分配可复用的 `AlphaMoERoutePlan`。本分配辅助不启动 router；把结果作为
``plan=`` 传回 `alphamoe_fused_router`。该形式在 JIT 模块于捕获外载入一次后是 CUDA-graph 安全的。

#### `alphamoe_fused_router`

```python
alphamoe_fused_router(
    logits: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    has_shared_expert: bool = False,
    plan: Optional[AlphaMoERoutePlan] = None,
) -> AlphaMoERoutePlan
```

**功能**：路由 FP32 logits 并构建 AlphaMoE 对齐计划（SM100/SM103）。一个 cooperative 生成 kernel
完成 top-k 选择、selected-logit softmax、per-expert 计数、block padding 与分组 route scatter。
``has_shared_expert=True`` 时，最后一个专家被强制进入最后一个 route 槽，其余 ``top_k - 1`` 个专家
从 ``[0, num_experts - 1)`` 中选择。

**参数**

- `logits`（`torch.Tensor`）：连续 FP32 CUDA 张量 ``(num_tokens, num_experts)``；``num_tokens``
  须为正，``num_experts`` 须在 ``[1, 512]``。logits 须有限。
- `top_k`（`int`）：每 token 路由数，在 ``[1, min(num_experts, 16)]``。
- `block_m`（`int`，默认 `8`）：per-expert 对齐，在 ``[1, 16]``。
- `has_shared_expert`（`bool`，默认 `False`）：强制最后一个专家进入每个 token 的最后一个 route 槽；
  需要 ``top_k >= 2``。
- `plan`（`Optional[AlphaMoERoutePlan]`）：可复用输出 / 工作区分配；省略时分配最坏容量计划。

**返回**：`AlphaMoERoutePlan`——`topk_weights` 与 `topk_ids` 形状 ``(num_tokens, top_k)``；
`sorted_token_ids` 按专家分组存 flattened ``token * top_k + route``（以 sentinel
``num_tokens * top_k`` padding）；`expert_ids` 每 ``block_m`` 项存一个专家；
`num_tokens_post_padded` 为设备端 int32 标量命名有效范围。组内原子 scatter 顺序故意未指定。

**注意事项**：当路由 logits 完全相等时，所选专家集合及其顺序未指定（不保证按较低 expert id
打破平局）。冻结的 CUDA 源以 ``--use_fast_math`` 编译。该操作不依赖 AlphaMoE 计算 kernel，其计划
既可喂给 W8A8 也可喂给 NVFP4 融合 up/down 路径。

#### `alphamoe_interleave_gated_weights`

```python
alphamoe_interleave_gated_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：把 gate/up 权重行交错到 alphamoe_sm100 设备布局。`alphamoe_fp8_block_scale_aligned_moe`
消费的 gemm1（gate+up）权重，其行轴交替八行 gate、八行 up，使一个 128 行 TMA box 携带匹配的
gate 与 up 特征。本离线 helper 转换常规 ``[gate; up]`` 堆叠布局。权重载入时运行一次；kernel 绝不能
看到未交错的权重。

**参数**

- `gemm1_weights`（`torch.Tensor`）：``(num_experts, 2 * intermediate_size, hidden_size)`` FP8 权重，
  gate 行在前，按八行块交错。
- `gemm1_weights_scale`（`torch.Tensor`）：``(num_experts, 2 * intermediate_size / 128,
  hidden_size / 128)`` float32 block scale，逐行交错，每 scale 行覆盖 128 权重的行。

**返回**：`(weights, scales)`，保持输入形状、dtype 与设备。

#### `alphamoe_fp8_block_scale_aligned_moe`

```python
alphamoe_fp8_block_scale_aligned_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：预对齐路由计划上的 W8A8 block-scale MoE（SM100/SM103）。运行生成的 Alpha-MoE megakernel：
routed gate/up 投影、带 per-token FP8 requantization 的 SwiGLU、down 投影、路由权重 / 缩放应用，
以及异步 BF16 reduce-add 到 `out`——单 kernel，无中间全局张量。路由计划是 vLLM/SGLang 的
``moe_align_block_size`` 契约。

**参数（要点）**：`hidden_states` ``(num_tokens, hidden_size)`` fp8_e4m3fn（按 token 沿
hidden_size 每 128 一组的量化；最内 stride 为 1，行 stride 正、不重叠、可被 16 整除，数据指针
16 字节对齐）；`hidden_states_scale` ``(num_tokens, hidden_size // 128)`` float32；
`gemm1_weights` ``(num_experts, 2*intermediate_size, hidden_size)`` fp8（**交错的设备布局**，由
`alphamoe_interleave_gated_weights` 产生；``2*intermediate_size`` 须为 256 倍数，``hidden_size``
128 倍数）；`gemm1_weights_scale` ``(num_experts, 2*intermediate_size/128, hidden_size/128)``
float32；`gemm2_weights` ``(num_experts, hidden_size, intermediate_size)`` fp8；
`gemm2_weights_scale` ``(num_experts, hidden_size/128, intermediate_size/128)`` float32；
`sorted_token_ids` int32 计划位置；`expert_ids` int32 每 ``block_m`` 项一个专家；
`num_tokens_post_padded` int32 单元素有效范围；`topk_weights` ``(num_tokens, top_k)`` float32；
`top_k` 每 token 路由数；`block_m` 计划块大小（正的 8 的倍数）；`routed_scaling_factor` float；
`out` ``(num_tokens, hidden_size)`` bfloat16 累加器（省略时分配零张量；提供时 kernel **累加**进它，
调用方负责清零或播种；数据指针须 16 字节对齐）。

**返回**：`torch.Tensor`——``(num_tokens, hidden_size)`` bfloat16 累加器。

**注意事项**：需要 SM100/SM103（B200/B300 类）设备；kernel 是冻结的生成调度，对所有零激活块有
unit-scale 保护。

#### `alphamoe_nvfp4_aligned_moe` / `alphamoe_nvfp4_routed_moe`

```python
alphamoe_nvfp4_aligned_moe(
    hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    output1_scale_gate_scalar, output1_scale_scalar, output2_scale_scalar,
    sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out,
    top_k: int, block_m: int = 8, routed_scaling_factor: float = 1.0,
    w1_scale_prepared=None, w2_scale_prepared=None,
) -> None

alphamoe_nvfp4_routed_moe(
    hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    output1_scale_gate_scalar, output1_scale_scalar, output2_scale_scalar,
    sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out,
    topk_ids, cumsum_buffer,
    top_k: int, block_m: int = 8, routed_scaling_factor: float = 1.0,
    w1_scale_prepared=None, w2_scale_prepared=None, w1_data_prepared=None,
    w1_gate_up_data_prepared=None, w1_gate_up_scale_prepared=None,
    w1_scale_prepared_interleaved=None, w2_data_prepared=None,
    w2_data_prepared_k256=None, w2_scale_prepared_k256=None, accumulate: bool = True,
) -> torch.Tensor
```

**功能**：AlphaMoE NVFP4 计算。aligned 入口消费已有的路由计划；routed 入口先对齐给定专家 id 再计算。
两者都消费 packed E2M1 激活与权重，带线性 per-16 E4M3 scale；形态选择阶段执行 gate/up 投影、SwiGLU、
NVFP4 requantization 与 down 投影。三个连续 FP32 ``[E]`` 张量提供 per-expert 静态 ModelOpt scale：
gate scale 在 SiLU 前应用，up scale 在 SwiGLU 相乘前应用，down scale 在路由加权前应用。

**参数（要点）**

- `hidden_states`：``[M, K/2]`` packed E2M1（每 ``uint8`` 两个 E2M1，偶逻辑值在低半字节）；
  `hidden_states_scale`：线性 E4M3 ``[M, K/16]``。
- `gemm1_weights`：``[E, N, K/2]``（约定 ``[gate; up]`` 行序；``N`` 可被 256 整除）；
  `gemm1_weights_scale`：``[E, N, K/16]``。
- `gemm2_weights`：``[E, K, N/4]``；`gemm2_weights_scale`：``[E, K, N/32]``。
- `output1_scale_gate_scalar` / `output1_scale_scalar` / `output2_scale_scalar`：连续 FP32 per-expert
  ``[E]`` gate / up / down 反量化 scale。
- `sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`：对齐 MoE 计划；`num_tokens_post_padded`
  为单元素设备张量，命名有效计划范围。
- `out`：连续 BF16 ``[M, K]``，**原地更新并返回**；aligned 入口把它当作初始值（累加进 FP32 临时
  缓冲后转换回 BF16），routed 入口保留调用方 `out`。
- `topk_ids` / `cumsum_buffer`（routed）：``[M, top_k]`` int32 路由专家 id 与至少 ``E + 2`` 项的
  int32 路由工作区。
- `w1_scale_prepared` / `w2_scale_prepared` / `w1_data_prepared` / `w1_gate_up_data_prepared` /
  `w1_gate_up_scale_prepared` / `w2_data_prepared` / `w2_data_prepared_k256` /
  `w2_scale_prepared_k256`（可选）：由 ``prepare_nvfp4_*`` 一次性产生的不变面板；不传则保留既有路径。
- `accumulate`（routed，`bool`，默认 `True`）：为 `True` 时把路由贡献加到 `out` 现有值；否则直接
  覆盖（无需清零）。

**返回**：aligned 入口 `None`（原地更新 `out`）；routed 入口返回 ``out``。

**注意事项**（转述上游 NVFP4 说明）

- 逻辑 ``K = 2 * hidden_states.shape[1]``，须至少 256 且可被 256 整除；scale 张量须为
  ``torch.float8_e4m3fn`` 且用上述线性 per-16 布局；FlashInfer 的 128x4-swizzled NVFP4 scale 布局
  是不同契约，不可传入。
- aligned 入口使用 FP32 scratch（``4 * M * K`` 字节），兼容 CUDA Graph 捕获；不重置调用方初始输出。
  FP32 批量归约顺序相关，并按 PTX ``cp.reduce.async.bulk.add.f32`` 的规定把次正规输入 / 结果刷为
  带符号零。
- routed 入口对选中的完整 route 使用私有 per-route 累加器并按 route 顺序 finalize，保留调用方现有
  输出；M=8/128/512 等特定形态在提供对应 prepared data 时选择 prepared-data 路径。相邻 gate/up M512
  route 需同时提供 ``w1_gate_up_data_prepared`` 与 ``w1_gate_up_scale_prepared``（两者取其一抛
  `ValueError`）。本调用内不发生任何 weight 准备。

#### NVFP4 准备辅助

```python
prepare_nvfp4_w1_scales(w1_scale) -> torch.Tensor          # [E*(N/128)*(K/256),16,128] uint8
prepare_nvfp4_w1_data(w1) -> torch.Tensor                  # [E*(N/128)*(K/256),128,128] uint8
prepare_nvfp4_w2_data(w2) -> torch.Tensor                  # [E*(K/128)*(N/256),128,64] uint8
prepare_nvfp4_w2_data_k256(w2) -> torch.Tensor             # [E*(K/128)*(N/512),128,128] uint8
prepare_nvfp4_w2_scales(w2_scale) -> torch.Tensor          # [E*(K/128)*B,8,128] uint8, B=N/256
prepare_nvfp4_w2_scales_k256(w2_scale) -> torch.Tensor    # [E*(K/128)*(N/512),16,128] uint8
prepare_nvfp4_w1_gate_up_data(w1) -> torch.Tensor         # [E*(N/256)*(K/256),256,128] uint8
prepare_nvfp4_w1_gate_up_scales(prepared_scales, original_weight_shape) -> torch.Tensor  # [R,32,128]
```

**功能**：在模型载入时一次性把 NVFP4 权重 / scale 字节重排为不变的面板（纯字节置换，无反量化 /
requantization）。除 ``prepare_nvfp4_w2_scales_k256`` / ``prepare_nvfp4_w1_gate_up_scales`` 外，输入
均为连续的原始 ``uint8`` 张量（``w1`` ``[E,N,K/2]``、``w2`` ``[E,K,N/4]``、``w1_scale``
``[E,N,K/16]``、``w2_scale`` ``[E,K,N/32]``）。

**注意事项**

- ``prepare_nvfp4_w1_scales`` / ``prepare_nvfp4_w2_scales`` 只置换现有 scale 字节为不可变 uint8
  tensor-map 面板，不做 scale 算术；在最终 device-local 权重布局确定之后、warmup 或 graph capture
  之前调用；保持原始 scale 张量与结果面板存活。
- ``prepare_nvfp4_w1_data`` 要求 ``N % 128 == 0`` 且 ``K % 256 == 0``；``prepare_nvfp4_w1_gate_up_data``
  / ``..._scales`` 要求 ``N % 256 == 0`` 且 ``K % 256 == 0``，且 scale helper 消费的是
  ``prepare_nvfp4_w1_scales`` 的输出（而非原始 E4M3 scale）。
- 所有准备时间与内存都独立于重复请求的 kernel 计时之外；派生面板无需保存进模型 checkpoint。

---

### Multi-LoRA MoE（BGMV）

#### `bgmv_moe`

```python
bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    output_dim: Optional[int] = None,
) -> torch.Tensor
```

**功能**：高层次多 LoRA MoE BGMV：一次调用完成 shrink + expand。计算
``delta[token] = Σ_expert (topk_weight * x[token] @ lora_a[expert, lora_id] @ lora_b[expert, lora_id])``。

**参数**

- `x`（`torch.Tensor`）：``[num_tokens, hidden_dim]`` 输入激活。
- `lora_a_weights`（`List[torch.Tensor]`）：每个 slice 一个 LoRA-A，形状
  ``[max_loras, num_experts, rank, hidden_dim]``。
- `lora_b_weights`（`List[torch.Tensor]`）：每个 slice 一个 LoRA-B，形状
  ``[max_loras, num_experts, feat_out, rank]``。
- `sorted_token_ids`（`torch.Tensor`）：``[num_pairs]`` 路由 token 索引。
- `expert_ids`（`torch.Tensor`）：每个 routed pair 的专家索引。
- `lora_indices`（`torch.Tensor`）：每个输入 token 的 LoRA 索引。
- `topk_weights`（`torch.Tensor`）：每个 routed pair 的 FP32 路由权重。
- `num_experts`（`int`）：LoRA 张量中的专家数。
- `output_dim`（`Optional[int]`）：总输出维度；`None` 时由 `lora_b_weights` 推断。

**返回**：`torch.Tensor`——``[num_tokens, total_feat_out]``，含 LoRA delta。

#### `prepare_bgmv_moe`

```python
prepare_bgmv_moe(
    x, lora_a_weights, lora_b_weights, sorted_token_ids, expert_ids, lora_indices,
    topk_weights, num_experts,
    *, backend: Literal["cake"] = "cake", fallback: bool = True,
    shrink_out=None, y_accum=None, grouped=None, order_remap=None,
) -> BGMVMoEPlan
```

**功能**：准备一个可图重放的 BGMV MoE shrink+expand 流水线。生成的 Cake 路径支持单个 LoRA slice、
rank 8/16/32/64、任意正向 8 倍数的 hidden size、BF16/FP16 输入，以及精确的 SM90（H100/H200）、
SM100（B200/GB200）、SM103（B300/GB300）或 SM107（R200）设备。hidden 2688 / 3072 在 rank 32 下
（至多 2048 token）使用特化实测 body（``plan.variant == "specialized"``）；其余（含这些形状在更大
token 数下）使用运行时 hidden 的通用 bundle（``plan.variant == "generic"``）。

**参数（要点）**

- `backend`（`Literal["cake"]`，默认 `"cake"`）：仅支持 ``"cake"``。
- `fallback`（`bool`，默认 `True`）：对不受支持的输入用 portable 路径代替抛错。
- `grouped`（可选，generic 变体专属）：`None` 时由计划在 route 明显多于 ``num_loras * num_experts``
  pair 时选择 pair-grouped 流水线。
- `order_remap`（可选，generic per-route Cake 流水线专属）：仅在 SM90 上 dispatch shrink 的
  (LoRA, expert)-bin 序。
- `shrink_out` / `y_accum`（可选）：pointer-stable 的 shrink 工作区 / FP32 输出累加器。

**返回**：可复用的图后端执行计划，其 ``run`` 返回 FP32 累加输出；``plan.backend_used`` 为 `"cake"`
或 `"portable"`。

**注意事项**：路由可任意；每个输出有一个 owner，按固定输入顺序累加 route，故相同 prepared 重放按位
可复现。不受支持的输入在 ``fallback=True`` 时由 `BGMVMoEPortablePlan`（portable
`bgmv_moe_shrink` + `bgmv_moe_expand`）服务，每个不同原因每进程发一次 `RuntimeWarning`；
``fallback=False`` 时抛 `ValueError`。非法输入总是抛错。

#### `BGMVMoECakePlan` / `BGMVMoEPortablePlan`

- `BGMVMoECakePlan`：pointer-stable 的 SM90/SM100/SM103/SM107 Cake BGMV MoE shrink+expand 执行
  计划。无输出原子操作、每个输出 token 一个 owner、重放按位可复现。``backend_used == "cake"``；
  ``variant`` 为 ``"specialized"`` 或 ``"generic"``。
- `BGMVMoEPortablePlan`：运行 portable `bgmv_moe_shrink` + `bgmv_moe_expand` 路径的已准备计划。
  接口同 `BGMVMoECakePlan`（``run`` 清零工作区、运行 shrink 与 expand、返回 FP32 累加器；首次 eager
  调用捕获 CUDA Graph）。portable expand 用原子累加，故重放不保证按位一致；
  ``backend_used == "portable"`` 且 ``schedule_id is None``。

#### `bgmv_moe_shrink` / `bgmv_moe_expand`

```python
bgmv_moe_shrink(
    y, x, w_ptr, sorted_token_ids, expert_ids, lora_indices, lora_stride, *,
    per_pair_input: bool = False,
) -> None

bgmv_moe_expand(
    y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights, lora_indices,
    slice_start_loc, output_slices, lora_stride, *, finalize: bool = True,
) -> None
```

**功能**：底层 shrink（经 LoRA-A 投影输入）与 expand（经 LoRA-B 投影）操作。shrink 对每个
(token, expert) pair 计算 ``y[slice, pair, rank] += x[token] @ lora_a[expert, lora_id, :, :]``。
expand 在 ``finalize=True``（默认）时对每个 pair 计算路由加权 combine 到 per-token 行：
``y[token, col:col+feat] += topk_weight * (x[slice, pair, :] @ lora_b[expert, lora_id])``；
``finalize=False``（FC1 LoRA delta）时写 per-pair、未加权结果（纯 store，无 topk_weight、无跨专家
combine），跳过的 pair 会 early-return，故 `y` 必须由调用方 zero-initialize。

**参数（要点）**：`y` 输出缓冲区（finalize 时 ``[num_tokens, total_feat_out]``，否则
``[num_pairs, total_feat_out]``，float32）；`x` 输入（shrink 时 ``[num_tokens, hidden_dim]``，
expand 时 shrink 输出 ``[num_slices, num_pairs, rank]``）；`w_ptr` ``[num_slices, num_experts]``
int64 指针表；`sorted_token_ids` / `expert_ids` 每个 pair 的 token / 专家索引；`lora_indices`
每个 token 的适配器 id（``-1`` 表示无 LoRA，pair 被跳过）；`lora_stride` 权重张量中相邻适配器之间
的元素 stride；`per_pair_input`（shrink，默认 `False`）为 `True` 时输入行是 pair 本身（FC2）而非
token（FC1）；`topk_weights`（expand）每个 pair 的 float32 路由权重（finalize=False 时忽略但仍须为
合法 ``[num_pairs]`` 张量）；`slice_start_loc`（expand）每个 slice 的列偏移（int64）；
`output_slices` 每个 slice 的输出特征维度。

#### `bgmv_moe_gemm1_lora_delta` / `bgmv_moe_gemm2_lora_delta`

```python
bgmv_moe_gemm1_lora_delta(
    hidden_states, w_ptr_a, lora_stride_a, w_ptr_b, lora_stride_b, topk_ids,
    lora_ids, rank, intermediate_size, *, lora_dtype=torch.bfloat16, scale=1.0,
    out_dtype=torch.bfloat16,
) -> torch.Tensor

bgmv_moe_gemm2_lora_delta(
    gemm1_activation_output, expanded_idx_to_permuted_idx, w_ptr_a, lora_stride_a,
    w_ptr_b, lora_stride_b, topk_ids, topk_weights, lora_ids, rank, hidden_size,
    *, lora_dtype=torch.bfloat16, scale=1.0, out_dtype=torch.bfloat16,
) -> torch.Tensor
```

**功能**：FC1 / FC2 的 LoRA delta，布局与 ``trtllm_*_moe`` 的 ``gemm1_lora_delta`` /
``gemm2_lora_delta`` 一致。gemm1 delta 在非线性 SwiGLU 之前加法，因此 unweighted 且保留
per-(token, slot)（不跨专家求和、不按路由权重缩放），返回 ``[T, top_k, 2*I]``。gemm2 delta 在 FC2
之后、combine 之后加法，因此 weighted 且跨专家 combine，返回 ``[T, H]``，需加到 MoE 输出上。

**参数（要点）**：`w_ptr_a` / `w_ptr_b` 为 `fill_w_ptr` 产生的 int64 base-pointer 表；``[2,
num_experts]`` 表示常规（gate/up 两 slice），1D ``[num_experts]`` 选择共享-A 的 performant LoRA；
`lora_stride_a` / `lora_stride_b` 为适配器间的元素 stride；`lora_ids` ``[T]`` 为 token 的适配器 id
（``-1`` 无适配器）；`rank` 为 LoRA rank；`scale` 为 ``alpha / rank``。

**返回**：gemm1 返回 ``[T, top_k, 2*I]``；gemm2 返回 ``[T, H]``，dtype 均为 `out_dtype`。

---

### 传统 TRT-LLM / CUTLASS Fused MoE

#### `cutlass_fused_moe`

```python
cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1, tp_rank: int = 0, ep_size: int = 1, ep_rank: int = 0,
    cluster_size: int = 1, cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_packed_weights: bool = False,
    use_wfp4afp8_humming: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    swizzled_input_sf: bool = True,
    use_fused_finalize: bool = True,
    profile_ids: Optional[List[int]] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    *,
    clamped_relu2_limit: Optional[torch.Tensor] = None,
    situ_beta: Optional[torch.Tensor] = None,
    situ_linear_beta: Optional[torch.Tensor] = None,
    backend: str = "cutlass",
) -> torch.Tensor
```

**功能**：使用 CUTLASS 后端计算 MoE 层。把专家选择、专家计算与输出组合融合为单一操作，支持多种
数据类型与并行策略。

**参数（要点）**

- `input`：``[num_tokens, hidden_size]``，支持 float / float16 / bfloat16 / float8_e4m3fn / nvfp4；
  FP8 输入须已量化，NVFP4 支持量化与未量化输入。
- `token_selected_experts` / `token_final_scales`：每个 token 选择的专家索引 / 各 token 专家输出的
  缩放因子。
- `fc1_expert_weights` / `fc2_expert_weights`：GEMM1 / GEMM2 每专家权重。
- `output_dtype`：期望输出 dtype。
- `quant_scales`（`List[torch.Tensor]`）：量化 scale。NVFP4 顺序为 gemm1 激活全局 scale、gemm1
  权重 block scale、gemm1 dequant scale、gemm2 激活全局 scale、gemm2 权重 block scale、gemm2
  dequant scale；FP8 顺序为 gemm1 dequant scale、gemm2 激活 quant scale、gemm2 dequant scale、
  gemm1 输入 dequant scale；Humming FP8×MXFP4 为 gemm1 折叠权重 block scale、gemm1 per-local-expert
  residual scale（含固定 ``2^6`` 补偿）、reserved scalar 或 per-local-expert gemm2 激活 scale、
  gemm2 折叠权重 block scale、gemm2 per-local-expert residual scale。
- `input_sf`：输入的量化 scale factor。
- `swiglu_alpha` / `swiglu_beta` / `swiglu_limit`：SwiGLU 激活的 alpha / beta / per-expert limit。
- `tp_size` / `tp_rank` / `ep_size` / `ep_rank` / `cluster_size` / `cluster_rank`：并行配置。
- `enable_alltoall`（默认 `False`）：是否启用专家输出的 all-to-all 通信。
- `use_deepseek_fp8_block_scale` / `use_w4_group_scaling` / `use_mxfp8_act_scaling` /
  `use_packed_weights` / `use_wfp4afp8_humming`：量化 / 打包路径选择。
- `min_latency_mode`：最小延迟模式（当前未实现，请求时抛 `NotImplementedError`）。
- `enable_pdl`（`Optional[bool]`）：PDL；`None` 由运行时按设备与相邻 stream 操作选安全值。
- `activation_type`（`ActivationType`，默认 `Swiglu`）：GEMM1 后激活；`Relu2` 与 `ClampedRelu2` 用
  非 gated GEMM1；SM10x 上 `ClampedRelu2` 把 ``(limit * tanh(relu(x) / limit))**2`` 融合进
  BF16xBF16 或 MXFP8xMXFP8 GEMM1 epilogue。
- `swizzled_input_sf`（默认 `True`）：`input_sf` 是否 swizzled 布局；FP4 allgather/alltoall 后为
  linear 时置 `False`。
- `use_fused_finalize`（默认 `True`）：是否把 top-k expert reduction（finalize）融合进 GEMM2
  epilogue（使用非结合原子操作，运行间不确定；置 `False` 使用非融合确定性路径）。
- `profile_ids`：``[gemm1_profile, gemm2_profile]`` 覆盖，绝对索引；``-1`` 保持默认 tactic。
- `workspace_buffer`：预分配 scratch，跨调用复用（大 batch 下可达 10–20 GiB）；详见下。
- `clamped_relu2_limit`（keyword-only）：ClampedRelu2 的模型级正 clamp scale（单元素 float32 CUDA）。
- `situ_beta` / `situ_linear_beta`（keyword-only）：Situ 激活的 per-expert ``beta`` / ``linear_beta``
  （float32 ``[num_experts_on_rank]``）；`None` 用 `DEFAULT_SITU_BETA` / `DEFAULT_SITU_LINEAR_BETA`。
- `backend`（`str`，默认 `"cutlass"`）：``"cutlass"`` 保持现有后端；``"cake"`` 选择 prepared
  TP8-local SiTU 完整调用（要求 output 与 workspace）。

**返回**：`out`（`torch.Tensor`）——``[seq_len, hidden_size]``。

**注意事项**

- 支持 FP32 / FP16 / BF16 / FP8 / NVFP4；实现张量并行与专家并行；Blackwell 上部分高级特性（FP8
  block scaling、minimum latency mode）未实现。
- `workspace_buffer` 用法：先用 `cutlass_fused_moe_workspace_size` 查询所需字节，一次性分配 1-D
  ``torch.uint8`` / ``torch.int8``（与 input 同 CUDA 设备、128 字节对齐），之后每次前向传入。每个
  CUDA stream 上下文一个 buffer；为最大 token 数分配的 buffer 对所有更小 token 数有效。

#### `cutlass_fused_moe_workspace_size`

```python
cutlass_fused_moe_workspace_size(
    max_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts_total: int,
    top_k: int,
    *,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype = torch.bfloat16,
    activation_type: ActivationType = ActivationType.Swiglu,
    tp_size: int = 1, tp_rank: int = 0, ep_size: int = 1, ep_rank: int = 0,
    min_latency_mode: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    use_fused_finalize: bool = True,
    use_packed_weights: bool = False,
    use_wfp4afp8_humming: bool = False,
    device: Optional[torch.device] = None,
    backend: str = "cutlass",
) -> int
```

**功能**：返回 `cutlass_fused_moe` 所需的工作区字节数。在模型载入时一次性分配该字节数（1-D
``torch.uint8``），并作为 ``workspace_buffer=`` 传入每次调用，以消除每次调用的多 GiB scratch 分配。

**参数（要点）**：`max_num_tokens` 为运行期将出现的最大 ``input.shape[0]``（buffer 单调按此值定
大小）；`intermediate_size` 为逻辑、未打包的中间维度（打包权重时为 ``fc2_expert_weights.shape[2]``
乘格式打包因子）；`num_experts_total` 为跨所有 EP rank 的全局专家数，须等于
``fc2_expert_weights.shape[0] * ep_size``；`x_dtype` / `weight_dtype` 用于选择 kernel runner；
`device` 默认当前 CUDA 设备；`backend="cake"` 返回 prepared SiTU scratch 大小（仅用 host 元数据，
要求 BF16/uint8、TP8、SiTU 与文档化的固定几何）。

**注意事项**：每个 CUDA stream 上下文分配一个 buffer；不同 stream 上重叠的 micro-batch 各自需要
自己的 buffer。

#### TRT-LLM MoE 函数族

以下 12 个函数共享几乎所有参数与返回契约，分 logits（内建路由）与 routed（预计算路由）两类：

```python
trtllm_bf16_moe(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights,
    num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
    local_num_experts, routed_scaling_factor=None, routing_method_type=0,
    use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK, do_finalize=True,
    enable_pdl=True, tune_max_num_tokens=8192, activation_type=ActivationType.Swiglu.value,
    norm_topk_prob=True, routing_replay_out=None, gemm1_alpha=None, gemm1_beta=None,
    gemm1_clamp_limit=None, output=None) -> Union[List[torch.Tensor], torch.Tensor]

trtllm_bf16_routed_moe(topk_ids, hidden_states, gemm1_weights, gemm2_weights, num_experts,
    top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts,
    routed_scaling_factor=None, routing_method_type=0, use_shuffled_weight=True,
    weight_layout=WeightLayout.BlockMajorK, do_finalize=True, enable_pdl=True,
    gemm1_lora_delta=None, tune_max_num_tokens=8192, activation_type=ActivationType.Swiglu.value,
    routing_replay_out=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
    output=None) -> Union[torch.Tensor, List[torch.Tensor]]

trtllm_fp8_per_tensor_scale_moe(...)          # 见文件内签名要点
trtllm_fp8_per_tensor_scale_routed_moe(...)
trtllm_fp8_per_channel_scale_moe(...)         # logits 版；无 routed_ 变体签名差异
trtllm_fp8_per_channel_scale_routed_moe(...)
trtllm_fp8_block_scale_moe(...)               # 支持 DeepSeekFp8 / MxFp8 / BlockMajorK
trtllm_fp8_block_scale_routed_moe(...)
trtllm_fp4_block_scale_moe(...)               # NVFP4/MXFP8/BF16，含 SiTU (activation_type=10)
trtllm_fp4_block_scale_routed_moe(...)
trtllm_mxint4_block_scale_moe(...)            # MXINT4，bfloat16 hidden states
trtllm_mxint4_block_scale_routed_moe(...)
```

**功能**：TensorRT-LLM 的融合 MoE 算子族。BF16 版实现 bfloat16 MoE 并使用 TRT-LLM 后端自动调优；
`*_routed_*` 版消费预计算路由（packed ``(expert_id << 16) | weight`` 单 int32 张量，或
``(topk_ids, topk_weights)`` 对），适合 CUDA Graph 捕获或分布式 MoE。FP8 / FP4 / MXINT4 版分别
消费对应量化权与激活。

**共享参数**（要点）

- `routing_logits`：``[seq_len, num_experts]``，float32 或 bfloat16（logits 版）。
- `routing_bias`：``[num_experts]``，bfloat16 或 float32，可为 `None`。
- `hidden_states`：形状 / dtype 依量化类型：BF16 版须 bfloat16；FP8 版逐块 scale 对应；FP4 版为
  ``[seq_len, hidden_size // 2]``（NVFP4）或 ``[seq_len, hidden_size]``（MXFP8 / bfloat16）；
  MXINT4 版为 bfloat16 ``[seq_len, hidden_size]``。
- `topk_ids`（routed）：``[seq_len, top_k]`` int32 packed 格式，或 ``(topk_ids, topk_weights)`` 对。
- `routing_method_type`（`int`，默认 `0`）：见 `RoutingMethodType`（0 Default、1 Renormalize、
  2 DeepSeekV3、3 Llama4、4 RenormalizeNaive、5 TopK、6 SigmoidRenorm、7 MiniMax2、8 Sigmoid、
  9 TopKSigmoid、10 Unspecified）。
- `use_shuffled_weight` / `weight_layout`：权重布局选择；BF16 版要求 `BlockMajorK`，其它函数依
  `fp8_quantization_type` 支持 `MajorK` / `BlockMajorK`。
- `do_finalize`（默认 `True`）：是否 finalize 输出。
- `enable_pdl`：BF16 版默认 `True`，其余 `Optional[bool]`（`None` 在 SM90+ 自动选择）。
- `activation_type`：BF16 版 ``3`` Swiglu / ``6`` Relu2 / ``7`` SwigluStep；FP4 版额外 ``10`` SiTU
  计算 ``beta*tanh(x0/beta) * alpha*tanh(x1/alpha)*sigmoid(x1)``；FP8 per-tensor 版额外
  ``0`` Gelu / ``4`` Geglu / ``9`` Identity。
- `gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit`：per-expert SwiGLU OA 参数 / clamp limit。
- `output`：可选原地 ``[seq_len, hidden_size]`` 输出。

**返回**：`do_finalize=True` 时返回最终 MoE 输出（BF16 版为标量返回，v0.8.0 将变为 ``[output]``）；
否则返回 ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``。带 `gemm1_lora_delta` 时
额外附加 post-activation FC1 输出。

**共有注意事项**

- `hidden_states_scale` 若提供，其布局经 `hidden_states_scale_layout`（`SfLayout`）声明；trtllm-gen
  routed GEMM **只**能消费 ``layout_linear``，声明任何 swizzled 布局抛 `NotImplementedError`。
  省略该参数已弃用（推断为 linear 并发出 `DeprecationWarning`）。
- `valid_hidden_size` / `valid_intermediate_size`（若存在）：把张量形状隐含的维度视为 padded，仅
  收缩有效区域；匹配 TensorRT-LLM 时返回张量的宽度恰为 ``valid_hidden_size``。
- `num_fused_shared_experts > 0`（部分函数）时，所有 per-expert 张量须有
  ``num_experts + num_fused_shared_experts`` 行；每个 token 无条件以权重 1.0 路由到共享专家；目前
  与专家并行不共存。
- FP4 版的 `routing_replay_out`：可选 int16 ``(num_tokens_or_larger, top_k)``，记录所选专家 id；
  用于 CUDA Graph 预分配时可大于 `num_tokens`。

---

### Prims-TS Fused MoE

```python
prims_ts_bf16_moe(...)                    # 同 trtllm_bf16_moe，权重布局默认 WeightLayout.MajorK
prims_ts_bf16_routed_moe(...)             # 同 trtllm_bf16_routed_moe
prims_ts_fp4_block_scale_moe(...)         # 同 trtllm_fp4_block_scale_moe，含 routing_input_mode/topk_ids
prims_ts_fp4_block_scale_routed_moe(...)  # 同 trtllm_fp4_block_scale_routed_moe
prims_ts_fp8_block_scale_moe(...)         # 同 trtllm_fp8_block_scale_moe
prims_ts_fp8_block_scale_routed_moe(...)  # 同上，另加 keyword-only FC1/FC2 bias
prims_ts_fp8_per_tensor_scale_moe(...)    # 同 trtllm_fp8_per_tensor_scale_moe
```

**功能**：实验性 Blackwell（SM100）Prims-TS 后端。公开入口与对应的 ``trtllm_*`` API 参数与返回值
一致。符号定义在 backend 模块并从 `flashinfer.fused_moe` 重导出：

- `flashinfer.fused_moe.backends.prims_ts.bf16_op.prims_ts_bf16_moe`
- `...bf16_op.prims_ts_bf16_routed_moe`
- `...fp4_op.prims_ts_fp4_block_scale_moe`
- `...fp4_op.prims_ts_fp4_block_scale_routed_moe`
- `...fp8_op.prims_ts_fp8_block_scale_moe`
- `...fp8_op.prims_ts_fp8_block_scale_routed_moe`
- `...fp8_op.prims_ts_fp8_per_tensor_scale_moe`

**注意事项**：为使普通 ``import flashinfer`` 不导入 CUTLASS Task Scheduling，这些传统扁平参数 API
是惰性导入的（经模块 ``__getattr__`` 首次使用时加载）。

### `PrimsTsConfig` / `PrimsTsRunner`

```python
@dataclass(frozen=True)
class PrimsTsConfig:                 # supported: SM100, SM103（opt-in，不在默认后端列表）
    def supported(cls, arch: int) -> bool: ...
    def prepare_weights(w1_bf16, w2_bf16, *, quant: QuantConfig, num_local_experts: int,
        hidden_size: int, intermediate_size: int, activation=None, device=None,
        permute_cache=None, hidden_states_scale_global=None): ...
    def prepare_activations(hidden_states_bf16, *, quant, hidden_states_scale_global=None): ...

class PrimsTsRunner(_TrtllmRunnerBase): ...
```

**功能**：SM100/SM103 的显式统一 MoE 后端。覆盖 NVFP4×NVFP4、MXFP4×MXFP8、MXFP4×BF16、
BF16×BF16、FP8PerTensor×FP8PerTensor、DeepSeekFp8×DeepSeekFp8 与 MXFP8×MXFP8。路由与 finalize 留在
TRT-LLM Gen 路径；只有 GEMM 中间阶段使用 Prims-TS。

**注意事项**：NVFP4 / MXFP4 复用 `TrtllmFp4Config`，BF16 复用 `TrtllmBf16Config`，MXFP8 复用
`TrtllmFp8BlockConfig`，per-tensor FP8 复用 `TrtllmFp8PerTensorConfig`。DeepSeekFp8 激活与 scale
匹配 `TrtllmFp8BlockConfig`，但权重 payload 以 epilogue tile 64 shuffle，**不得**注册到
``"trtllm_fp8_block"`` 下；同一共享 view 可为 ``"prims_ts"`` 与 ``"trtllm_fp4_routed"`` 两键注册。
``intermediate_size`` 须为 128 的倍数；融合共享专家与 LoRA 不在范围内。

---

### Standalone TRT-LLM Gen Routing

#### `trtllm_gen_routing`

```python
trtllm_gen_routing(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    routing_method: RoutingMethodType,
    top_k: int,
    *,
    num_fused_shared_experts: int = 0,
    n_group: int = 0,
    topk_group: int = 0,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
    tile_tokens_dim: int = 8,
    norm_topk_prob: bool = True,
    enable_pdl: Optional[bool] = None,
) -> TrtllmGenRoutingResult
```

**功能**：独立的 trtllm-gen MoE 路由（专家选择 + 置换）。运行 trtllm-gen 融合 MoE launcher 在其
GEMM 之前执行的同一路由 kernel（``Routing::Runner::run``），直接返回路由输出而非喂给 GEMM，使路由
阶段可脱离量化 / GEMM 轴单独单元测试。

**参数**

- `routing_logits`：``(num_tokens, num_experts)``，float32 或 bfloat16。
- `routing_bias`：``(num_experts,)``，float32 或 bfloat16（DeepSeekV3 / MiniMax2 风格方法使用）。
- `routing_method`（`RoutingMethodType`）：路由方法。
- `top_k`（`int`）：每 token 选择数。
- `num_fused_shared_experts` / `n_group` / `topk_group` / `local_expert_offset` /
  `local_num_experts` / `routed_scaling_factor` / `tile_tokens_dim` / `norm_topk_prob` /
  `enable_pdl`（keyword-only）：与融合 MoE 路由阶段一致的配置。

**返回**：`TrtllmGenRoutingResult`——见下。

**注意事项**：路由 kernel 在 from-logits 模式下从不直接输出 expert id（专家身份存在于置换布局中），
``topk_ids`` 因此在 torch 侧重构。

#### `TrtllmGenRoutingResult`

```python
class TrtllmGenRoutingResult(NamedTuple)
```

**功能**：trtllm-gen MoE 路由阶段的输出。`topk_ids` / `topk_weights` 为每 token 的专家选择；其余
字段为融合 MoE kernel 消费的置换 / bookkeeping 张量（超过实际 padded 计数
``total_num_padded_tokens`` 的项未定义）。在该计数内，未使用的
``permuted_idx_to_token_idx`` 项为 ``-1``，使下游 TMA gather 不为 padding 行载入激活。

---

### CuteDSL Fused MoE

```python
cute_dsl_fused_moe(
    x, x_sf, token_selected_experts, token_final_scales,
    w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
    w2_weight, w2_weight_sf, w2_alpha,
    num_experts, top_k, num_local_experts=None, local_expert_offset=0,
    output_dtype=torch.bfloat16, use_fused_finalize=True, moe_output=None,
    aux_stream=None, enable_pdl=True, activation_type=ActivationType.Swiglu.value,
    swiglu_alpha=DEFAULT_SWIGLU_ALPHA, swiglu_beta=DEFAULT_SWIGLU_BETA,
    swiglu_limit=DEFAULT_SWIGLU_LIMIT, situ_beta=None, situ_linear_beta=None,
    *, quant_mode="w4a4", per_token_scale=None, tactic=None,
    localized_weights=None, localized_streams=None, localized_sm_count=None,
    localized_memset_stream=None, localized_allow_nonlocalized=False,
    autotune_use_cuda_graph=False,
) -> torch.Tensor

cute_dsl_fused_moe_bf16(...)      # SM90（Hopper），bf16/fp16 未量化
cute_dsl_fused_moe_nvfp4(...)     # 已弃用，改用 cute_dsl_fused_moe(quant_mode="w4a4")
cute_dsl_fused_moe_mxfp8_mxfp4(...)  # 已弃用，改用 cute_dsl_fused_moe(quant_mode="w4a8")
b12x_fused_moe(...)               # SM120/SM121 b12x CuTe-DSL
```

**功能**：使用 CuTe-DSL block-scaled kernel 的融合 MoE 前向。`cute_dsl_fused_moe` 是简单函数式
API（CUDA Graph 支持请用 `CuteDslMoEWrapper`）；支持 SM100 / SM103 / SM107，W4A8 仅限 SM100 / SM103。
`x` 为 ``quant_mode="w4a4"`` 的 packed NVFP4、``"w4a8"`` 的 MXFP8、``"w4a16"`` 的 BF16。

**注意事项**：CuteDSL 后端在安装 ``nvidia-cutlass-dsl`` 包时才有条件可用。自动调优由
``autotune`` 上下文管理器控制：``with autotune(True): output = cute_dsl_fused_moe(...)``。NVFP4 /
MXFP8×MXFP4 入口点将来会弃用，请改用带对应 ``quant_mode`` 的 `cute_dsl_fused_moe`。
`b12x_fused_moe` 自动按 routed 行数选择 micro（decode）/ static / dynamic 后端。

#### 包装类

- `CuteDslMoEWrapper`：CuteDSL MoE 包装类，支持 CUDA Graph 与自动调优。``use_cuda_graph=True`` 时
  在捕获外创建持久 CUDA stream 与 event 资源，使捕获与重放期间可做 async-memset / GEMM1 overlap。
  支持 SM100 / SM103 / SM107（W4A8 限 SM100 / SM103）。属性包括 `num_experts`、`top_k`、
  `hidden_size`、`intermediate_size`、`use_cuda_graph`。
- `CuteDslBf16MoEWrapper`：SM90 CuTe-DSL 融合 MoE 包装（bf16/fp16，未量化）。持有静态 MoE 配置，
  ``run`` 可被 CUDA Graph 捕获，委托给 `cute_dsl_fused_moe_bf16`。
- `CuteDslMxfp8Mxfp4MoEWrapper`（继承 `CuteDslMoEWrapper`）：MXFP8 × MXFP4 流水线生产包装，
  已弃用（改用 `CuteDslMoEWrapper(quant_mode="w4a8")`）。由于复用 stream / event 资源，一个实例不可
  重入或并发安全；首个 ``run`` 把实例绑定到该次调用的 CUDA stream，每个 stream 创建一个包装。
- `B12xMoEWrapper`：SM120/SM121 b12x 融合 MoE 包装，支持 CUDA Graph（预分配工作区缓冲），每次
  调用自动选择 micro/static/dynamic 后端。构造参数包括 `num_experts`、`top_k`、`hidden_size`、
  `intermediate_size`、`use_cuda_graph`、`max_num_tokens`、`num_local_experts`（EP）、`expert_map`。

---

### MonoMoE（单 kernel block-FP8，SM90a）

```python
has_monomoe() -> bool
get_scratchpad_size_bytes() -> int
alloc_scratchpad(device: torch.device) -> torch.Tensor
interleave_for_tma_wgmma_up(w_fp8: torch.Tensor) -> torch.Tensor
mono_moe(
    activations_in, router_logits, expert_weights_up, expert_scales_up,
    expert_weights_down, expert_scales_down, top_k, scoring_func="softmax",
    renormalize=True, out=None, scratchpad=None, interleave_up=True,
) -> torch.Tensor
```

**功能**：针对 Hopper（SM90a）上 Qwen3.5-35B block-FP8 形态的单 kernel top-K MoE。整条流水线
（路由、up 投影、SiLU、down 投影与归约）在一次 kernel 启动内完成。调用前用 `has_monomoe` 检查可用性。

**参数/函数说明**

- `has_monomoe()`：返回 monomoe CUDA 扩展能否构建并加载。
- `get_scratchpad_size_bytes()`：返回 kernel 所需全局 scratchpad 字节数（取自 C++ 的
  ``sizeof(MoEGemmSpec<Dims>)``，永远不会与 kernel 结构布局失配）。
- `alloc_scratchpad(device)`：在 `device` 上分配零初始化 scratchpad（1-D ``uint8``，大小为
  `get_scratchpad_size_bytes`）。零填充建立 kernel 的交接不变量（0.0f 激活 scale sentinel、launch
  parity 计数器与就绪标志）；之后 kernel 自维护，因此分配一次并复用同一张量。
- `interleave_for_tma_wgmma_up(w_fp8)`：为 Pair_Layout WGMMA A-tile 重打包 fp8 up 投影权重。输入
  ``[E, 2*N, K]`` 行主 fp8（每专家前 ``N`` 行为 gate，后 ``N`` 行为 up，``N`` 须为 64 的倍数）；
  输出仍为 ``[E, 2*N, K]``（字节占用相同），按 warpgroup/warp 重排使 ``silu(gate) * up`` 成为
  每 lane 的寄存器操作（无跨 warp 交换）。
- `mono_moe(...)`：单 kernel block-FP8 top-K MoE，固定 E256 / N512 / K2048 形态。`(E, N, K)` 由权重
  张量推出，须为该固定形态；至多 8 token（BS8 kernel）。`activations_in` 为 bf16 ``[M, K]``
  （``M <= 8``）；`router_logits` 为 bf16 ``[M, E]``；`expert_weights_up` 为 fp8_e4m3 up/gate 权重
  ``[E, 2*N, K]``（默认应用 `interleave_for_tma_wgmma_up`，已交错时传 ``interleave_up=False``）；
  `expert_scales_up` 为 fp32 block-wise scale ``[E, 2N/128, K/128]``；down 权重与 scale 类似。

**返回**：`torch.Tensor`——MoE 输出。

---

### Cake NVFP4 Warp Decode（SM100/SM103）

#### `CakeWarpDecodeConfig` / `CakeWarpDecodeRunner`

```python
@dataclass(frozen=True)
class CakeWarpDecodeConfig:
    backend: Literal["cake"] = "cake"
    def supported(cls, arch: int) -> bool: ...
    def prepare_weights(w1_bf16, w2_bf16, *, quant: QuantConfig = _NVFP4_NVFP4,
        num_local_experts: int, hidden_size: int, intermediate_size: int,
        activation=None, device=None, permute_cache=None): ...
    def prepare_activations(hidden_states_bf16, *, quant: QuantConfig = _NVFP4_NVFP4): ...

class CakeWarpDecodeRunner(MoERunner): ...
```

**功能**：exact SM100 与 exact SM103 的 Cake warp-decode runner，是显式统一 MoE 后端。以
``CakeWarpDecodeConfig(backend="cake")`` 选择；不在默认后端列表中；每个设备只加载其精确的生成目标。

**支持的契约（要点）**：``(activation, hidden_size, intermediate_size, num_experts, top_k)`` 必须是
文档化的一组精确组合（如 ``(SwiGLU(), 2048, 512, 512, 10)``、``(SwiGLU(), 2048, 1536, 60, 4)``、
``(SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 3072, 128, 4)``、
``(SiTU(gate_scale=4.0, linear_scale=25.0), 3584, 3072, 896, 16)`` 等，含 sharded 分区切片）；token
数 1–32；路由为 `UnpackedPrecomputed` 且为连续 int32 专家 id 与 BF16 路由权重；量化为 NVFP4；
finalization 与 PDL 启用；专家并行、融合共享专家、bias 与 LoRA 禁用。

**注意事项**（转述上游说明）

- 物理权重 / 激活布局与 `TrtllmFp4Config` 的 NVFP4×NVFP4 相同，使同一量化表示可被两个 runner 使用。
- 使用生产 ``MajorK`` 32 行 MMA shuffle（物理行 ``p`` 在逻辑行
  ``(p & ~31) + ((p & 7) << 2) + ((p & 31) >> 3)`` 恢复）与生产 ``R128c4`` E4M3 block scale 布局。
- runner 在计时启动或 CUDA Graph 捕获前准备工作区并复用；未见过的工作区形状或未验证的路由张量代次
  在捕获期间被拒绝，而非隐式初始化。最多保留 64 个活路由张量；验证另一个不同张量会显式失败。
- CUDA Graph 捕获需要已 warmup 的工作区；共享工作区的图重放与 eager 调用须由调用方排序；需要并发
  执行的图请使用各自独立的 `MoELayer` 实例。
- 在 exact SM100 与 SM103 AOT 构建中（启用 MoE kernel 时）也会注册匹配模块；SM100 不会加载或回退
  到 SM103 模块，其它 compute capability 被拒绝。

---

### cuTile Fused MoE

cuTile 后端以 Config / Runner 成对提供，所有 Config 提供 `supported` 与 `prepare_weights`
（部分提供 `prepare_activations`），注册到对应 ``cutile_*`` backend key：

| Config | 权重 / 激活格式 | 备注 |
| --- | --- | --- |
| `CuTileBf16Config` | BF16 / BF16 | `prepare_weights(w1_bf16, w2_bf16, *, num_local_experts, hidden_size, intermediate_size, activation=None, device=None)`。 |
| `CuTileNvfp4Config` | NVFP4 / NVFP4 | `prepare_weights(w1_fp4, w1_block_scale, w1_global_scale, w2_fp4, w2_block_scale, w2_global_scale, *, ..., source_format=...)`。 |
| `CuTileNvfp4Bf16Config` | NVFP4 / BF16 | 继承 `CuTileNvfp4Config`。 |
| `CuTileMxfp4Config` | MXFP4 / MXFP4 | `prepare_weights(w1_fp4, w1_block_scale, w2_fp4, w2_block_scale, *, ...)`。 |
| `CuTileMxfp4Bf16Config` | MXFP4 / BF16 | 继承 `CuTileMxfp4Config`。 |
| `CuTileFp8PerTensorConfig` | FP8 / FP8（per-tensor） | `prepare_weights(cls, w1_fp8, w1_scale, w2_fp8, w2_scale, *, ...)`。 |
| `CuTileFp8PerTensorBf16Config` | FP8 权 / BF16 输入 | 继承 `CuTileFp8PerTensorConfig`。 |
| `CuTileMxfp8Config` | MXFP8 / MXFP8（E8M0 per 32） | `prepare_weights(w1_fp8, w1_scale, w2_fp8, w2_scale, *, ...)`。 |
| `CuTileMxfp8Bf16Config` | MXFP8 权 / BF16 输入 | 继承 `CuTileMxfp8Config`。 |
| `CuTileMxfp4Mxfp8Config` | MXFP4 权 / MXFP8 输入 | 继承 `CuTileMxfp4Config`。 |

对应的 Runner 类（`CuTileBf16Runner`、`CuTileNvfp4Runner`、`CuTileNvfp4Bf16Runner`、
`CuTileMxfp4Runner`、`CuTileMxfp4Bf16Runner`、`CuTileFp8PerTensorRunner`、
`CuTileFp8PerTensorBf16Runner`、`CuTileMxfp8Runner`、`CuTileMxfp8Bf16Runner`、
`CuTileMxfp4Mxfp8Runner`）为统一 MoE runner 适配器。

---

### cuDNN Grouped-GEMM MoE

| Config / Runner | 底层算子 | 备注 |
| --- | --- | --- |
| `CudnnGroupedGemmBf16Config` / `CudnnGroupedGemmBf16Runner` | `grouped_mm_bf16` | key `cudnn_grouped_gemm_bf16`。 |
| `CudnnGroupedGemmFp8PerTensorConfig` / `CudnnGroupedGemmFp8PerTensorRunner` | `grouped_mm_fp8` | key `cudnn_grouped_gemm_fp8_per_tensor`；`prepare_weights` 需 `hidden_states_scale_global` 与 `intermediate_scale_global`。 |
| `CudnnGroupedGemmMxfp8Config` / `CudnnGroupedGemmMxfp8Runner` | `grouped_mm_mxfp8` | key `cudnn_grouped_gemm_mxfp8`；提供 `prepare_activations(hidden_states_bf16)`。 |
| `CudnnGroupedGemmNvfp4Config` / `CudnnGroupedGemmNvfp4Runner` | `grouped_mm_fp4`（16-wide block） | key `cudnn_grouped_gemm_nvfp4`；提供 `prepare_activations(hidden_states_bf16)`。 |

这些 config 提供 `supported(arch)` 与 `prepare_weights(w1_bf16, w2_bf16, *, num_local_experts,
hidden_size, intermediate_size, activation=None, device=None)`。

---

## 说明 / 注意事项（模块级设计说明）

- **两层 API 共存**：`MoELayer` 是统一 MoE API 的正式入口（构造与调用都带 ``@flashinfer_api``，
  参与 ``FLASHINFER_LOGLEVEL`` 与 ``FLASHINFER_DUMP_*``）；各后端的 per-backend 函数（``trtllm_*``、
  ``cute_dsl_*``、``prims_ts_*`` 等）同样正式。两层设计为共存，互不取代。
- **cuDNN Frost**：Frost 是 SM107a 上稳定的 `MoELayer` 后端，SM120a 上额外支持 BF16。可显式选择或
  在 autotune 中与合格后端一起被接纳，不需要 ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS``。
  详细的依赖基线、PTXAS 汇编器版本要求、AOT / JIT 交付与 trace 支持见上游 rst 的 Frost 小节
  （可用 ``FLASHINFER_CUDNN_FROST_PTXAS``、``FLASHINFER_TRACE_DUMP=1`` 与
  ``MoELayer.__call__.fi_trace(...)``）。SM107 的 SM 相关约束与 patch 级最低要求也记录在其中。
- **AlphaMoE Router**：独立的 AlphaMoE 前端把 FP32 logits 转为 top-k 权重与专家分组、block 对齐的
  route plan。完全相等的路由 logits 之间的选择与顺序未指定。计划的原子 scatter 顺序故意未指定。
- **Cake NVFP4 Warp Decode / Prims-TS**：均为显式统一 MoE 后端，不在默认后端列表中；选择它们即
  opt-in。二者对支持契约均采取 fail-closed（不满足契约即拒绝）。
- **MonoMoE**：仅 SM90a；调用前用 `has_monomoe` 检查。
- **模块内未在公开 rst 中列出的内部符号**（如 `RoutingInputMode`、`TrtllmMoERoutingMetadata`、
  `gen_cutlass_fused_moe_*_module`、`cudnn_frost_grouped_gemm1_swiglu` 等）虽由
  `flashinfer.fused_moe` 重导出，但不在本文件详述；其中 ``cudnn_frost_grouped_gemm1_swiglu`` 仍为
  实验性 API。

## 相关链接 / 参考

- 上游文档 `docs/api/fused_moe.rst`（含 Frost / Cake / Prims-TS / AlphaMoE 的详细设计说明与
  kernel 形式表）。
- `flashinfer.tllm_enums`：`RoutingMethodType`、`ActivationType`、`WeightLayout`、`SfLayout`、
  `is_gated_activation` 的定义（由 `flashinfer.fused_moe` 重导出）。
- Humming（``https://github.com/inclusionAI/humming``）：`preprocess_moe_weights_for_sm90_mixed_gemm_humming`
  所采用的 E8M0 范围夹取 / residual-scale 分解 / FP4 payload 重写方案来源。

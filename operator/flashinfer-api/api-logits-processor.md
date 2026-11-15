# Logits 处理器（`flashinfer.logits_processor`）

> 源码：`flashinfer/logits_processor/`（`__init__.py`、`processors.py`、`pipeline.py`、`types.py`、`op.py`、`compiler.py`、`fusion_rules.py`、`legalization.py`、`validators.py`）
> 上游文档：`docs/api/logits_processor.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.logits_processor` 是一个用于构建 LLM 输出处理流水线的**声明式、可插拔**框架。用户以高层
处理器（`LogitsProcessor`）声明流水线，框架将其**合法化（legalize）** 为带具体输入/输出类型的低层算子
（`Op` / `ParameterizedOp`），再经类型检查、有效性校验与**融合（fusion）** 编译为高效算子序列。

流水线的数据在内部以带语义类型标记的 `TaggedTensor` 流动，类型包括 `LOGITS`（原始或掩码后的实值分数）、
`PROBS`（非负概率，可归一化）与 `INDICES`（整数 token id，终态类型）。类型系统确保算子连接正确，并让融合
规则可以在编译期做出决策。

## 导入

```python
import torch
from flashinfer.logits_processor import (
    LogitsPipe,        # 流水线
    LogitsProcessor,   # 处理器基类
    Temperature, Softmax, TopK, TopP, MinP, Sample,  # 内置处理器
    TensorType, TaggedTensor,  # 类型
    Op, ParameterizedOp,       # 低层算子基类
    FusionRule,                # 自定义融合规则
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `LogitsPipe` | 类 | 声明式处理流水线容器与执行入口。 |
| `LogitsProcessor` | 类（ABC） | 高层处理器抽象基类。 |
| `Temperature` | 类 | 温度缩放处理器。 |
| `Softmax` | 类 | Softmax 处理器（logits → probs）。 |
| `TopK` | 类 | Top-k 过滤处理器。 |
| `TopP` | 类 | Top-p（nucleus）过滤处理器。 |
| `MinP` | 类 | Min-p 过滤处理器。 |
| `Sample` | 类 | 采样处理器（→ indices）。 |
| `TensorType` | 枚举 | 张量语义类型。 |
| `TaggedTensor` | 数据类 | 携带语义类型的张量包装。 |

## API 详情

### `LogitsPipe`

```python
LogitsPipe(
    processors: List[LogitsProcessor],
    compile: bool = True,
    input_type: Optional[TensorType] = None,
    custom_fusion_rules: Optional[List[FusionRule]] = None,
    custom_validity_checks: Optional[List[ValidityCheck]] = None,
)
```

**功能**：提供构建 LLM 输出处理流水线的声明式方式。构造时依次执行：推断初始输入类型 → 合法化处理器为低层
算子 → 编译（类型检查、有效性校验、融合）。

**参数**

- `processors`（`List[LogitsProcessor]`）：按顺序应用的处理器列表（不可为空）。
- `compile`（`bool`，默认 `True`）：是否在编译时应用融合优化。可在实例化后调用 `LogitsPipe.compile()`
  再编译。
- `input_type`（`Optional[TensorType]`，默认 `None`）：期望的输入张量类型（`TensorType.LOGITS` 或
  `TensorType.PROBS`）。若第一个处理器可同时接受两种类型则必须指定；其他情况下会从第一个处理器自动推断。
- `custom_fusion_rules`（`Optional[List[FusionRule]]`，默认 `None`）：编译时应用的额外融合规则。
- `custom_validity_checks`（`Optional[List[ValidityCheck]]`，默认 `None`）：编译时应用的额外有效性检查。

**方法**

- `__call__(x, **kwargs) -> torch.Tensor`：执行流水线。`x` 为 `torch.Tensor` 或 `TaggedTensor`；运行时参数
  （如 `temperature`、`top_k`、`top_p`、`min_p`）通过 `**kwargs` 传入。返回普通的 `torch.Tensor`（非
  `TaggedTensor`）。若流水线未编译则记录 warning 并按离散算子运行。
- `compile(custom_fusion_rules=None, custom_validity_checks=None) -> None`：编译流水线（类型检查、有效性校验、
  融合）。
- `initial_type`（属性，`TensorType`）：流水线的初始输入张量类型。

**注意事项**

- 流水线自动校验算子间的类型兼容性；算子会尽可能被融合。
- 输出始终为普通 `torch.Tensor`，而非 `TaggedTensor`。
- 构造/编译失败时抛出 `ValueError`。

**示例**

```python
import torch
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, Sample

pipe = LogitsPipe([
    Temperature(),      # 按 temperature 缩放 logits
    Softmax(),          # logits -> 概率
    TopP(),             # top-p 过滤
    Sample()            # 采样
])

batch_size = 4
vocab_size = 5
logits = torch.randn(batch_size, vocab_size, device="cuda")
output_ids = pipe(logits, temperature=0.7, top_p=0.9)

# 以概率为输入起点
from flashinfer.logits_processor import TensorType, TopK
prob_pipe = LogitsPipe([TopK(), Sample()], input_type=TensorType.PROBS)
probs = torch.softmax(logits, dim=-1)
token_ids = prob_pipe(probs, top_k=40)
```

---

### `LogitsProcessor`

```python
class LogitsProcessor(ABC):
    def __init__(self, **params: Any): ...
    @abstractmethod
    def legalize(self, input_type: TensorType) -> List[Op]: ...
```

**功能**：定义可应用于 logits 或概率的高层变换。每个处理器会自动合法化为可被类型检查、校验并融合的低层
`Op` 或 `ParameterizedOp`。用户可继承该类实现自己的处理器。

**参数**

- `**params`（`Any`）：处理器专用的编译期参数。

**方法**

- `legalize(input_type: TensorType) -> List[Op]`：将处理器合法化为低层算子列表。`input_type` 为处理器的期望
  输入张量类型；返回低层算子列表。子类必须实现。

**注意事项**

- 子类必须实现 `legalize`，将高层处理器转换为一个或多个具具体输入/输出类型的低层算子。

**示例**

```python
>>> import torch
>>> from flashinfer.logits_processor import LogitsPipe, TopK, Sample, TensorType
>>> pipe = LogitsPipe([TopK(), Sample()], input_type=TensorType.PROBS)
>>> pipe
LogitsPipe([TopK -> Sample], ops=[ProbsTopKOp -> ProbsSampleOp], compiled_ops=[FusedProbsTopKSampleOp])
```

---

### `Temperature`

```python
Temperature(**params: Any)
```

**功能**：logits 的温度缩放处理器。将 logits 除以温度值。类型变换：`TensorType.LOGITS -> TensorType.LOGITS`。

**参数**

- 无编译期参数。
- 运行时参数：`temperature`（`float` 或 `torch.Tensor`）：缩放用的温度值，必须为正，可为标量或逐 batch 张量。

**示例**

```python
>>> import torch
>>> from flashinfer.logits_processor import LogitsPipe, Temperature
>>> pipe = LogitsPipe([Temperature()])
>>> logits = torch.randn(2, 2, device="cuda")
>>> pipe(logits, temperature=0.8)
```

---

### `Softmax`

```python
Softmax(enable_pdl: Optional[bool] = None, **params: Any)
```

**功能**：将 logits 转换为概率的 Softmax 处理器。类型变换：`TensorType.LOGITS -> TensorType.PROBS`。

**参数**

- `enable_pdl`（`Optional[bool]`，默认 `None`，编译期）：是否为内核实现启用 PDL。默认 `None` 表示设备支持
  PDL 时自动启用。

**注意事项**

- 每条流水线中只能出现一次。

**示例**

```python
>>> from flashinfer.logits_processor import LogitsPipe, Softmax
>>> pipe = LogitsPipe([Softmax()])
>>> probs = pipe(torch.randn(2, 2, device="cuda"))
```

---

### `TopK`

```python
TopK(joint_topk_topp: bool = False, **params: Any)
```

**功能**：Top-k 过滤处理器，仅保留概率最高的 top-k 个 token，掩掉其余。类型变换：
`TensorType.LOGITS -> TensorType.LOGITS` 或 `TensorType.PROBS -> TensorType.PROBS`。

**参数**

- `joint_topk_topp`（`bool`，默认 `False`，编译期）：当后接 `TopP` 时是否启用 top-k 与 top-p 的联合过滤。
- 运行时参数：`top_k`（`int` 或 `torch.Tensor`）：保留的 top token 数，可为标量或逐 batch 张量。

**注意事项**

- 应用于 `TensorType.LOGITS` 时，将非 top-k 值置为 `-inf`。
- 应用于 `TensorType.PROBS` 时，将非 top-k 值置零并重归一化。

**示例**

```python
>>> from flashinfer.logits_processor import LogitsPipe, TopK, TensorType
>>> pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
>>> pipe(torch.randn(2, 2, device="cuda"), top_k=1)
```

**另见**：`flashinfer.sampling.top_k_mask_logits`、`flashinfer.sampling.top_k_renorm_probs`

---

### `TopP`

```python
TopP(**params: Any)
```

**功能**：Top-p（nucleus）过滤处理器，保留累计概率不超过阈值 p 的 token。类型变换：
`TensorType.PROBS -> TensorType.PROBS`。

**参数**

- 无编译期参数。
- 运行时参数：`top_p`（`float` 或 `torch.Tensor`）：累计概率阈值，落在 `(0, 1]`，可为标量或逐 batch 张量。

**示例**

```python
>>> from flashinfer.logits_processor import LogitsPipe, TopP
>>> pipe = LogitsPipe([TopP()])
>>> probs = torch.randn(2, 2, device="cuda")
>>> pipe(probs / probs.sum(dim=-1, keepdim=True), top_p=0.9)
```

**另见**：`flashinfer.sampling.top_p_renorm_probs`

---

### `MinP`

```python
MinP(**params: Any)
```

**功能**：Min-p 过滤处理器，保留概率至少为最大概率 p 倍的 token。类型变换：
`TensorType.PROBS -> TensorType.PROBS`。

**参数**

- 无编译期参数。
- 运行时参数：`min_p`（`float` 或 `torch.Tensor`）：最小概率阈值（相对最大概率的比例），必须落在 `(0, 1]`，
  可为标量或逐 batch 张量。

**示例**

```python
>>> from flashinfer.logits_processor import LogitsPipe, MinP
>>> pipe = LogitsPipe([MinP()])
>>> probs = torch.randn(2, 2, device="cuda")
>>> pipe(probs / probs.sum(dim=-1, keepdim=True), min_p=0.05)
```

---

### `Sample`

```python
Sample(deterministic: bool = True, **params: Any)
```

**功能**：从 logits 或概率分布中采样 token 索引的采样处理器。类型变换：
`TensorType.LOGITS -> TensorType.INDICES` 或 `TensorType.PROBS -> TensorType.INDICES`。

**参数**

- `deterministic`（`bool`，默认 `True`，编译期）：是否使用确定性内核实现。
- 运行时参数：
  - `indices`（`torch.Tensor`，可选）：当多个输出共享同一概率张量时用于批量采样的索引。
  - `generator`（`torch.Generator`，可选）：用于可复现采样的随机数生成器。

**注意事项**

- 输出为 `TensorType.INDICES`——其后不能跟随任何算子。

**示例**

```python
>>> from flashinfer.logits_processor import LogitsPipe, Sample, TensorType
>>> pipe = LogitsPipe([Sample(deterministic=True)], input_type=TensorType.LOGITS)
>>> tokens = pipe(torch.randn(2, 5, device="cuda"), top_k=1)
```

**另见**：`flashinfer.sampling.sampling_from_logits`、`flashinfer.sampling.sampling_from_probs`

---

### `TensorType`

```python
class TensorType(Enum):
    LOGITS = auto()    # 原始或掩码后的实值分数
    PROBS = auto()     # 非负概率，可归一化
    INDICES = auto()   # 整数 token id，终态类型
```

**功能**：表示流水线中张量的语义类型。

**成员**

- `LOGITS`：原始或掩码后的 float 分数，实值向量。
- `PROBS`：非负 float 概率，可归一化。
- `INDICES`：整数 token id，终态类型。

---

### `TaggedTensor`

```python
@dataclass(slots=True, frozen=True)
class TaggedTensor:
    data: torch.Tensor
    type: TensorType
```

**功能**：张量包装，在流水线执行过程中维护语义类型信息，确保类型安全，同时对下游 PyTorch 代码保持无缝。

**属性 / 方法**

- `data`（`torch.Tensor`）：底层张量。
- `type`（`TensorType`）：张量的语义类型。
- `TaggedTensor.logits(t)`（静态方法）：以 `TensorType.LOGITS` 创建 `TaggedTensor`。
- `TaggedTensor.probs(t)`（静态方法）：以 `TensorType.PROBS` 创建 `TaggedTensor`。
- `TaggedTensor.indices(t)`（静态方法）：以 `TensorType.INDICES` 创建 `TaggedTensor`。
- `shape` / `device` / `dtype`（属性）：底层张量的形状 / 设备 / dtype。
- `size(dim=None)`：底层张量的 size。
- `__torch_function__`：将 `TaggedTensor` 参数解包为底层张量后调用 PyTorch 函数。

**注意事项**

- `TaggedTensor` 主要供 `LogitsPipe` 内部使用。
- 用户通常使用普通张量；标记会自动发生。

## 自定义功能（Customization Features）

### 自定义 Logits 处理器

可以通过继承 `LogitsProcessor` 创建自己的 logits 处理器：

```python
class CustomLogitsProcessor(LogitsProcessor):

    def __init__(self, **params: Any):
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List["Op"]:
        return [CustomOp(**self.params)]


class CustomOp(Op):
    # 定义输入与输出张量类型
    IN = TensorType.LOGITS
    OUT = TensorType.LOGITS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        pass


pipe = LogitsPipe([CustomLogitsProcessor()])  # 流水线将被编译为 [CustomOp]
```

### 自定义融合规则

可以注册自定义融合规则以优化特定的处理器组合：

```python
def custom_fusion_guard(window: List[Op]) -> bool:
    # 是否应用该融合
    return True


def build_custom_fusion(window: List[Op]) -> Op:
    # 通过设置参数等创建一个融合算子
    return CustomOp()


custom_rule = FusionRule(
    pattern=(Temperature, Softmax),
    guard=custom_fusion_guard,
    build=build_custom_fusion,
    prio=20,
)

pipe = LogitsPipe(
    [Temperature(), Softmax(), Sample()],
    custom_fusion_rules=[custom_rule],
)   # 流水线中编译后的算子将为 [CustomOp, Sample]
```

### `FusionRule`（自定义融合规则所使用的元组类型）

```python
class FusionRule(NamedTuple):
    pattern: Tuple[type, ...]
    guard: Callable[[List[Op]], bool]
    build: Callable[[List[Op]], Op]
    prio: int = 0
```

- `pattern`：要匹配的算子类型元组（例如 `(TopK, Sampling)`）。
- `guard`：接收匹配到的算子（是流水线中算子窗口，精确匹配 `pattern`）并返回是否应融合的布尔值的函数。
- `build`：接收匹配到的算子并返回融合算子的函数。
- `prio`：规则应用优先级（越大越先尝试），默认 `0`。

## 相关链接 / 参考

- 上游文档：`docs/api/logits_processor.rst`
- 编译器与合法化：`flashinfer.logits_processor.compiler.Compiler`、
  `flashinfer.logits_processor.legalization.legalize_processors`
- 底层算子：`flashinfer.logits_processor.operators`（`TemperatureOp`、`SoftmaxOp`、
  `LogitsTopKOp`、`ProbsTopKOp`、`TopPOp`、`MinPOp`、`LogitsSampleOp`、`ProbsSampleOp` 及各 `Fused*` 算子）

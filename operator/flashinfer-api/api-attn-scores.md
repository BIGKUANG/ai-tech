# 分页 MQA 注意力分数（`flashinfer.attn_scores`）

> 源码：`flashinfer/attn_scores/__init__.py`、`flashinfer/attn_scores/attn_scores.py`
> 上游文档：`docs/api/attn_scores.rst`
> 适用版本：FlashInfer 0.7.2

`flashinfer.attn_scores` 提供分页 MQA 逻辑（“注意力分数”）内核，用于数据中心 Blackwell
（SM100/SM103）与 Rubin（SM107）GPU。它针对每个请求、每个 KV 位置，计算一个稀疏注意力
索引器（indexer）用来挑选保留哪些 KV token 的、按头加权的整流 query-key 分数：

```text
logits[t, p] = sum_h w[t, h] * relu(q[t, h] · k[p])
```

关于该表达式有两点容易出错：

- 整流（ReLU）是**逐头、在加权求和之前**施加的，而不是作用在求和后的分数上。由于权重可能为负，
  结果也可能为负。
- `fp8_paged_mqa_logits` 还会乘以每个 token 携带在 `kv_fused` 块尾部的 FP32 KV scale，因此其
  结果为上式再乘以 `s_p`。`fp4_paged_mqa_logits` 没有这种逐位置因子：MXFP4 块 scale 已被折叠进
  反量化后的值本身。这种不对称是代数性的——一个 token 整个 K 向量上统一的 scale 可以从点积中
  提取出来（且为正，与 ReLU 可交换），而两个操作数上每 32 元素的分组 scale 则不能。

两个变体在 scale 之外还有差异：`fp4_paged_mqa_logits` 额外接收 `q_sf` scale 张量
（以及 `sf_vec_size`、`is_kv_sf_interleaved`），且每字节打包两个 FP4 值；只有
`fp8_paged_mqa_logits` 暴露 `acc_dtype`（fp4 的 MMA 累加器固定为 float32）；默认
`output_dtype` 分别为 float32（fp8）与 bfloat16（fp4）。fp8 接受参数化形状，而 fp4 要求恰好
`num_heads=64`、`head_dim=128`。最后，`next_n=4` 对 fp8 在所有受支持架构上都是单趟，但对 fp4
只在 Rubin（SM107）上如此——SM100/SM103 会把它作为两次内部 KV 趟运行；任何调用方构造的
`schedule_meta` 都必须用调度辅助函数的匹配 `next_n`/`variant` 参数生成。

分页参数、输出契约以及 `out=`/`schedule_meta` 语义在这一对函数之间是完全一致的。

## 导入

```python
from flashinfer import (
    fp8_paged_mqa_logits,
    fp4_paged_mqa_logits,
    compute_paged_mqa_logits_schedule,
    min_block_table_width,
    padded_seq_len,
    precompile_paged_mqa_logits,
)
# 或
from flashinfer.attn_scores import fp8_paged_mqa_logits
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `fp8_paged_mqa_logits` | 函数 | FP8 分页 MQA 逻辑（注意力分数），Blackwell / Rubin。 |
| `fp4_paged_mqa_logits` | 函数 | MXFP4 分页 MQA 逻辑，Blackwell / Rubin。 |
| `compute_paged_mqa_logits_schedule` | 函数 | 计算 CTA 调度张量，可作为 `schedule_meta` 传入。 |
| `min_block_table_width` | 函数 | 返回给定序列长度所需的 `block_tables` 最小列数。 |
| `padded_seq_len` | 函数 | 返回 `out` 缓冲区所需的最小列数（向上取整到内部存储粒度）。 |
| `precompile_paged_mqa_logits` | 函数 | 预热常见静态配置的内核编译缓存。 |

## API 详情

### `fp8_paged_mqa_logits`

```python
fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    output_dtype: torch.dtype = torch.float32,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    schedule_meta: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：面向 Blackwell（SM100/SM103）与 Rubin（SM107）的 FP8 分页 MQA 逻辑。针对分页
FP8 KV cache 计算按头加权的 ReLU 注意力分数（DeepSeek 风格稀疏注意力索引器）：

```text
logits[b*next_n + t, pos] =            (以 output_dtype 存储)
    scale[pos] * sum_h weights[b*next_n + t, h] * relu(q[b,t,h,:] @ k[pos,:])
    └────────────── 尾声：epi_dtype ─────────────┘   └─ MMA：acc_dtype ───┘
```

每个头的点积在 tensor core 内以 `acc_dtype` 沿 head_dim 累加；其后的所有操作——ReLU、权重相乘、
跨头求和、`scale[pos]` 相乘——在 CUDA core 上以 `epi_dtype` 执行；结果在存储时转换为
`output_dtype`。ReLU 是逐头、在加权求和之前施加的，因此负权重可使输出为负。

`next_n = q.shape[1]` 是每个请求被打分的 query 位置数（推测/MTP draft 深度；普通 decode 用 1），
它是 `q` 的一个维度而非参数；内核会自动针对其特化。槽位 `t = 0` 是最旧的 draft 位置，
`t = next_n - 1` 是最新的；`seq_lens[b]` 是最新槽位可见的 KV 长度。

**参数**

- `q`（`torch.Tensor`，`[batch_size, next_n, num_heads, head_dim]`，`float8_e4m3fn`）：query。
- `kv_fused`（`torch.Tensor`，`[num_blocks, block_size, 1, head_dim + 4]`，`uint8`）：分页 KV 池。
  “fused” 指每个块把量化值与 scale 因子一起存在同一个 uint8 缓冲区中。4-D 形状只是尺寸契约；
  内核把每块的 `block_size*(head_dim+4)` 字节读作两个连续区域，而非逐 token 行：
  - 字节 `[0, block_size*head_dim)`：FP8 值，token 主序（token i 在偏移 `i*head_dim`）；
  - 字节 `[block_size*head_dim, end)`：`block_size` 个小端 float32 逐 token scale（token i 的
    scale 在 `block_size*head_dim + 4*i`）。
- `weights`（`torch.Tensor`，`[batch_size*next_n, num_heads]`，`float32`）：逐头混合权重（非模型
  参数）。行 `b*next_n + t` 与 `q[b, t]` 配对。当 `epi_dtype` 为 float16 时内部转换。
- `block_tables`（`torch.Tensor`，`[batch_size, max_blocks_per_seq]`，`int32`）：物理块索引，每行
  连续（`stride(1)==1`）。`max_blocks_per_seq` 必须至少为
  `min_block_table_width(max(seq_lens), block_size) = ceil(max(seq_lens)/block_size)`。列数过少是
  设备端越界读（未定义行为）。
- `seq_lens`（`torch.Tensor`，`[batch_size]`，`int32`）：每个请求的 KV 长度；不得有元素超过
  `max_seq_len`。接受任意 stride（如 `seq_lens[::next_n]` 零拷贝）。
- `max_seq_len`（`int`）：最大 KV 序列长度，必须 `>= max(seq_lens)`。输出行按其定尺寸；值过小是
  设备端越界写（未定义行为）。
- `output_dtype`（`torch.dtype`，默认 `torch.float32`）：输出类型，可为 `torch.float32` 或
  `torch.float16`。**不支持 bfloat16**（FP8 尾声无 bf16 分支，会静默误算）。注意
  `fp4_paged_mqa_logits` 默认是 bfloat16。
- `epi_dtype`（`torch.dtype`，默认 `torch.float32`）：尾声累加类型（float32 或 float16）。
  float16 使尾声流量减半，但在 `|logit| > 65504` 时饱和，长序列累加可能上溢为 inf。
- `acc_dtype`（`torch.dtype`，默认 `torch.float32`）：MMA 累加器类型（float32 或 float16）。
- `schedule_meta`（`Optional[torch.Tensor]`，默认 `None`）：可选、预先算好的
  `[num_sms+1, 2]` int32 CTA 调度，连续，位于 `q` 的设备上。除非 profiling 表明每次调用的调度计算
  在 eager 热循环中确有影响，否则建议保持 `None`。仅在 `seq_lens` 内容与目标设备 SM 数不变时才能
  复用；`seq_lens` 变化（包括 CUDA Graph 重放）时必须重算。陈旧调度会挂起内核。用
  `compute_paged_mqa_logits_schedule()` 生成。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：可选的预分配输出存储（PyTorch `out=` 风格）。
  形状至少 `[batch_size*next_n, padded_seq_len(max_seq_len)]`；内核会把列数补齐到内部存储粒度并
  无条件写入填充列，因此更窄的缓冲是越界写。必须位于 `q` 的设备上、dtype 为 `output_dtype`、
  行连续（`stride(1)==1`）且互不重叠（`stride(0) >= padded_seq_len(max_seq_len)`）。CUDA Graph
  捕获时需要。允许更大（例如跨捕获共享一个地址稳定的 max-batch 缓冲）。

**返回**

- `torch.Tensor`：`[batch_size*next_n, max_seq_len]`，dtype 为 `output_dtype`。始终是上述填充存储
  的视图（调用方提供时为其缓冲的视图，否则为每次调用内部分配的视图）。行 `b*next_n + t` 对 KV
  位置 `0 .. seq_lens[b] - next_n + t`（含）有意义——最新槽看到整个序列，每个更早槽少一个位置。
  超过该限制的列以及 `out` 超出 `max_seq_len` 的填充列均为**未指定**：消费前请按该限制切片或
  掩码（例如 top-k 之前）。`seq_lens[b] == 0` 的请求行永不写入。

**注意事项**

- **CUDA Graph 捕获**：传入用 `padded_seq_len()` 定尺寸的、地址稳定的 `out`；保持
  `q`/`kv_fused`/`weights`/`seq_lens`/`block_tables` 地址静态；保持 `schedule_meta=None`，调度由
  捕获内部的 GPU 内核计算并在每次重放时依据 `seq_lens` 当前内容重新导出。
- 调用方自行管理调度时，必须在每次重放前将新调度重算进同一缓冲；陈旧调度会挂起内核。
- **当前内核限制**：`head_dim` 必须是 32 的倍数；`num_heads` 必须是 4 的倍数；
  `next_n*num_heads` 必须是 `[8, 256]` 内 8 的倍数；`block_size`（`kv_fused.shape[1]`）必须是
  32、64 或 128；`head_dim`、`num_heads`、`next_n` 合计须适配每 CTA 共享内存（SM100 上
  `head_dim <= 192` 在 `num_heads=64, next_n=1` 时可行，256 不可行）；输出尺寸
  `batch_size*next_n * padded_seq_len(max_seq_len)` 必须 `< 2**31` 个元素（内核用 32 位偏移索引）。
- `block_tables` 宽度与 `max_seq_len` 都是硬性前置条件而非被检查的参数：违反时是设备端越界读写。
  `FLASHINFER_VALIDATE_INPUTS=1` 会在违反时报错，但仅是开发辅助，且 CUDA Graph 捕获期间会跳过。

**示例**

```python
import torch
import flashinfer

B, next_n, num_heads, head_dim = 2, 3, 64, 128
block_size, max_seq_len = 32, 8000
device = "cuda"

seq_lens = torch.tensor([3000, 8000], dtype=torch.int32, device=device)
blocks_per_seq = flashinfer.min_block_table_width(max_seq_len, block_size)  # 250
num_blocks = B * blocks_per_seq
block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).view(B, blocks_per_seq)

kv = torch.randn(num_blocks, block_size, head_dim, device=device).to(torch.float8_e4m3fn)
kv_scales = torch.ones(num_blocks, block_size, device=device)
kv_fused = torch.cat(
    [kv.view(torch.uint8).flatten(1), kv_scales.view(torch.uint8).flatten(1)], dim=1
).view(num_blocks, block_size, 1, head_dim + 4)

q = torch.randn(B, next_n, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
weights = torch.rand(B * next_n, num_heads, device=device)

logits = flashinfer.fp8_paged_mqa_logits(
    q, kv_fused, weights, block_tables, seq_lens, max_seq_len
)  # [B*next_n, max_seq_len] float32

out = torch.empty(
    B * next_n, flashinfer.padded_seq_len(max_seq_len), dtype=torch.float32, device=device
)
logits = flashinfer.fp8_paged_mqa_logits(
    q, kv_fused, weights, block_tables, seq_lens, max_seq_len, out=out
)
```

---

### `fp4_paged_mqa_logits`

```python
fp4_paged_mqa_logits(
    q: torch.Tensor,
    q_sf: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    sf_vec_size: int = 32,
    output_dtype: torch.dtype = torch.bfloat16,
    epi_dtype: torch.dtype = torch.float32,
    is_kv_sf_interleaved: bool = False,
    schedule_meta: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：面向 Blackwell（SM100/SM103）与 Rubin（SM107）的 FP4（MXFP4）分页 MQA 逻辑，针对分页
MXFP4 KV cache 计算按头加权的 ReLU 注意力分数：

```text
logits[b*next_n + t, pos] =          (以 output_dtype 存储)
    sum_h weights[b*next_n + t, h] * relu(q[b,t,h,:] @ k[pos,:])
    └──────── 尾声：epi_dtype ───────┘   └─ MMA：float32 ─────┘
```

每个头的点积在 tensor core 内以 float32 累加（不同于 FP8 变体，此处不可配置）；ReLU、权重相乘与
跨头求和以 `epi_dtype` 在 CUDA core 上执行。UE8M0 块 scale 被折叠进 q 和 k 的反量化，因此不像
FP8 变体那样有尾部逐 token scale。

`next_n` 可为 1..4：在 Rubin（SM107+）上每个 `next_n` 都是对 KV 的单趟；在 SM100/SM103 上，
`next_n=4` 以两次内部趟执行、每趟读取整个 KV（数值相同，KV 带宽成本约 2 倍，16K 序列长度下实测
运行时约为 `next_n=3` 的 1.3 倍），此时调用方提供的 `schedule_meta` 必须用本次调用的 `next_n` 与
`variant="fp4"` 构造。

**参数**

- `q`（`torch.Tensor`，`[batch_size, next_n, num_heads, head_dim/2]`，`uint8`）：每字节打包两个
  FP4（E2M1）值：一个头某行的第 j 字节低位半字节存元素 `2j`，高位半字节存元素 `2j+1`。
- `q_sf`（`torch.Tensor`，`[batch_size, next_n, num_heads]`，`int32`）：每个 (token, head) 的打包
  UE8M0 scale。`head_dim/sf_vec_size = 4` 个单字节 scale 装进一个 int32；第 k 个（最低有效、
  小端）字节是元素 `[k*sf_vec_size, (k+1)*sf_vec_size)` 的 UE8M0 scale。等价于 uint8
  `[batch_size, next_n, num_heads, 4]` 经 `.view(torch.int32)`。
- `kv_fused`（`torch.Tensor`，`[num_blocks, block_size, 1, head_dim/2 + 4]`，`uint8`）：分页 KV 池。
  每块字节读作两个连续区域：`[0, block_size*head_dim/2)` 为打包 FP4 值（token 主序，半字节规则同
  `q`）；`[block_size*head_dim/2, end)` 为 UE8M0 scale（每 token `head_dim/sf_vec_size = 4` 字节，
  默认 token 主序，见 `is_kv_sf_interleaved`）。
- `weights`（`torch.Tensor`，`[batch_size*next_n, num_heads]`，`float32`）：逐头混合权重。
- `block_tables`（`torch.Tensor`，`[batch_size, max_blocks_per_seq]`，`int32`）：同 fp8，硬性宽度
  前置条件。
- `seq_lens`（`torch.Tensor`，`[batch_size]`，`int32`）：每个请求的 KV 长度。
- `max_seq_len`（`int`）：最大 KV 序列长度，必须 `>= max(seq_lens)`。
- `sf_vec_size`（`int`，默认 `32`）：共享同一 UE8M0 scale 的 FP4 值个数，决定 `q_sf` 的打包以及
  每个融合 KV 行的 scale 字节。**必须等于 32**。
- `output_dtype`（`torch.dtype`，默认 `torch.bfloat16`）：`bfloat16`、`float32` 或 `float16`。
  默认与 fp8 变体（float32）不同；同时使用两者时请显式传入。
- `epi_dtype`（`torch.dtype`，默认 `torch.float32`）：尾声累加类型（float32、float16 或
  bfloat16）。无 `acc_dtype` 旋钮：块 scaled FP4 MMA 始终以 float32 累加。
- `is_kv_sf_interleaved`（`bool`，默认 `False`）：声明每个 `kv_fused` 块 scale 尾部的顺序。
  `False` 表示 token 顺序；`True` 表示块 scale 被拆成 4 段等长 run 再轮转交织，槽 k 持有 token
  `(k % 4) * (block_size // 4) + k // 4` 的 scale。此为传入 KV cache 的描述，须与实际写入一致；
  提供交织顺序可能更快。
- `schedule_meta`（`Optional[torch.Tensor]`，默认 `None`）：可选 CTA 调度。始终用
  `compute_paged_mqa_logits_schedule(seq_lens, device=..., next_n=..., variant="fp4")` 生成，
  参数须与本次调用一致。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：可选预分配输出，契约同 fp8（dtype 须匹配
  `output_dtype`）。

**返回**

- `torch.Tensor`：`[batch_size*next_n, max_seq_len]`，`output_dtype`。行 `b*next_n + t` 对 KV 位置
  `0 .. seq_lens[b] - next_n + t`（含）有意义；超过该限制的列未指定。`seq_lens[b] == 0` 的行永不
  写入。

**注意事项**

- **当前内核限制**：`num_heads` 必须等于 64；`head_dim` 必须等于 128；`next_n` 必须在 1..4；
  `schedule_meta` 必须用本次调用的 `next_n` 与 `variant="fp4"` 生成；`sf_vec_size` 必须等于 32；
  `block_size` 必须是 32、64 或 128；`is_kv_sf_interleaved` 仅当 `block_size == 128` 时可为
  `True`；输出尺寸必须 `< 2**31` 个元素。
- **CUDA Graph 捕获**：传入地址稳定的 `out`；保持输入地址静态；保持 `schedule_meta=None`。
- 违反 `block_tables` 宽度或 `max_seq_len` 前置条件是设备端越界读写。

**示例**

```python
import torch
import flashinfer

B, next_n, num_heads, head_dim = 2, 3, 64, 128
block_size, max_seq_len = 32, 8000
device = "cuda"

seq_lens = torch.tensor([3000, 8000], dtype=torch.int32, device=device)
blocks_per_seq = flashinfer.min_block_table_width(max_seq_len, block_size)
num_blocks = B * blocks_per_seq
block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).view(B, blocks_per_seq)
weights = torch.rand(B * next_n, num_heads, device=device)

q = torch.zeros(B, next_n, num_heads, head_dim // 2, dtype=torch.uint8, device=device)
q_sf = torch.zeros(B, next_n, num_heads, dtype=torch.int32, device=device)
kv_fused = torch.zeros(
    num_blocks, block_size, 1, head_dim // 2 + 4, dtype=torch.uint8, device=device
)

logits = flashinfer.fp4_paged_mqa_logits(
    q, q_sf, kv_fused, weights, block_tables, seq_lens, max_seq_len
)  # [B*next_n, max_seq_len] bfloat16
```

---

### `compute_paged_mqa_logits_schedule`

```python
compute_paged_mqa_logits_schedule(
    seq_lens: torch.Tensor,
    device: Optional[torch.device] = None,
    *,
    next_n: int = 1,
    variant: str = "fp8",
    use_gpu_kernel: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

**功能**：计算分页 MQA 逻辑内核的 CTA 调度张量。返回 CUDA 上的 `[num_sms+1, 2]` int32，可作
`schedule_meta` 传给 `fp8_paged_mqa_logits` 或 `fp4_paged_mqa_logits`。请传入与即将进行的主调用
相同的 `variant` 和 `next_n`：某些 `(variant, next_n, device)` 组合下内核会内部重构问题，本辅助
函数会施加相同的内部策略，确保返回的调度总是描述内核实际运行的工作。把内容视为**不透明**——
绝不要检查、序列化，或跨 FlashInfer 版本/设备/(variant, next_n) 复用。

**参数**

- `seq_lens`（`torch.Tensor`，`[B]` int32，一维）：可在 CPU 或 `device` 上，任意 stride。
- `device`（`Optional[torch.device]`，默认 `None`）：目标 CUDA 设备。默认 `seq_lens.device`，
  当 `seq_lens` 在 CPU 上时用当前 CUDA 设备。
- `next_n`（`int`，默认 `1`）：即将进行的主调用的 `next_n`（`q.shape[1]`），默认 1（普通 decode）。
- `variant`（`str`，默认 `"fp8"`）：`"fp8"` 或 `"fp4"`，调度的目标 API。
- `use_gpu_kernel`（`bool`，默认 `True`）：为 `True` 时完全在 GPU 上用小型专用调度内核计算——无
  D2H 拷贝，可 CUDA Graph 捕获。当 CuTe DSL 不可用、无法定位设备架构，或 `seq_lens` 在 CPU 上时，
  回退到 CPU numpy（不可图捕获）。
- `out`（`Optional[torch.Tensor]`，默认 `None`）：可选的预分配 CUDA `[num_sms+1, 2]` int32。
  CUDA Graph 捕获需要（静态缓冲）。这提供静态**存储**而非静态**内容**：地址稳定，但每当
  `seq_lens` 内容变化（包括每次重放）都必须把值重算进去。

**返回**

- `torch.Tensor`：`schedule_meta`，CUDA 上的 `[num_sms+1, 2]` int32（提供 `out` 时即 `out`）。

**注意事项**

- 复用的调度仅在每个请求的 `seq_lens` 内容不变时有效。`seq_lens` 变化时（包括每次 CUDA Graph
  重放）必须重算。

**示例**

```python
import torch
import flashinfer

seq_lens = torch.tensor([3000, 8000], dtype=torch.int32, device="cuda")
sched = flashinfer.compute_paged_mqa_logits_schedule(seq_lens, next_n=1, variant="fp8")
```

---

### `min_block_table_width`

```python
min_block_table_width(seq_len: int, block_size: int) -> int
```

**功能**：返回给定序列长度所需的 `block_tables` 最小列数，即自然宽度
`ceil(seq_len / block_size)`——序列占用的每个 KV 块一个条目，正是分页 KV 服务栈每请求已有的表。
内核对每次块表读取都以该行自身的块数为谓词，因此无需填充列，超出序列块数的条目永不读取。

**参数**

- `seq_len`（`int`）：序列长度。
- `block_size`（`int`）：KV 块大小。

**返回**

- `int`：`ceil(seq_len / block_size)`。

**示例**

```python
blocks_per_seq = flashinfer.min_block_table_width(max_seq_len, block_size)
block_tables = torch.arange(...).view(B, blocks_per_seq)
```

---

### `padded_seq_len`

```python
padded_seq_len(max_seq_len: int) -> int
```

**功能**：返回分页 MQA 逻辑 `out` 缓冲区的最小列数，把 `max_seq_len` 向上取整到内核内部输出存储
粒度。内核会无条件写入填充的尾部位置，因此输出张量必须至少分配这么多列（更窄的 `out` 是越界写）。
粒度为实现细节、可能变化——请始终调用本辅助函数而非硬编码。填充列存放未指定的暂存内容，切勿消费。

**参数**

- `max_seq_len`（`int`）：最大 KV 序列长度。

**返回**

- `int`：`out` 所需的最小列数。

**示例**

```python
out = torch.empty(
    (B * next_n, flashinfer.padded_seq_len(max_seq_len)), dtype=torch.float32, device="cuda"
)
logits = flashinfer.fp8_paged_mqa_logits(..., out=out)
```

---

### `precompile_paged_mqa_logits`

```python
precompile_paged_mqa_logits(
    device: Optional[torch.device] = None,
    variants: Tuple[str, ...] = ("fp8", "fp4"),
    output_dtypes: Optional[Tuple[torch.dtype, ...]] = None,
    batch_sizes: Optional[Sequence[int]] = None,
) -> None
```

**功能**：为常见静态配置预编译分页 MQA 逻辑内核，填充磁盘上的 CuTe-DSL 内核缓存，使后续对
`fp8_paged_mqa_logits` 和 `fp4_paged_mqa_logits` 的调用跳过首次编译。在部署初始化或包构建步骤中
调用一次。

预热的集合恰为：fp8——`num_heads=64`、`head_dim=128`、`block_size` 在 `{32,64,128}`、`next_n`
1..4、fp32 尾声/累加器（12 个内核）；fp4——`num_heads=64`、`head_dim=128`、`block_size` 在
`{32,64,128}`、`next_n` 1..4、fp32 尾声（每个输出 dtype 12 个内核）。其他情况（fp16 尾声、其他
头几何）仍在首次使用时编译。

**参数**

- `device`（`Optional[torch.device]`，默认 `None`）：目标 CUDA 设备。默认为当前 CUDA 设备。
- `variants`（`Tuple[str, ...]`，默认 `("fp8", "fp4")`）：要构建的精度。部署通常只运行一种索引器
  精度，例如传 `("fp8",)` 以避免编译永不被调用的内核。
- `output_dtypes`（`Optional[Tuple[torch.dtype, ...]]`，默认 `None`）：要预热的输出 dtype。
  `output_dtype` 是编译缓存键的一部分，未预热的 dtype 仍在首次使用时编译。默认每个变体的常见
  集合：FP8 为 float32，FP4 为 bfloat16 和 float32。
- `batch_sizes`（`Optional[Sequence[int]]`，默认 `None`）：要预热其 GPU 调度内核的批大小（调用方
  单位）。调度内核按调度器行数的 32 行桶特化。默认 `None` 表示不预热任何调度桶——为已知批大小
  集合捕获 CUDA Graph 的部署应传入它们。

**返回**

- `None`。

**注意事项**

- 若未安装 `nvidia-cutlass-dsl`，会发出 `RuntimeWarning` 并且不做任何预编译，首次 fp8/fp4 调用将
  失败。
- 传入未知 `variants` 或变体不支持的 `output_dtypes` 会抛出 `ValueError`。

**示例**

```python
import flashinfer
flashinfer.precompile_paged_mqa_logits(device="cuda", variants=("fp8",), batch_sizes=[1, 8, 32])
```

## 相关链接 / 参考

- 上游 `docs/api/attn_scores.rst`
- 源码 `flashinfer/attn_scores/attn_scores.py`

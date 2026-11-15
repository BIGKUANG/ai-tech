# Cake 采样（`flashinfer.cake_sampling`）

> 源码：`flashinfer/cake_sampling.py`
> 上游文档：`docs/api/cake_sampling.rst`
> 适用版本：FlashInfer 0.7.2

`cake_sampling` 是面向 Hopper 及更新 GPU（计算能力 9.x、10.x、11.x、12.x）的
**thread-block-cluster（线程块簇）**实现的 "先 top-k 再 top-p" 概率采样。它已在 H100（9.0）、B200（10.0）、
B300 / GB300（10.3）与 Rubin R200（10.7）上测量；11.x 与 12.x 是编译目标，尚未在硬件上运行。它把
`flashinfer.sampling.top_k_top_p_sampling_from_probs`（`filter_apply_order="top_k_first"`）的三个阶段
融合为每次调用最多两个内核：

1. 一个 thread-block-cluster radix select，写入精确的逐行 top-k slab（`[batch, 1024]` 的值与索引，外加
   逐行 count）：一次 11-bit cluster pass 通过分布式共享内存归约 2048 桶直方图，被选中桶的条目以 64-bit
   `(key, ~index)` 复合键收集进每个 CTA，其余 key 位与边界并列（tie）用本地 pass 解决（桶溢出收集容量的行
   走精确的三趟 cluster 路径）；
2. 一个 programmatic-dependent-launch 的稀疏 top-p 内核，用单个复合键 bitonic 网络对 slab 前缀排序，
   保留最短的、其独占质量（exclusive mass）低于 `top_p` 乘 top-k 质量的前缀，重归一化，并按
   `curand_init(seed, row, offset)` 通过逆 CDF 为每行抽取一个 token。

## 启动形式（Launch forms）

下列每种形式都产生 **位级相同** 的输出；选择只影响时序，由 host 针对
`(device, batch, vocab, top_k_max)` 决策一次（该决策被 memoise，故路由检查与 launch 共享它）。

- **融合双 warp 尾部**（`top_k_max <= 64`，manifest 中的 `fused_tail_kcap`）：阶段 2/3 在阶段 1 内核内、
  于 cluster 第一个 CTA 的两个 warp 上运行，仅一次 launch。dispatcher 能为这类 top-k 选择的每个变体都带
  这个尾部（manifest `fused_tail`）；但 `(8, 48)` resident（一个大 k 的 pick）不带。
- **整 CTA 尾部**（`64 < top_k_max <= 1024`，`fused_block_tail_kcap`；launch flag bit 3）：阶段 2/3 在
  变体的一个单独 build（manifest `fused_block_tail`，符号后缀 `_bt`）中于整个第一个 CTA（512 线程，每个
  两个 slab 条目）上运行。仅在计算能力 9.0、10.0、10.3 上、单波（one-wave）网格、cluster-8 streaming 变体
  且 `top_k_max > 768` 时采用（10.3 上最高 `vocab = 196608`）；其他所有大 k launch 走链式形式。这类 launch
  中 k 不超过 64 的行走双 warp 尾部。
- **两趟链式**（其余全部）：阶段 2/3 内核通过 PDL 链在阶段 1 之后。阶段 1 仅当 batch 能放入其最后一波留下
  的空闲 SM 时才提前发出 `griddepcontrol.launch_dependents`（bit 1）；streaming 变体在计算能力 10.x 上于
  第一趟前发信号（bit 2），在 Hopper 上于其 filter pass 之后发信号。
- **构建选择**（host 侧；内核看不到这些位）：bit 4 选择 streaming 变体的 coarse-sample build
  （`_cs`：被采样的第一趟读取行的 1/4 改为 1/8），bit 6 为其 speculative-sample build（`_sp`：第一个寄存
  器 chunk 兼作样本），bit 7 为任一 sample build 的 slab-tail 形式（`_cs_lb` / `_sp_lb`），bit 8 为默认
  build 的 pushed-coarse-sums 形式（`_lg`），bit 9 为多 CTA stream 的 leader-push exchange 形式
  （`_lp` / `_cs_lp` / `_sp_lp`，会取代 bit 7 与 bit 8），bit 10 为 leader-push sample build 的 CTA-local
  select 形式（`_cs_lp_l1` / `_sp_lp_l1`），bit 11 为整 CTA 尾部 build 的整数测试形式（`_bt_tia`），bit 5
  为双 warp 尾部之上 cluster-8 stream 的 row-span filter arm。每个 build 只在测量到更快的计算能力、cluster
  尺寸与行长上采用；策略常量位于 `flashinfer.cake_sampling` 中。每个多 CTA streaming build 内部，leader 的
  radix pass 会在被选中桶的计数等于剩余 rank 时精确提前停止（无 host flag；结果与完整 pass 序列位级相同）。
- **阶段 2/3 静态形式**：阶段 2/3 内核存在三种仅在指令选择上不同的形式——基形式（f64 top-p 测试，max/min
  bitonic 交换）、将 bitonic 交换实现为一次 64-bit 比较与选择的 `one_cmp_select`（计算能力 9.0、10.0），
  以及在前者之上加入整数 top-p 测试的 `int_tests`（10.3）。其他设备运行基形式。manifest 为每种形式列出每个
  slab 及其 `features`、`variant_flags` 和触发调度的 `capabilities`，`stage23_variant_flags` 返回设备的
  形式，binding 按 `(threads, items, variant_flags)` 选择内核。

## 调度与回退（Dispatch and fallback）

冻结内核无法服务的请求会调度到 `flashinfer.sampling` 的 `top_k_first` 路由（语义相同、generator 前进相同）：
`probs` 不是行连续的（`stride(0) == vocab`）CUDA `float32` 张量、无 `top_k`、`top_k` 不在 `[1, vocab)` 内或
超过 slab（1024）、设备不在 build 目标内、或词表无冻结变体覆盖。build 目标是 FlashInfer 的
`CompilationContext` 目标（设置了 `FLASHINFER_CUDA_ARCH_LIST` 时用它，否则用可见设备的计算能力），限定在
受支持的主版本 9-12；架构不在其中的设备走 `top_k_first` 路由而非在 launch 时失败。streaming 变体服务任何
不超过 2^21 的词表；动态共享内存超过设备 opt-in 限制的冻结变体不是调度候选，故在 12.x（99 KB）上
streaming 变体退出，词表超过寄存器驻留容量（196608）的请求在那里走 `top_k_first` 路由。
`cake_sampling_route` 在不 launch 的情况下报告决策。当 `top_k` 是张量且未给 `top_k_max` 时，host 读取
`top_k.max()`（一次设备同步）。

阶段 1 变体由代价模型逐次调用选择，其单波 CTA 容量表与常量以设备 SM 数为键（B200 / B300 为 148，H100 为
132，Rubin R200 为 212；其他设备用最近的实测表）。在 212-SM 表上，若一个小 k（`top_k_max` 至多 64 或未知）
的 ept-32 streaming pick 其网格与 cluster-8 网格都放入一波，会被重选为 cluster-8 ept-16 stream（R200 上
batch <= 16 时实测快 8-15%；多波网格、大 k 与其他表保留排名结果）。在 132-SM 表上，某个小 k 的 cluster-2
ept-32 pick（行的单 CTA chunk 数至少 16，V262144）从 batch 64 起且两网格都放入一波时，会被重选为 cluster-1
ept-32 stream（H100 上实测快 1.4-1.9%）。`renorm_out` 与 `workspace` 可暴露某次调用的已排序 slab。

## 源码产物（Source product）

冻结源码位于 `csrc/cake_sampling/generated/`：一个 translation unit（`cake_sampling_kernels.cu`，按顺序
include `cake_sampling_kernels_part<N>.cuh` body 文件；每个文件保持在仓库 5 MiB 限制内）加上
`manifest.json`，后者记录每个冻结变体的 launch 资源与 build flag、各阶段 2/3 形式被调度的计算能力、slab
几何与文件名。`flashinfer.jit.cake_sampling` 将 manifest 渲染为 binding 的变体表，并把该 unit 编译为单个
fatbin，每个 build 目标一个 `-gencode`。模块名由内容派生（`cake_sampling_` + 对冻结源码部分、manifest、
binding 头与渲染后 binding 密封的 20 位十六进制），每进程计算一次，因此另一个 bundle 版本的 JIT build 或
已安装 AOT 产物绝不会被本版本加载。

## 正确性契约（Correctness contract）

路由受容差级契约约束（内核可能在版本间改变位；精度级别与选择精确性不可变）：

- **Top-k 集合**：对每一行与每种形状，等于 float64 排序参考，包括并列规则（key 上并列由更小索引打破：
  64-bit `(key, ~index)` 复合键是全序，故精确选择都得到相同集合与相同 slab 顺序）。
- **Top-p 截断**：等于 float64 参考，除非精确累积质量落在 `top_p` 乘 top-k 质量的 `eps = 1e-6` 内，此时截断
  可能相差一个元素（`tests/utils/test_cake_sampling.py` 在该边界上放置行并检查两侧）。
- **采样**：始终落在精确支撑集（保留前缀）内；对相同输入、`philox_seed`、`philox_offset` 运行时确定
  （含 CUDA-graph 重放、并发流与重复调用）；多 seed 的下一 token 直方图处于按行类别的精确分布的 99% 二项带内。
  未显式给出 Philox 参数而被捕获进 CUDA graph 的调用绝不在捕获内读取 generator（PyTorch 会将其注册进图并在
  每次重放触发两个 `FillFunctor` 内核，GB300 上每次重放 +46 us）；它使用 generator 的初始 seed 加上在捕获间
  不同的 host 侧 offset。
- **重归一化 slab**（`renorm_out`）：相对 float64 参考 fp32 `rtol 1e-6`、`atol 1e-7`。key 保持精确 fp32 位
  模式；每次累加至少 fp32（任何地方都无 bf16 / fp16）；不放宽容差以接纳内核改动。
- 变体的每个阶段 1 build（`_bt`、`_cs`、`_sp`、`_cs_lb`、`_sp_lb`、`_lg`、`_lp`、`_cs_lp`、`_sp_lp`）与
  每个阶段 2/3 形式，此外还在每个测试行上与默认 build 位级相同作为门控。改变位的改动必须记录哪些行可能不同
  （并列、eps 边界）及其原因。
- 精确回退保持在内核侧：溢出收集容量的候选列表走三趟 cluster 路径，候选少于 k 的行被整体保留；超过 2^21 的
  词表走参考 `top_k_first` 路由。

## Benchmark

```bash
python benchmarks/bench_cake_sampling.py --cupti --skip-joint \
    --batches 1,2,4,8,16,32,64,128 \
    --vocabs 32768,128256,151936,262144 --top-k {10,50,1000} [--cuda-graph]
```

分别对 `top_k_first` 路由与本路由计时（使用 `flashinfer.testing.bench_gpu_time`，CUPTI，冷 L2）。

## 导入

```python
from flashinfer.cake_sampling import (
    top_k_top_p_sampling_from_probs,
    top_k_probs_to_slab,
    cake_sampling_route,
    choose_stage1,
    choose_stage23,
    stage23_variant_flags,
)
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `top_k_top_p_sampling_from_probs` | 函数 | 融合 top-k-then-top-p 概率采样（cluster radix 流水线）。 |
| `top_k_probs_to_slab` | 函数 | 仅阶段 1：精确逐行 top-k 写入 `[batch, 1024]` slab。 |
| `cake_sampling_route` | 函数 | 报告请求走 pipeline 还是 fallback 路由。 |
| `choose_stage1` | 函数 | 返回阶段 1 的 `(cluster, ept, stream)` 选择。 |
| `choose_stage23` | 函数 | 返回能容纳 `top_k_max` 的最小冻结 `(threads, items)` slab。 |
| `stage23_variant_flags` | 函数 | 返回设备上被调度的阶段 2/3 静态特征位掩码。 |

## API 详情

### `top_k_top_p_sampling_from_probs`

```python
top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    renorm_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    enable_pdl: bool = True,
) -> torch.Tensor
```

**功能**：融合 top-k-then-top-p 概率采样（thread-block-cluster radix 流水线）。

**参数**

- `probs`（`torch.Tensor`）：`float32 [batch, vocab]` 概率，连续。
- `top_k`（`Union[int, torch.Tensor]`）：每行保留的候选数（`int` 或 `int32 [batch]`），`1 <= k <= 1024`。
- `top_p`（`Union[float, torch.Tensor]`）：nucleus 阈值，落在 `(0, 1]`（`float` 或 `float32 [batch]`）。
- `top_k_max`（`Optional[int]`，默认 `None`）：`top_k` 为张量时的上界（避免一次设备同步）。
- `generator`（`Optional[torch.Generator]`）：Philox seed/offset 来源（省略时用默认 CUDA generator），前进
  方式与 `flashinfer.sampling.top_k_top_p_sampling_from_probs` 一致。CUDA graph 捕获期间 generator 不被读取
  或推进（否则会将其注册进图并给每次重放增加两个 fill 内核）；捕获的 launch 使用 generator 的初始 seed 加
  在捕获间不同的 host 侧 offset。
- `philox_seed`、`philox_offset`（`Optional[int]`）：显式 Philox 参数（须同时提供）；此时不触碰 `generator`。
- `out`（`Optional[torch.Tensor]`）：`int32 [batch]` 输出缓冲（省略时分配）。
- `renorm_out`（`Optional[torch.Tensor]`）：可选的 `float32 [batch, 1024]` 缓冲，接收按已排序 slab 顺序
  （概率降序、索引升序；被丢弃条目为零；每行仅写入前 `count` 个条目）的重归一化保留概率。仅 pipeline 路由
  提供。
- `workspace`（`Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]`）：阶段 1 slab 缓冲
  `(values float32 [batch, 1024], indices int32 [batch, 1024], counts int32 [batch])`。给定时，本次调用的已
  排序 top-k slab 留在其中（与 `renorm_out` 对齐）；否则使用按形状缓存的 workspace。
- `enable_pdl`（`bool`，默认 `True`）：以 PDL 启动阶段 2/3。

**返回**

- `samples`（`torch.Tensor`）：`int32 [batch]` 采样得到的 token id。

**注意事项**

- `philox_seed` 与 `philox_offset` 必须成对给出。
- 若路由不是 `pipeline`（即回退），会转调 `flashinfer.sampling.top_k_top_p_sampling_from_probs`，且
  `filter_apply_order="top_k_first"`、`deterministic=True`，并复用同一 generator/seed/offset；此时传入
  `renorm_out` 会抛出 `ValueError`。

---

### `top_k_probs_to_slab`

```python
top_k_probs_to_slab(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    out_vals: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
    out_count: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**功能**：仅阶段 1：精确的逐行 top-k 写入 `[batch, 1024]` slab。slab 保存 `lexsort(-prob, index)` 的前 `k`
个条目（NaN/负概率被清洗为 `+0`）。其布局是确定性的（输入的纯函数）但 **未排序**；超过 `count` 的条目未定义。
调度条件同 `top_k_top_p_sampling_from_probs`；冻结内核无法服务的请求会抛出带 `cake_sampling_route` 原因的
`ValueError`。

**参数**

- `probs`（`torch.Tensor`）：`float32 [batch, vocab]` 概率，连续。
- `top_k`（`Union[int, torch.Tensor]`）：每行保留的候选数（`int` 或 `int32 [batch]`），`1 <= k <= 1024`。
- `top_k_max`（`Optional[int]`，默认 `None`）：`top_k` 为张量时的上界（避免一次设备同步）。
- `out_vals`（`Optional[torch.Tensor]`）：`float32 [batch, 1024]` slab 值缓冲（省略时分配）。
- `out_idx`（`Optional[torch.Tensor]`）：`int32 [batch, 1024]` slab 索引缓冲（省略时分配）。
- `out_count`（`Optional[torch.Tensor]`）：`int32 [batch]` 逐行保留计数缓冲（省略时分配）。

**返回**

- `values, indices, counts`：slab `(values [batch, 1024], indices [batch, 1024], counts [batch])`。

---

### `cake_sampling_route`

```python
cake_sampling_route(
    probs: torch.Tensor,
    top_k: Optional[Union[int, torch.Tensor]],
    top_k_max: Optional[int] = None,
) -> str
```

**功能**：当冻结内核能服务该请求时返回 `"pipeline"`，否则返回 `"fallback:<reason>"`。`top_k_max` 可在
`top_k` 为张量时避免一次设备同步。

**参数**

- `probs`（`torch.Tensor`）：待采样的概率。
- `top_k`（`Optional[Union[int, torch.Tensor]]`）：top-k 阈值。
- `top_k_max`（`Optional[int]`，默认 `None`）：`top_k` 为张量时的上界。

**返回**

- `str`：`"pipeline"` 或 `"fallback:<reason>"`。可能的 reason 包括 `device`、`dtype`、`layout`、`arch`、
  `no_top_k`、`top_k_disabled`、`top_k_gt_slab`、`vocab_too_large`。

**注意事项**

- 该函数只报告决策，不执行 launch。

---

### `choose_stage1`

```python
choose_stage1(
    batch: int,
    vocab: int,
    sm_count: Optional[int] = None,
    smem_limit: Optional[int] = None,
    top_k_max: Optional[int] = None,
    two_launch: Optional[bool] = None,
) -> tuple[int, int, bool]
```

**功能**：返回 `batch` 行、每行 `vocab` 个条目（最大 top-k 为 `top_k_max`）时的 `(cluster, ept, stream)`
阶段 1 选择。设备相关默认值在此解析，排名本身按解析后的参数元组 memoise（`_choose_stage1_resolved`）：被服务
的路由每次调用询问两次，而 round-7 融合尾部项使排名代价翻倍，host-bound 的双 launch 链会在其 launch 间隙
承担这一开销。

寄存器驻留候选的选择顺序：波数最少，其次寄存器 chunk 至少 16 个条目，再取较大 cluster。该驻留选择会与每个
streaming 变体通过设备波表的拟合代价模型比较（`_STAGE1_COST_BY_SM_COUNT`：驻留按
`waves * (base + per_ept * ept)`，streaming 按
`waves * (base + chunk * chunks + cluster_cta * (cluster - 1))`，两者都加上
`launch_cta * ctas / wave_ctas[1]`）；驻留变体胜出并列，streaming 并列时偏好更小 cluster。波表与常量由
`sm_count` 选择（省略时为当前设备的 SM 数；B200 148、H100 132、Rubin R200 212 为实测，其他取最近）。
`top_k_max` 超过 `_STAGE1_LARGE_K` 时加上该表的大 k streaming 项（`None` 按小 top-k 排名）；`two_launch`
表示这类 launch 是否单独走阶段 2/3 内核（`None` 时对当前设备用 `_launch_is_two_kernels`，即其整 CTA 尾部
策略）并选择双 launch 项。动态共享内存需求超过 `smem_limit`（省略时为当前设备 opt-in 限制）的变体不是候选。

**参数**

- `batch`（`int`）：行数。
- `vocab`（`int`）：每行条目数。
- `sm_count`（`Optional[int]`，默认 `None`）：设备 SM 数。
- `smem_limit`（`Optional[int]`，默认 `None`）：动态共享内存上限。
- `top_k_max`（`Optional[int]`，默认 `None`）：最大 top-k。
- `two_launch`（`Optional[bool]`，默认 `None`）：是否采用双 launch。

**返回**

- `tuple[int, int, bool]`：`(cluster, ept, stream)`。

---

### `choose_stage23`

```python
choose_stage23(top_k_max: int) -> tuple[int, int]
```

**功能**：返回能容纳 `top_k_max` 个条目的最小冻结 `(threads, items)` slab。

**参数**

- `top_k_max`（`int`）：最大 top-k。

**返回**

- `tuple[int, int]`：`(threads, items)`。

**注意事项**

- 若 `top_k_max` 超过冻结阶段 2/3 容量，抛出 `ValueError`。
- 该函数被 `functools.cache` 缓存。

---

### `stage23_variant_flags`

```python
stage23_variant_flags(device: torch.device | int | None = None) -> int
```

**功能**：返回 `device` 上被调度的阶段 2/3 静态特征位掩码（`STAGE23_FEATURE_BITS`）；`0` 表示基形式
（f64 测试，max/min 交换），即每个未列出架构所运行的形式。

**参数**

- `device`（`torch.device | int | None`，默认 `None`）：目标设备；为 `int` 时视为设备索引，为 `None` 或
  设备无索引时使用 `torch.cuda.current_device()`。

**返回**

- `int`：特征位掩码。按设备索引缓存，因为 eager 的 k > 64 路径每次 launch 都会调用它。

## 相关链接 / 参考

- 上游文档：`docs/api/cake_sampling.rst`
- 回退实现：`flashinfer.sampling.top_k_top_p_sampling_from_probs`
- 基准脚本：`benchmarks/bench_cake_sampling.py`
- 测试：`tests/utils/test_cake_sampling.py`

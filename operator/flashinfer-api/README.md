# FlashInfer API 参考文档

本目录是 `flashinfer` 仓库中 Python 包（`flashinfer`）的 API 参考文档。
文档按模块拆分，每个文件介绍一组相关的公共 API，包含函数/类签名、参数说明、返回值、
注意事项和示例。内容由上游 Sphinx 源文档（`flashinfer/docs/api/*.rst`）与源码
（`flashinfer/**/*.py`）整理而成。

> 源码路径：`/data/home/shendakuang/workspace/framework/flashinfer/flashinfer`
> 版本：FlashInfer **0.7.2**（`version.txt`）

## 文档索引

### 注意力（Attention）

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-attention.md](./api-attention.md) | 单/批量 decode、单/批量 prefill、BatchAttention、MLA、XQA、SM120 NVFP4 注意力、DCP 推测解码 | `flashinfer.decode`, `flashinfer.prefill`, `flashinfer.attention`, `flashinfer.mla`, `flashinfer.xqa` |
| [api-attn-scores.md](./api-attn-scores.md) | 分页 MQA logits / 注意力分数 | `flashinfer.attn_scores` |
| [api-page.md](./api-page.md) | 分页 KV-Cache 的追加、量化写入与索引工具 | `flashinfer.page` |
| [api-cascade.md](./api-cascade.md) | 级联注意力（共享前缀）与状态合并 | `flashinfer.cascade` |
| [api-pod.md](./api-pod.md) | POD（Prefill-On-Decode）注意力 | `flashinfer.pod` |
| [api-sparse.md](./api-sparse.md) | 块稀疏 / 变长块稀疏注意力 | `flashinfer.sparse`, `flashinfer.cake_vsa`, `flashinfer.msa_ops` |
| [api-cake-fmha.md](./api-cake-fmha.md) | Cake FMHA context / decode、请求序 paged decode 计划 | `flashinfer.cake_fmha`, `flashinfer.cake_fmha_request_ordered` |

### 量化与 GEMM

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-quantization.md](./api-quantization.md) | NVFP4 / MXFP4 / MXFP8 量化、4over6 配方、packbits、KV Cache 量化 | `flashinfer.quantization` |
| [api-fp4-quantization.md](./api-fp4-quantization.md) | FP4 量化（向后兼容垫片） | `flashinfer.fp4_quantization` |
| [api-gemm.md](./api-gemm.md) | BF16/FP8/FP4/MXFP8 GEMM、SVDQuant、SegmentGEMMWrapper | `flashinfer.gemm` |
| [api-grouped-mm.md](./api-grouped-mm.md) | 分组矩阵乘（BF16/FP8/MXFP8/FP4） | `flashinfer.grouped_mm` |
| [api-concat-ops.md](./api-concat-ops.md) | 拼接 / 打包算子、MLA KV 打包量化 | `flashinfer.concat_ops`, `flashinfer.mla_kv_pack` |
| [api-diffusion-ops.md](./api-diffusion-ops.md) | 扩散 Transformer（DiT）融合算子 | `flashinfer.diffusion_ops` |

### MoE

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-fused-moe.md](./api-fused-moe.md) | cutlass / cuDNN / TRT-LLM / CuTe-DSL / PrimTS 融合 MoE | `flashinfer.fused_moe` |
| [api-cake-moe-grouped-gemm.md](./api-cake-moe-grouped-gemm.md) | Cake MoE 分组 GEMM | `flashinfer.experimental.cake_moe_grouped_gemm` |
| [api-cake-mxfp4-situ-moe.md](./api-cake-mxfp4-situ-moe.md) | Cake MXFP4 × MXFP8 SiTU 路由 MoE | `flashinfer.fused_moe.cake` / `cake_cute` |
| [api-mhc.md](./api-mhc.md) | 多头残差组合（mHC） | `flashinfer.mhc` |

### 归一化 / 激活 / 位置编码 / 采样

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-norm.md](./api-norm.md) | RMSNorm / LayerNorm / 残差融合 / 量化变体 / 训练用 RMSNorm | `flashinfer.norm`, `flashinfer.cake_rmsnorm_train` |
| [api-activation.md](./api-activation.md) | up/gate 层激活（SiLU/GELU and Mul） | `flashinfer.activation` |
| [api-rope.md](./api-rope.md) | 旋转位置编码（RoPE）及 FP8 量化变体 | `flashinfer.rope` |
| [api-sampling.md](./api-sampling.md) | Top-K/Top-P/Min-P 采样、softmax、投机采样 | `flashinfer.sampling` |
| [api-cake-sampling.md](./api-cake-sampling.md) | Cake 采样（SM120 Blackwell） | `flashinfer.cake_sampling` |
| [api-topk.md](./api-topk.md) | Top-K 选择与 page-table / ragged 变换 | `flashinfer.topk`, `flashinfer.topk_varlen` |
| [api-logits-processor.md](./api-logits-processor.md) | logits 后处理（过滤、惩罚等） | `flashinfer.logits_processor` |

### 线性 / 循环序列模型

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-gdn-decode.md](./api-gdn-decode.md) | Gated Delta-Rule 解码 | `flashinfer.gdn_decode` |
| [api-gdn-fused-decode.md](./api-gdn-fused-decode.md) | 融合 GDN 解码步 | `flashinfer.gdn_kernels.experimental` |
| [api-gdn-prefill.md](./api-gdn-prefill.md) | Gated Delta-Rule 预填充 | `flashinfer.gdn_prefill` |
| [api-gdn2-prefill.md](./api-gdn2-prefill.md) | Gated Delta-Rule 2 预填充 | `flashinfer.gdn2_prefill` |
| [api-gdp-prefill.md](./api-gdp-prefill.md) | Gated DeltaProduct 预填充 | `flashinfer.gdp_prefill` |
| [api-kda.md](./api-kda.md) | Kimi Delta Attention（KDA）prefill / decode | `flashinfer.kda`, `flashinfer.kda_prefill`, `flashinfer.kda_decode` |
| [api-mamba.md](./api-mamba.md) | Mamba / Mamba-2 状态空间模型 kernel | `flashinfer.mamba` |

### 基础设施

| 文档 | 内容 | 对应模块 / 源码 |
| --- | --- | --- |
| [api-comm.md](./api-comm.md) | AllReduce / AllGather / All-to-All、MNNVL、PCIe IPC、Ulysses、MoE 通信 | `flashinfer.comm` |
| [api-cudnn.md](./api-cudnn.md) | cuDNN 后端（注意力 / MoE grouped GEMM） | `flashinfer.cudnn` |
| [api-cute-dsl.md](./api-cute-dsl.md) | CuTe-DSL 高级接口与包装器 | `flashinfer.cute_dsl` |
| [api-testing.md](./api-testing.md) | 基准与性能分析工具（FLOPs/TFLOPs/吞吐） | `flashinfer.testing` |
| [api-green-ctx.md](./api-green-ctx.md) | CUDA 绿色上下文（SM 分区） | `flashinfer.green_ctx` |

## 安装

```bash
# 从 PyPI 安装（首次使用时编译 / 下载 kernel）
pip install flashinfer-python

# CUDA 13 变体
pip install flashinfer-python[cu13]
```

`flashinfer-python` 是核心包，会在首次使用时编译或下载 kernel；此外还有
`flashinfer-cubin`（预编译 kernel 二进制）与 `flashinfer-jit-cache`（JIT 缓存）等可选包。
更多安装方式（源码、nightly）见仓库 `README.md`。

## 通用约定

- **张量设备**：所有算子运行在 CUDA 设备上，输入张量需位于同一 GPU。
- **数据类型**：注意力/GEMM 常见输入为 `torch.float16`、`torch.bfloat16`；量化算子支持
  `fp8`（E4M3/E5M2）、`fp4`（E2M1，含 NVFP4 / MXFP4）、`mxfp8` 等。
- **分页 KV-Cache**：`paged_kv_cache` 形状通常为 `(num_pages, 2, page_size, num_kv_heads, head_dim)`，
  页面布局与 `page_size` 需与 kernel 契约一致；`block_tables` / `indices` 描述逻辑序列到物理页的映射。
- **变长批处理**：使用 `indptr`（累积长度，`int32`）描述每个请求的 token/序列区间。
- **包装器（Wrapper）**：多数批量算子提供 `plan(...)` / `begin_forward(...)` 预先规划，
  `forward(...)` / `run(...)` 执行；规划结果可缓存，配合 CUDA Graph 捕获使用。
- **`@flashinfer_api`**：绝大多数公开函数经此装饰，提供统一的日志、trace 与自定义算子注册行为。
- **后端选择**：许多模块支持 `backend=` 参数（如 `"auto"`、`"cutlass"`、`"trtllm"`、`"cudnn"`、
  `"cake"`、`"cute-dsl"`、`"cuda"` 等），在可用后端间自动或显式分派；具体可选值与架构门控见各页说明。

## 快速开始

```python
import torch
import flashinfer

# 采样
probs = torch.softmax(torch.randn(4, 5, device="cuda"), dim=-1)
samples = flashinfer.sampling.sampling_from_probs(probs)

# RMSNorm
x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
w = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
y = flashinfer.norm.rmsnorm(x, w)

# 激活
gate_up = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
out = flashinfer.activation.silu_and_mul(gate_up)
```

## 说明

- 每个模块页顶部标注源码路径与对应上游 `.rst`；API 清单以 `.rst` 的 `autosummary` 为准。
- 签名、默认值与文档字符串取自对应源码；若某条目在当前源码中未找到实现，页内会明确标注。
- 本目录为离线 Markdown 参考，官方在线文档见 <https://docs.flashinfer.ai/>。

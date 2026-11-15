# DeepGEMM API 参考文档

本目录是 `DeepGEMM` 仓库的 API 参考文档。DeepGEMM 是一个面向 NVIDIA GPU 的高性能张量核心算子库，把现代大语言模型的关键计算原语（FP8/FP4/BF16 GEMM、融合通信的 MoE、MQA indexer 打分、HyperConnection 等）统一到一个 CUDA 代码库中。所有内核在运行时通过 DeepJIT 编译，安装时无需 CUDA 编译。

文档按模块拆分，每个文件介绍一组相关的公共 API，包含函数签名、参数说明、返回值、注意事项和示例。

> 源码路径：`/data/home/shendakuang/workspace/framework/DeepGEMM`
> Python 公共入口：`deep_gemm/__init__.py`（C++ 扩展 `deep_gemm._C`，Python 封装在 `deep_gemm/`）
> 底层 C++ 注册：`csrc/python_api.cpp` 及 `csrc/apis/*.hpp`

## 文档索引

| 文档 | 内容 | 对应源码 |
| --- | --- | --- |
| [api-gemm.md](./api-gemm.md) | 稠密（非分组）GEMM：FP8/FP4 `fp8_fp4_gemm_*`、BF16 `bf16_gemm_*` 及别名 | `csrc/apis/gemm.hpp` |
| [api-gemm-grouped.md](./api-gemm-grouped.md) | M 分组（contiguous / masked）与 K 分组 GEMM | `csrc/apis/gemm.hpp` |
| [api-cublaslt.md](./api-cublaslt.md) | cuBLASLt 包装算子：`cublaslt_gemm_*`、NVFP4、`batched_syrk`、`batched_symm` | `csrc/apis/gemm.hpp` |
| [api-einsum.md](./api-einsum.md) | 硬编码 Einstein 求和：`einsum`、`fp8_einsum` | `csrc/apis/einsum.hpp` |
| [api-attention.md](./api-attention.md) | V3.2 indexer MQA logits（非分页 / 分页 / 稀疏 与元数据） | `csrc/apis/attention.hpp` |
| [api-hyperconnection.md](./api-hyperconnection.md) | HyperConnection 预归一化 TF32 GEMM | `csrc/apis/hyperconnection.hpp` |
| [api-mega-gate.md](./api-mega-gate.md) | 融合 BF16 gate GEMM 与 MoE top-k 路由 | `csrc/apis/mega_gate.hpp` |
| [api-mega-mhc.md](./api-mega-mhc.md) | Mega mHC：mHC 混合 + RMSNorm（可 FP8 输出） | `csrc/apis/mega_mhc.hpp` |
| [api-mega-moe.md](./api-mega-moe.md) | Mega MoE：融合 EP dispatch/combine、GEMM、SwiGLU | `csrc/apis/mega_moe.hpp`, `deep_gemm/mega/__init__.py` |
| [api-layout.md](./api-layout.md) | Scaling Factor 布局变换、TMA 对齐、M/K 对齐 | `csrc/apis/layout.hpp`, `deep_gemm/utils/layout.py` |
| [api-locality-domain.md](./api-locality-domain.md) | Locality Domain 内存分配与 `localize` | `csrc/apis/locality_domain.hpp`, `deep_gemm/locality_domain.py` |
| [api-epilogue.md](./api-epilogue.md) | GEMM epilogue 类（Alpha / FP8 量化 / BF16 随机舍入） | `csrc/apis/epilogue_class.hpp` |
| [api-config.md](./api-config.md) | 运行时配置：SM 数、TC 利用率、PDL、确定性、编译维度等 | `csrc/apis/config.hpp` |
| [api-utils.md](./api-utils.md) | 量化/数学/分布式工具、测试基准工具、Legacy Triton 内核 | `deep_gemm/utils/`, `deep_gemm/testing/`, `deep_gemm/legacy/` |
| [scaling-factor-format.md](./scaling-factor-format.md) | **专题**：FP8/FP4 Scaling Factor 格式规范 | `csrc/apis/layout.hpp`, `csrc/utils/layout.hpp` |

## 通用约定

DeepGEMM 使用 `D = C + A @ B` 的命名约定，函数名的后缀表示内存布局（NT 表示非转置 A、转置 B）：

- `_nt`：`D = C + A @ B.T`，即 `A[M,K]`、`B[N,K]`（K-major / 行主序）
- `_nn`：`D = C + A @ B`，即 `A[M,K]`、`B[K,N]`
- `_tn`：`D = C + A.T @ B`，即 `A[K,M]`、`B[K,N]`
- `_tt`：`D = C + A.T @ B.T`，即 `A[K,M]`、`B[N,K]`

其它约定：

- `d` 为输出张量，所有算子只支持 `d` 最后一维连续（row-major / N-major）。
- `c` 为可选累加输入；省略时等价于从零累加。`c` 与 `d` 为同一个张量时可原地累加，否则会先执行一次 `d.copy_(c)`。
- FP8/FP4 算子的 A、B 需要传入 `(data, scale_factor)` 元组；缩放因子布局见 [scaling-factor-format.md](./scaling-factor-format.md)。
- SM90（Hopper）仅支持 NT 布局；SM100（Blackwell）支持 NT/TN/NN/TT。
- SM90 缩放因子为 FP32；SM100 为打包的 UE8M0（每个 `int32` 打包 4 个 UE8M0）。
- `compiled_dims` 控制 JIT 编译时把哪些维度当作常量（`"nk"`、`"mn"` 或 `""`），影响编译特化与缓存键。
- 所有内核默认启用 JIT 缓存；首次调用会触发编译。

## 安装

```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# 建立必要的 include 链接并构建 C++ 扩展
./develop.sh

# 或安装
./install.sh
```

依赖：NVIDIA SM90/SM100 GPU、Python ≥ 3.8、CUDA Toolkit ≥ 12.9、PyTorch ≥ 2.3、CUTLASS ≥ 4.0、支持 C++20 `<format>` 的编译器。

## 快速开始（FP8 NT GEMM）

```python
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

m, n, k = 4096, 4096, 4096
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')

# 量化（SM100 使用打包 UE8M0，gran_k=128）
a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=True, use_packed_ue8m0=True)
b_fp8, sfb = per_token_cast_to_fp8(b, use_ue8m0=True, use_packed_ue8m0=True)
sfa = deep_gemm.transform_sf_into_required_layout(sfa, m, k, (1, 128))
sfb = deep_gemm.transform_sf_into_required_layout(sfb, n, k, (1, 128))

d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d, recipe=(1, 1, 128))
```

## 快速开始（BF16 GEMM）

```python
import torch
import deep_gemm

a = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
b = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
d = torch.empty((4096, 4096), dtype=torch.bfloat16, device='cuda')

deep_gemm.bf16_gemm_nt(a, b, d)
```

## 命名空间总览

`deep_gemm` 顶层导出以下内容（见 `deep_gemm/__init__.py`）：

- 配置：`set_num_sms` / `get_num_sms`、`set_tc_util` / `get_tc_util`、`set_pdl` / `get_pdl`、`use_deterministic_algorithms`、`set_ignore_compile_dims`、`set_block_size_multiple_of`
- GEMM：`fp8_fp4_gemm_{nt,nn,tn,tt}`、`m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous`、`m_grouped_fp8_fp4_gemm_nt_masked`、`fp8_gemm_*` / `fp4_gemm_nt`（别名）、`bf16_gemm_*`、`m_grouped_bf16_gemm_*`、`k_grouped_*_gemm_*_contiguous`
- cuBLASLt：`cublaslt_gemm_{nt,nn,tn,tt}`、`cublaslt_nvfp4_gemm_nt`、`batched_syrk`、`batched_symm`
- 注意力：`fp8_fp4_mqa_logits`、`fp8_fp4_paged_mqa_logits`、`fp8_fp4_sparse_mqa_logits`、`fp8_fp4_paged_sparse_mqa_logits` 及其元数据构造函数
- Mega 内核：`bf16_mega_gate`、`mega_mhc`、`fp8_fp4_mega_moe`、`bf16_mega_moe`、`SymmBuffer`、`get_symm_buffer_for_mega_moe`、`transform_weights_for_mega_moe`
- 其它：`fp8_einsum`、`einsum`、`tf32_hc_prenorm_gemm`、`transform_sf_into_required_layout`
- 子模块：`deep_gemm.epilogue`、`deep_gemm.locality_domain`、`deep_gemm.utils`、`deep_gemm.testing`、`deep_gemm.legacy`

## 环境变量

每个 `DG_JIT_*` 变量在未设置时会回退到对应的全局 `DJ_JIT_*` 变量。

| 变量 | 取值 | 说明 | 默认 |
| --- | --- | --- | --- |
| `DG_JIT_DEBUG` | 0/1 | 打开 JIT 调试（编译命令、PTXAS 输出、行号、PTX/SASS dump） | 0 |
| `DG_PRINT_CONFIGS` | 0/1 | 为每个 shape 打印选中的配置 | 0 |
| `DG_JIT_CACHE_DIR` | 字符串 | 编译缓存目录（可用 `:` 分隔多个路径，按前到后查找） | `$HOME/.dj` |
| `DG_JIT_NVCC_COMPILER` | 字符串 | NVCC 路径 | 自动探测 |
| `DG_JIT_CPP_STANDARD` | 整数 | C++ 标准版本 | 20 |
| `DG_JIT_PRINT_COMPILER_COMMAND` | 0/1 | 打印编译命令 | 0 |
| `DG_JIT_PTXAS_VERBOSE` | 0/1 | 输出详细 PTXAS 信息 | 0 |
| `DG_JIT_CHECK_NO_SPILLS` | 0/1 | 断言内核无寄存器溢出 | 0 |
| `DG_JIT_CHECK_NO_LOCAL_MEMORY` | 0/1 | 断言内核无 local memory 使用 | 0 |
| `DG_JIT_PRINT_LOAD_TIME` | 0/1 | 打印内核加载时间 | 0 |
| `DG_JIT_WITH_LINEINFO` | 0/1 | 嵌入源码行号信息 | 0 |
| `DG_JIT_DUMP_ASM` | 0/1 | dump PTX 与 SASS | 0 |
| `DG_JIT_DUMP_PTX` | 0/1 | dump PTX | 0 |
| `DG_JIT_DUMP_SASS` | 0/1 | dump SASS | 0 |
| `DG_COMM_KERNEL_DEBUG` | 0/1 | 每次 Mega MoE 调用前清零对称缓冲区（调试） | 0 |
| `DG_USE_NVIDIA_TOOLS` | 0/1 | 在外部 NVIDIA 工具下跳过内部 profiling | 0 |
| `DG_SKIP_CUDA_BUILD` | 0/1 | 安装时跳过 CUDA 扩展构建 | 0 |
| `DG_FORCE_BUILD` | 0/1 | 强制本地构建而不下载预编译 wheel | 0 |

## 参考

- 仓库 README：`README.md`
- Scaling factor 规范：[scaling-factor-format.md](./scaling-factor-format.md)
- 测试与示例：`tests/`

# ai_tech

GPU 体系结构、CUDA 运行时、LLM 推理框架与算子库 API 的中文学习资料与实测笔记。

所有文档均基于本地 **NVIDIA H20-3e（Hopper，sm_90）+ CUDA 13.0** 环境整理，代码示例可直接编译运行。

## 目录结构

```
.
├── arch/                          # GPU 体系结构
│   ├── gpu_arch_notes/            # 原书精读笔记与译文（含插图、配套代码）
│   │   ├── 通用图形处理器架构（中文全译本）.md/.html
│   │   ├── General-Purpose Graphics Processor Architecture (2018).pdf
│   │   ├── H20_共享存储器Bank冲突详解.md
│   │   ├── code/                  # 笔记配套 CUDA 微基准
│   │   └── figures/               # 书中插图
│   └── cuda_programming/          # CUDA 指南译本 + 编程实战 + 代码
├── runtime/
│   └── cuda-runtime-guide-zh.md   # CUDA Runtime 库详解（目录/架构/模块/用法/对比）
├── compile/
│   ├── learning_report.md         # 编译技术学习报告
│   └── learning_report_outline.md # 学习报告提纲
├── framework/
│   ├── sglang-intro.md            # SGLang 推理框架介绍（结构/架构/模块）
│   ├── sglang-intro-prompt.md     # 生成该文档的提示词
│   ├── vllm-intro.md              # vLLM 推理框架介绍
│   └── vllm-intro-prompt.md       # 生成该文档的提示词
└── operator/                      # 算子库 API 参考（离线中文文档）
    ├── flashinfer-api/            # FlashInfer 0.7.2 Python API 参考
    ├── flash-attention-api/       # FlashAttention 2.8.4（FA2/FA3/FA4）API 参考
    └── deepgemm-api/              # DeepGEMM 2.8.1 API 参考
```

## 内容简介

### arch — GPU 体系结构
- **中文全译本**（`gpu_arch_notes/`）：Aamodt 等《General-Purpose Graphics Processor Architectures》全文翻译。
- **gpu_arch_notes**：按章节拆解的精读笔记，含 SIMT 核心、存储系统、交叉研究等，并补充 Volta → Hopper 之后的进展；`code/` 为配套微基准。
- **Bank 冲突专题**：以 H20 实测数据讲解 bank 冲突的成因、测量与优化，附完整可运行程序。
- **cuda_programming**：CUDA Programming Guide 中文全译本，以及《CUDA 编程实战》（H20/sm_90，16 个可编译运行示例，含实测数据与指令级优化）。

### runtime — CUDA Runtime
以实际安装的 CUDA Toolkit 为基线，讲解 `libcudart` 的安装目录组织、公开头文件模块、API 用法与代码案例，并与 Driver API、OpenCL、HIP、SYCL 做对比。

### framework — LLM 推理框架
以 SGLang 为例，梳理代码结构与分层架构（入口 → 调度 → 执行 → 显存/缓存 → 并行与运行时），并逐模块介绍 RadixAttention、PD 分离、推测解码、约束解码等机制。

### operator — 算子库 API 参考
- **flashinfer-api**：FlashInfer **0.7.2** Python 包，按注意力、量化与 GEMM、MoE、归一化/激活/RoPE/采样、线性/循环序列模型、通信与后端等模块拆分，含签名、参数说明与示例。
- **flash-attention-api**：FlashAttention **2.8.4**，覆盖 FA2/FA3/FA4（CuTeDSL）注意力、变长与 KV Cache 专题、RoPE、归一化、融合 Linear/MLP、块稀疏注意力及训练组件。
- **deepgemm-api**：DeepGEMM **2.8.1**，覆盖 FP8/FP4/BF16 稠密与分组 GEMM、cuBLASLt、Einsum、V3.2 indexer、HyperConnection、Mega Gate/mHC/MoE 及 Scaling Factor 格式规范。

## 使用方式

阅读文档无需额外环境。运行 `arch/gpu_arch_notes/code` 下的示例需要 CUDA Toolkit：

```bash
nvcc -arch=sm_90 arch/gpu_arch_notes/code/props.cu -o props
./props
```

## 许可

文档内容仅供学习交流；原书译文与插图的版权归原作者及出版方所有。

# ai_tech

GPU 体系结构、CUDA 运行时与 LLM 推理框架的中文学习资料与实测笔记。

所有文档均基于本地 **NVIDIA H20-3e（Hopper，sm_90）+ CUDA 13.0** 环境整理，代码示例可直接编译运行。

## 目录结构

```
.
├── arch/                          # GPU 体系结构
│   ├── 通用图形处理器架构（中文全译本）.md/.html
│   ├── General-Purpose Graphics Processor Architecture (2018).pdf
│   ├── gpu_arch_notes/            # 原书 5 章精读笔记（解读 + 前沿补充 + H20 实测）
│   ├── H20_共享存储器Bank冲突详解.md
│   ├── figures/                   # 书中插图
│   └── cuda_examples/             # 微基准 CUDA 示例
├── runtime/
│   └── cuda-runtime-guide-zh.md   # CUDA Runtime 库详解（目录/架构/模块/用法/对比）
└── framework/
    ├── sglang-intro.md            # SGLang 推理框架介绍（结构/架构/模块）
    └── sglang-intro-prompt.md     # 生成该文档的提示词
```

## 内容简介

### arch — GPU 体系结构
- **中文全译本**：Aamodt 等《General-Purpose Graphics Processor Architectures》全文翻译。
- **gpu_arch_notes**：按章节拆解的精读笔记，含 SIMT 核心、存储系统、交叉研究等，并补充 Volta → Hopper 之后的进展。
- **Bank 冲突专题**：以 H20 实测数据讲解 bank 冲突的成因、测量与优化。
- **cuda_examples**：合并访问、占用率、分支发散、bank 冲突、带宽、SAXPY 等微基准。

### runtime — CUDA Runtime
以实际安装的 CUDA Toolkit 为基线，讲解 `libcudart` 的安装目录组织、公开头文件模块、API 用法与代码案例，并与 Driver API、OpenCL、HIP、SYCL 做对比。

### framework — LLM 推理框架
以 SGLang 为例，梳理代码结构与分层架构（入口 → 调度 → 执行 → 显存/缓存 → 并行与运行时），并逐模块介绍 RadixAttention、PD 分离、推测解码、约束解码等机制。

## 使用方式

阅读文档无需额外环境。运行 `arch/cuda_examples` 下的示例需要 CUDA Toolkit：

```bash
nvcc -arch=sm_90 arch/cuda_examples/bank_conflict_demo.cu -o bank_conflict_demo
./bank_conflict_demo
```

## 许可

文档内容仅供学习交流；原书译文与插图的版权归原作者及出版方所有。

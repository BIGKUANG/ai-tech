# 《General-Purpose Graphics Processor Architectures》精读笔记

本目录把 Tor M. Aamodt、Wilson Wai Lun Fung、Timothy G. Rogers 所著
*General-Purpose Graphics Processor Architectures*（Morgan & Claypool, 2018）拆成 5 章，
每章在忠实原书内容的基础上做了三件事：

1. **由浅入深的中文解读**：先建立直觉，再进入微架构细节；
2. **外部补充**：用现代 GPU（Volta → Ampere → Hopper/H20）的资料补足原书 2018 年之后的进展，并明确标注"延伸"；
3. **H20 实测**：在本地 NVIDIA H20-3e（sm_90）上运行微基准，给出真实数字。

> 原书是综述性著作，许多"基准体系结构"描述来自专利与白皮书，可能与实际产品有差异。本笔记保留这种不确定性，并在补充处注明出处类型。

> **图号约定**：本笔记自绘的 Mermaid 图使用 **`图 X-Y`**（章-序号）；引用原书插图使用 **`图 X.Y`**（原书编号）。

---

## 五篇文档

| # | 文档 | 对应原书 | 核心问题 |
|---|---|---|---|
| 1 | [01_引言与GPU硬件基础.md](01_引言与GPU硬件基础.md) | 第 1 章 | GPU 为什么存在？它在系统中和芯片内部长什么样？ |
| 2 | [02_编程模型与指令集体系结构.md](02_编程模型与指令集体系结构.md) | 第 2 章 | CUDA 线程如何变成 PTX/SASS？MIMD 外观如何落到 SIMD 硬件？ |
| 3 | [03_SIMT核心-指令与寄存器数据流.md](03_SIMT核心-指令与寄存器数据流.md) | 第 3 章 | 一个核心如何用一条 SIMD 流水线执行上千线程？分支发散怎么办？ |
| 4 | [04_存储系统.md](04_存储系统.md) | 第 4 章 | 从 load/store 到 DRAM 的完整路径；合并、bank 冲突、缓存、调度 |
| 5 | [05_交叉研究与前沿方向.md](05_交叉研究与前沿方向.md) | 第 5 章 | 线程调度、替代并行、事务内存、异构系统，以及现代 GPU 的演进 |

## 专题文档

| 文档 | 内容 |
|---|---|
| [H20 共享存储器 Bank 冲突详解](H20_共享存储器Bank冲突详解.md) | 以 H20 的 32 bank × 4B / 228KB 为实例，讲清 bank 映射、冲突度数、padding、XOR swizzle、广播，并附 11 张图和可复现的 CUDA 实测 |

## 建议阅读顺序

建议阅读顺序如 **图 0-1** 所示：

```mermaid
flowchart LR
    A["1. 引言<br/>建立全局观"] --> B["2. 编程模型<br/>软件如何映射"]
    B --> C["3. SIMT 核心<br/>指令与寄存器"]
    C --> D["4. 存储系统<br/>带宽与延迟"]
    D --> E["5. 交叉研究<br/>调度/TM/异构"]
    E -. "回看 3、4 章的研究方向" .-> C
```

**图 0-1**：建议阅读顺序

- **第一遍**：按 1→5 顺序读，重点看每章的"小结"和"思考题"。
- **第二遍**：带着问题读。例如"为什么 H20 的 occupancy 上限是 2048 线程/SM"→ 回第 3 章寄存器堆与第 5 章 TLP。
- **动手**：每章末尾都有可复现命令；先跑实验，再读对应小节，效果最好。

## 实测环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA H20-3e × 8 |
| Compute Capability | 9.0（Hopper） |
| SM 数 | 78 |
| 每 SM 线程上限 | 2048 |
| 每 SM 寄存器 | 65536 × 32-bit（256 KB） |
| 每 SM 共享存储器 | 228 KB |
| L2 | 60 MB |
| 显存位宽 | 6016 bit |
| CUDA | 13.0 |
| 驱动 | 590.44.01 |

## 实测结果速查

| 实验 | 关键结论 | 文档位置 |
|---|---|---|
| 峰值带宽（float4 拷贝） | 2 GB 数组达 **3915 GB/s**，接近 HBM3 标称 4 TB/s | 第 1、4 章 |
| 合并访存 | coalesced **1635 GB/s** → stride=32 **396 GB/s**（0.24×） | 第 4 章 |
| 共享存储器 bank 冲突 | 2-way **1.85×**、4-way **3.70×**、32-way **29.56×** | 第 4 章 |
| 分支发散 | 交替分支路径 **1.66×** 于无发散 | 第 3 章 |
| TLP/occupancy | 256→2048 线程/SM，吞吐 **1.76×** 后饱和 | 第 5 章 |
| PTX/SASS | sm_90 的 SASS 已使用统一寄存器 UR、内存描述符、控制位 | 第 2 章 |

## 示例代码

所有代码在 [`code/`](code/)，编译统一使用 `-arch=sm_90`：

```bash
cd code

# 设备参数
nvcc -arch=sm_90 props.cu -o props && ./props

# 峰值带宽
nvcc -O3 -arch=sm_90 --extended-lambda bandwidth.cu -o bandwidth && ./bandwidth

# 合并访存
nvcc -O3 -arch=sm_90 --extended-lambda coalescing.cu -o coalescing && ./coalescing

# 共享存储器 bank 冲突
nvcc -O3 -arch=sm_90 --extended-lambda bank_conflict.cu -o bank_conflict && ./bank_conflict

# 分支发散
nvcc -O3 -arch=sm_90 --extended-lambda divergence.cu -o divergence && ./divergence

# TLP / occupancy
nvcc -O3 -arch=sm_90 --extended-lambda occupancy.cu -o occupancy && ./occupancy

# PTX / SASS
nvcc -arch=sm_90 -ptx   saxpy.cu -o saxpy.ptx
nvcc -arch=sm_90 -cubin saxpy.cu -o saxpy.cubin
cuobjdump -sass saxpy.cubin
```

> 若在其他架构上复现，请替换 `sm_90`：Ampere `sm_80`、Ada `sm_89`、Blackwell `sm_100`/`sm_120`。不同架构的绝对数字会变化，但趋势一致。

## 原书插图

正文引用的原书插图位于 [`figures/`](figures/)，按原书编号命名（如 `Figure3.4.png`），便于与英文原书交叉对照。

## 术语对照

| 英文 | 本书译名 |
|---|---|
| warp / wavefront | 线程束 / 波前 |
| thread block / CTA | 线程块 / 协作线程数组 |
| SIMT / SIMD / MIMD | 单指令多线程 / 单指令多数据 / 多指令多数据 |
| register file | 寄存器堆 |
| warp scheduler | 线程束调度器 |
| operand collector | 操作数收集器 |
| scoreboard | 记分牌 |
| scratchpad memory | 便笺式存储器 |
| memory coalescing | 访存合并 |
| divergence / reconvergence | 发散 / 重汇聚 |
| occupancy | 占用率 |
| memory partition | 存储分区 |
| MSHR | 缺失状态处理寄存器 |

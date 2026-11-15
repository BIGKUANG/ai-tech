# CUDA 编程实战：从入门到高手（H20 / sm_90）

> 版本：2026-11　目标设备：NVIDIA H20-3e（Hopper，sm_90）　工具链：CUDA 13.0
>
> 配套代码：`code/`（16 个可编译示例，`make run` 一键运行）
>
> 本文以《CUDA 编程指南（中文全译本）》的概念为骨架，结合 H20 实测数据，按“能跑 → 跑对 → 跑快 → 看懂指令”的路径组织。所有数字都来自本机 `code/` 目录下的程序。

---

## 0. 先看全局：这本书怎么用

### 0.1 你将获得什么

学完并动手做完本文的 16 个实验，你应该能够：

- 独立写出正确、可维护的 CUDA 程序（向量加法、转置、归约、直方图、GEMM）；
- 用“带宽 / 延迟 / 占用率 / 算术强度”四个词解释一个内核快慢的原因；
- 定位并修复最常见的性能问题：未合并访存、bank 冲突、分支发散、原子争用、占用率不足；
- 读懂 PTX 与 SASS，知道一条 C++ 语句大致会变成哪些指令；
- 用 `nvcc`、`cuobjdump`、`nvdisasm`、`ncu`、`nsys` 做编译与性能分析；
- 知道从指令层还能怎么继续加速（向量化、FFMA/MUFU、异步拷贝、张量核心）。

### 0.2 学习路线（由浅入深）

| 阶段 | 章节 | 你会写/会用的东西 |
|---|---|---|
| 入门 | 1–3 | 编译、设备信息、向量加法、grid-stride、占用率 |
| 基础优化 | 4–7 | 合并访存、共享内存、bank 冲突、归约、原子 |
| 中级优化 | 8–10 | 分支发散、ILP、数学函数、内联 PTX |
| 系统级 | 11–13 | 流与重叠、异步拷贝、张量核心、性能分析 |
| 高手 | 14–16 | 指令级加速清单、陷阱、进阶路线 |

### 0.3 代码目录

```
code/
├── Makefile
├── run_all.sh
├── 01_hello.cu            设备信息 + 第一个内核
├── 02_vecadd.cu           向量加法（基础范式）
├── 03_grid_stride.cu      固定网格 + 循环
├── 04_coalescing.cu       合并访存 vs 跨步
├── 05_transpose.cu        矩阵转置：全局 vs 共享内存
├── 06_bank_conflict.cu    共享内存 bank 冲突扫描
├── 07_reduction.cu        归约：shared / warp shuffle / 两趟
├── 08_atomic.cu           直方图：全局原子 vs 共享原子
├── 09_occupancy.cu        占用率与 block size
├── 10_divergence.cu       分支发散
├── 11_streams.cu          多流重叠传输与计算
├── 12_async_copy.cu       异步拷贝 / 软件流水
├── 13_wmma_gemm.cu        张量核心 WMMA GEMM
├── 14_ilp.cu              指令级并行（多累加器）
├── 15_ptx_fastmath.cu     内联 PTX + 快速数学
└── 16_saxpy.cu            SAXPY 工程化写法
```

编译与运行：

```bash
cd code
make            # 编译全部，产物在 bin/
make run        # 编译并运行全部，结果写入 results/
./bin/02_vecadd # 单独运行
```

默认 `ARCH=sm_90`，可用 `make ARCH=sm_90` 显式指定。

---

## 1. 环境、编译与“看见指令”

### 1.1 H20 的关键参数

运行 `code/01_hello.cu` 得到的真实信息：

| 参数 | 值 |
|---|---|
| 设备名 | NVIDIA H20-3e |
| 计算能力 | 9.0（Hopper） |
| SM 数量 | 78 |
| 每 SM 最大线程数 | 2048 |
| 每线程块最大线程数 | 1024 |
| 每 SM 寄存器数 | 65536（32-bit） |
| 每 SM 共享内存（上限） | 233472 B（228 KB） |
| 每块共享内存（显式选择启用） | 232448 B |
| L2 缓存 | 62914560 B（60 MB） |
| warp 大小 | 32 |

记住三个数：**78 个 SM、每 SM 2048 线程、warp=32**。后文计算占用率、网格大小时会反复用到。

### 1.2 nvcc 的编译流水线

```text
.cu / .cuh
   │  (预处理 + 主机/设备代码分离)
   ├── 主机代码 ──> 主机编译器 (gcc/clang) ──> .o
   └── 设备代码 ──> cicc ──> PTX ──> ptxas ──> cubin ──> fatbin
                                                   │
                                            链接进可执行文件
```

- **PTX**：虚拟指令集（`compute_90`），前向兼容，运行时可由驱动 JIT 成目标架构的 SASS。
- **SASS**：真实 GPU 机器码（`sm_90`），与具体架构绑定。
- **cubin / fatbin**：SASS 或 PTX 的容器，可同时打包多个架构。

常用命令：

```bash
# 生成 PTX（看虚拟指令）
nvcc -O3 -arch=sm_90 -ptx 02_vecadd.cu -o 02_vecadd.ptx

# 生成 SASS（看真实机器码）
cuobjdump -sass bin/02_vecadd | less
# 或先把 cubin 抽出来
cuobjdump -xelf all bin/02_vecadd
nvdisasm -g 02_vecadd.cubin | less

# 看寄存器/共享内存用量
nvcc -O3 -arch=sm_90 -Xptxas -v -c 02_vecadd.cu -o /tmp/02.o
```

`02_vecadd` 的真实结果：

```text
ptxas info : Function properties for _Z6vecAddPKfS0_Pfi
ptxas info : Used 12 registers, used 0 barriers
```

### 1.3 架构目标：`sm_90` / `compute_90` / `sm_90a`

- `-arch=compute_90`：只生成 PTX，可被更新的 GPU JIT 执行（前向兼容）。
- `-arch=sm_90`：生成 SASS + 可选 PTX，直接在本机执行。
- `-arch=sm_90a`：启用 Hopper 的**架构特定特性**（如部分 TMA/张量核心高级指令），只能在 sm_90 上跑。
- 家族特定：`sm_100f` 等，面向 Blackwell 家族。本文不涉及。

对 H20 本地开发，`-arch=sm_90` 是最常用的选择。

### 1.4 常用编译选项（性能相关）

| 选项 | 作用 | 建议 |
|---|---|---|
| `-O3` | 最高优化 | 默认开启 |
| `-lineinfo` | 行号信息，供 ncu/nsys 关联源码 | 性能分析时开 |
| `-Xptxas -v` | 打印寄存器/共享内存/spill | 调优时开 |
| `-maxrregcount=N` | 限制每线程寄存器数 | 占用率不足且无 spill 时尝试 |
| `__launch_bounds__(T, B)` | 告知编译器块大小/每 SM 块数 | 稳定占用率 |
| `--use_fast_math` | 用近似数学函数、放宽 IEEE | 精度可接受时用 |
| `-rdc=true` (+`-lcudadevrt`) | 分离编译/动态并行 | 多文件设备函数时用 |
| `-dlto` | 链接时优化 | 分离编译且追求性能时用 |
| `--generate-line-info` | 同 `-lineinfo` | — |

### 1.5 错误检查：从第一天就养成

```cpp
#define CK(x) do { cudaError_t _ck = (x); if (_ck != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck)); exit(1); } } while (0)
```

内核启动本身不返回错误码，启动后要检查：

```cpp
kernel<<<grid, block>>>(...);
CK(cudaGetLastError());        // 捕获启动参数错误
CK(cudaDeviceSynchronize());   // 捕获执行期异步错误（调试时）
```

生产代码里尽量用 `cudaStreamSynchronize` / 事件同步代替全局 `cudaDeviceSynchronize`，避免无谓地等待所有流。

---

## 2. 第一个内核：线程、网格、索引

### 2.1 最小示例

`code/01_hello.cu`：

```cpp
__global__ void hello_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 8)
        printf("  [device] block %d thread %d: hello from GPU\n", blockIdx.x, threadIdx.x);
}
// 主机：
hello_kernel<<<1, 8>>>();
cudaDeviceSynchronize();
```

要点：

- `__global__`：内核，主机调用、设备执行。
- `<<<grid, block>>>`：执行配置（网格多少个块、每块多少线程）。
- `blockIdx.x`、`threadIdx.x`、`blockDim.x`、`gridDim.x`：内建变量。
- 内核启动是**异步**的，必须同步才能看到结果。

### 2.2 向量加法：CUDA 的 “Hello World”

`code/02_vecadd.cu` 的完整骨架：

```cpp
__global__ void vecAdd(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

主机侧四步：分配 → 拷贝 → 启动 → 拷回。

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
```

对于 `n = 16M`：

```text
n=16777216 blocks=65536 threads=256
time=0.089 ms  effective BW=2270.3 GB/s  verify=OK
```

**为什么有效带宽按 3 倍字节算？** 读 A、读 B、写 C，共 `3 × 16M × 4B = 192MB`，除以时间得到约 2.27 TB/s。这已经接近 H20 这类 HBM3 设备在纯流式访问下的可观水平，说明这个内核基本是**带宽受限**的。

### 2.3 索引与边界

- 一维：`i = blockIdx.x * blockDim.x + threadIdx.x`
- 二维：`x = blockIdx.x * blockDim.x + threadIdx.x; y = blockIdx.y * blockDim.y + threadIdx.y;`
- `blockDim` 是每块线程数，`gridDim` 是块数。
- 当 `n` 不是 `blockDim` 的整数倍时，必须 `if (i < n)` 做边界检查，否则越界读写。

### 2.4 事件计时

```cpp
cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s);
for (int k = 0; k < 20; ++k) vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
cudaEventRecord(e); cudaEventSynchronize(e);
float ms; cudaEventElapsedTime(&ms, s, e); ms /= 20.0f;
```

事件计时基于 GPU 时间戳，比 `clock()`/`gettimeofday` 更贴近内核真实耗时；多次取平均以降低抖动。

---

## 3. 性能的第一性原理

### 3.1 三个词：带宽、延迟、占用率

- **带宽（bandwidth）**：单位时间能搬多少字节。流式内核的上限。
- **延迟（latency）**：一次访存/指令从发出到结果可用要多久。GPU 靠**大量并发 warp** 隐藏它。
- **占用率（occupancy）**：SM 上实际驻留的 warp 数 / 最大 warp 数。占用率越高，越有能力用别的 warp 填满延迟。

经验法则：

1. 先判断内核是**带宽受限**还是**计算受限**（看算术强度）。
2. 带宽受限：优化访存（合并、向量化、减少流量）。
3. 计算受限：优化指令（FFMA、减少指令数、用专用单元）。
4. 两者都不饱和：多半是延迟/占用率/同步问题。

### 3.2 grid-stride：让网格规模与问题规模解耦

`code/03_grid_stride.cu` 对比两种写法：

```cpp
// A：一元素一线程，线程数随 n 增长
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];

// B：固定网格 + 循环
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    c[i] = a[i] + b[i];
```

H20 实测：

| n | 一元素一线程 | grid-stride |
|---|---|---|
| 1M | 0.006 ms (2131 GB/s) | 0.005 ms (2343 GB/s) |
| 16M | 0.090 ms (2235 GB/s) | 0.071 ms (2818 GB/s) |
| 64M | 0.355 ms (2268 GB/s) | 0.275 ms (2925 GB/s) |

结论：网格过大时，块调度与尾部开销会拉低带宽；固定为 `SM 数 × 每 SM 块数` 再用循环覆盖，通常更稳。

### 3.3 占用率：不是越高越好，但不能太低

`code/09_occupancy.cu` 固定总线程数（78×2048），只改变 block size，测 FMA 吞吐：

| block | blocks | occupancy | time(ms) | TFLOPS |
|---|---|---|---|---|
| 32 | 4992 | 50.0% | 0.082 | 31.99 |
| 64 | 2496 | 100.0% | 0.081 | 32.25 |
| 128 | 1248 | 100.0% | 0.081 | 32.30 |
| 256 | 624 | 100.0% | 0.081 | 32.29 |
| 512 | 312 | 100.0% | 0.081 | 32.30 |
| 1024 | 156 | 100.0% | 0.081 | 32.35 |

解释：

- block=32 时，每 SM 最多 32 个块 → 32×32=1024 线程，占用率只有 50%，但仍能达到 ~32 TFLOPS，因为 4 路 ILP 仍能填满流水线。
- block≥64 时占用率 100%，吞吐几乎不变 —— 说明该内核已经**计算/发射受限**，而非延迟受限。

**结论**：占用率是“隐藏延迟的手段”，不是目标本身。先用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 估算，再看实测。

计算占用率：

```cpp
int active;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, blockSize, dynamicSmem);
double occ = (double)active * blockSize / maxThreadsPerSM;   // 0..1
```

### 3.4 Roofline 速算

对一个内核粗算：

```
算术强度 AI = FLOPs / Bytes
若 AI < 峰值算力 / 峰值带宽  → 带宽受限
否则                          → 计算受限
```

H20 上 `FFMA` 峰值约 30+ TFLOPS（本机实测 FP32 约 32 TFLOPS），有效带宽约 2.8–3.3 TB/s，拐点 AI ≈ 10 FLOP/B。向量加法 AI≈0.02，显然带宽受限；GEMM 的 AI 很高，属于计算受限。

---

## 4. 全局内存与合并访存

### 4.1 warp 与内存事务

一个 warp 有 32 个 lane。若 32 个 lane 访问**连续**的 4B 元素，硬件可以把它们合并成少量 32B 扇区事务；若每个 lane 落在不同缓存行，就要发很多事务。

`code/04_coalescing.cu` 对比：

```cpp
// 合并：lane i 读 a[i]
out[i] = a[i];
// 跨步：lane i 读 a[(i*32)%n]
out[i] = a[(size_t)i * 32 % n];
```

H20 实测（64MB 读 + 64MB 写）：

```text
coalesced :   0.082 ms  1639.8 GB/s
strided   :   0.294 ms   456.7 GB/s  (slowdown 3.6x)
```

跨步 32（=128B）时，一个 warp 的 32 个 lane 落在 32 条不同缓存行上，事务数暴涨，有效带宽掉到 1/3.6。

### 4.2 向量化：一次搬 16 字节

对于连续访存，用 `float4` 让每个线程处理 4 个元素，指令数减少、事务更整齐：

```cpp
__global__ void copy4(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
```

SASS 中会出现 `LDG.E.128` / `STG.E.128`（128 位加载/存储）。注意 16 字节对齐，`n` 要按 4 个 float 计。

### 4.3 `__restrict__` 与只读缓存

`__restrict__` 告诉编译器指针不别名，允许它重排、合并、用只读路径。`02_vecadd` 的 SASS：

```text
LDG.E.CONSTANT R3, desc[UR4][R2.64] ;
LDG.E.CONSTANT R4, desc[UR4][R4.64] ;
FADD R9, R4, R3 ;
STG.E desc[UR4][R6.64], R9 ;
```

`LDG.E.CONSTANT` 表示走了常量/只读缓存路径，这是 `const __restrict__` 带来的收益。

### 4.4 访存优化清单

- 连续线程访问连续地址（合并）。
- 尽量 16 字节对齐，使用向量类型。
- 减少不必要的读写（融合、原地、复用）。
- 只读数据加 `const __restrict__`。
- 用 `cudaMemcpyAsync` + 页锁定内存重叠传输（第 11 章）。

---

## 5. 共享内存、bank 冲突与矩阵转置

### 5.1 为什么需要共享内存

全局内存带宽高但延迟也高。把会被**同一线程块反复使用**的数据搬到共享内存（片上、低延迟、高带宽），可以显著减少全局访存。共享内存按 bank 组织：32 个 bank，连续的 32 位字映射到连续 bank。

### 5.2 矩阵转置：从 467 GB/s 到 1112 GB/s

`code/05_transpose.cu` 的朴素写法读是合并的，写是跨步的：

```cpp
out[(size_t)x * n + y] = in[(size_t)y * n + x];   // 写不合并
```

共享内存分块（32×32 tile，padding 1）：

```cpp
__shared__ float tile[TILE][TILE + 1];
tile[threadIdx.y][threadIdx.x] = in[(size_t)y * n + x];
__syncthreads();
out[(size_t)y2 * n + x2] = tile[threadIdx.x][threadIdx.y];
```

H20 实测（4096×4096）：

```text
naive  (global) :   0.287 ms   467.1 GB/s
shared (+1 pad) :   0.121 ms  1112.3 GB/s  speedup 2.38x
```

`ptxas -v` 显示共享内存 4224 B = 32×33×4，正好是 padding 后的 tile。

### 5.3 bank 冲突：stride 的代价

`code/06_bank_conflict.cu` 让每个线程以 `stride` 在共享内存中跨步读取。H20 实测（相对 stride=1）：

| stride | 相对耗时 |
|---|---|
| 1,3,5,…（与 32 互素） | 1.00x |
| 2,6,10,…,30 | 1.41x |
| 4,12,20,28 | 2.76x |
| 8,24 | 5.50x |
| 16 | 10.98x |
| 32 | 21.96x |

规律：**冲突倍数 = 32 / gcd(stride, 32)**。stride=32 时 32 个 lane 全部命中同一 bank，串行化 32 次。

修复手段：

1. **padding**：`float tile[32][33]`，让列访问错开 bank。
2. **swizzle**：按 `(row ^ col)` 重排索引。
3. 改变数据布局，让同一 warp 访问不同 bank。

SASS 中共享内存访问是 `LDS` / `STS`；冲突不会体现在指令数量上，而是体现在 `LDS` 的重放/串行化（用 ncu 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 观察）。

---

## 6. 归约与 warp 原语

### 6.1 三种归约

`code/07_reduction.cu` 对 16M 个 1 求和，对比：

| 实现 | 时间 | 结果 |
|---|---|---|
| shared + atomic | 0.052 ms | 16777216 OK |
| warp shuffle + atomic | 0.070 ms | 16777216 OK |
| 两趟（partial + final） | 0.068 ms | 16777216 OK |

关键代码：

```cpp
// 每线程先 grid-stride 累加
float local = 0.f;
for (int i = blockIdx.x*blockDim.x + tid; i < n; i += blockDim.x*gridDim.x)
    local += in[i];

// warp 内归约
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
if ((threadIdx.x & 31) == 0) atomicAdd(out, v);
```

要点：

- `__shfl_down_sync` 是 warp 内寄存器交换，无共享内存、无块同步，延迟低。
- `atomicAdd` 只由每个 warp 的 leader 执行，把争用从“每线程”降到“每 warp”。
- 两趟法彻底避免全局原子争用：第一趟每块写一个部分和，第二趟单块汇总。
- 累加顺序不同会导致浮点结果有细微差异，这是正常的。

### 6.2 同步原语速查

| 原语 | 作用域 | 说明 |
|---|---|---|
| `__syncthreads()` | 线程块 | 屏障 + 内存可见 |
| `__syncwarp(mask)` | warp | 独立线程调度下重汇聚 |
| `__shfl_*_sync` | warp | 寄存器交换 |
| `__threadfence_block/()/system()` | 块/设备/系统 | 内存栅栏 |
| `cuda::barrier` / mbarrier | 块/集群 | 异步屏障（高级） |

---

## 7. 原子操作与直方图

### 7.1 全局原子 vs 共享原子

`code/08_atomic.cu`：64M 字节的直方图（256 个 bin）。

```cpp
// 直接对全局直方图原子加
atomicAdd(&hist[in[i]], 1);

// 先在共享内存里统计，再合并
__shared__ int sh[256];
atomicAdd(&sh[in[i]], 1);
__syncthreads();
for (int k = t; k < 256; k += blockDim.x)
    if (sh[k]) atomicAdd(&hist[k], sh[k]);
```

H20 实测：

```text
global atomic :  20.186 ms  total=67108864
shared atomic :   0.172 ms  total=67108864  speedup 117.33x
```

同一 bin 被大量线程同时原子更新时，全局原子会严重串行化；共享内存把争用限制在块内，并大幅减少全局原子次数。

### 7.2 使用建议

- 能在共享内存/寄存器里局部聚合的，就不要直接打全局原子。
- 能用 warp 归约代替原子加的，优先归约。
- 需要跨块累加时，两趟法或“每块一个部分和”通常比海量原子更快。
- `atomicAdd` 返回旧值，若不需要旧值，编译器/硬件有更轻的归约路径（如 `red.global.add`）。

---

## 8. 分支发散与指令级并行

### 8.1 分支发散：一半 lane 在“陪跑”

`code/10_divergence.cu`：奇数 lane 做 heavy 次 FMA，偶数 lane 不做。

```text
uniform   (all lanes work):   0.289 ms
divergent (half lanes work):  0.290 ms
发散版本只做了一半有效工作，却耗时 1.00x 于均匀版本；
即 warp 的有效 lane 利用率约为 50%。
```

warp 以 SIMT 方式执行：只要 warp 内有 lane 走 heavy 分支，整个 warp 就要为这条路径付出时间，其余 lane 被 mask 掉。**有效算力 = 实际工作 / 时间**，所以发散会让有效利用率下降。

修复思路：

- 让同一 warp 内的线程走相同分支（重排数据/任务）。
- 用**谓词化/选择**代替短分支：`cond ? a : b` 可能编译成 `SEL`，避免控制流分叉。
- 把分支提到更粗的粒度（按块/按 warp 分工）。
- 用 warp 专门化（warp specialization）。

### 8.2 ILP：用多条独立链喂满流水线

`code/14_ilp.cu`：三种内核每线程完成相同 FMA 总数，只改变独立链数量。

小网格（1 block × 32 线程，延迟受限）：

```text
chain1  :   0.022 ms
chain4  :   0.008 ms
chain8  :   0.009 ms
```

大网格（近似占满 GPU）：

```text
chain1  :   0.093 ms
chain4  :   0.081 ms
chain8  :   0.087 ms
```

解释：

- 单链时，每条 `FFMA` 依赖上一条结果，必须等延迟；发射槽被浪费。
- 4/8 条独立链让硬件可以交错发射，隐藏 FMA 延迟，小网格下提升约 2.7 倍。
- 大网格下本来就有大量 warp 隐藏延迟，ILP 的收益变小（0.093 → 0.081）。

**结论**：占用率不足时，ILP 是“穷人版占用率”；占用率足够时，ILP 主要帮助降低寄存器压力下的依赖停顿。

---

## 9. 数学函数、内联 PTX 与 SASS

### 9.1 标准库 vs 内建快速函数

`code/15_ptx_fastmath.cu` 对比 `expf` 与 `__expf`：

```text
expf (libm)   :   0.018 ms
__expf (MUFU) :   0.016 ms  speedup 1.11x
```

`__expf` 直接映射到 `MUFU.EX2`（特殊功能单元），`expf` 则包含更多范围约简与修正指令。SASS 中能看到：

```text
MUFU.EX2 R4, R0 ;
MUFU.EX2 R7, R6 ;
MUFU.EX2 R9, R8 ;
FFMA.SAT R0, R2.reuse, R9.reuse, 0.5 ;
FFMA.RM R0, R0, R11, 12582913 ;
```

### 9.2 内联 PTX

```cpp
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
```

- `fma.rn.f32`：融合乘加，单次舍入。
- 可用 `asm volatile` 防止被优化掉；注意约束符（`=f`、`f`、`r`、`l`）。
- 读特殊寄存器：`mov.u32 %0, %%laneid;`、`%%smid`。

### 9.3 从 C++ 到 SASS 的典型映射

| C++ | PTX | SASS |
|---|---|---|
| `c[i] = a[i] + b[i]` | `ld.global.nc.f32` / `add.f32` / `st.global.f32` | `LDG.E.CONSTANT` / `FADD` / `STG.E` |
| `y[i] = a*x[i] + y[i]` | `fma.rn.f32` | `FFMA` |
| 共享内存读写 | `ld.shared` / `st.shared` | `LDS` / `STS` |
| `expf` | 一长串 | `FFMA` + `MUFU.EX2` |
| `float4` 拷贝 | `ld.global.v4.f32` | `LDG.E.128` |
| 线程块屏障 | `bar.sync` | `BAR.SYNC` |

`02_vecadd` 的 PTX：

```text
ld.global.nc.f32 	%f1, [%rd8];
ld.global.nc.f32 	%f2, [%rd6];
add.f32 	%f3, %f2, %f1;
st.global.f32 	[%rd10], %f3;
```

---

## 10. 异步执行：流、事件与重叠

### 10.1 为什么需要流

默认情况下，同一个流里的操作串行；不同流之间可以并发（在资源允许且没有默认流干扰时）。把数据分成多个 chunk，每个 chunk 用独立流，可以重叠 H2D 拷贝、计算、D2H 拷贝。

`code/11_streams.cu`：

```cpp
for (int c = 0; c < chunks; ++c) {
    cudaMemcpyAsync(d + c*per, h + c*per, perBytes, cudaMemcpyHostToDevice, st[c]);
    scale_kernel<<<per/256, 256, 0, st[c]>>>(d + c*per, per, 2.0f);
    cudaMemcpyAsync(h + c*per, d + c*per, perBytes, cudaMemcpyDeviceToHost, st[c]);
}
```

H20 实测（64MB，4 个 chunk）：

```text
sequential :   2.833 ms
streams    :   1.753 ms  speedup 1.62x
```

要点：

- 异步拷贝必须使用**页锁定内存**（`cudaMallocHost`），否则会退化为同步。
- 用事件表达跨流依赖，避免全局同步。
- 默认流会与阻塞流同步；需要并发时使用 `cudaStreamNonBlocking` 或每线程默认流。

### 10.2 事件：计时与依赖

```cpp
cudaEventRecord(start, stream);
kernel<<<..., stream>>>(...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

事件还可以做跨流等待（`cudaStreamWaitEvent`），构建有向依赖。

---

## 11. 异步拷贝与软件流水

### 11.1 同步流水 vs 异步流水

`code/12_async_copy.cu`：把 global 数据分块搬到 shared，再累加。

```cpp
// 同步：读 -> 同步 -> 算 -> 同步
for (int i = threadIdx.x; i < TILE; i += blockDim.x) buf[i] = in[t*TILE + i];
__syncthreads();
acc += ...;
__syncthreads();

// 异步：块协作 memcpy_async（底层 LDGSTS），拷贝与后续工作重叠
cg::memcpy_async(block, buf, in + t*TILE, sizeof(float)*TILE);
cg::wait(block);
acc += ...;
block.sync();
```

H20 实测（256MB）：

```text
sync  pipeline :   0.241 ms  1113.1 GB/s
async pipeline :   0.148 ms  1817.9 GB/s  speedup 1.63x
```

### 11.2 从 LDGSTS 到 TMA

- **LDGSTS**（CC 8.0+）：逐元素/小块的 global→shared 异步拷贝，本文的 `memcpy_async` 在满足对齐时会下沉到它。
- **TMA**（CC 9.0+，Hopper）：张量内存加速器，支持**多维、大批量**的 global↔shared 异步拷贝，由专门的 TMA 引擎执行，几乎不占 SM 发射资源。使用 `cuda::ptx::cp_async_bulk` / `cp_async_bulk_tensor`，配合 `mbarrier` 完成通知；多维版本需要主机侧 `cuTensorMapEncodeTiled` 创建 tensor map。
- TMA 特别适合 GEMM 的 tile 加载、卷积、转置等规则多维访问。

一个 TMA 一维拷贝的内核骨架（cuda::ptx）：

```cpp
__shared__ alignas(16) float smem[TILE];
__shared__ uint64_t bar;
if (threadIdx.x == 0) cuda::ptx::mbarrier_init(&bar, blockDim.x);
__syncthreads();

if (is_elected()) {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                             smem, gmem + off, sizeof(smem),
                             cuda::device::barrier_native_handle(bar));
    cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
}
bar.arrive_and_wait();
// 此时 smem 可用
```

细节（对齐、swizzle、tensor map）见《CUDA 编程指南（中文全译本）》4.13 节。

### 11.3 软件流水的通用模式

```text
for t in tiles:
    prefetch(t + stages)     // 异步发起
    wait(t)                  // 等当前 tile 就绪
    compute(t)               // 计算与下一批拷贝重叠
```

双缓冲/多缓冲的本质：用“在途数据”填满内存延迟。

---

## 12. 张量核心与 WMMA GEMM

### 12.1 什么是张量核心

张量核心（Tensor Core）是 Hopper 上专门做矩阵乘累加的硬件单元，一条 `HMMA` 指令完成一个小矩阵块的 `D = A*B + C`。相比标量 FFMA，吞吐高一个数量级以上。

### 12.2 用 WMMA 写一个 16×16×16 的 GEMM

`code/13_wmma_gemm.cu`：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);

for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A + m0*K + k, K);
    wmma::load_matrix_sync(b_frag, B + k*N + n0, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
wmma::store_matrix_sync(C + m0*N + n0, c_frag, N, wmma::mem_row_major);
```

运行结果：

```text
M=N=K=512, grid=(32,32)
sampled check: OK (mismatches=0)
```

`ptxas -v`：该内核使用 32 个寄存器。

### 12.3 从 WMMA 到真正的 GEMM

生产级 GEMM 还需要：

- 更大的 CTA tile（如 128×128×32），多 warp 协作；
- 多级流水：TMA 加载 A/B tile、warp 级 WMMA、累加器驻留寄存器；
- swizzle 共享内存布局消除 bank 冲突；
- split-K、流水线、epilogue 融合（bias/激活）；
- 参考 CUTLASS / cuBLAS，不要从零手写。

WMMA 示例的意义是让你**看懂张量核心的编程接口**，而不是替代 CUTLASS。

---

## 13. 性能分析：ncu 与 nsys

### 13.1 Nsight Compute（单内核）

```bash
# 基本指标
ncu --set basic ./bin/05_transpose

# 关键指标：吞吐、占用率、内存、bank 冲突
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__time_duration.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  ./bin/06_bank_conflict
```

关注：

- `dram__throughput...`：是否撞到带宽墙。
- `sm__throughput...`：计算单元是否饱和。
- `sm__warps_active...`：实际占用率。
- `l1tex__data_bank_conflicts...`：共享内存 bank 冲突次数。
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`：全局加载扇区数（合并好坏）。

### 13.2 Nsight Systems（时间线）

```bash
nsys profile -o report ./bin/11_streams
nsys stats report.nsys-rep
```

看：H2D/D2H 与 kernel 是否重叠、流之间是否有空洞、启动间隙是否过大。

### 13.3 命令行拿到 SASS 的寄存器/指令

```bash
cuobjdump -res-usage bin/09_occupancy      # 资源使用
cuobjdump -sass bin/09_occupancy | grep -c FFMA   # 统计指令条数
nvdisasm -c bin/09_occupancy.cubin         # 控制流图
```

---

## 14. 指令级加速清单（重点）

下面每一条都能在 SASS/PTX 层面找到对应证据。

### 14.1 访存指令

| 目标 | 做法 | 指令层证据 |
|---|---|---|
| 合并访存 | 连续 lane 访问连续地址 | `LDG.E` 扇区数减少；避免多事务 |
| 向量化 | `float4` / `int4` | `LDG.E.128` / `STG.E.128`，指令数 ÷4 |
| 只读数据 | `const __restrict__` | `LDG.E.CONSTANT`（只读缓存） |
| 减少流量 | 融合算子、原地更新、复用 | 总 `LDG/STG` 字节数下降 |
| 共享内存 | 复用块内数据 | `LDS`/`STS` 替代重复 `LDG` |
| 消除 bank 冲突 | padding / swizzle | `LDS` 重放减少（ncu 指标） |

### 14.2 计算指令

| 目标 | 做法 | 指令层证据 |
|---|---|---|
| 乘加融合 | 写 `a*b+c` | `FFMA` 替代 `FMUL`+`FADD` |
| 减少指令 | 用 `fmaf`、位运算代替取模 | `IADD3`、`LOP3`、`IMAD` 组合 |
| 特殊函数 | `__expf`、`__sinf`、`__fdividef` | `MUFU.EX2/SIN/RCP` |
| 精度换速度 | `--use_fast_math`、`-ftz=true` | 更短的数学指令序列 |
| 整数优化 | 用移位代替乘除 2 的幂 | `SHF` 替代 `IMAD` |
| 谓词化 | 短分支用 `cond ? a : b` | `SEL` / 谓词执行，避免 `BRA` |

### 14.3 控制流与并行

| 目标 | 做法 |
|---|---|
| 减少发散 | 同 warp 同路径；按 warp/块分工；warp 专门化 |
| 提高 ILP | 多累加器、循环展开（`#pragma unroll`） |
| 提高占用率 | 减小寄存器/共享内存；`__launch_bounds__`；`-maxrregcount` |
| 隐藏延迟 | 增大在途数据；异步拷贝；多流 |
| 减少同步 | 用 warp shuffle 代替块同步；缩小 barrier 作用域 |

### 14.4 异步与专用单元

| 目标 | 做法 | 指令层证据 |
|---|---|---|
| 拷贝与计算重叠 | `cp.async` / `memcpy_async` | `LDGSTS` |
| 大批量多维搬运 | TMA | TMA 引擎，SM 发射压力低 |
| 矩阵乘加速 | WMMA / MMA | `HMMA` / `MMA` |
| 归约加速 | warp reduce | `REDUX`（CC 8.0+） |
| 原子聚合 | shared atomic / warp reduce | `ATOMS` / `RED` |

### 14.5 编译期

- `-O3 -lineinfo`（分析时）。
- `-Xptxas -v` 看寄存器与 spill；spill 会生成 `LDL/STL`（局部内存），要尽量避免。
- `-maxrregcount` / `__launch_bounds__` 在“占用率不足”与“spill”之间权衡。
- `-dlto` 改善分离编译的跨文件优化。
- 减少不必要的 `__forceinline__` 与过度展开，控制编译时间与代码体积。

### 14.6 一个“从 SASS 反推优化”的例子

`02_vecadd` 的 SASS：

```text
IMAD.WIDE R2, R9, 0x4, R2 ;          // 地址计算
LDG.E.CONSTANT R3, desc[UR4][R2.64] ;
LDG.E.CONSTANT R4, desc[UR4][R4.64] ;
FADD R9, R4, R3 ;
STG.E desc[UR4][R6.64], R9 ;
ISETP.GE.AND P0, PT, R9, UR4, PT ;   // 边界判断
```

观察与优化：

- 两条 `LDG` + 一条 `FADD` + 一条 `STG`，很干净；如果换成 `float4`，会变成 `LDG.E.128`，指令数降到 1/4。
- `IMAD.WIDE` 是 64 位地址计算；若能用 32 位索引/`__restrict__`，可减少地址指令。
- 边界 `ISETP` 在 `n` 是块大小整数倍时可省掉（但会牺牲通用性）。

---

## 15. 常见陷阱与调试

1. **忘记同步**：内核异步，主机立刻读结果会读到旧数据；计时/验证前要同步。
2. **越界访问**：没有 `if (i < n)`；`cuda-memcheck` / `compute-sanitizer` 可查。
3. **未初始化**：`cudaMalloc` 不初始化；需要 `cudaMemset`。
4. **共享内存竞争**：写后读之间缺 `__syncthreads()`；读后写之间也要同步。
5. **bank 冲突**：二维数组按列访问；用 padding/swizzle。
6. **原子争用**：大量线程打同一地址；先局部聚合。
7. **占用率过低**：寄存器/共享内存过大；用 `-Xptxas -v` 与 occupancy API 检查。
8. **寄存器 spill**：局部数组动态索引、结构体过大；看 SASS 的 `LDL/STL`。
9. **伪共享/默认流**：默认流隐式同步；用非阻塞流或每线程默认流。
10. **计时错误**：把第一次启动（含模块加载/JIT）算进去；先预热再计时。

调试工具：

```bash
compute-sanitizer ./bin/02_vecadd        # 越界、未初始化、竞争
cuda-gdb ./bin/02_vecadd                 # 断点调试
CUDA_LAUNCH_BLOCKING=1 ./bin/02_vecadd   # 串行化，定位是哪个内核出错
```

---

## 16. 从入门到高手：路线与练习

### 16.1 四周路线

| 周 | 目标 | 练习 |
|---|---|---|
| 第 1 周 | 会写正确内核 | 01–04；自己实现 `axpy`、`dot`、`relu` |
| 第 2 周 | 会做基础优化 | 05–08；把转置改成 64×64 tile，比较 bank 冲突 |
| 第 3 周 | 会做中级优化 | 09–12；用 ncu 找出自己内核的瓶颈并优化 20% |
| 第 4 周 | 系统与高级 | 13–15；读 CUTLASS 的一个 GEMM kernel，写读书笔记 |

### 16.2 进阶练习（按难度）

1. 把 `02_vecadd` 改成 `float4` 版本，比较 GB/s。
2. 给 `05_transpose` 加 swizzle，去掉 padding，比较性能。
3. 实现一个支持任意 `n` 的 `reduce`，要求结果与 CPU 双精度和的相对误差 < 1e-5。
4. 实现“分段归约 + 前缀和”，用于 softmax 的分母。
5. 用 `cp.async` 实现双缓冲矩阵乘，比较与朴素版本。
6. 用 `wmma` 实现 128×128×32 的分块 GEMM，加入共享内存与双缓冲。
7. 用 `ncu` 分析第 6 题，找出瓶颈（DRAM、SM、bank、占用率）。
8. 读《CUDA 编程指南（中文全译本）》第 4 章的 TMA、集群、异步屏障章节，写一个 TMA 加载 tile 的 demo。

### 16.3 推荐资料

- 《CUDA 编程指南（中文全译本）》（本目录）。
- CUDA C++ Programming Guide / Best Practices Guide（英文原版，最新）。
- CUTLASS、cuBLAS、cuDNN 源码与文档。
- Nsight Compute / Nsight Systems 用户手册。
- NVIDIA Hopper 架构白皮书。

---

## 附录 A：代码索引与编译命令

| 文件 | 主题 | 关键点 |
|---|---|---|
| `01_hello.cu` | 设备信息、第一个内核 | `cudaGetDeviceProperties` |
| `02_vecadd.cu` | 向量加法 | 边界、事件计时、带宽 |
| `03_grid_stride.cu` | grid-stride | 固定网格 |
| `04_coalescing.cu` | 合并访存 | 连续 vs 跨步 |
| `05_transpose.cu` | 转置 | 共享内存 + padding |
| `06_bank_conflict.cu` | bank 冲突 | stride 扫描 |
| `07_reduction.cu` | 归约 | shared / shuffle / 两趟 |
| `08_atomic.cu` | 直方图 | 全局 vs 共享原子 |
| `09_occupancy.cu` | 占用率 | block size 扫描 |
| `10_divergence.cu` | 发散 | lane 利用率 |
| `11_streams.cu` | 多流 | H2D/计算/D2H 重叠 |
| `12_async_copy.cu` | 异步拷贝 | `memcpy_async` 流水 |
| `13_wmma_gemm.cu` | 张量核心 | WMMA GEMM |
| `14_ilp.cu` | ILP | 多累加器 |
| `15_ptx_fastmath.cu` | 内联 PTX | FFMA/MUFU |
| `16_saxpy.cu` | SAXPY | 工程化写法 |

统一编译：

```bash
cd code
make -j4                 # 全部编译
make run                 # 全部运行
ARCH=sm_90 make -j4      # 显式指定架构
```

查看指令：

```bash
cuobjdump -sass bin/02_vecadd | less
nvcc -O3 -arch=sm_90 -ptx 02_vecadd.cu -o 02_vecadd.ptx
nvcc -O3 -arch=sm_90 -Xptxas -v -c 02_vecadd.cu -o /tmp/02.o
```

---

## 附录 B：H20 实测数据汇总

> 以下均为本机（H20-3e，78 SM，CUDA 13.0）`make run` 输出的代表性结果，单次运行会有抖动。

| 实验 | 指标 | 结果 |
|---|---|---|
| 02 向量加法 (16M) | 时间 / 有效带宽 | 0.089 ms / 2270 GB/s |
| 03 grid-stride (64M) | 一元素一线程 vs 循环 | 2268 vs 2925 GB/s |
| 04 合并访存 (64MB) | 连续 vs 跨步 | 1640 vs 457 GB/s（3.6×） |
| 05 转置 (4096²) | 全局 vs 共享 | 467 vs 1112 GB/s（2.38×） |
| 06 bank 冲突 | stride=1/2/4/8/16/32 | 1.00/1.41/2.76/5.50/10.98/21.96× |
| 07 归约 (16M) | shared / warp / 两趟 | 0.052 / 0.070 / 0.068 ms |
| 08 直方图 (64M) | 全局 vs 共享原子 | 20.19 vs 0.172 ms（117×） |
| 09 占用率 | block=32 vs 128 | 50% vs 100% 占用，吞吐均 ~32 TFLOPS |
| 10 分支发散 | 均匀 vs 发散 | 0.289 vs 0.290 ms（有效 lane 50%） |
| 11 多流 (64MB) | 串行 vs 多流 | 2.833 vs 1.753 ms（1.62×） |
| 12 异步拷贝 (256MB) | 同步 vs 异步流水 | 1113 vs 1818 GB/s（1.63×） |
| 13 WMMA GEMM (512³) | 正确性 | 抽样校验通过 |
| 14 ILP（小网格） | chain1 / chain4 / chain8 | 0.022 / 0.008 / 0.009 ms |
| 15 快速数学 | expf vs __expf | 0.018 vs 0.016 ms（1.11×） |
| 16 SAXPY (16M) | 时间 / 带宽 | 0.073 ms / 2769 GB/s |

---

## 结语

CUDA 性能优化的本质只有一句话：**让每个 SM 的每个周期都在做有用的工作，并且尽量少搬数据。**

- 先保证正确：边界、同步、初始化、错误检查。
- 再保证喂饱：占用率、ILP、异步拷贝。
- 再减少搬运：合并、向量化、共享内存、复用。
- 最后抠指令：FFMA、MUFU、SEL、REDUX、HMMA。
- 全程用数据说话：`nvcc -Xptxas -v` 看资源，`ncu` 看瓶颈，`nsys` 看时间线。

把这 16 个实验亲手跑一遍、改一遍、用 ncu 看一遍，你就已经跨过了“会写 CUDA”到“会调 CUDA”的门槛。剩下的，是在真实算子（GEMM、Attention、Norm、MoE）上反复实践。

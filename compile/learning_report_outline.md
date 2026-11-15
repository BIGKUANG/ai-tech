# TileLang 学习报告大纲

> 目标：系统梳理 TileLang 的代码目录、整体架构与各核心模块，形成一份可扩展的学习/技术报告。

## 1. 引言
- 1.1 TileLang 是什么：面向 GPU/CPU/NPU 高性能 kernel 的领域特定语言（DSL）
- 1.2 设计目标：Pythonic 语法 + TVM 编译基础设施，兼顾生产力与极致性能
- 1.3 报告范围与阅读指引

## 2. 背景与生态
- 2.1 与 TVM / TIR / TIRX 的关系
  - 2.1.1 TVM 的现状与 TileLang 对 TIR 的扩展（`tirx` 命名空间、`SBlock`、tile 级算子）
  - 2.1.2 复用与 fork 的边界：哪些来自 upstream TVM，哪些是 TileLang 自研（transform/pass、layout、backend registry）
  - 2.1.3 为什么在 TVM 之上做 DSL 而不是从零写编译器
- 2.2 与 CUTLASS（CuTe）的对比
  - 2.2.1 CUTLASS/CuTe 作为 C++ 模板库 vs TileLang 作为 Python DSL
  - 2.2.2 CuTe layout algebra 的借鉴与在 TileLang 中的落地（`cute.py`、`src/layout/cute_layout.cc`）
  - 2.2.3 生产力 / 可维护性 / 学习曲线对比
- 2.3 与 Triton 的多维度详细对比（重点章节）
  - 2.3.1 设计哲学与抽象层次：Triton 的 block-level「编译器尽量决定」 vs TileLang 的 tile-level「用户显式控制内存层次」
  - 2.3.2 编程模型对比：
    - 程序实例化：`tl.program_id` + `tl.arange` vs `T.Kernel(bx, by, threads=...)` + tile 形状
    - 数据搬运：`tl.load/tl.store`（隐式向量化、mask）vs `T.copy`（tile→tile，自动 TMA/cp.async）
    - 计算：`tl.dot`（编译器选 MMA）vs `T.gemm`（显式 WGMMA/TCGEN5MMA/TMEM）
    - 归约：`tl.max/tl.sum` vs `T.reduce_max/T.reduce_sum` + `T.alloc_fragment` 布局
  - 2.3.3 内存层次与布局控制：
    - Triton：无显式 shared memory API，编译器管理 shared/register；blocked layout 由编译器选择
    - TileLang：`T.alloc_shared/alloc_fragment/alloc_local/alloc_barrier` 显式驻留；`LayoutInference` + `Layout/Fragment/Swizzle` + CuTe 布局代数 + GEMM 专用布局 + 成本模型（`layout_cost_model.cc`）
  - 2.3.4 编译栈与 IR 体系：
    - Triton：自研 MLIR 系 IR（TTIR → TTGIR → LLVM IR → PTX/cubin），与 LLVM 紧耦合
    - TileLang：TVM/TIR(X) + 大量 Python/C++ pass + backend registry；多目标、可插拔
    - 用两张 IR 流水线图并排对比
  - 2.3.5 调度与流水线：
    - Triton：`num_stages` 自动软件流水；编译器自动 warp 调度
    - TileLang：`T.Pipelined` + `PipelinePlanning`/`InjectSoftwarePipeline`，显式 warp specialization（`AutoWarpSpecialization`/`ProducerConsumerWarpSpecialized`）、persistent kernel、L2 persistent
  - 2.3.6 硬件特性映射：
    - TMA descriptor / mbarrier / WGMMA / Hopper→Blackwell（`LowerHopperIntrin`、`LowerBlackwell2SM`、tcgen05/TMEM、`InjectFenceProxy`、`InjectTcgen05Fence`）
    - Triton 的 TMA/WGMMA 由编译器在受限模式下自动生成
  - 2.3.7 后端与可移植性：Triton（NVIDIA CUDA、AMD ROCm、CPU） vs TileLang（CUDA/ROCm/Metal/Ascend/CPU/WebGPU），后端注册表机制
  - 2.3.8 可调试性与可观测性：Triton 的 MLIR dump vs TileLang 的 `TL_LOWER_TRACE`/`pass_diff`/`Analyzer`/`plot_layout`/pass timing
  - 2.3.9 自动调优生态：Triton `@triton.autotune` vs TileLang `autotuner` + `carver`（matmul 调度生成）
  - 2.3.10 性能与生产力权衡：Triton 上手快、性能上限受编译器约束；TileLang 控制力强、上限高但需理解内存层次
  - 2.3.11 代码级对比示例（同一算子两种写法并排展示，取自仓库现有成对样例）：
    - 示例 A：per-token FP8 量化 —— `examples/cast/example_triton_cast_to_fp8.py` vs `examples/cast/example_per_token_cast_to_fp8.py`
      - 对比点：`tl.load+mask+tl.max` vs `T.copy+T.reduce_absmax+T.Parallel`；同步/布局/向量化由谁决定
    - 示例 B：NSA 稀疏注意力 —— `examples/deepseek_nsa/example_triton_nsa_fwd.py` vs `example_tilelang_nsa_fwd.py`
    - 示例 C：MLA 解码 —— `examples/deepseek_mla/README.md` 中 TileLang / Triton / FlashMLA / FlashInfer 的性能对比（bs64/bs128）
    - 逐例给出「代码行数 / 关键 API 映射 / 编译产物差异 / 性能对比」小结表
  - 2.3.12 横向总览对比表（维度：抽象层次、内存控制、布局控制、流水线、硬件 intrinsic、IR 栈、后端、调试、生态、学习曲线、性能上限）
  - 2.3.13 选型建议：何时选 Triton（快速原型、生态集成、规整算子）、何时选 TileLang（极致性能、复杂流水线、多后端、Ascend 等非 NVIDIA 目标）
- 2.4 支持的后端矩阵（CUDA / ROCm / Metal / Ascend / CPU / WebGPU）与支持等级
  - 2.4.1 各后端成熟度、特性覆盖、已知限制
  - 2.4.2 与 Triton 后端覆盖的差异对照表

## 3. 代码目录结构
- 3.1 顶层目录总览（`src/`、`tilelang/`、`examples/`、`benchmark/`、`testing/`、`3rdparty/`、`docs/`、`maint/`）
- 3.2 Python 主包 `tilelang/` 目录树（按子模块逐级列出）
- 3.3 C++ 核心源码 `src/` 目录树（`op/`、`transform/`、`layout/`、`runtime/`、各 target 子目录）
- 3.4 构建与工程文件（`CMakeLists.txt`、`pyproject.toml`、`cmake/`、`version_provider.py`）

## 4. 代码架构图
- 4.1 分层架构总览（Frontend DSL → IR → Transform Passes → Backend CodeGen → Runtime）
- 4.2 多后端抽象架构图（`tilelang/backend` 注册表 + 各 target 后端）
- 4.3 编译流水线数据流图（Python kernel → JIT → Lower → Host/Device IR → CodeGen → 二进制）
- 4.4 关键 Mermaid 图（架构图、模块依赖图、编译流程图）

## 5. 核心模块详细介绍
- **配图要求（全章通用）**：5.1–5.13 每个模块至少配 1 张图，统一用 Mermaid（架构图/流程图/时序图/类图/状态图），并在图中标注关键文件与数据流，做到「一图对应一节」。
- 5.1 语言层（`tilelang/language/`）
  - 5.1.1 DSL 入口与 kernel 定义（`kernel.py`、`frame.py`、`allocate.py`）
  - 5.1.2 数据移动：`copy_op.py`（`T.copy`/`T.tma_copy`）
  - 5.1.3 计算：`gemm_op.py`、`reduce_op.py`、`scan_op.py`、`fill_op.py`、`atomic.py`、`logical.py`、`math_intrinsics.py`、`fp8.py`
  - 5.1.4 控制流与调度：`loop.py`（`T.Pipelined`）、`tile_schedule.py`、`warp_specialize.py`
  - 5.1.5 注解与元信息：`annotations.py`、`meta.py`、`customize.py`、`symbolics.py`
  - 5.1.6 TileOp 抽象（`tilelang/tileop/`）：`base.py`
  - ▶ 配图：DSL→TileOp→IR 调用关系图；语言 API 分类图（搬运/计算/控制流/注解）
- 5.2 类型系统与数据类型（`tilelang/dtypes.py`、`language/dtypes.py`）
  - ▶ 配图：标量/向量/FP8/自定义类型类图与类型提升/转换关系图
- 5.3 布局系统（`tilelang/layout/` + `src/layout/`）
  - 5.3.1 `Layout`/`Fragment`/`PartialFragment`
  - 5.3.2 CuTe 布局代数与转换（`cute.py`、`_cute_ffi_api.py`、`src/layout/cute_layout.cc`）
  - 5.3.3 Swizzle（`swizzle.py`、`swizzle_mode.py`）
  - 5.3.4 GEMM 专用布局（`gemm_sp.py`、`src/layout/gemm_layouts.cc`、`tcgen05_layout.cc`）
  - 5.3.5 Ascend 布局（`layout/ascend.py`）
  - ▶ 配图：Layout/Fragment 类图；全局内存→shared→fragment 的布局映射图；swizzle 前后对照图
- 5.4 前端解析与分析（`tilelang/analysis/`）
  - AST 打印、嵌套循环检查、fragment 循环检查、并行局部索引检查、布局可视化
  - ▶ 配图：AST→IR 解析流程图；各 analysis 检查的触发点与失败诊断图
- 5.5 引擎与编译流程（`tilelang/engine/`、`tilelang/jit/`）
  - 5.5.1 `lower.py`：Lower 主流程（Host/Device IR 分离）
  - 5.5.2 `semantic_check.py`：前置语义检查
  - 5.5.3 `param.py`：KernelParam / CompiledArtifact
  - 5.5.4 `tilelang/jit/`：JIT 适配、kernel 封装、ABI、诊断与异常
  - ▶ 配图：Lower 主流程时序图（lower.py 调用链）；JIT adapter 选择与 ABI 封装图（详见第 6 章）
- 5.6 变换与 Pass（`tilelang/transform/` + `src/transform/`）
  - 5.6.1 Python 侧 pass 封装（`pass_config.py`、`simplify.py`、`hoist_broadcast_values.py`、`add_bufstore_wrapper.py`、`decouple_type_cast.py`）
  - 5.6.2 C++ 侧核心 pass 清单（分类：frontend legalize、loop 变换、存储/内存、lowering、验证）
    - frontend_legalize、loop_partition、loop_vectorize、unroll_loop、loop_unswitching
    - flatten_buffer、storage_rewrite、merge_shared_memory_allocations、hoist_global_buffer_allocations
    - inject_pipeline、pipeline_planning、lower_tile_op、lower_intrin、lower_thread_allreduce
    - split_host_device、make_packed_api、materialize_kernel_launch
    - verify_* 系列（buffer_init、parallel_loop、reducer_consumed、reducer_epoch）
  - 5.6.3 Pass 管理与编排（`backend/pass_pipeline`）
  - ▶ 配图：Pass 分类图；Pass 依赖/顺序图；单个 pass 的 before→after IR 图
- 5.7 后端抽象层（`tilelang/backend/`）
  - 5.7.1 `BackendContext` / Target / Module（`module.py`、`target.py`）
  - 5.7.2 后端注册表与分发（`device_codegen.py`、`host_codegen.py`、`execution_backend.py`）
  - ▶ 配图：BackendModule 类图；target→pipeline/codegen/execution_backend 分发图
- 5.8 具体后端实现
  - 5.8.1 CUDA（`tilelang/cuda/` + `src/cuda/`）：codegen、pipeline、execution_backend、cutedsl_backend、NVRTC
  - 5.8.2 ROCm（`tilelang/rocm/` + `src/rocm/`）
  - 5.8.3 Metal（`tilelang/metal/` + `src/metal/`）
  - 5.8.4 Ascend（`tilelang/ascend/` + `src/ascend/`）：target、pipeline、codegen、SIMD/SIMT、torch 交换
  - 5.8.5 CPU / LLVM（`tilelang/cpu/` + `src/cpu/`）
  - 5.8.6 WebGPU（`tilelang/webgpu/`）
  - ▶ 配图：后端总览架构图；各后端 pipeline/codegen 对照图（可用一张分组图覆盖 5.8.1–5.8.6）
- 5.9 算子层（`src/op/`）
  - 算子注册与基类（`operator.cc`、`operator.h`、`builtin_registry.h`）
  - copy / fill / gemm（含 blockscaled、sparse）/ reduce / scan / atomic / logical / math / parallel / transpose
  - ▶ 配图：算子继承/注册类图；算子→tile op→硬件指令映射图
- 5.10 运行时与执行（`tilelang/runtime`、`src/runtime/`）
  - 5.10.1 二进制加载与缓存（`tilelang/cache/`：kernel_cache、binary_cache、cuda/ascend 变体）
  - 5.10.2 运行时支撑（logging、error_helpers、device 管理）
  - ▶ 配图：二进制加载与缓存流程图；kernel 调用时序图（host→device）
- 5.11 自动调优（`tilelang/autotuner/`、`tilelang/carver/`）
  - 5.11.1 `tuner.py`、`param.py`、`capture.py`、`grouped_compile.py`
  - 5.11.2 `carver`：matmul 分析与调度生成
  - ▶ 配图：autotune 搜索流程图（config 空间→编译→bench→最优）；carver 调度生成流程图
- 5.12 性能剖析与可观测性
  - 5.12.1 `tilelang/profiler/`（bench、wall、torch_bench、msprof）
  - 5.12.2 `tilelang/instrumentation/`（pass 计时、lower trace）
  - 5.12.3 `tilelang/tools/`（Analyzer、compile_only、pass_timing、plot_layout）
  - ▶ 配图：instrumentation 事件采集图；profiler/工具在编译-运行链路中的位置图
- 5.13 工具与辅助（`tilelang/tools/`、`tilelang/contrib/`、`tilelang/utils/`）
  - 编译器封装（nvcc、nvrtc、hipcc、msvc、bisheng、rocm）
  - 通用工具（device、tensor、language、pass_diff、version）
  - ▶ 配图：工具与 contrib/utils 的功能分类图及依赖关系图

## 6. 编译流水线深度剖析（重点章节：配图 + 实例 IR + 全模块详解）
- 6.0 本章导读
  - 说明：先用一张总图建立整体心智模型，再用一个真实 GEMM 逐阶段展示 IR 演变，最后按模块逐一拆解
  - 阅读依赖：先读第 4 章（架构图）与 5.5/5.6/5.7（引擎、Pass、后端抽象）
- 6.1 编译模块总览（先建立整体视图）
  - 6.1.1 编译相关目录与职责：
    - 前端与解析：`tilelang/language/`、`tilelang/analysis/`、`tilelang/tileop/`
    - 引擎编排：`tilelang/engine/`（`lower.py`、`semantic_check.py`、`param.py`、`callback.py`）
    - Pass 封装：`tilelang/transform/`（Python 侧） + `src/transform/`（C++ 侧核心 pass）
    - 后端抽象：`tilelang/backend/`（`module.py`、`target.py`、`device_codegen.py`、`host_codegen.py`、`execution_backend.py`、`pass_pipeline/`）
    - 具体后端：`tilelang/{cuda,rocm,metal,ascend,cpu,webgpu}/` 与 `src/{cuda,rocm,metal,ascend,cpu,webgpu}/`
    - JIT 与运行时：`tilelang/jit/`（`kernel.py`、`adapter/`、`abi.py`）、`tilelang/cache/`、`tilelang/runtime/`
    - 可观测性：`tilelang/instrumentation/`、`tilelang/tools/`、`tilelang/profiler/`
  - 6.1.2 编译流程分层图（Mermaid，必画）：
    ```mermaid
    flowchart TD
      A[Python DSL 函数 / T.prim_func] --> B[前端解析 frame/AST 捕获]
      B --> C[IRModule / PrimFunc]
      C --> D[create_backend_context: 解析 target + execution_backend]
      D --> E[PreLowerSemanticCheck 语义检查]
      E --> F[BackendContext.lower → PassPipeline]
      F --> G[host_mod / device_mod Filter 分离]
      G --> H[device codegen: TIR→CUDA/HIP/...]
      G --> I[host codegen: TIR→C/LLVM]
      H --> J[CompiledArtifact]
      I --> J
      J --> K[execution_backend adapter: tvm_ffi/cython/nvrtc/cutedsl/torch/pto]
      K --> L[KernelCache / binary_cache 命中或落盘]
      L --> M[可执行 kernel + torch 调用封装]
    ```
- 6.2 从 Python 函数到 IRModule 的完整链路
  - 6.2.1 编译入口盘点：`tilelang.compile`、`@tilelang.jit`、`tilelang.lower`、`tilelang.autotuner` / `grouped_compile`、carver
  - 6.2.2 `@tilelang.jit` 装饰器解析与参数符号化（`T.const`、`T.symbolic`）
  - 6.2.3 AST 解析与 frame 栈：`language/frame.py`、`language/allocate.py`、`analysis/`（嵌套循环检查、fragment 循环检查、并行局部索引检查）
  - 6.2.4 TileOp 抽象与前端合法化：`tileop/base.py` → `frontend_legalize`
  - 6.2.5 配图：Python → PrimFunc/IRModule 数据流图（Mermaid sequenceDiagram 或 flowchart）
- 6.3 Lower 阶段拆解（以 `tilelang/engine/lower.py` 为主线）
  - 6.3.1 `BackendContext` 解析：`create_backend_context` → `determine_target` → `_resolve_backend_module` → `resolve_execution_backend`（target 与 execution_backend 的合法性校验）
  - 6.3.2 `PreLowerSemanticCheck` 前置语义检查（可用 `tl.disable_prelower_semantic_check` 关闭）
  - 6.3.3 `context.lower(mod)` → `BackendModule.get_pipeline(target).lower` → `PassPipeline.lower`（含 phase 窗口与错误富化 `enrich_error`）
  - 6.3.4 CUDA 主流水线 `CUDAPassPipelineBody` 的阶段化拆解（对照 `tilelang/cuda/pipeline.py`）：
    - 阶段 A · Prologue 前处理：`BindTarget`、`MaterializeKernelLaunch`、`AnnotateDeviceBoundTmaCopies`、`LetInline`、`AddWrapperForSingleBufStore`、`LegalizeNegativeIndex`、`VerifyParallelLoop`、`InjectAssumes`、`Simplify`
    - 阶段 B · reducer/语义校验：`CanonicalizeLegacyReducer`、`VerifyReducerEpoch`、`VerifyBufferInit`
    - 阶段 C · CUDA/Blackwell 专用前置：warp specialization（`AutoWarpSpecialization`/`MaterializeWSSchedule`/`ProducerConsumerWarpSpecialized`）、`LowerBlackwell2SM`
    - 阶段 D · 流水线规划：`IfStmtBinding`、`UnrollLoop`、`PipelinePlanning`、`InjectSoftwarePipeline`
    - 阶段 E · 布局推断与 tile 下降：`LayoutInference`、`ReducerPlanAndMaterialize`、`LowerTileOp`、`VerifyReducerConsumed`
    - 阶段 F · 低层合法化与向量化：`DecoupleTypeCast`、`LegalizeVectorizedLoop`、`LegalizeSafeMemoryAccess`、`LowerAccessPtr`、`HoistNonRestrictParams`
    - 阶段 G · 存储/分配/同步：`LowerSharedTmem`、`PlanAndUpdateBufferAllocationLocation`、`LowerSharedBarrier`、`FuseMBarrierArriveExpectTx`、`HoistGlobalBufferAllocations`、`FlattenBuffer`、`ConfigIndexBitwidth`、`VectorizeLoop`、`StorageRewrite`、`MergeSharedMemoryAllocations`、`ThreadSync`
    - 阶段 H · API/设备分离：`AnnotateDeviceRegions`、`SplitHostDevice`、`MakePackedAPI`、`LowerDeviceKernelLaunch`、`PersistThreadblock`
    - 每阶段配「输入 IR 片段 → 输出 IR 片段」示意
  - 6.3.5 Host/Device 分离的判定逻辑：`is_device_call` / `get_host_call`（`CallingConv.DEVICE_KERNEL_LAUNCH`、C packed 函数）
  - 6.3.6 逐 pass 清单与分类表（frontend legalize / loop 变换 / 存储内存 / lowering / 验证），标注 pass config 开关（`tilelang/transform/pass_config.py`）
- 6.4 实例走读：一个 GEMM 的 IR 逐级演变（本章核心示例，必做）
  - 选取 `examples/gemm` 中一个标准 `T.gemm` + `T.Pipelined` + `T.copy` 的 kernel
  - 6.4.1 源程序（Python DSL）与关键构造说明（`T.Kernel`、`alloc_shared`、`alloc_fragment`、`T.copy`、`T.Pipelined`、`T.gemm`、`T.clear`）
  - 6.4.2 初始 tile-level IR（前端合法化后）：展示 `T.gemm`/`T.copy`/`T.Parallel` 仍为高层 tile op
  - 6.4.3 `PipelinePlanning` / `InjectSoftwarePipeline` 之后：展示 pipeline stage/order、double buffering、mbarrier/cp.async 的引入
  - 6.4.4 `LayoutInference` 之后：展示 `Fragment`/`Layout` 注解如何挂到 buffer 上（配合 `TL_LAYOUT_VISUALIZATION_ENABLE` 或 `plot_layout` 截图）
  - 6.4.5 `LowerTileOp` 之后：tile op 展开为线程级循环（`T.Parallel` → thread binding、`T.copy` → 向量化 load/store、`T.gemm` → wgmma/cp.async）
  - 6.4.6 最终 device IR 与生成的 CUDA 片段（`FlattenBuffer`/`StorageRewrite`/`ThreadSync`/`MakePackedAPI` 之后）
  - 6.4.7 每步给出「IR before / IR after / 触发 pass / 观测命令」四列表；IR 用真实 dump 截取（可用 `TL_ENABLE_DUMP_IR` 或 `TL_LOWER_TRACE` 生成）
- 6.5 Pass 编排与可观测性（编译模块的「控制面」）
  - 6.5.1 `PassPipeline` / `BackendModule` 注册与校验（`module.py`：pipeline/codegen/execution backend 一一对应校验）
  - 6.5.2 `pass_configs` 与 `PassConfigKey`：常用开关（`tl.enable_async_copy`、`tl.disable_tma_lower`、`tl.disable_wgmma`、`tl.layout_cost_model`、`tl.enable_dump_ir`、`tl.pass_profile` 等）
  - 6.5.3 编译期插桩：`tilelang/instrumentation/`（`compile_pass_instrumentation`、`pass_phase`、`PassEvent`、`CodegenEvent`）
  - 6.5.4 调试与可视化工具：
    - `TL_LOWER_TRACE`（逐 pass before/after IR + codegen 捕获，推荐）
    - `pass_diff`（旧版）、`Analyzer`、`compile_only`、`pass_timing`、`pass_visualizer`、`plot_layout`
- 6.6 Host 与 Device 代码生成
  - 6.6.1 codegen 注册表：`device_codegen.py`（`DeviceCodegen.build`/`build_without_compile`）、`host_codegen.py`（`HostCodegen`、`HostCodegenHook`、`STANDARD_HOST_CODEGENS`）、`execution_backend.py`
  - 6.6.2 Host 侧 pass：`BindTarget` → `FP8/BF16StorageLegalize` → `LowerTVMBuiltin` → `LowerCustomDatatypes` → `LowerIntrin` → `CombineContextCall` → `preprocess_host_codegen` → `codegen_host`
  - 6.6.3 Device 侧准备：`_prepare_device_codegen_mod`（`LowerIntrin`、`Simplify`、`HoistBroadcastValues`）→ `codegen_device(compile_device=...)`
  - 6.6.4 CUDA codegen 路径（`tilelang/cuda/codegen.py` + `src/cuda/codegen/`）、postproc 回调（`register_cuda_postproc`）
  - 6.6.5 `CompiledArtifact` 结构与 `KernelParam`（`engine/param.py`）
  - 6.6.6 execution backend 对比表：`tvm_ffi`（默认，预编译二进制） / `cython` / `nvrtc` / `cutedsl` / `torch` / `pto`，及各自 `enable_host_codegen`/`enable_device_compile` 语义
- 6.7 缓存与二次编译命中路径
  - 6.7.1 `KernelCache`：cache key 组成（源码哈希 + target + pass_configs + 版本等）、`kernel.cu`/`kernel_lib.so`/`params.json`/`manifest.json` 布局
  - 6.7.2 `binary_cache` / `cuda_binary_cache` / `ascend_binary_cache` 的二进制级缓存
  - 6.7.3 `JITKernel.from_database` 命中路径：跳过 lower、直接恢复 adapter 与 ABI
  - 6.7.4 流程图：首次编译 vs 二次命中的两条路径对比（Mermaid）
- 6.8 多后端流水线差异对照
  - 6.8.1 ROCm pipeline（`tilelang/rocm/pipeline.py`）与 CUDA 的差异
  - 6.8.2 Metal、Ascend（SIMD/SIMT、torch_npu 交换）、CPU/LLVM、WebGPU pipeline 要点
  - 6.8.3 backend registry 如何按 target kind 选择 pipeline/codegen/execution backend
- 6.9 本章配图与产物清单（供写作时按图索骥）
  - 图 1：编译总流程图（Mermaid，见 6.1.2）
  - 图 2：Python→IRModule 前端链路图
  - 图 3：CUDA PassPipeline 阶段图（8 个阶段，标注关键 pass）
  - 图 4：GEMM 实例 IR 演变阶梯图（tile-level → pipelined → layout → threaded → CUDA）
  - 图 5：Host/Device codegen 与 execution backend 分发图
  - 图 6：缓存命中/未命中路径图
  - 表 1：逐 pass 输入/输出/开关/观测方式
  - 表 2：execution backend 能力对照表
  - 产物：真实 `.tir` dump 片段、生成的 CUDA 源码片段、`report.html`（lower trace）截图

## 7. 关键概念专题
- 7.1 Layout 推断与成本模型
- 7.2 软件流水线（`T.Pipelined`）与 warp specialization
- 7.3 TMA / WGMMA / TMEM（Blackwell）等硬件特性映射
- 7.4 多后端方言（language dialect）设计
- 7.5 数据竞争 / 同步 / 异步生命周期（语义规则）

## 8. 案例与示例分析
- 8.1 经典 GEMM 走读（`examples/gemm`）
- 8.2 FlashAttention（`examples/flash_attention`）
- 8.3 稀疏 / 分组 / FP8 / 量化算子示例
- 8.4 Ascend 后端示例（`examples/ascend`）
- 8.5 调优与工具示例（`examples/analyze`、`examples/plot_layout`）

## 9. 测试与质量保障
- 9.1 测试体系（`testing/python`、`testing/cpp`、`testing/ascend`）
- 9.2 benchmark 基准（`benchmark/`：matmul、flash_attention、mamba、fp8 等）
- 9.3 回归与性能回归（`tilelang/testing/perf_regression.py`）

## 10. 附录
- 10.1 术语表（TIR / TIRX / Layout / Fragment / Swizzle / TileOp 等）
- 10.2 参考文献与相关 PR/文档链接
- 10.3 后续学习路线建议

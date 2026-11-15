# DeepInfer 学习报告大纲

> 目标：以当前仓库（commit `9bd9cf2`，2026-10-21）为准，系统梳理 DeepInfer 的定位、代码结构、引擎架构、V4.1 模型实现、调度与 KV Cache、并行与负载均衡、DFlash 投机解码、构建测试与部署调优。
>
> 说明：本次开源版本聚焦**离线推理**，在线服务封装、KV Cache 存储系统、完整的 E/P/D 分离与跨副本负载均衡未包含在仓库中；报告会明确区分“已开源实现”与“生产系统依赖”。

## 1. 引言
- 1.1 DeepInfer 是什么：DeepSeek 生产推理框架的开源版本
- 1.2 能力矩阵：模型、算子、并行、调度、投机、结构化输出、双硬件
- 1.3 开源边界：哪些已提供，哪些属于生产基础设施
- 1.4 报告范围、源码版本与阅读指引

## 2. 背景与生态
- 2.1 DeepSeek-V4.1-Flash 模型架构
  - 2.1.1 MLA、压缩 KV（C1/C2）与稀疏 indexer
  - 2.1.2 SWA、attention sink 与 YoCo/consumer 层
  - 2.1.3 MoE：noaux_tc、shared expert、FP8/FP4
  - 2.1.4 mHC 多路残差主干
  - 2.1.5 Engram n-gram 记忆
  - 2.1.6 DFlash / DSpark 投机解码
- 2.2 大模型推理的核心矛盾
  - 2.2.1 Prefill 与 Decode 的不同 roofline
  - 2.2.2 KV Cache 容量、带宽与命中率
  - 2.2.3 MoE 专家并行与负载均衡
  - 2.2.4 CPU 调度开销与 GPU 空泡
- 2.3 与相关项目的关系
  - 2.3.1 vLLM / SGLang / TensorRT-LLM 的继承与差异
  - 2.3.2 FlashMLA / DeepGEMM / DeepEP / DeepSelect / TileKernels
  - 2.3.3 TileLang 自定义算子
  - 2.3.4 与纯 Kernel 库、DSL 推理栈的横向对比
- 2.4 硬件与部署形态
  - 2.4.1 NVIDIA SM100 与昇腾 950
  - 2.4.2 EPD 分离、Wide-EP、KV 存储集群

## 3. 代码目录结构
- 3.1 顶层目录总览
- 3.2 Python 包 `deepinfer/`
  - 3.2.1 `engine/`：引擎、调度、参数
  - 3.2.2 `worker/`：执行、缓存、CUDA Graph
  - 3.2.3 `model_executor/`：模型、算子、采样、并行
  - 3.2.4 `structure_output/`、`ve/`、`distributed/`、`npu_ops/`
- 3.3 C++ 扩展 `csrc/`
  - 3.3.1 `engine/`：Scheduler、BlockManager、Sequence、ZOS、InputMetadata
  - 3.3.2 `utils/`、`graph_timer/`、`cuda_utils` / `asc_utils`
- 3.4 `vendor/` 与 `deep_cluster/`
- 3.5 `examples/`、`tests/`、`scripts/`、`doc/`

## 4. 代码架构图
- 4.1 分层架构总览（Entrypoint → LLMEngine → Worker → ModelRunner → Model → Kernel）
- 4.2 一次请求的完整生命周期
- 4.3 模块依赖图
- 4.4 并行拓扑与进程组
- 4.5 构建与运行架构（vendor 编译、`_so/<tag>`、平台分派）

## 5. 核心模块详细介绍
- 5.1 入口与配置
  - 5.1.1 `entrypoints/llm.py`：离线 `LLM`
  - 5.1.2 `EngineArgs` / `ModelConfig` / `CacheConfig` / `ParallelConfig` / `SchedulerConfig`
  - 5.1.3 `environ.py`：集中式环境变量
- 5.2 `LLMEngine`
  - 5.2.1 初始化顺序：tokenizer → workers → load_model → profile → cache → scheduler
  - 5.2.2 `add_request`、`step`、`post_process`
  - 5.2.3 `DeferredRelease` 与跨步资源生命周期
- 5.3 Worker 与 ModelRunner
  - 5.3.1 `execute_model_step` 的四段式流水
  - 5.3.2 输入准备、CUDA Graph 选择与 padding
  - 5.3.3 `execute_main_model` / `execute_sample` 的无同步约束
  - 5.3.4 profile run、capture、decode 性能曲线
- 5.4 C++ Scheduler
  - 5.4.1 `EngineType`：NORMAL / PROMPT / COMPLETION
  - 5.4.2 Prefill 调度：chunked、CP 对齐、多模态边界、checkpoint
  - 5.4.3 Decode 调度：LCFS、抢占与重算、prompted 队列
  - 5.4.4 DP 步型投票（ALL / ANY）
  - 5.4.5 `SchedulerOutputs` 与统计量
- 5.5 序列、BlockManager 与状态管理
  - 5.5.1 `Sequence` / `SequenceMetadata`
  - 5.5.2 `Blocks` 空闲链表与引用计数
  - 5.5.3 block + state 联合分配
- 5.6 InputMetadata 与 C++ 预处理
  - 5.6.1 `InputMetadataV4` 字段地图
  - 5.6.2 `prepare_prompt_v4` / `prepare_decode_v4`
  - 5.6.3 CP mirror、YoCo 截断、DFlash 字段
- 5.7 CacheEngine 与 KV Cache
  - 5.7.1 `KVCache` 五类视图
  - 5.7.2 `StateBlock`：stored / dropped 分区
  - 5.7.3 `KVCacheManager` 逐层布局
  - 5.7.4 分配、profile、落盘接口
- 5.8 模型构建与加载
  - 5.8.1 `V41Builder` 与 meta 构建
  - 5.8.2 HF 覆盖校验、materialize、FP8/FP4 打包
  - 5.8.3 权重合并与 dummy 初始化
- 5.9 Sampler 与投机验证
  - 5.9.1 Prompt / Decode 两条路径
  - 5.9.2 force-topk、golden、blacklist、prompt logprobs
  - 5.9.3 验证、接受/拒绝、残差重采样
  - 5.9.4 MTP info pool 与置信度调度
- 5.10 结构化输出
  - 5.10.1 xgrammar 编译与缓存
  - 5.10.2 Matcher：触发词、KMP、rollback、EOS
  - 5.10.3 bitmask 应用与异步构建
- 5.11 多模态与 VE
  - 5.11.1 `MultimodalProcessor` 与共享池
  - 5.11.2 ViT / aligner / patch 元数据
- 5.12 平台抽象与 deep_cluster
- 5.13 EPLB
  - 5.13.1 logical / physical / redundant expert
  - 5.13.2 复制策略与贪心放置
  - 5.13.3 统计窗口、重排周期、权重 shuffle
- 5.14 并行与通信
  - 5.14.1 DP/CP/EP/ENP 进程组
  - 5.14.2 CP mirror 与 all-gather
  - 5.14.3 DeepEP dispatch/combine
  - 5.14.4 userbuffer 与 SM 配额
- 5.15 测试、构建与工具链

## 6. 模型执行与算子实现
- 6.1 `DeepseekModel` 前向总览
- 6.2 DecoderLayer：Attn-Norm / Attention / FFN-Norm / MoE
- 6.3 V4Attention
  - 6.3.1 Q/KV 低秩投影与融合
  - 6.3.2 KV 压缩（C1/C2）与 Compressor
  - 6.3.3 SWA 路径与 ring cache
  - 6.3.4 稀疏 indexer：MQA logits、candidate blocks、topk
  - 6.3.5 Prefill / Decode 的 FlashMLA 调用
  - 6.3.6 跨层共享：share-KV / share-index / share-index-K
  - 6.3.7 YoCo 截断：mask / compact、边界重排
- 6.4 DFlashAttention 与 DFlash 模块
- 6.5 MoE 实现
  - 6.5.1 权重重排与 FP4 打包
  - 6.5.2 gate、EPLB 映射、dispatch/combine
  - 6.5.3 grouped GEMM、SwiGLU、共享专家
  - 6.5.4 Mega MoE
- 6.6 mHC 与 mHC Head
- 6.7 Engram
- 6.8 量化与 GEMM（ReplicatedLinear、DeepGEMM、TileKernels）
- 6.9 TileLang 自定义算子清单与平台分派
- 6.10 采样与结构化输出的融合点

## 7. 调度与执行流水线
- 7.1 同步执行基线：schedule → forward → sample → post-process
- 7.2 Zero-Overhead Scheduling（ZOS）
  - 7.2.1 流水重排与延迟 post-process
  - 7.2.2 seqlens 修正、remap、D2H/H2D
  - 7.2.3 抢占与在途请求
- 7.3 CUDA Graph
  - 7.3.1 分档捕获、padding、输入元数据
  - 7.3.2 DFlash 图的合并捕获
  - 7.3.3 decode 性能曲线与缓存
- 7.4 MTP varlen 调度
- 7.5 PD 分离接口与开源边界
- 7.6 端到端时序图（同步 vs ZOS）

## 8. KV Cache、内存与缓存复用
- 8.1 分页 Block 布局
- 8.2 主 KV / SWA / index KV 的格式
- 8.3 State Cache 与 128 token 落盘粒度
- 8.4 前缀缓存、fork 与重算
- 8.5 SWA bounded replay
- 8.6 KV 存储集群与 swap-in（生产依赖）
- 8.7 显存核算与 profile
- 8.8 Batch invariant 对缓存/算子的约束

## 9. 并行、通信与负载均衡
- 9.1 并行维度与 rank 重排
- 9.2 Attention DP + Balanced CP
- 9.3 MoE EP + Wide-EP
- 9.4 EPLB 算法与运行时
- 9.5 DeepEP 通信与 userbuffer
- 9.6 EPD 分离与生产部署
- 9.7 故障、扩缩容与配额（生产）

## 10. 正确性、测试与性能
- 10.1 精度基线：PPL golden 与逐样本比对
- 10.2 Batch invariant 与确定性采样
- 10.3 测试矩阵与 CI 标记
- 10.4 性能基准方法与指标
- 10.5 性能图表与生产数据
- 10.6 常见问题定位（CUDA Graph OOM、illegal memory access、同步守卫等）

## 11. 调优与部署指南
- 11.1 关键参数表
- 11.2 Prefill 调优
- 11.3 Decode 调优
- 11.4 显存与并行度选择
- 11.5 YoCo / DFlash / EPLB 调优
- 11.6 昇腾平台注意事项
- 11.7 常见故障与排查顺序

## 12. 附录
- 12.1 术语表
- 12.2 文件与 API 索引
- 12.3 与 vLLM / SGLang / TensorRT-LLM 的对比表
- 12.4 后续学习路线

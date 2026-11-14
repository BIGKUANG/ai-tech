# 提示词：生成 SGLang 项目介绍文档

## 角色
你是一名资深 LLM 推理框架架构师与技术文档作者，擅长把大型工程代码讲清楚。

## 任务
阅读目标项目 `/data/home/shendakuang/workspace/framework/sglang`，输出一份中文技术介绍文档（Markdown，写入 `/data/home/shendakuang/workspace/framework/docs/`）。以真实代码目录与文件为准，不得臆造模块；引用处尽量标注 `文件路径:行号`。

## 输出要求（按此结构，逐条覆盖）
1. **代码结构**：给出从仓库根到 `python/sglang/` 的目录树，说明 `python/`（Python 主体）、`rust/` 与 `sgl-model-gateway/`（Rust 组件）、`proto/`、`docs/`、`examples/`、`benchmark/` 等顶层目录职责。
2. **架构设计**：用 mermaid 绘图说明分层——前端入口（`launch_server.py`/`api`）→ 调度（`srt/managers` 的 scheduler/tokenizer/detokenizer）→ 执行（`model_executor`、`layers`、`kernels`）→ 显存与缓存（`mem_cache` RadixAttention、`mem_cache`/`disaggregation`）→ 并行与运行时（`distributed`、`platforms`、`speculative`）。画出请求流向与关键数据流。
3. **模块功能（本章篇幅最大）**：对应第 2 章逐个展开，至少覆盖：`entrypoints`、`managers`（调度）、`model_executor`、`models`+`model_loader`、`layers`+`kernels`、`mem_cache`（RadixAttention）、`mem_cache`/PD 分离 `disaggregation`、`speculative`（推测解码）、`sampling`、`constrained`（约束解码）、`lora`、`multimodal`、`distributed`+`eplb`/`elastic_ep`、`platforms`（多硬件后端）、`compilation`、`observability`。每个模块写清：职责、关键文件、与其他模块的交互。
4. **创新点与优缺点**：重点写 RadixAttention 前缀缓存复用、以 `sglang` 前端 DSL 表达复杂控制流（并行/分支/循环/约束）、PD 分离、投机解码、前后端协同调度、多硬件统一抽象。分别列出优点与局限/代价。
5. **项目应用**：列举典型场景（大规模在线服务、Agentic 工作流、结构化/JSON 输出、多模态、RL rollout 数据生成、扩散模型生成）及生态集成（SGLang Diffusion/Omni、SpecForge、Miles/slime、SMG/Ray Serve/llm-d 等）。
6. **未来展望**：结合仓库现状给出可验证的演进方向（更多后端硬件、更优调度、长上下文与前缀缓存增强、编译与内核融合、多模态/扩散、分布式弹性与容错等）。

## 约束
- 语言：中文；术语保留英文原名（如 RadixAttention）。
- 结构清晰、可读，善用表格、mermaid 图与代码引用，避免空话。
- 全文基于实际代码，不确定处标注“待确认”，不要编造 API。

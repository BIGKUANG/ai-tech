# Mega Gate API

- 源码：`csrc/apis/mega_gate.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM100

## bf16_mega_gate

```python
bf16_mega_gate(
    x, weight, num_topk,
    use_shared_as_routed, num_shared_experts,
    routed_scaling_factor, ep_rank,
    scoring_func="sqrtsoftplus",
    mask=None,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    to_physical_map=None,
    logical_count=None,
    unmapped_topk_idx=None,
    force_random=None,
    out=None,
) -> (topk_idx, topk_weights)
```

把 **BF16 gate GEMM** 与 **MoE top-k 路由**融合成单个内核：先计算 `logits = X @ W.T`，再用 `sqrtsoftplus` 打分，并完成 top-k 选择、偏置、物理 expert 映射、权重归一化。

**形状与类型**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `x` | `[T, H]`，`torch.bfloat16`，连续 | token 特征；`H % 256 == 0` |
| `weight` | `[E, H]`，`torch.bfloat16`，连续 | routed expert 门控权重；`E <= 512` 且 `E % kNumExpertsPerLaneVector == 0` |
| `num_topk` | int | `0 < num_topk <= E` |
| `use_shared_as_routed` | bool | 是否把 shared expert 当作 routed 一并路由 |
| `num_shared_experts` | int | `use_shared_as_routed=True` 时为 1 或 2；要求 `num_topk % num_shared_experts == 0` 且 `E % (num_topk / num_shared_experts) == 0` |
| `routed_scaling_factor` | float | 有限值，作用于输出的权重 |
| `ep_rank` | int | 当前 EP rank，`>= 0` |
| `scoring_func` | str | 目前仅支持 `"sqrtsoftplus"` |
| `mask` | `[T]`，`torch.bool`，可选 | token 掩码（无效 token 输出 `-1` / `0`） |
| `bias` | `[E]`，`torch.float`，可选 | routing bias（参与排序） |
| `image_bias` | `[E]`，`torch.float`，可选 | 图像 token 专用 bias；必须与 `image_token_mask` 同时提供 |
| `image_token_mask` | `[T]`，`torch.bool`，可选 | 图像 token 掩码 |
| `fix_routing_mask` | `[T]`，`torch.bool`，可选 | 固定路由的 token；**要求同时提供 `unmapped_topk_idx`** |
| `to_physical_map` | `[E + S, dup]`，`torch.int`，可选 | 逻辑 expert → 物理 expert 映射；必须与 `logical_count` 同时提供 |
| `logical_count` | `[E + S]`，`torch.int`，可选 | 每个逻辑 expert 的物理副本数 |
| `unmapped_topk_idx` | `[T, num_topk]`，`torch.int64`，可选 | 输入 / 输出：`stride(1) == 1`（行可有 stride，槽位连续）。接收逻辑 top-k 索引；`fix_routing_mask` 置位的 token 还会从中读取要保留的路由 |
| `force_random` | `[T]`，`torch.bool`，可选 | 强制随机路由 |
| `out` | `(topk_idx, topk_weights)`，可选 | 预分配输出：`topk_idx` `[T, K']` `int64`、`topk_weights` `[T, K']` `FP32`，二者连续 |

其中 `S = num_shared_experts`（仅当 `use_shared_as_routed` 时非零），`K' = num_topk + S`。

**返回**

- `topk_idx`：`[T, K']`，`int64`（物理 expert 索引；无效 token 为 `-1`）。
- `topk_weights`：`[T, K']`，`float32`。

**行为说明**

- 内核按 `score + bias` 排序 expert，但输出权重是**未加 bias 的 score**，在 top-k 上归一化后乘 `routed_scaling_factor`。
- `unmapped_topk_idx` 接收逻辑 top-k 索引（便于调用方在逻辑空间使用）。
- 置位 `fix_routing_mask` 的 token：其路由被固定，`unmapped_topk_idx` 同时作为输入被读取。
- 无效 token（`mask == False`）：`topk_idx == -1`、`topk_weights == 0`；若提供 `unmapped_topk_idx`，其对应行也为 `-1`。
- 无 physical map 时，索引直接落在 `[0, E)`；有 map 时，映射到物理索引。
- `num_tokens == 0` 时返回空输出。
- 仅 SM100；`num_physical_topk <= kNumMaxTopk (32)`、`num_tokens <= kNumMaxTokens`。

**示例**

```python
import torch, deep_gemm

num_tokens, hidden, num_routed_experts = 4096, 7168, 384
num_topk, num_shared_experts = 6, 1
x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
weight = torch.randn((num_routed_experts, hidden), dtype=torch.bfloat16, device='cuda')

topk_idx, topk_weights = deep_gemm.bf16_mega_gate(
    x, weight, num_topk,
    use_shared_as_routed=False, num_shared_experts=num_shared_experts,
    routed_scaling_factor=1.5, ep_rank=0,
)
```

## 相关

- MoE 后端（Mega MoE）：[api-mega-moe.md](./api-mega-moe.md)
- 测试参考：`tests/test_mega_gate.py`

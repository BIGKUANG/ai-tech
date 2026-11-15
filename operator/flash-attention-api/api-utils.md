# 工具 API（测试 / 生成 / 基准 / 分布式）

- 源码：`flash_attn/utils/`

---

## 测试工具（`utils/testing.py`）

| 名称 | 说明 |
| --- | --- |
| `generate_random_padding_mask(max_seqlen, batch_size, device, mode="random", zero_lengths=False)` | 生成随机 padding mask（bool） |
| `generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None, ...)` | 由 padded 张量生成 varlen 张量与还原函数 |
| `construct_local_mask(seqlen_q, seqlen_k, window_size, sink_token_length, ..., device=None)` | 构造滑动窗口局部掩码 |
| `construct_chunk_mask(seqlen_q, seqlen_k, attention_chunk, ..., device=None)` | 构造 chunk 掩码 |
| `attention_ref(...)` | 纯 PyTorch 参考注意力，用于校验 |

### attention_ref

```python
attention_ref(
    q, k, v,
    query_padding_mask=None, key_padding_mask=None, key_leftpad=None,
    attn_bias=None, dropout_p=0.0, dropout_mask=None, causal=False,
    qv=None, q_descale=None, k_descale=None, v_descale=None,
    window_size=(None, None), attention_chunk=0, sink_token_length=0,
    learnable_sink=None, softcap=0.0,
    upcast=True, reorder_ops=False, intermediate_dtype=None,
)
```

**参数**

- `q`：`(batch, seqlen_q, nheads, head_dim)`
- `k`：`(batch, seqlen_k, nheads, head_dim)`
- `v`：`(batch, seqlen_k, nheads, head_dim_v)`
- `qv`：`(batch, seqlen_q, nheads, head_dim_v)`（MLA）
- `query_padding_mask` / `key_padding_mask`：`(batch, seqlen)`
- `attn_bias`：可广播到 `(batch, nheads, seqlen_q, seqlen_k)`
- `upcast`：是否升到 fp32 计算再转回
- `reorder_ops`：是否改变运算顺序（不改变数学）以估计数值误差

**返回**：`(output, attention)`，其中 `output` 为 `(batch, seqlen_q, nheads, head_dim_v)`，`attention` 为 `(batch, nheads, seqlen_q, seqlen_k)`（dropout 后的 softmax）。

**示例**

```python
from flash_attn.utils.testing import attention_ref

ref_out, attn = attention_ref(q, k, v, causal=True)
```

---

## 基准工具（`utils/benchmark.py`）

| 名称 | 说明 |
| --- | --- |
| `benchmark_forward(fn, *inputs, desc="", verbose=True, amp=True, ...)` | 测前向耗时/显存 |
| `benchmark_backward(fn, *inputs, grad, desc="", ...)` | 测反向 |
| `benchmark_combined(fn, grad, *inputs, ...)` | 前向+反向合并测 |
| `benchmark_fwd_bwd(fn, *inputs, ...)` | 前向与反向分别测 |
| `benchmark_all(fn, *inputs, ...)` | 综合测量 |
| `pytorch_profiler(fn, *inputs, ...)` | 用 torch profiler 分析 |
| `benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs)` | 测显存占用 |

**示例**

```python
from flash_attn.utils.benchmark import benchmark_forward

benchmark_forward(flash_attn_func, q, k, v, causal=True)
```

---

## 生成工具（`utils/generation.py`）

### InferenceParams

```python
@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size): ...
```

传给模型以高效复用 KV Cache 的推理参数。

### 采样函数

| 名称 | 说明 |
| --- | --- |
| `modify_logits_for_top_k_filtering(logits, top_k)` | 原地把非 top-k 的 logits 置 `-inf` |
| `modify_logits_for_top_p_filtering(logits, top_p)` | 原地 nucleus(top-p) 过滤 |
| `sample(logits, top_k=1, top_p=0.0, temperature=1.0)` | 采样 |
| `sample_speculative(logits, logits_draft, tokens_draft, ...)` | 投机采样 |

### 解码

| 名称 | 说明 |
| --- | --- |
| `decode(input_ids, model, max_length, temperature=1.0, top_k=None, top_p=None, ...)` | 自回归解码 |
| `decode_speculative(input_ids, model, model_draft, ...)` | 投机解码 |
| `allocate_inference_cache(batch_size, max_seqlen, ...)` | 分配 KV Cache |
| `update_graph_cache(...)` / `capture_graph(...)` | CUDA Graph 捕获缓存 |
| `GenerationMixin` | 提供 `generate(...)` 与 `allocate_inference_cache(...)` 的 mixin |

---

## 分布式工具（`utils/distributed.py`）

| 名称 | 说明 |
| --- | --- |
| `all_gather_raw(input_, process_group, async_op=False)` | 原始 all-gather |
| `reduce_scatter_raw(input_, process_group, async_op=False)` | 原始 reduce-scatter |
| `all_reduce_raw(input_, process_group, async_op=False)` | 原始 all-reduce |
| `AllGatherFunc` / `ReduceScatterFunc` / `AllReduceFunc` | 对应 autograd.Function |
| `sync_shared_params(model, process_group)` | 同步共享参数 |
| `allreduce_sequence_parallel_grad(model, process_group)` | 序列并行的梯度 all-reduce |
| `get_dim_for_local_rank(dim, world_size, local_rank, multiple_of=1)` | 计算张量并行下的局部分维 |

---

## 其他工具

| 文件 | 名称 | 说明 |
| --- | --- | --- |
| `library.py` | `triton_op(...)` | 把 Triton 函数包装为 `torch.library` custom op |
| `pretrained.py` | `state_dict_from_pretrained(model_name, device=None, dtype=None)` | 从 HuggingFace 加载 state dict |
| `torch.py` | `custom_amp_decorator(...)` | 自定义 AMP 装饰器 |
| `benchmark.py` | 见上文 | 性能基准 |

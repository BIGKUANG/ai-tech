# POD（Prefill-On-Decode）注意力（`flashinfer.pod`）

> 源码：`flashinfer/pod.py`
> 上游文档：`docs/api/pod.rst`
> 适用版本：FlashInfer 0.7.2

POD（Preflight-On-Decode）注意力在**一次内核启动**中并发执行单请求 prefill 内核与批量 decode
内核，适合把分块 prefill（chunked prefill）与进行中的 decode 请求重叠的服务栈。该机制首次提出
于论文 <https://arxiv.org/abs/2410.18038>。本模块提供两个 Wrapper：`PODWithPagedKVCacheWrapper`
（prefill 侧为连续单请求 K/V，decode 侧为分页）与 `BatchPODWithPagedKVCacheWrapper`（两侧均为
批量分页）。辅助数据结构可在多个 Transformer 层间复用。

## 导入

```python
from flashinfer import PODWithPagedKVCacheWrapper, BatchPODWithPagedKVCacheWrapper
# 或
from flashinfer.pod import PODWithPagedKVCacheWrapper
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `PODWithPagedKVCacheWrapper` | 类 | 单请求 prefill + 批量分页 decode 的 POD 注意力。 |
| `BatchPODWithPagedKVCacheWrapper` | 类 | 批量分页 prefill + 批量分页 decode 的 POD 注意力。 |

## API 详情

### `PODWithPagedKVCacheWrapper`

```python
PODWithPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
    use_cuda_graph: bool = False,
    paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
    paged_kv_indices_buffer: Optional[torch.Tensor] = None,
    paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    jit_args: Optional[List[Any]] = None,
) -> None
```

**功能**：带分页 kv-cache 的 POD 注意力 Wrapper（prefill 侧使用单请求连续 K/V）。辅助数据结构可
跨多层 decode 复用。

**生命周期**：构造 `__init__` → `plan`（构造辅助数据结构）→ `run`。`begin_forward` 是 `plan` 的
别名。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲，用于 split-k 中间
  结果。推荐 128MB，设备须与输入张量一致。
- `kv_layout`（`str`，默认 `"NHD"`）：输入 k/v 张量的 layout，`"NHD"` 或 `"HND"`。
- `use_cuda_graph`（`bool`，默认 `False`）：是否为批量 decode 启用 CUDA Graph；启用时辅助数据结构
  存于提供的缓冲中，且 `batch_size` 在 wrapper 生命周期内不能改变。
- `paged_kv_indptr_buffer`（`Optional[torch.Tensor]`，默认 `None`）：用户预留的 GPU 缓冲，存分页
  kv cache 的 indptr，大小 `[batch_size + 1]`。仅 `use_cuda_graph=True` 时需要。
- `paged_kv_indices_buffer`（`Optional[torch.Tensor]`，默认 `None`）：存分页 kv cache 页索引，须
  足以存放生命周期内最大页数（`max_num_pages`）。仅 CUDA Graph 模式需要。
- `paged_kv_last_page_len_buffer`（`Optional[torch.Tensor]`，默认 `None`）：存最后一页 entry 数，
  大小 `[batch_size]`。仅 CUDA Graph 模式需要。
- `jit_args`（`Optional[List[Any]]`，默认 `None`）：提供时用于创建 JIT 模块，否则使用默认注意力
  实现。

**属性**

- `is_cuda_graph_enabled`（`bool`）：返回 `use_cuda_graph`。

#### `plan`（= `begin_forward`）

```python
plan(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    q_data_type: Optional[Union[str, torch.dtype]] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
) -> None
```

**功能**：为给定问题规格规划 POD 的批量 decode。必须在任何 `run` 之前调用；辅助数据结构在此
创建并缓存供多次 run 复用。`num_qo_heads` 必须是 `num_kv_heads` 的倍数；不等时使用 GQA。

**参数**

- `indptr`（`torch.Tensor`）：分页 kv cache 的 indptr，形状 `[batch_size + 1]`。
- `indices`（`torch.Tensor`）：分页 kv cache 的页索引，形状 `[qo_indptr[-1]]`。
- `last_page_len`（`torch.Tensor`）：每个请求最后一页的 entry 数，形状 `[batch_size]`。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：头维度。
- `page_size`（`int`）：分页 kv cache 的页大小。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）：`NONE`/`ROPE_LLAMA`/`ALIBI`。
- `window_left`（`int`，默认 `-1`）：注意力窗口左边界（含）；`-1` 表示全序列长度。
- `q_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `"float16"`）：query 数据类型。
- `kv_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：key/value 数据类型；默认
  `q_data_type`。
- `data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：query 与 key/value 共同的数据
  类型。**已废弃**，请改用 `q_data_type`/`kv_data_type`。
- `sm_scale`（`Optional[float]`，默认 `None`）：softmax scale；默认 `1 / sqrt(head_dim)`。缓存在
  wrapper 上并在 `run` 时复用。
- `rope_scale`（`Optional[float]`，默认 `None`）：RoPE 插值 scale；仅在
  `pos_encoding_mode != "NONE"` 时参考；默认 `1.0`。
- `rope_theta`（`Optional[float]`，默认 `None`）：RoPE 频率 base；仅在非 `NONE` 时参考；默认 `1e4`。
- `non_blocking`（`bool`，默认 `True`）：是否异步把输入张量拷贝到设备。

**注意事项**

- `plan` 不能在 CUDA Graph 或 `torch.compile` 中使用。
- CUDA Graph 模式下 `batch_size` 必须与初始化时一致，且 `indices` 长度不能超过分配的缓冲。

#### `run`

```python
run(
    q_p: torch.Tensor,
    k_p: torch.Tensor,
    v_p: torch.Tensor,
    q_d: torch.Tensor,
    paged_kv_cache_d: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    custom_mask_p: Optional[torch.Tensor] = None,
    packed_custom_mask_p: Optional[torch.Tensor] = None,
    causal_p: bool = False,
    kv_layout_p: str = "NHD",
    pos_encoding_mode_p: str = "NONE",
    sm_scale_p: Optional[float] = None,
    window_left_p: int = -1,
    rope_scale_p: Optional[float] = None,
    rope_theta_p: Optional[float] = None,
    return_lse_p: bool = False,
    custom_mask_d: Optional[torch.Tensor] = None,
    packed_custom_mask_d: Optional[torch.Tensor] = None,
    causal_d: bool = False,
    kv_layout_d: str = "NHD",
    pos_encoding_mode_d: str = "NONE",
    sm_scale_d: Optional[float] = None,
    window_left_d: int = -1,
    rope_scale_d: Optional[float] = None,
    rope_theta_d: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    return_lse_d: bool = False,
    use_fp16_qk_reduction: bool = False,
    enable_pdl: Optional[bool] = None,
    *args,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**功能**：为一批请求计算 POD 融合注意力。单次启动同时运行单请求 prefill（`q_p` 对 `k_p`/`v_p`）
与批量 decode（`q_d` 对 `paged_kv_cache_d`），使分块 prefill/连续批处理迭代中的 prefill 与 decode
token 共享内核启动与调度资源。

**参数**

- `q_p`（`torch.Tensor`）：prefill query，形状 `[qo_len, num_qo_heads, head_dim]`。
- `k_p`（`torch.Tensor`）：prefill key，layout 与 `kv_layout_p` 匹配。
- `v_p`（`torch.Tensor`）：prefill value，layout 与 `kv_layout_p` 匹配。
- `q_d`（`torch.Tensor`）：decode query，形状 `[batch_size, num_qo_heads, head_dim]`。
- `paged_kv_cache_d`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：decode 请求的
  分页 KV cache。layout 与 `__init__` 设置的 wrapper `kv_layout` 匹配。
- `custom_mask_p`、`packed_custom_mask_p`（`Optional[torch.Tensor]`）：prefill 侧可选稠密/位打包
  自定义掩码，layout 见 `single_prefill_with_kv_cache`。
- `causal_p`（`bool`，默认 `False`）：是否对 prefill 侧施加因果掩码。
- `kv_layout_p`（`str`，默认 `"NHD"`）：`k_p`/`v_p` 的 layout，`"NHD"` 或 `"HND"`。
- `pos_encoding_mode_p`（`str`，默认 `"NONE"`）：prefill 侧位置编码模式。
- `sm_scale_p`（`Optional[float]`，默认 `None`）：prefill 侧 softmax scale；默认 `1/sqrt(head_dim)`。
- `window_left_p`（`int`，默认 `-1`）：滑窗 prefill 左窗口；`-1` 禁用。
- `rope_scale_p`、`rope_theta_p`（`Optional[float]`）：prefill 侧 RoPE scale/theta；默认 `1.0`/`1e4`。
- `return_lse_p`（`bool`，默认 `False`）：为 prefill 内核分配 LSE 缓冲。**注意：缓冲会被内核分配并
  填充，但当前不返回给调用方**——`run` 始终只返回 `(out_p, out_d)`。这是当前 POD wrapper 的已知
  限制。
- `custom_mask_d`、`packed_custom_mask_d`（`Optional[torch.Tensor]`）：decode 侧可选自定义掩码。
- `causal_d`（`bool`，默认 `False`）：是否对 decode 侧施加因果掩码。
- `kv_layout_d`（`str`，默认 `"NHD"`）：**当前被忽略**：decode KV layout 始终取自 `__init__` 的
  `kv_layout`；此参数为与 `kv_layout_p` 对称而接受，但内核不使用其值。
- `pos_encoding_mode_d`（`str`，默认 `"NONE"`）：**当前被忽略**：由 `plan` 的
  `self._pos_encoding_mode` 覆盖。
- `sm_scale_d`（`Optional[float]`，默认 `None`）：**当前被忽略**：由 `plan` 的 `self._sm_scale`
  覆盖（其本身默认 `1/sqrt(head_dim)`）。
- `window_left_d`（`int`，默认 `-1`）：**当前被忽略**：由 `plan` 的 `self._window_left` 覆盖。
- `rope_scale_d`、`rope_theta_d`（`Optional[float]`）：**当前被忽略**：由 `plan` 的
  `self._rope_scale`/`self._rope_theta` 覆盖。
- `q_scale`、`k_scale`、`v_scale`（`Optional[float]`）：decode 侧 FP8 标定 scale，折入 decode
  `sm_scale`（`q_scale`、`k_scale`）或内核输出（`v_scale`）。
- `return_lse_d`（`bool`，默认 `False`）：为 decode 内核分配 LSE 缓冲。同 `return_lse_p` 的注意
  事项：分配但不返回。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否以 FP16 累加 QK（精度更低、吞吐更高）。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：Programmatic Dependent Launch 开关。为 `None`
  时由 wrapper 从 query 设备自动探测。
- `*args`：保留给未来内核参数的前向兼容。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(out_p, out_d)`，prefill 输出（形状
  `[qo_len, num_qo_heads, head_dim]`）与 decode 输出（形状
  `[batch_size, num_qo_heads, head_dim]`）。即使 `return_lse_p`/`return_lse_d` 为 `True`，LSE
  也**不**包含在返回值中。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
decode_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace_buffer, "NHD")
decode_wrapper.plan(kv_page_indptr, kv_page_indices, kv_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    pos_encoding_mode="NONE", data_type=torch.float16)
out_p, out_d = decode_wrapper.run(q_p, k_p, v_p, q_d, kv_cache_d)
```

---

### `BatchPODWithPagedKVCacheWrapper`

```python
BatchPODWithPagedKVCacheWrapper(
    float_workspace_buffer: torch.Tensor,
    kv_layout: str = "NHD",
) -> None
```

**功能**：带分页 kv-cache 的批量 POD 注意力（prefill 与 decode 两侧均为批量分页）。辅助数据结构
可跨多层 decode 复用。

**生命周期**：构造 `__init__` → `plan` → `run`。

#### `__init__`

**参数**

- `float_workspace_buffer`（`torch.Tensor`）：用户预留的 float workspace 缓冲，用于 split-k 中间
  结果。推荐 128MB，设备须与输入张量一致。内部会沿第 0 维均分为 prefill 与 decode 两份。
- `kv_layout`（`str`，默认 `"NHD"`）：输入 k/v 张量的 layout，`"NHD"` 或 `"HND"`。

**属性**

- `is_cuda_graph_enabled`（`bool`）：当前实现返回 `False`（该 wrapper 不启用 CUDA Graph 辅助缓冲）。

#### `plan`

```python
plan(
    qo_indptr_p: torch.Tensor,
    kv_indptr_p: torch.Tensor,
    kv_indices_p: torch.Tensor,
    last_page_len_p: torch.Tensor,
    qo_indptr_d: torch.Tensor,
    kv_indptr_d: torch.Tensor,
    kv_indices_d: torch.Tensor,
    last_page_len_d: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    q_data_type: Optional[Union[str, torch.dtype]] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
) -> None
```

**功能**：为给定问题规格规划 POD 的批量 prefill 与 decode。必须在任何 `run` 之前调用。`num_qo_heads`
必须是 `num_kv_heads` 的倍数；不等时使用 GQA。

**参数**

- `qo_indptr_p`（`torch.Tensor`）：prefill query/output 张量的 indptr，形状 `[batch_size + 1]`。
- `kv_indptr_p`（`torch.Tensor`）：prefill 分页 kv-cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_indices_p`（`torch.Tensor`）：prefill 分页 kv-cache 的页索引，形状 `[kv_indptr[-1]]`。
- `last_page_len_p`（`torch.Tensor`）：每个 prefill 请求最后一页的 entry 数，形状 `[batch_size]`。
- `qo_indptr_d`（`torch.Tensor`）：decode query/output 张量的 indptr，形状 `[batch_size + 1]`。
- `kv_indptr_d`（`torch.Tensor`）：decode 分页 kv-cache 的 indptr，形状 `[batch_size + 1]`。
- `kv_indices_d`（`torch.Tensor`）：decode 分页 kv-cache 的页索引，形状 `[kv_indptr[-1]]`。
- `last_page_len_d`（`torch.Tensor`）：每个 decode 请求最后一页的 entry 数，形状 `[batch_size]`。
- `num_qo_heads`（`int`）：query/output 头数。
- `num_kv_heads`（`int`）：key/value 头数。
- `head_dim`（`int`）：头维度。
- `page_size`（`int`）：分页 kv cache 的页大小。
- `pos_encoding_mode`（`str`，默认 `"NONE"`）：`NONE`/`ROPE_LLAMA`/`ALIBI`。
- `window_left`（`int`，默认 `-1`）：注意力窗口左边界（含）；`-1` 表示全序列长度。
- `q_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `"float16"`）：query 数据类型。
- `kv_data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：key/value 数据类型；默认
  `q_data_type`。
- `data_type`（`Optional[Union[str, torch.dtype]]`，默认 `None`）：query 与 key/value 共用的数据
  类型。**已废弃**，请改用 `q_data_type`/`kv_data_type`。
- `sm_scale`（`Optional[float]`，默认 `None`）：softmax scale；默认 `1.0 / sqrt(head_dim_qk)`。
- `rope_scale`（`Optional[float]`，默认 `None`）：RoPE 插值 scale；默认 `1.0`。
- `rope_theta`（`Optional[float]`，默认 `None`）：RoPE theta；默认 `1e4`。
- `non_blocking`（`bool`，默认 `True`）：是否异步拷贝输入张量到设备。

**注意事项**

- `plan` 不能在 CUDA Graph 或 `torch.compile` 中使用。

#### `run`

```python
run(
    q_p: torch.Tensor,
    paged_kv_cache_p: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    q_d: torch.Tensor,
    paged_kv_cache_d: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    custom_mask_p: Optional[torch.Tensor] = None,
    packed_custom_mask_p: Optional[torch.Tensor] = None,
    causal_p: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    return_lse: bool = False,
    use_fp16_qk_reduction: bool = False,
    enable_pdl: Optional[bool] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
]
```

**功能**：计算批量 POD 融合注意力。在同一次内核启动中运行批量分页 prefill（`q_p` 对
`paged_kv_cache_p`）与批量分页 decode（`q_d` 对 `paged_kv_cache_d`），共享调度与执行资源。所有
prefill/decode 的形状与策略参数（位置编码、滑窗、sm_scale、RoPE 等）取自 `plan` 缓存的值。

**参数**

- `q_p`（`torch.Tensor`）：prefill query，形状 `[qo_indptr_p[-1], num_qo_heads, head_dim]`。
- `paged_kv_cache_p`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：prefill 请求的
  分页 KV cache。layout 与 `__init__` 的 `kv_layout` 匹配。
- `q_d`（`torch.Tensor`）：decode query，形状 `[batch_size_d, num_qo_heads, head_dim]`。
- `paged_kv_cache_d`（`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`）：decode 请求的
  分页 KV cache。
- `custom_mask_p`（`Optional[torch.Tensor]`，默认 `None`）：prefill 侧可选稠密自定义掩码（当
  `packed_custom_mask_p` 为 `None` 时自动打包）。
- `packed_custom_mask_p`（`Optional[torch.Tensor]`，默认 `None`）：prefill 侧可选位打包自定义掩码。
- `causal_p`（`bool`，默认 `False`）：是否对 prefill 侧施加因果掩码。
- `q_scale`、`k_scale`、`v_scale`（`Optional[float]`，默认 `None`）：decode 侧 FP8 标定 scale，
  折入 decode `sm_scale`（`q_scale`、`k_scale`）或内核输出（`v_scale`）。
- `return_lse`（`bool`，默认 `False`）：是否返回 prefill 与 decode 的 LSE 张量。
- `use_fp16_qk_reduction`（`bool`，默认 `False`）：是否以 FP16 累加 QK。
- `enable_pdl`（`Optional[bool]`，默认 `None`）：PDL 开关；`None` 时自动探测。

**返回**

- `Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple, Tuple]]`：默认 `(out_p, out_d)`。当
  `return_lse=True` 时返回 `((out_p, lse_p), (out_d, lse_d))`。

**示例**

```python
import torch
import flashinfer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(workspace_buffer, "NHD")
wrapper.plan(
    p_q_indptr, p_kv_indptr, kv_indices_p, last_page_len_p,
    d_q_indptr, d_kv_indptr, kv_indices_d, last_page_len_d,
    num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
    head_dim=head_dim, page_size=page_block_size,
    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
)
o_p_batch, o_d_batch = wrapper.run(q_p, kv_p, q_d, kv_d, causal_p=True)
```

## 相关链接 / 参考

- 上游 `docs/api/pod.rst`
- 源码 `flashinfer/pod.py`
- POD 论文：https://arxiv.org/abs/2410.18038

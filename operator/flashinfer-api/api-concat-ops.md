# 拼接 / 打包算子（`flashinfer.concat_ops` 与 `flashinfer.mla_kv_pack`）

> 源码：`flashinfer/concat_ops.py`、`flashinfer/mla_kv_pack.py`
> 上游文档：`docs/api/concat_ops.rst`
> 适用版本：FlashInfer 0.7.2

本页覆盖两个相关联的模块：

- `flashinfer.concat_ops`：将张量拼接 / 打包成 MLA 风格注意力布局的辅助算子。
  `flashinfer.dsv3_ops` 为 DeepSeek-V3 调用点重新导出 `concat_mla_k`。
- `flashinfer.mla_kv_pack`：融合的 MLA context K/V 打包与 fp8 e4m3 量化
  （`key = [k_nope ‖ k_pe]`，`value = v`），面向接受分离 fp8 Q/K/V 的 ragged MLA prefill 内核。

## 导入

```python
from flashinfer.concat_ops import concat_mla_k
# DeepSeek-V3 调用点亦可：
from flashinfer.dsv3_ops import concat_mla_k

from flashinfer.mla_kv_pack import concat_mla_kv_quant_fp8
```

## API 列表

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `concat_mla_k` | 函数 | 为 MLA 注意力拼接 `k_nope` 与 `k_rope`。 |
| `concat_mla_kv_quant_fp8` | 函数 | 融合 MLA context K/V 打包与 fp8 e4m3 量化。 |

## API 详情

### `concat_mla_k`

```python
concat_mla_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    backend: ConcatMLABackend = 'default',
) -> None
```

**功能**：为 MLA 注意力拼接 `k_nope` 与 `k_rope` 张量。该函数高效拼接：

- `k_nope`：逐 head 的 nope 值；
- `k_rope`：共享的 rope 值（广播到所有 head）。

支持的 dtype：`torch.bfloat16`、`torch.float16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`。

关键优化：

- 基于 warp 的处理与软件流水；
- 向量化内存访问（按 dtype 编译期分发）；
- 处理当前行时对下一行做 L2 预取；
- 在一个 chunk 内对所有 head 复用 rope 值的寄存器。

**参数**

- `k`（`torch.Tensor`）：输出张量，形状 `[num_tokens, num_heads, nope_dim + rope_dim]`，
  就地修改。
- `k_nope`（`torch.Tensor`）：`k` 的 nope 部分，形状 `[num_tokens, num_heads, nope_dim]`。
- `k_rope`（`torch.Tensor`）：`k` 的 rope 部分（共享），形状 `[num_tokens, 1, rope_dim]`，
  广播到所有 head。
- `backend`（`{"default", "cake"}`，可选，默认 `"default"`）：后端实现。仅源码可用的
  `"cake"` 后端在 SM100f 与精确 SM103a 上、针对文档所述固定形状与布局可用。

**返回**

- `None`：就地写入 `k`。

**注意事项**

- 该内核专门针对以下形状优化：`num_heads = 128`、`nope_dim = 128`、`rope_dim = 64`。

**示例**

```python
>>> import torch
>>> import flashinfer
>>> num_tokens = 2048
>>> num_heads = 128
>>> nope_dim = 128
>>> rope_dim = 64
>>> k = torch.empty(num_tokens, num_heads, nope_dim + rope_dim, dtype=torch.bfloat16, device="cuda")
>>> k_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=torch.bfloat16, device="cuda")
>>> k_rope = torch.randn(num_tokens, 1, rope_dim, dtype=torch.bfloat16, device="cuda")
>>> flashinfer.concat_ops.concat_mla_k(k, k_nope, k_rope)
```

---

### `concat_mla_kv_quant_fp8`

```python
concat_mla_kv_quant_fp8(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    *,
    nope_dim: int = NOPE_DIM,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能**：融合的 MLA context K/V 打包与 fp8 e4m3 量化。将逐 head 的 `kv_b_proj` 输出
`kv_nope = [k_nope | v]` 与共享的、已 RoPE 的 `k_pe` 转换为 ragged MLA prefill 内核消费的
fp8 `key = [k_nope | k_pe]` / `value = v` 缓冲区（例如
`trtllm_ragged_attention_deepseek` 使用分离 Q/K/V 时），一次内存遍历完成，而非两次类型转换
加两次跨步拷贝。

数值：该转换是饱和 e4m3 转换（round to nearest even；有限溢出与 `+-inf` → `+-448`；
NaN → `0x7F`），即 GPU 上自 torch 2.13 起 `Tensor.to(torch.float8_e4m3fn)` 产生的行为。
输出与该转换加切片拷贝逐位相等。

融合内核是一个生成的 Cake 程序（每个 head group 一个，即一个 warp 拥有的 head 对数，
由 token 数与 head 数规划），针对精确计算能力 10.0 与 10.3 构建。它服务 bf16 输入，
head 几何为 `nope_dim=128, rope_dim=64, v_dim=128`，张量 32 字节对齐（`kv_nope`、`key`、
`value` 连续；`k_pe` 连续或为更宽行主序工作区的列切片：最后一维 stride 为 1，行 stride
是 16 个元素的倍数且至少为 `rope_dim`），并落在包数据 allowlist
`flashinfer/mla_kv_pack_fp8_workloads.json` 的 head 数与 token 数范围内（1..128 heads，
1..131072 tokens）；其它所有调用走同等的饱和 e4m3 数值的可组合 torch 路径。设置
`FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`（调用时读取）强制可组合路径；
`mla_kv_pack._concat_mla_kv_quant_fp8_stats()` 报告分发计数器。

**参数**

- `kv_nope`（`torch.Tensor`）：形状 `[num_tokens, num_heads, nope_dim + v_dim]`，bf16
  （融合路径要求连续）。
- `k_pe`（`torch.Tensor`）：形状 `[num_tokens, rope_dim]` 或 `[num_tokens, 1, rope_dim]`，
  bf16，跨 head 共享；融合路径下连续，或为行主序工作区的列切片，其行 stride 是 16 个元素的
  倍数（例如 `[num_tokens, 576]` 缓冲区的 `latent[:, 512:]`）。
- `key`（`torch.Tensor`，可选）：形状 `[num_tokens, num_heads, nope_dim + rope_dim]`、
  dtype `float8_e4m3fn` 的输出；`None` 时分配。
- `value`（`torch.Tensor`，可选）：形状 `[num_tokens, num_heads, v_dim]`、dtype
  `float8_e4m3fn` 的输出；`None` 时分配。
- `nope_dim`（`int`，关键字，默认 `NOPE_DIM`）：`kv_nope` 最后一维的分割点（默认 128）。

**返回**

- `Tuple[torch.Tensor, torch.Tensor]`：`(key, value)`。

## 相关链接 / 参考

- `flashinfer.trtllm_ragged_attention_deepseek`：消费 `concat_mla_kv_quant_fp8` 产出的分离
  fp8 Q/K/V 的 ragged MLA prefill 内核。
- `flashinfer.dsv3_ops`：为 DeepSeek-V3 调用点重新导出 `concat_mla_k`。

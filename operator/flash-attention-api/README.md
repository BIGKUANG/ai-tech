# FlashAttention API 参考文档

本目录是 `flash-attention` 仓库中 `flash_attn` Python 包的 API 参考文档。
文档按模块拆分，每个文件介绍一组相关的公共 API，包含函数签名、参数说明、返回值、注意事项和示例。

> 源码路径：`/data/home/shendakuang/workspace/framework/flash-attention/flash_attn`

## 文档索引

| 文档 | 内容 | 对应源码 |
| --- | --- | --- |
| [api-attention-fa2.md](./api-attention-fa2.md) | FA2 标准注意力接口（`flash_attn_func` 系列） | `flash_attn/flash_attn_interface.py` |
| [flash_attn_varlen_func.md](./flash_attn_varlen_func.md) | **专题**：`flash_attn_varlen_func` API、算子实现、Triton 版本与 prefill/decode 数据流 | `flash_attn/flash_attn_interface.py`, `flash_attn/flash_attn_triton.py` |
| [flash_attn_with_kvcache.md](./flash_attn_with_kvcache.md) | **专题**：`flash_attn_with_kvcache` API、KV Cache 调度、Triton 参考实现与 prefill/decode 数据流 | `flash_attn/flash_attn_interface.py`, `hopper/flash_attn_interface.py` |
| [api-attention-fa3.md](./api-attention-fa3.md) | FA3 Hopper 注意力接口 | `hopper/flash_attn_interface.py` |
| [api-cute-fa4.md](./api-cute-fa4.md) | FA4 CuTeDSL 注意力接口（Blackwell/Hopper） | `flash_attn/cute/interface.py` |
| [api-rotary.md](./api-rotary.md) | 旋转位置编码（RoPE） | `flash_attn/layers/rotary.py` |
| [api-normalization.md](./api-normalization.md) | LayerNorm / RMSNorm / Dropout+Add 融合算子 | `flash_attn/ops/layer_norm.py`, `flash_attn/ops/rms_norm.py` |
| [api-linear-mlp.md](./api-linear-mlp.md) | 融合 Linear/MLP、激活函数、Triton 算子 | `flash_attn/ops/fused_dense.py`, `flash_attn/ops/activations.py`, `flash_attn/ops/triton/` |
| [api-padding.md](./api-padding.md) | BERT 变长 padding 工具 | `flash_attn/bert_padding.py` |
| [api-losses.md](./api-losses.md) | CrossEntropy 损失 | `flash_attn/losses/cross_entropy.py` |
| [api-modules.md](./api-modules.md) | MHA 层与 Transformer Block | `flash_attn/modules/` |
| [api-blocksparse.md](./api-blocksparse.md) | 块稀疏注意力 | `flash_attn/flash_blocksparse_attn_interface.py` |
| [api-utils.md](./api-utils.md) | 测试/生成/基准/分布式工具 | `flash_attn/utils/` |

## 版本与代际

`flash_attn` 是一个多代际共存的包，`flash_attn/__init__.py` 通过 `pkgutil.extend_path` 支持 FA2 与 FA4 同时安装：

```python
__version__ = "2.8.4"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
```

| 代际 | 位置 | 目标架构 | 公共入口 |
| --- | --- | --- | --- |
| **FA2** | `flash_attn/flash_attn_interface.py` | Ampere (SM80) 及以上 | `flash_attn_func` 等（顶层包直接导出） |
| **FA3** | `hopper/flash_attn_interface.py` | Hopper (SM90) | `flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_qkvpacked_func`, `flash_attn_combine`, `flash_attn_with_kvcache` |
| **FA4** | `flash_attn/cute/interface.py` | Hopper/Blackwell (SM90/SM100/SM110) | `flash_attn_func`, `flash_attn_varlen_func`（`flash_attn.cute` 导出） |

## 安装

```bash
# FA2 / 通用包
pip install flash-attn --no-build-isolation

# FA4 CuTeDSL 版本（包名 flash-attn-4）
pip install flash-attn-4
# 开发安装：
pip install -e "flash_attn/cute[dev]"
```

FA4 依赖：`nvidia-cutlass-dsl>=4.6.2`, `torch`, `einops`, `apache-tvm-ffi`, `quack-kernels>=0.5.3`。

## 快速开始（FA2）

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda", requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
```

## 快速开始（FA4 CuTe）

```python
import torch
import flash_attn.cute as fa4

q = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(2, 1024, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out = fa4.flash_attn_func(q, k, v, causal=True)
```

## 通用约定

- 张量布局：`(batch, seqlen, num_heads, head_dim)`，最后一维连续（`stride(-1) == 1`）。
- 支持的数据类型：`torch.float16`、`torch.bfloat16`（部分算子支持 `fp8`）。
- 变长接口使用 `cu_seqlens`（累积序列长度，`int32`）与 `max_seqlen`。
- 变长布局：`(total_tokens, num_heads, head_dim)`，无 batch 维度。
- 所有接口均支持前向/反向自动微分，除非另有说明（如 `flash_attn_with_kvcache` 仅支持前向）。
- `softmax_scale` 默认 `1 / sqrt(head_dim)`。
- `causal=True` 时因果掩码对齐到注意力矩阵右下角。
- `window_size=(left, right)` 非 `(-1, -1)` 时启用滑动窗口局部注意力。

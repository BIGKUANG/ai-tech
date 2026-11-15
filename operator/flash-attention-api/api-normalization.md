# 归一化融合算子 API (LayerNorm / RMSNorm)

提供 CUDA 融合实现（`flash_attn/ops/layer_norm.py`、`flash_attn/ops/rms_norm.py`）与 Triton 实现（`flash_attn/ops/triton/layer_norm.py`）。融合了 **Dropout + Residual Add + LayerNorm / RMSNorm**，并支持 rowscale、layerscale、prenorm 等变体。

---

## flash_attn.ops.layer_norm

### layer_norm

```python
layer_norm(x, weight, bias, epsilon)
```

标准 LayerNorm。

### dropout_add_layer_norm

```python
dropout_add_layer_norm(
    x0, residual, weight, bias,
    dropout_p, epsilon,
    rowscale=None, layerscale=None,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

融合算子：`x0 + residual` → dropout → LayerNorm。

- `residual_in_fp32` 仅在 `residual is None` 时生效；否则残差 dtype 取 `residual.dtype`。
- `prenorm=True` 时，返回值同时包含归一化前的残差输出（用于 post-norm 架构的残差连接）。

### dropout_add_layer_norm_subset

```python
dropout_add_layer_norm_subset(
    x0, residual, weight, bias,
    dropout_p, epsilon,
    layerscale=None, x0_subset=None, out_subset=None,
    rowscale_const=1.0, out_numrows=0,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

只对 `x0` 的一个子集做 dropout + add + norm，减少计算量。

### dropout_add_layer_norm_parallel_residual

```python
dropout_add_layer_norm_parallel_residual(
    x0, x1, residual,
    weight0, bias0, weight1, bias1,
    dropout_p, epsilon,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

并行残差架构（两个子层共享同一个残差），如 GPT-J 风格。

### DropoutAddLayerNorm (nn.Module)

```python
DropoutAddLayerNorm(hidden_size, prenorm=False, p=0.0, eps=1e-5, residual_in_fp32=False, device=None, dtype=None)
```

`forward(x0, residual=None)`。

---

## flash_attn.ops.rms_norm

### rms_norm

```python
rms_norm(x, weight, epsilon)
```

RMSNorm（基于 `DropoutAddLayerNormFn` 的 RMS 变体实现）。

### dropout_add_rms_norm

```python
dropout_add_rms_norm(
    x0, residual, weight, bias,
    dropout_p, epsilon,
    rowscale=None, layerscale=None,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

### dropout_add_rms_norm_subset

```python
dropout_add_rms_norm_subset(
    x0, residual, weight, bias,
    dropout_p, epsilon,
    layerscale=None, x0_subset=None, out_subset=None,
    rowscale_const=1.0, out_numrows=0,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

### dropout_add_rms_norm_parallel_residual

```python
dropout_add_rms_norm_parallel_residual(
    x0, x1, residual,
    weight0, bias0, weight1, bias1,
    dropout_p, epsilon,
    prenorm=False, residual_in_fp32=False,
    return_dropout_mask=False,
)
```

### RMSNorm (nn.Module)

```python
RMSNorm(hidden_size, eps=1e-5, device=None, dtype=None)
```

`forward(x)`。

### DropoutAddRMSNorm (nn.Module)

```python
DropoutAddRMSNorm(hidden_size, prenorm=False, p=0.0, eps=1e-5, residual_in_fp32=False, device=None, dtype=None)
```

`forward(x0, residual=None)`。

---

## flash_attn.ops.triton.layer_norm（Triton 实现）

无需编译 CUDA 扩展，纯 Triton，可用于推理或在不支持 CUDA 扩展时使用。

### layer_norm_fn

```python
layer_norm_fn(
    x, weight, bias,
    residual=None,
    x1=None, weight1=None, bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
)
```

通用融合归一化前端，`is_rms_norm=True` 时为 RMSNorm，否则 LayerNorm。

### rms_norm_fn

```python
rms_norm_fn(
    x, weight, bias,
    residual=None,
    x1=None, weight1=None, bias1=None,
    eps=1e-6, dropout_p=0.0, rowscale=None,
    prenorm=False, residual_in_fp32=False,
    zero_centered_weight=False,
    return_dropout_mask=False,
    out_dtype=None, out=None, residual_out=None,
)
```

### layer_norm_linear_fn

```python
layer_norm_linear_fn(...)
```

融合 LayerNorm/RMSNorm + Linear（见源码 `LayerNormLinearFn`）。

### RMSNorm (nn.Module, Triton)

```python
RMSNorm(hidden_size, eps=1e-5, dropout_p=0.0, zero_centered_weight=False, device=None, dtype=None)
```

`forward(x, residual=None, prenorm=False, residual_in_fp32=False)`。

**示例**

```python
import torch
from flash_attn.ops.rms_norm import RMSNorm

norm = RMSNorm(hidden_size=512).cuda()
x = torch.randn(4, 128, 512, device="cuda", dtype=torch.float16, requires_grad=True)
y = norm(x)
y.sum().backward()
```

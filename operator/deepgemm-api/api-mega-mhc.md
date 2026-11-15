# Mega mHC API

- 源码：`csrc/apis/mega_mhc.hpp`
- Python 导入：`import deep_gemm`
- 目标架构：SM100

## mega_mhc

```python
mega_mhc(
    x, residual, shifted_prev_mix, post_mix, comb_res_mix,
    fn, mix_scales, mix_bases,
    hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale,
    sinkhorn_eps, num_sinkhorn_iters,
    rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
    new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
    y_bf16=None, y_fp8=None,
    y_gemm_sf=None, y_routed_sf=None, y_shared_sf=None,
    shared_sf_block_m=0,
)
```

Mega mHC：执行 shifted 或 normal mHC，生成下一组混合系数，并施加带可选 FP8 E4M3 cast 的 RMSNorm。

**输入**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `x` | `[T, H]`，`torch.bfloat16` | 输入 |
| `residual` | `[T, hc_mult, H]`，`torch.bfloat16` | 残差 |
| `shifted_prev_mix` | `[T, hc_mult, 1]`，`torch.float`，可选 | shifted mHC 的旧 mix；与 `new_prev_mix` 同时为 `None`（normal mHC）或同时非空 |
| `post_mix` | `[T, hc_mult, 1]`，`torch.float` | post mix |
| `comb_res_mix` | `[T, hc_mult, hc_mult]`，`torch.float` | 组合残差系数 |
| `fn` | `[hc_mult*(hc_mult+2), hc_mult*H]`，`torch.float` | 混合投影权重 |
| `mix_scales` | `[3]`，`torch.float` | 缩放 |
| `mix_bases` | `[hc_mult*(hc_mult+2)]`，`torch.float` | 偏置 |
| `hc_mult` | int | 必须等于 `kNumRoutes`（4） |
| `hc_norm_eps` | float | 归一化 eps |
| `hc_pre_eps` | float | pre mix eps |
| `hc_post_scale` | float | post mix 缩放 |
| `sinkhorn_eps` | float | Sinkhorn eps |
| `num_sinkhorn_iters` | int | `>= 1` |
| `rmsnorm_weight` | `[H]`，`torch.bfloat16` | RMSNorm 权重 |
| `rmsnorm_eps` | float | RMSNorm eps |
| `rmsnorm_scale` | float | RMSNorm 缩放 |

**输出**

| 名称 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `new_residual` | `[T, hc_mult, H]`，`torch.bfloat16` | 新残差 |
| `new_prev_mix` | `[T, hc_mult, 1]`，`torch.float`，可选 | 仅 shifted mHC |
| `new_post_mix` | `[T, hc_mult, 1]`，`torch.float` | 新 post mix |
| `new_comb_res_mix` | `[T, hc_mult, hc_mult]`，`torch.float` | 新组合残差系数 |
| `y_bf16` | `[T, H]`，`torch.bfloat16`，可选 | BF16 归一化输出 |
| `y_fp8` | `[T, H]`，`torch.float8_e4m3fn`，可选 | FP8 归一化输出 |
| `y_gemm_sf` | `[T, H/128]`，`torch.int`，可选 | FP8 输出的 GEMM SF 布局（TMA 对齐的列主序） |
| `y_routed_sf` | `[T, H/128]`，`torch.int`，可选 | FP8 输出的 routed SF 布局（连续行主序） |
| `y_shared_sf` | `[T, H/128]`，`torch.int`，可选 | FP8 输出的 shared-expert SF 布局（Mega MoE 布局，要求 `shared_sf_block_m > 0`） |
| `shared_sf_block_m` | int | shared SF 的 block M；仅 `y_shared_sf` 非空时 `> 0`，否则为 0 |

**约束**

- shifted 状态“全有或全无”：`shifted_prev_mix` 与 `new_prev_mix` 必须同时非空 / 同时为空。
- `y_routed_sf` 与 `y_shared_sf` 必须同时提供或同时省略。
- `y_gemm_sf` 与 `y_routed_sf` 互斥。
- `y_bf16` 与 `y_fp8` 相互独立，至少提供一个。
- 提供 `y_fp8` 时，必须恰好提供以下之一：单独的 `y_gemm_sf`，或成对的 `y_routed_sf` + `y_shared_sf`。
- `hidden > 0` 且 `hidden % (kDefaultNumSplits * BLOCK_K) == 0`。
- FP8 gran_k 为 32，每个 `int32` 打包 4 个 UE8M0 缩放因子。
- `num_tokens == 0` 时直接返回。
- Split-K 每次调用自动选择：确定性算法下固定为 16；否则选择“最少的不等整块切分”以最小化最长生产者任务。
- 同时使用 `y_shared_sf` 时，`y_shared_sf` 存储的行数按 `ceil_div(T, block_m) * align(block_m, 128)` 计。
- 每个 stream 首次调用会初始化 split barrier；若要在 CUDA Graph 中捕获，必须在捕获前先 warm-up 执行一次（否则报 `CaptureStatus::None` 断言错误）。

**示例**

```python
import torch, deep_gemm

T, H, hc_mult = 4096, 7168, 4
num_hc_outputs = hc_mult * (hc_mult + 2)
x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
residual = torch.randn((T, hc_mult, H), dtype=torch.bfloat16, device='cuda')
post_mix = torch.randn((T, hc_mult, 1), dtype=torch.float, device='cuda').sigmoid()
comb_res_mix = torch.rand((T, hc_mult, hc_mult), device='cuda')
fn = torch.randn((num_hc_outputs, hc_mult * H), dtype=torch.float, device='cuda') * 1e-2
mix_scales = torch.randn((3,), dtype=torch.float, device='cuda') * 0.1
mix_bases = torch.randn((num_hc_outputs,), dtype=torch.float, device='cuda') * 0.1
rmsnorm_weight = (torch.randn((H,), dtype=torch.bfloat16, device='cuda') * 0.1 + 1)

new_residual = torch.empty_like(residual)
new_post_mix = torch.empty_like(post_mix)
new_comb_res_mix = torch.empty_like(comb_res_mix)
y_bf16 = torch.empty_like(x)

deep_gemm.mega_mhc(
    x, residual, None, post_mix, comb_res_mix,
    fn, mix_scales, mix_bases,
    hc_mult, 2e-5, 3e-4, 1.25, 2e-6, 10,
    rmsnorm_weight, 7e-6, 1.25,
    new_residual, None, new_post_mix, new_comb_res_mix,
    y_bf16=y_bf16)
```

## 相关

- HyperConnection 预归一化 GEMM：[api-hyperconnection.md](./api-hyperconnection.md)
- Mega MoE（消费 FP8 输出与 SF）：[api-mega-moe.md](./api-mega-moe.md)
- 测试参考：`tests/test_mega_mhc.py`

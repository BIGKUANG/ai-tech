// 10_divergence.cu —— 分支发散：一半 lane 空闲，warp 仍要跑完整路径
// 编译: nvcc -O3 -arch=sm_90 10_divergence.cu -o 10_divergence
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 均匀：所有线程都做 heavy 次 FMA（参考基准）
__global__ void uniform(const float* in, float* out, int n, int heavy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int k = 0; k < heavy; ++k) v = fmaf(v, 1.000001f, 1e-6f);
    out[i] = v;
}

// 发散：只有奇数 lane 做 heavy，偶数 lane 直接退出分支；
// 但 warp 必须等 heavy 路径执行完，偶数 lane 处于 masked-off 状态
__global__ void divergent(const float* in, float* out, int n, int heavy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    if (threadIdx.x & 1) {
        for (int k = 0; k < heavy; ++k) v = fmaf(v, 1.000001f, 1e-6f);
    }
    out[i] = v;
}

static float bench(void (*k)(const float*, float*, int, int), const float* in, float* o, int n, int heavy) {
    k<<<(n + 255) / 256, 256>>>(in, o, n, heavy);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) k<<<(n + 255) / 256, 256>>>(in, o, n, heavy);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 20.0f;
}

int main() {
    int n = 1 << 20, heavy = 4096;   // 计算受限，避免被访存掩盖
    float *in, *o;
    CK(cudaMalloc(&in, (size_t)n * 4)); CK(cudaMalloc(&o, (size_t)n * 4));
    CK(cudaMemset(in, 1, (size_t)n * 4));

    float tu = bench(uniform,   in, o, n, heavy);
    float td = bench(divergent, in, o, n, heavy);
    printf("uniform   (all lanes work): %7.3f ms  -> 100%% lane utilization\n", tu);
    printf("divergent (half lanes work): %7.3f ms\n", td);
    printf("发散版本只做了一半有效工作，却耗时 %.2fx 于均匀版本；\n", td / tu);
    printf("即 warp 的有效 lane 利用率约为 %.0f%%。\n", 50.0 * tu / td);
    cudaFree(in); cudaFree(o);
    return 0;
}

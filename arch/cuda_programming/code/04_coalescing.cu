// 04_coalescing.cu —— 合并访存：连续 vs 跨步
// 关键结论：warp(32 线程) 访问连续地址可合并为少量内存事务；跨步访问会产生大量事务。
// 编译: nvcc -O3 -arch=sm_90 04_coalescing.cu -o 04_coalescing
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

const int N = 1 << 24;   // 16M floats = 64MB

// 连续读：thread i 读 a[i]
__global__ void copy_coalesced(const float* __restrict__ a, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

// 跨步读：thread i 读 a[(i*32)%n]，同一 warp 的 32 个线程落在 32 条不同缓存行上
__global__ void copy_strided(const float* __restrict__ a, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[(size_t)i * 32 % n];
}

static float bench(void (*k)(const float*, float*, int), const float* a, float* o, int n) {
    k<<<(n + 255) / 256, 256>>>(a, o, n);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) k<<<(n + 255) / 256, 256>>>(a, o, n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 10.0f;
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    float *a, *o;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&o, bytes));
    CK(cudaMemset(a, 1, bytes));

    float t1 = bench(copy_coalesced, a, o, N);
    float t2 = bench(copy_strided,   a, o, N);
    double bw1 = 2.0 * bytes / (t1 * 1e-3) / 1e9;
    double bw2 = 2.0 * bytes / (t2 * 1e-3) / 1e9;
    printf("coalesced : %7.3f ms  %6.1f GB/s\n", t1, bw1);
    printf("strided   : %7.3f ms  %6.1f GB/s  (slowdown %.1fx)\n", t2, bw2, t2 / t1);

    cudaFree(a); cudaFree(o);
    return 0;
}

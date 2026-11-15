// 05_transpose.cu —— 矩阵转置：全局内存未合并写 vs 共享内存分块
// 编译: nvcc -O3 -arch=sm_90 05_transpose.cu -o 05_transpose
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

const int TILE = 32;

// 朴素实现：读合并，写跨步（列方向写），全局内存写不合并
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;   // 列
    int y = blockIdx.y * TILE + threadIdx.y;   // 行
    if (x < n && y < n) out[(size_t)x * n + y] = in[(size_t)y * n + x];
}

// 共享内存分块：读合并、写合并；用 +1 padding 消除 bank 冲突
__global__ void transpose_shared(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[(size_t)y * n + x];
    __syncthreads();
    // 转置后写出：x/y 交换
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < n && y < n) out[(size_t)y * n + x] = tile[threadIdx.x][threadIdx.y];
}

static float bench(void (*k)(const float*, float*, int), const float* a, float* o, int n, bool& ok) {
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    k<<<grid, block>>>(a, o, n);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) k<<<grid, block>>>(a, o, n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    ok = true;
    return ms / 10.0f;
}

int main() {
    int n = 4096;   // 4096^2 * 4B = 64MB
    size_t bytes = (size_t)n * n * sizeof(float);
    float *a, *o;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&o, bytes));
    CK(cudaMemset(a, 2, bytes));

    bool ok;
    float t1 = bench(transpose_naive,  a, o, n, ok);
    float t2 = bench(transpose_shared, a, o, n, ok);
    double bw1 = 2.0 * bytes / (t1 * 1e-3) / 1e9;
    double bw2 = 2.0 * bytes / (t2 * 1e-3) / 1e9;
    printf("n=%d\n", n);
    printf("naive  (global) : %7.3f ms  %6.1f GB/s\n", t1, bw1);
    printf("shared (+1 pad) : %7.3f ms  %6.1f GB/s  speedup %.2fx\n", t2, bw2, t1 / t2);

    cudaFree(a); cudaFree(o);
    return 0;
}

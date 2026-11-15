// 08_atomic.cu —— 直方图：全局原子争用 vs 共享内存局部直方图
// 编译: nvcc -O3 -arch=sm_90 08_atomic.cu -o 08_atomic
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// v1: 直接对全局直方图做原子加（同一 bin 高度争用）
__global__ void hist_global(const unsigned char* __restrict__ in, int n, int* hist) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&hist[in[i]], 1);
}

// v2: 每块先在共享内存里统计，再合并到全局，显著降低全局原子次数
__global__ void hist_shared(const unsigned char* __restrict__ in, int n, int* hist) {
    __shared__ int sh[256];
    int t = threadIdx.x;
    for (int k = t; k < 256; k += blockDim.x) sh[k] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + t; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&sh[in[i]], 1);
    __syncthreads();
    for (int k = t; k < 256; k += blockDim.x)
        if (sh[k]) atomicAdd(&hist[k], sh[k]);
}

static float bench(void (*k)(const unsigned char*, int, int*), const unsigned char* in, int n, int* h, int blocks, int threads) {
    k<<<blocks, threads>>>(in, n, h);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) k<<<blocks, threads>>>(in, n, h);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 10.0f;
}

int main() {
    int n = 1 << 26;   // 64M bytes
    unsigned char* d_in; CK(cudaMalloc(&d_in, n));
    unsigned char* h_in = (unsigned char*)malloc(n);
    for (int i = 0; i < n; ++i) h_in[i] = (unsigned char)(i * 7 & 255);   // 分布较均匀
    CK(cudaMemcpy(d_in, h_in, n, cudaMemcpyHostToDevice));
    free(h_in);

    int* d_hist; CK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    int threads = 256, blocks = 8192;

    CK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    float t1 = bench(hist_global, d_in, n, d_hist, blocks, threads);
    CK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    hist_global<<<blocks, threads>>>(d_in, n, d_hist);
    CK(cudaDeviceSynchronize());
    int h1[256]; CK(cudaMemcpy(h1, d_hist, sizeof(h1), cudaMemcpyDeviceToHost));

    CK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    float t2 = bench(hist_shared, d_in, n, d_hist, blocks, threads);
    CK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    hist_shared<<<blocks, threads>>>(d_in, n, d_hist);
    CK(cudaDeviceSynchronize());
    int h2[256]; CK(cudaMemcpy(h2, d_hist, sizeof(h2), cudaMemcpyDeviceToHost));

    long s1 = 0, s2 = 0;
    for (int i = 0; i < 256; ++i) { s1 += h1[i]; s2 += h2[i]; }
    printf("global atomic : %7.3f ms  total=%ld (n=%d)\n", t1, s1, n);
    printf("shared atomic : %7.3f ms  total=%ld  speedup %.2fx\n", t2, s2, t1 / t2);

    cudaFree(d_in); cudaFree(d_hist);
    return 0;
}

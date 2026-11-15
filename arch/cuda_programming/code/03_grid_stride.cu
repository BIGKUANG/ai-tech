// 03_grid_stride.cu —— grid-stride 循环：让任意规模都能跑满 GPU
// 对比：固定“一元素一线程” vs 固定线程总数 + 循环
// 编译: nvcc -O3 -arch=sm_90 03_grid_stride.cu -o 03_grid_stride
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 方式 A：一元素一线程（线程数随 n 增长）
__global__ void add_one_thread(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 方式 B：固定网格 + grid-stride 循环
__global__ void add_grid_stride(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = a[i] + b[i];
}

static float bench(void (*launch)(const float*, const float*, float*, int, int, int),
                   const float* a, const float* b, float* c, int n, int blocks, int threads) {
    launch(a, b, c, n, blocks, threads);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int k = 0; k < 20; ++k) launch(a, b, c, n, blocks, threads);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 20.0f;
}

static void launch_one(const float* a, const float* b, float* c, int n, int, int t) {
    add_one_thread<<<(n + t - 1) / t, t>>>(a, b, c, n);
}
static void launch_gs(const float* a, const float* b, float* c, int n, int g, int t) {
    add_grid_stride<<<g, t>>>(a, b, c, n);
}

int main() {
    int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    int sms = p.multiProcessorCount;
    int threads = 256;
    int gs_blocks = sms * 8;                 // 每 SM 8 个块，常用起点

    for (int n : {1 << 20, 1 << 24, 1 << 26}) {
        size_t bytes = (size_t)n * sizeof(float);
        float *a, *b, *c;
        CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&c, bytes));
        CK(cudaMemset(a, 0, bytes)); CK(cudaMemset(b, 0, bytes));

        float t1 = bench(launch_one, a, b, c, n, 0, threads);
        float t2 = bench(launch_gs,  a, b, c, n, gs_blocks, threads);
        double bw1 = 3.0 * bytes / (t1 * 1e-3) / 1e9;
        double bw2 = 3.0 * bytes / (t2 * 1e-3) / 1e9;
        printf("n=%9d | one-thread %7.3f ms (%5.1f GB/s) | grid-stride %7.3f ms (%5.1f GB/s)\n",
               n, t1, bw1, t2, bw2);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    return 0;
}

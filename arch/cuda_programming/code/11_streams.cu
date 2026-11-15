// 11_streams.cu —— CUDA 流与异步：把 H2D / 计算 / D2H 重叠起来
// 编译: nvcc -O3 -arch=sm_90 11_streams.cu -o 11_streams
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

__global__ void scale_kernel(float* x, int n, float a) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] *= a;
}

int main() {
    int n = 1 << 24;                       // 16M floats = 64MB
    size_t bytes = (size_t)n * sizeof(float);
    int chunks = 4;
    int per = n / chunks;
    size_t perBytes = (size_t)per * sizeof(float);

    float *h, *d;
    CK(cudaMallocHost(&h, bytes));         // 页锁定内存才能异步传输
    CK(cudaMalloc(&d, bytes));
    for (int i = 0; i < n; ++i) h[i] = 1.0f;

    // ---------- 串行：传输与计算不重叠 ----------
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int c = 0; c < chunks; ++c) {
        CK(cudaMemcpy(d + (size_t)c * per, h + (size_t)c * per, perBytes, cudaMemcpyHostToDevice));
        scale_kernel<<<per / 256, 256>>>(d + (size_t)c * per, per, 2.0f);
        CK(cudaMemcpy(h + (size_t)c * per, d + (size_t)c * per, perBytes, cudaMemcpyDeviceToHost));
    }
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms_seq; cudaEventElapsedTime(&ms_seq, s, e);

    // ---------- 多流：每个 chunk 一个流，传输/计算重叠 ----------
    const int NS = chunks;
    cudaStream_t st[NS];
    for (int i = 0; i < NS; ++i) CK(cudaStreamCreate(&st[i]));
    for (int i = 0; i < n; ++i) h[i] = 1.0f;

    cudaEventRecord(s);
    for (int c = 0; c < chunks; ++c) {
        CK(cudaMemcpyAsync(d + (size_t)c * per, h + (size_t)c * per, perBytes,
                           cudaMemcpyHostToDevice, st[c]));
        scale_kernel<<<per / 256, 256, 0, st[c]>>>(d + (size_t)c * per, per, 2.0f);
        CK(cudaMemcpyAsync(h + (size_t)c * per, d + (size_t)c * per, perBytes,
                           cudaMemcpyDeviceToHost, st[c]));
    }
    for (int i = 0; i < NS; ++i) CK(cudaStreamSynchronize(st[i]));
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms_par; cudaEventElapsedTime(&ms_par, s, e);

    printf("sequential : %7.3f ms\n", ms_seq);
    printf("streams    : %7.3f ms  speedup %.2fx\n", ms_par, ms_seq / ms_par);

    for (int i = 0; i < NS; ++i) cudaStreamDestroy(st[i]);
    cudaFreeHost(h); cudaFree(d);
    return 0;
}

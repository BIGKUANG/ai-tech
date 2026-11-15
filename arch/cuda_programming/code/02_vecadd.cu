// 02_vecadd.cu —— 最经典的向量加法 C = A + B
// 覆盖：数据搬运、内核启动、边界检查、正确性校验、事件计时、有效带宽
// 编译: nvcc -O3 -arch=sm_90 02_vecadd.cu -o 02_vecadd
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

__global__ void vecAdd(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);   // 16M 元素 = 64MB/数组
    size_t bytes = (size_t)n * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 预热
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int k = 0; k < 20; ++k) vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= 20.0f;

    CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) if (fabsf(hC[i] - 3.0f) > 1e-5f) { bad++; if (bad < 3) printf("mismatch @%d: %f\n", i, hC[i]); }

    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   // 读 A、读 B、写 C
    printf("n=%d blocks=%d threads=%d\n", n, blocks, threads);
    printf("time=%.3f ms  effective BW=%.1f GB/s  verify=%s\n", ms, gbps, bad ? "FAIL" : "OK");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}

// 16_saxpy.cu —— 经典 SAXPY：y = a*x + y，工程化写法
// 编译: nvcc -O3 -arch=sm_90 16_saxpy.cu -o 16_saxpy
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = a * x[i] + y[i];   // 编译器会生成 FFMA
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    float a = 2.0f;
    size_t bytes = (size_t)n * sizeof(float);

    float *x, *y, *d_x, *d_y;
    CK(cudaMallocHost(&x, bytes)); CK(cudaMallocHost(&y, bytes));
    for (int i = 0; i < n; ++i) { x[i] = 1.0f; y[i] = 1.0f; }
    CK(cudaMalloc(&d_x, bytes)); CK(cudaMalloc(&d_y, bytes));
    CK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));

    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sms * 16, threads = 256;

    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 50; ++i) saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 50.0f;

    // 计时循环里 kernel 执行了 50 次（会累加 50 次），验证前重置并只跑一次
    CK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));
    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) if (fabsf(y[i] - 3.0f) > 1e-5f) bad++;
    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   // 读 x、读 y、写 y

    printf("SAXPY n=%d a=%.1f blocks=%d threads=%d\n", n, a, blocks, threads);
    printf("time=%.3f ms  BW=%.1f GB/s  %s\n", ms, gbps, bad ? "FAIL" : "OK");

    cudaFreeHost(x); cudaFreeHost(y); cudaFree(d_x); cudaFree(d_y);
    return 0;
}

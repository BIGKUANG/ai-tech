// 06_bank_conflict.cu —— 共享内存 bank 冲突：跨步扫描
// 共享内存 32 个 bank；同一 warp 内多线程访问同一 bank 的不同地址 -> 冲突串行化。
// 编译: nvcc -O3 -arch=sm_90 06_bank_conflict.cu -o 06_bank_conflict
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 每个线程在共享内存里以 stride 跨步读取，制造 bank 冲突
__global__ void bank_kernel(float* out, int stride, int reps) {
    __shared__ float s[1024];
    int t = threadIdx.x;
    for (int k = t; k < 1024; k += blockDim.x) s[k] = k * 0.5f;
    __syncthreads();

    float acc = 0.f;
    for (int r = 0; r < reps; ++r) {
        int idx = (t * stride + r) & 1023;
        acc += s[idx];
    }
    if (acc == 12345.f) out[t] = acc;   // 防止优化掉
}

int main() {
    int reps = 4096;
    float* d_out; CK(cudaMalloc(&d_out, 256 * sizeof(float)));
    int threads = 256;
    int blocks = 512;
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    (void)sms;

    printf("stride |   time(ms) | relative\n");
    printf("-------+------------+---------\n");
    float base = 0.f;
    for (int stride = 1; stride <= 32; ++stride) {
        // 预热
        bank_kernel<<<blocks, threads>>>(d_out, stride, reps);
        CK(cudaDeviceSynchronize());
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) bank_kernel<<<blocks, threads>>>(d_out, stride, reps);
        cudaEventRecord(e); CK(cudaEventSynchronize(e));
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= 10.0f;
        if (stride == 1) base = ms;
        printf(" %5d | %10.3f | %6.2fx\n", stride, ms, ms / base);
    }
    printf("\n注：stride=1/3/5... 等与 32 互素的步长基本无冲突；stride=2/4/8/16/32 冲突明显。\n");
    cudaFree(d_out);
    return 0;
}

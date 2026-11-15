// 09_occupancy.cu —— 占用率与线程块大小：固定总线程数，只改 block size
// 编译: nvcc -O3 -arch=sm_90 09_occupancy.cu -o 09_occupancy
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 4 个独立累加器 -> 指令级并行(ILP)
__global__ void fma_kernel(float* out, int reps) {
    float a0 = threadIdx.x * 1e-6f + 1.f;
    float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    for (int r = 0; r < reps; ++r) {
        a0 = fmaf(a0, 1.0000001f, 1e-7f);
        a1 = fmaf(a1, 1.0000001f, 1e-7f);
        a2 = fmaf(a2, 1.0000001f, 1e-7f);
        a3 = fmaf(a3, 1.0000001f, 1e-7f);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

int main() {
    int sms = 0, maxThreadsPerSM = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    // 固定总线程数 = GPU 满载线程数，只改变 block 大小，保证总工作量一致
    int totalThreads = sms * maxThreadsPerSM;
    int reps = 2048;

    printf("SMs=%d maxThreadsPerSM=%d totalThreads=%d\n", sms, maxThreadsPerSM, totalThreads);
    printf("block | blocks | occupancy |   time(ms) |   TFLOPS\n");
    printf("------+--------+-----------+------------+---------\n");

    for (int threads : {32, 64, 128, 256, 512, 1024}) {
        int blocks = totalThreads / threads;
        int allocThreads = blocks * threads;
        float* d_out; CK(cudaMalloc(&d_out, (size_t)allocThreads * sizeof(float)));

        int active = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fma_kernel, threads, 0));
        double occ = (double)active * threads / maxThreadsPerSM * 100.0;

        fma_kernel<<<blocks, threads>>>(d_out, reps);
        CK(cudaDeviceSynchronize());
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) fma_kernel<<<blocks, threads>>>(d_out, reps);
        cudaEventRecord(e); CK(cudaEventSynchronize(e));
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= 10.0f;

        double flops = (double)allocThreads * reps * 4 * 2;   // 每线程 reps*4 次 FMA
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("%5d | %6d | %8.1f%% | %10.3f | %8.2f\n", threads, blocks, occ, ms, tflops);
        cudaFree(d_out);
    }
    return 0;
}

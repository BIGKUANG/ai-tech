// 14_ilp.cu —— 指令级并行(ILP)：用多个独立累加器打散 FMA 依赖链
// 三种内核每线程完成相同的 FMA 总数，只改变“独立链的数量”。
// 编译: nvcc -O3 -arch=sm_90 14_ilp.cu -o 14_ilp
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 1 条依赖链：每次 fma 依赖上次结果，延迟受限
__global__ void chain1(float* out, int reps) {
    float a = threadIdx.x * 1e-6f + 1.f;
    for (int r = 0; r < reps; ++r) a = fmaf(a, 1.0000001f, 1e-7f);
    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

// 4 条独立链：流水线可同时推进
__global__ void chain4(float* out, int reps) {
    float a0 = threadIdx.x * 1e-6f + 1.f, a1 = a0 + .1f, a2 = a0 + .2f, a3 = a0 + .3f;
    for (int r = 0; r < reps; ++r) {
        a0 = fmaf(a0, 1.0000001f, 1e-7f);
        a1 = fmaf(a1, 1.0000001f, 1e-7f);
        a2 = fmaf(a2, 1.0000001f, 1e-7f);
        a3 = fmaf(a3, 1.0000001f, 1e-7f);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

// 8 条独立链
__global__ void chain8(float* out, int reps) {
    float a[8];
    for (int i = 0; i < 8; ++i) a[i] = threadIdx.x * 1e-6f + 1.f + i * 0.1f;
    for (int r = 0; r < reps; ++r)
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fmaf(a[i], 1.0000001f, 1e-7f);
    float s = 0; for (int i = 0; i < 8; ++i) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

static float bench(void (*k)(float*, int), float* o, int reps, int blocks, int threads) {
    k<<<blocks, threads>>>(o, reps);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) k<<<blocks, threads>>>(o, reps);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 20.0f;
}

struct Case { const char* name; void (*fn)(float*, int); int ilp; };

int main() {
    float* o; CK(cudaMalloc(&o, 4096 * 1024 * sizeof(float)));
    int total = 8192;   // 每线程总 FMA 次数（各版本一致）
    Case cases[] = {{"chain1", chain1, 1}, {"chain4", chain4, 4}, {"chain8", chain8, 8}};

    printf("每线程 FMA 总数 = %d（三个版本相同）\n", total);
    printf("\n=== 小网格(1 block x 32 thread)，延迟受限 ===\n");
    for (auto& c : cases) {
        int reps = total / c.ilp;
        float ms = bench(c.fn, o, reps, 1, 32);
        printf("%-7s : %7.3f ms\n", c.name, ms);
    }

    printf("\n=== 大网格(近似占满 GPU) ===\n");
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    for (auto& c : cases) {
        int reps = total / c.ilp;
        float ms = bench(c.fn, o, reps, sms * 8, 256);
        printf("%-7s : %7.3f ms\n", c.name, ms);
    }
    cudaFree(o);
    return 0;
}

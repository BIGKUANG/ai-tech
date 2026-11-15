// 15_ptx_fastmath.cu —— 内联 PTX 与快速数学：从指令层看加速
// 编译: nvcc -O3 -arch=sm_90 15_ptx_fastmath.cu -o 15_ptx_fastmath
// 观察 SASS: cuobjdump -sass 15_ptx_fastmath | grep -E "FFMA|MUFU|FADD"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 内联 PTX：直接写 FFMA（融合乘加），并显式指定舍入
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// 内联 PTX：读取特殊寄存器（lane id / sm id），进阶调试用
__device__ __forceinline__ unsigned ptx_laneid() {
    unsigned r; asm("mov.u32 %0, %%laneid;" : "=r"(r)); return r;
}

__global__ void k_ptx_fma(const float* in, float* out, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = in[i], b = 1.0000001f, c = 1e-7f;
    for (int r = 0; r < reps; ++r) a = ptx_fma(a, b, c);
    out[i] = a + ptx_laneid() * 0.f;
}

// 标准 expf：调用数学库，指令多
__global__ void k_exp(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]) + expf(in[i] * 1.1f) + expf(in[i] * 1.2f);
}

// 快速 expf：__expf 直接映射到 MUFU.EX2 近似指令，快很多、精度低
__global__ void k_exp_fast(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(in[i]) + __expf(in[i] * 1.1f) + __expf(in[i] * 1.2f);
}

static float bench(void (*k)(const float*, float*, int), const float* in, float* o, int n) {
    k<<<(n + 255) / 256, 256>>>(in, o, n);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 50; ++i) k<<<(n + 255) / 256, 256>>>(in, o, n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 50.0f;
}

int main() {
    int n = 1 << 22;
    size_t bytes = (size_t)n * sizeof(float);
    float *in, *o; CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&o, bytes));
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = 0.001f * (i % 100);
    CK(cudaMemcpy(in, h, bytes, cudaMemcpyHostToDevice));

    // PTX FMA 链
    float* dummy; CK(cudaMalloc(&dummy, bytes));
    k_ptx_fma<<<(n + 255) / 256, 256>>>(in, dummy, n, 64);
    CK(cudaDeviceSynchronize());

    float t_exp = bench(k_exp, in, o, n);
    float t_fast = bench(k_exp_fast, in, o, n);
    printf("expf (libm)   : %7.3f ms\n", t_exp);
    printf("__expf (MUFU) : %7.3f ms  speedup %.2fx\n", t_fast, t_exp / t_fast);
    printf("\n查看指令：\n  cuobjdump -sass 15_ptx_fastmath | grep -E 'FFMA|MUFU' | head\n");
    printf("  nvdisasm -g 15_ptx_fastmath.cubin（若保留中间产物）\n");

    cudaFree(in); cudaFree(o); cudaFree(dummy); free(h);
    return 0;
}

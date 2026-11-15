// 07_reduction.cu —— 归约求和：共享内存 -> warp shuffle -> 两趟多块
// 编译: nvcc -O3 -arch=sm_90 07_reduction.cu -o 07_reduction
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// v1: 每线程先 grid-stride 累加，再块内共享内存树形归约，最后 atomicAdd
__global__ void reduce_shared(const float* __restrict__ in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x)
        local += in[i];
    s[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// v2: 每线程 grid-stride 累加，然后 warp shuffle 归约，warp leader atomicAdd
__global__ void reduce_warp(const float* __restrict__ in, float* out, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, v);
}

// v3: 两趟归约：第一趟每块写一个部分和，第二趟单块汇总（无全局原子争用）
__global__ void reduce_partial(const float* __restrict__ in, float* part, int n) {
    __shared__ float w[256 / 32];
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < blockDim.x / 32) {
        v = w[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (threadIdx.x == 0) part[blockIdx.x] = v;
    }
}
__global__ void reduce_final(const float* __restrict__ part, float* out, int m) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < m; i += blockDim.x) local += part[i];
    s[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

static float timeit(void (*run)(const float*, float*, int, int, int),
                    const float* in, float* out, int n, int blocks, int threads) {
    run(in, out, n, blocks, threads);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int k = 0; k < 20; ++k) run(in, out, n, blocks, threads);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 20.0f;
}

static void run_shared(const float* in, float* out, int n, int b, int t) {
    reduce_shared<<<b, t>>>(in, out, n);
}
static void run_warp(const float* in, float* out, int n, int b, int t) {
    reduce_warp<<<b, t>>>(in, out, n);
}
static void run_two(const float* in, float* out, int n, int b, int t) {
    float* part; cudaMalloc(&part, b * sizeof(float));
    reduce_partial<<<b, t>>>(in, part, n);
    reduce_final<<<1, 256>>>(part, out, b);
    cudaFree(part);
}

int main() {
    int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);
    float* d_in; CK(cudaMalloc(&d_in, bytes));
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    CK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
    free(h);

    float* d_out; CK(cudaMalloc(&d_out, sizeof(float)));
    int threads = 256, blocks = 4096;

    struct Case { const char* name; void (*fn)(const float*, float*, int, int, int); } cases[] = {
        {"shared+atomic", run_shared},
        {"warp+atomic",   run_warp},
        {"two-pass",      run_two},
    };
    printf("sum of %d ones, blocks=%d threads=%d\n", n, blocks, threads);
    for (auto& c : cases) {
        CK(cudaMemset(d_out, 0, sizeof(float)));
        float ms = timeit(c.fn, d_in, d_out, n, blocks, threads);
        // timeit 内部执行了 21 次；用原子累加的版本会累积，验证前单独跑一次并清零
        CK(cudaMemset(d_out, 0, sizeof(float)));
        c.fn(d_in, d_out, n, blocks, threads);
        CK(cudaDeviceSynchronize());
        float r; CK(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("%-14s : %7.3f ms  result=%.0f  %s\n", c.name, ms, r,
               (fabsf(r - n) < 0.5f) ? "OK" : "FAIL");
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

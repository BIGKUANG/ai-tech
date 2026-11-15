// 12_async_copy.cu —— 异步拷贝(LDGSTS)：块协作 memcpy_async + 计算重叠
// 编译: nvcc -O3 -arch=sm_90 12_async_copy.cu -o 12_async_copy
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

namespace cg = cooperative_groups;
constexpr int TILE = 1024;   // 每块每批元素数（4KB）

// 同步：global -> shared -> 计算，两次 block 同步
__global__ void sync_pipeline(const float* __restrict__ in, float* __restrict__ out, int ntiles) {
    __shared__ float buf[TILE];
    float acc = 0.f;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        for (int i = threadIdx.x; i < TILE; i += blockDim.x) buf[i] = in[(size_t)t * TILE + i];
        __syncthreads();
        for (int i = threadIdx.x; i < TILE; i += blockDim.x) acc += buf[i];
        __syncthreads();
    }
    if (acc == 1.2345e30f) out[0] = acc;
}

// 异步：块协作 memcpy_async（底层 LDGSTS），拷贝与后续计算/循环重叠
__global__ void async_pipeline(const float* __restrict__ in, float* __restrict__ out, int ntiles) {
    __shared__ float buf[TILE];
    auto block = cg::this_thread_block();
    float acc = 0.f;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        cg::memcpy_async(block, buf, in + (size_t)t * TILE, sizeof(float) * TILE);
        // 在等待期间可做与 buf 无关的工作（这里用下一次迭代的地址计算占位）
        cg::wait(block);
        for (int i = threadIdx.x; i < TILE; i += blockDim.x) acc += buf[i];
        block.sync();
    }
    if (acc == 1.2345e30f) out[0] = acc;
}

static float bench(void (*k)(const float*, float*, int), const float* in, float* o, int ntiles, int blocks) {
    k<<<blocks, 256>>>(in, o, ntiles);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) k<<<blocks, 256>>>(in, o, ntiles);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / 20.0f;
}

int main() {
    int ntiles = 1 << 16;                    // 65536 * 1024 * 4B = 256MB
    size_t bytes = (size_t)ntiles * TILE * sizeof(float);
    float *in, *o;
    CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&o, sizeof(float)));
    CK(cudaMemset(in, 0, bytes));
    int blocks = 1024;

    float t1 = bench(sync_pipeline,  in, o, ntiles, blocks);
    float t2 = bench(async_pipeline, in, o, ntiles, blocks);
    double bw1 = bytes / (t1 * 1e-3) / 1e9;
    double bw2 = bytes / (t2 * 1e-3) / 1e9;
    printf("sync  pipeline : %7.3f ms  %.1f GB/s\n", t1, bw1);
    printf("async pipeline : %7.3f ms  %.1f GB/s  speedup %.2fx\n", t2, bw2, t1 / t2);
    cudaFree(in); cudaFree(o);
    return 0;
}

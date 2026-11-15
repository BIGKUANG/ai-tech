// 01_hello.cu —— 第一个 CUDA 程序 + 设备信息
// 编译: nvcc -O3 -arch=sm_90 01_hello.cu -o 01_hello
// 运行: ./01_hello
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf 在内核中可用，但开销大，仅用于调试/演示
    if (tid < 8)
        printf("  [device] block %d thread %d: hello from GPU\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("=== 01 hello: 主机端 ===\n");
    printf("hello from CPU\n\n");

    int dev = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("=== 设备信息 ===\n");
    printf("name                 : %s\n", p.name);
    printf("compute capability   : %d.%d\n", p.major, p.minor);
    printf("SMs                  : %d\n", p.multiProcessorCount);
    printf("maxThreadsPerSM      : %d\n", p.maxThreadsPerMultiProcessor);
    printf("maxThreadsPerBlock   : %d\n", p.maxThreadsPerBlock);
    printf("regsPerSM            : %d\n", p.regsPerMultiprocessor);
    printf("sharedMemPerSM       : %zu bytes\n", p.sharedMemPerMultiprocessor);
    printf("sharedMemPerBlock    : %zu bytes\n", p.sharedMemPerBlockOptin);
    printf("L2 cache             : %d bytes\n", p.l2CacheSize);
    printf("warpSize             : %d\n\n", p.warpSize);

    printf("=== 02 hello: 设备端 ===\n");
    hello_kernel<<<1, 8>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("\ndone\n");
    return 0;
}

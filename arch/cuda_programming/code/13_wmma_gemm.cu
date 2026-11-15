// 13_wmma_gemm.cu —— 张量核心 GEMM（16x16x16，half 输入 / float 累加）
// 编译: nvcc -O3 -arch=sm_90 13_wmma_gemm.cu -o 13_wmma_gemm
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t _ck_err = (x); if (_ck_err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_err)); exit(1); } } while (0)

// 每个 warp 负责 C 的一个 16x16 tile，沿 K 维累加
__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    int warp_m = blockIdx.y;     // 第几个 16 行块
    int warp_n = blockIdx.x;     // 第几个 16 列块

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + (size_t)warp_m * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + (size_t)k * N + warp_n * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + (size_t)warp_m * 16 * N + warp_n * 16, c_frag, N,
                            wmma::mem_row_major);
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t aBytes = (size_t)M * K * sizeof(half);
    size_t bBytes = (size_t)K * N * sizeof(half);
    size_t cBytes = (size_t)M * N * sizeof(float);

    half *hA = (half*)malloc(aBytes), *hB = (half*)malloc(bBytes);
    float *hC = (float*)malloc(cBytes), *ref = (float*)malloc(cBytes);
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half((i % 7) * 0.1f);
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half((i % 5) * 0.2f);

    half *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, aBytes)); CK(cudaMalloc(&dB, bBytes)); CK(cudaMalloc(&dC, cBytes));
    CK(cudaMemcpy(dA, hA, aBytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bBytes, cudaMemcpyHostToDevice));

    dim3 grid(N / 16, M / 16);
    gemm_wmma<<<grid, 32>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC, dC, cBytes, cudaMemcpyDeviceToHost));

    // CPU 参考（用小规模抽样校验，避免太慢）
    int bad = 0;
    for (int m = 0; m < M; m += 37) {
        for (int n = 0; n < N; n += 41) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += __half2float(hA[m * K + k]) * __half2float(hB[k * N + n]);
            if (fabsf(s - hC[m * N + n]) > 1e-2f) bad++;
        }
    }
    printf("M=N=K=%d, grid=(%d,%d)\n", M, grid.x, grid.y);
    printf("sampled check: %s (mismatches=%d)\n", bad ? "FAIL" : "OK", bad);

    free(hA); free(hB); free(hC); free(ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

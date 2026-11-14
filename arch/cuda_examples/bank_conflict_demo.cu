// H20 / sm_90 共享存储器 bank 冲突完整演示
// 编译: nvcc -O3 -arch=sm_90 --extended-lambda bank_conflict_demo.cu -o bank_conflict_demo
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e){printf("ERR %d: %s\n",__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---------- 1. 冲突度数扫描: idx = lane * STRIDE ----------
template<int STRIDE>
__global__ void __launch_bounds__(256) stride_kernel(float* out, int reps){
  __shared__ volatile float s[1024];
  int t = threadIdx.x & 31;                      // 每个 warp 的 lane 0..31
  for(int i=threadIdx.x; i<1024; i+=blockDim.x) s[i] = i*0.5f;
  __syncthreads();
  float acc = 0.f;
  int idx = (t*STRIDE) & 1023;
  #pragma unroll 1
  for(int r=0; r<reps; ++r) acc += s[idx];
  out[blockIdx.x*blockDim.x + threadIdx.x] = acc;
}

// ---------- 2. 32x32 矩阵列读: 冲突 / padding / XOR swizzle ----------
template<int MODE>
__global__ void __launch_bounds__(256) col_kernel(float* out, int reps){
  __shared__ volatile float s[32*33];
  int t = threadIdx.x & 31;
  for(int i=threadIdx.x; i<32*33; i+=blockDim.x) s[i] = i*0.5f;
  __syncthreads();
  const int col = 17;
  int idx;
  if(MODE==0)      idx = t*32 + col;                 // 32-way 冲突
  else if(MODE==1) idx = t*33 + col;                 // padding 33 -> 无冲突
  else             idx = t*32 + ((col ^ t) & 31);    // XOR swizzle -> 无冲突
  float acc = 0.f;
  #pragma unroll 1
  for(int r=0; r<reps; ++r) acc += s[idx];
  out[blockIdx.x*blockDim.x + threadIdx.x] = acc;
}

// ---------- 3. 广播 / 无冲突 / 冲突 ----------
template<int MODE>
__global__ void __launch_bounds__(256) bcast_kernel(float* out, int reps){
  __shared__ volatile float s[1024];
  int t = threadIdx.x & 31;
  for(int i=threadIdx.x; i<1024; i+=blockDim.x) s[i] = i*0.5f;
  __syncthreads();
  int idx = (MODE==0) ? t : (MODE==1 ? 0 : t*32);
  float acc = 0.f;
  #pragma unroll 1
  for(int r=0; r<reps; ++r) acc += s[idx];
  out[blockIdx.x*blockDim.x + threadIdx.x] = acc;
}

template<class F> float bench(F launch, int iters=100){
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for(int i=0;i<iters;i++){ launch(); CK(cudaGetLastError()); }
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms/iters;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("GPU: %s  SMs=%d  sharedMemPerSM=%zu B  sharedMemPerBlockOptin=%zu B\n",
         p.name, p.multiProcessorCount,
         p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin);
  const int NB = p.multiProcessorCount*8;
  const int REPS = 20000;
  float* d; CK(cudaMalloc(&d, (size_t)NB*256*4));

  printf("\n== 1) 冲突度数扫描 (idx = lane*STRIDE, %d 次读/线程) ==\n", REPS);
  float b1 = bench([&]{ stride_kernel<1><<<NB,256>>>(d,REPS); });
  auto show=[&](int ways,float t){ printf(" %2d-way : %8.3f ms  %5.2fx\n",ways,t,t/b1); };
  show(1, b1);
  show(2, bench([&]{ stride_kernel<2><<<NB,256>>>(d,REPS); }));
  show(4, bench([&]{ stride_kernel<4><<<NB,256>>>(d,REPS); }));
  show(8, bench([&]{ stride_kernel<8><<<NB,256>>>(d,REPS); }));
  show(16,bench([&]{ stride_kernel<16><<<NB,256>>>(d,REPS); }));
  show(32,bench([&]{ stride_kernel<32><<<NB,256>>>(d,REPS); }));

  printf("\n== 2) 32x32 矩阵列读 (col=17) ==\n");
  float c0 = bench([&]{ col_kernel<0><<<NB,256>>>(d,REPS); });
  float c1 = bench([&]{ col_kernel<1><<<NB,256>>>(d,REPS); });
  float c2 = bench([&]{ col_kernel<2><<<NB,256>>>(d,REPS); });
  printf(" stride32 冲突   : %8.3f ms  1.00x\n", c0);
  printf(" stride33 padding: %8.3f ms  %.2fx\n", c1, c0/c1);
  printf(" XOR swizzle     : %8.3f ms  %.2fx\n", c2, c0/c2);

  printf("\n== 3) 广播 vs 无冲突 vs 冲突 ==\n");
  float x = bench([&]{ bcast_kernel<0><<<NB,256>>>(d,REPS); });
  float y = bench([&]{ bcast_kernel<1><<<NB,256>>>(d,REPS); });
  float z = bench([&]{ bcast_kernel<2><<<NB,256>>>(d,REPS); });
  printf(" s[lane] 无冲突  : %8.3f ms  1.00x\n", x);
  printf(" s[0]    广播    : %8.3f ms  %.2fx\n", y, x/y);
  printf(" s[lane*32] 冲突 : %8.3f ms  %.2fx\n", z, x/z);

  CK(cudaFree(d));
  return 0;
}

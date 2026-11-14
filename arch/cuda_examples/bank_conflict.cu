// 共享存储器 bank 冲突 (H20, sm_90)
#include <cstdio>
#include <cuda_runtime.h>
template<int STRIDE>
__global__ void __launch_bounds__(256) bank_kernel(float* out, int reps){
  __shared__ volatile float s[1024];
  int t=threadIdx.x;
  for(int k=t;k<1024;k+=blockDim.x) s[k]=k*0.5f;
  __syncthreads();
  float acc=0.f; int base=(t*STRIDE)&1023;
  for(int r=0;r<reps;r++){
    #pragma unroll 32
    for(int k=0;k<32;k++) acc += s[(base + (r*7+k)*STRIDE)&1023];
  }
  out[blockIdx.x*blockDim.x+t]=acc;
}
template<class F> float bench(F f,int it=100){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize(); cudaEventRecord(a);
  for(int i=0;i<it;i++) f(); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}
int main(){
  const int NB=78*8; float* d; cudaMalloc(&d,NB*256*4); int reps=2000;
  float t1=bench([&]{ bank_kernel<1><<<NB,256>>>(d,reps); });
  float t2=bench([&]{ bank_kernel<2><<<NB,256>>>(d,reps); });
  float t4=bench([&]{ bank_kernel<4><<<NB,256>>>(d,reps); });
  float t32=bench([&]{ bank_kernel<32><<<NB,256>>>(d,reps); });
  printf("shared memory bank conflict (%d blocks, %d reads/thread):\n",NB,reps*32);
  printf(" stride=1  no conflict : %8.3f ms  1.00x\n",t1);
  printf(" stride=2  2-way       : %8.3f ms  %.2fx\n",t2,t2/t1);
  printf(" stride=4  4-way       : %8.3f ms  %.2fx\n",t4,t4/t1);
  printf(" stride=32 32-way      : %8.3f ms  %.2fx\n",t32,t32/t1);
  return 0;
}

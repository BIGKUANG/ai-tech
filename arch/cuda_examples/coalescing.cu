// 合并访存对有效带宽的影响 (H20, sm_90)
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
const int N = 1<<24; // 16M floats = 64MB

__global__ void copy_coalesced(const float* __restrict__ in, float* __restrict__ out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) out[i]=in[i]*2.0f;
}
__global__ void copy_strided(const float* __restrict__ in, float* __restrict__ out, int n, int stride){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=(i%32)*stride + (i/32)*32;
  if(j<n && i<n) out[j]=in[j]*2.0f;
}
template<class F> float bench(F f,int it=50){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int i=0;i<it;i++) f(); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}
int main(){
  float *din,*dout; CK(cudaMalloc(&din,N*4)); CK(cudaMalloc(&dout,N*4)); CK(cudaMemset(din,1,N*4));
  double bytes=2.0*N*4;
  int nb=(N+255)/256;
  printf("global memory coalescing (64MB array, effective=%.1f MB):\n",bytes/1e6);
  float c=bench([&]{ copy_coalesced<<<nb,256>>>(din,dout,N); });
  float s8=bench([&]{ copy_strided<<<nb,256>>>(din,dout,N,8); });
  float s32=bench([&]{ copy_strided<<<nb,256>>>(din,dout,N,32); });
  printf(" coalesced  : %.4f ms  %.0f GB/s  1.00x\n",c,bytes/c/1e6);
  printf(" stride=8   : %.4f ms  %.0f GB/s  %.2fx\n",s8,bytes/s8/1e6,c/s8);
  printf(" stride=32  : %.4f ms  %.0f GB/s  %.2fx\n",s32,bytes/s32/1e6,c/s32);
  return 0;
}

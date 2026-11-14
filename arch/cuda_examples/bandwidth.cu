#include <cstdio>
#include <cuda_runtime.h>
__global__ void copy4(const float4* __restrict__ in, float4* __restrict__ out, size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) out[i]=in[i];
}
template<class F> float bench(F f,int it=30){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();cudaDeviceSynchronize();cudaEventRecord(a);for(int i=0;i<it;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/it;}
int main(){
  for(size_t N : {1u<<22, 1u<<24, 1u<<26, 1u<<27}){ // float4 count
    float4 *a,*b; cudaMalloc(&a,N*16); cudaMalloc(&b,N*16); cudaMemset(a,1,N*16);
    int th=256; size_t blk=(N+th-1)/th;
    float ms=bench([&]{copy4<<<blk,th>>>(a,b,N);});
    double bytes=2.0*N*16;
    printf("size=%6.1f MB  %.3f ms  %.1f GB/s\n",N*16.0/1e6,ms,bytes/ms/1e6);
    cudaFree(a);cudaFree(b);
  }
  return 0;
}

// 分支发散对吞吐率的影响 (H20, sm_90)
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void __launch_bounds__(256) div_heavy(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; float v=in[i%n];
  float a=0.f,b=0.f;
  #pragma unroll
  for(int k=0;k<256;k++){ a=fmaf(a,1.000001f,v); b=fmaf(b,0.999999f,a); }
  out[i]=a+b;
}
__global__ void __launch_bounds__(256) div_branch(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; float v=in[i%n];
  float a=0.f,b=0.f;
  if((threadIdx.x&1)==0){
    #pragma unroll
    for(int k=0;k<256;k++){ a=fmaf(a,1.000001f,v); b=fmaf(b,0.999999f,a); }
  } else {
    #pragma unroll
    for(int k=0;k<256;k++){ a=fmaf(a,0.999999f,v); b=fmaf(b,1.000001f,a); }
  }
  out[i]=a+b;
}
template<class F> float bench(F f,int it=200){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int i=0;i<it;i++) f(); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}
int main(){
  const int NB=78*8, n=NB*256;
  float *in,*out; CK(cudaMalloc(&in,n*4)); CK(cudaMalloc(&out,n*4)); CK(cudaMemset(in,1,n*4));
  float t1=bench([&]{ div_heavy<<<NB,256>>>(in,out,n); });
  float t2=bench([&]{ div_branch<<<NB,256>>>(in,out,n); });
  printf("uniform   (all threads heavy) : %.4f ms\n",t1);
  printf("divergent (alternating paths): %.4f ms  x%.2f\n",t2,t2/t1);
  return 0;
}

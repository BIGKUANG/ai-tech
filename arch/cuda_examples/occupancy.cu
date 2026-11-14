// TLP / 占用率扫描: 依赖 FMA 链 (延迟受限) (H20, sm_90)
#include <cstdio>
#include <cuda_runtime.h>
__global__ void chain(float* out, int reps){
  float a=threadIdx.x*1e-6f+1.f;
  for(int r=0;r<reps;r++) a=fmaf(a,1.0000001f,1e-7f);
  out[blockIdx.x*blockDim.x+threadIdx.x]=a;
}
template<class F> float bench(F f,int it=20){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize(); cudaEventRecord(a);
  for(int i=0;i<it;i++) f(); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}
int main(){
  const int SM=78, TPB=256, BASE=4096;
  float* d; cudaMalloc(&d, SM*64*TPB*4);
  printf("dependent-FMA chain, threads/SM sweep (H20, 78 SM):\n");
  float base=0;
  for(int bpsm : {1,2,4,8}){
    int nb=SM*bpsm; int reps=BASE*8/bpsm; // keep total work approx constant
    float ms=bench([&]{ chain<<<nb,TPB>>>(d,reps); });
    double ops=1.0*nb*TPB*reps;
    double g=ops/ms/1e6;
    if(bpsm==1) base=g;
    printf(" blocks/SM=%d threads/SM=%4d  %8.4f ms  %8.2f GFMA/s  %.2fx\n",
           bpsm,bpsm*TPB,ms,g,g/base);
  }
  return 0;
}

#include <cstdio>
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("name=%s\n",p.name);
  printf("SMs=%d\n",p.multiProcessorCount);
  printf("maxThreadsPerSM=%d\n",p.maxThreadsPerMultiProcessor);
  printf("regsPerSM=%d\n",p.regsPerMultiprocessor);
  printf("sharedMemPerSM=%zu\n",p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlockOptin=%zu\n",p.sharedMemPerBlockOptin);
  printf("L2=%d\n",p.l2CacheSize);
  printf("memBusWidth=%d\n",p.memoryBusWidth);
  printf("warpSize=%d\n",p.warpSize);
  printf("maxThreadsPerBlock=%d\n",p.maxThreadsPerBlock);
  return 0;
}

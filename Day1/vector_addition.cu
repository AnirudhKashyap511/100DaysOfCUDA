#include <iostream>
#include <cuda_runtime.h>
__global__ void vector_addition(int*a,int*b,int*c,int N)
{
    int i = threadIdx.x;
    if(i<N)
    {
        c[i] = a[i]+b[i];
    }
}

int main(void)
{
    int N = 1000;
    int *h_a = (int*)malloc(sizeof(int)*N);
    int *h_b = (int*)malloc(sizeof(int)*N);
    int *h_c = (int*)malloc(sizeof(int)*N);
    for(int i=0; i<1000; i++)
    {
        h_a[i]=i;
        h_b[i]=i;
    }
    int *d_c;
    int *d_a;
    int *d_b;
    cudaMalloc((void**)&d_c,sizeof(int)*N);
    cudaMalloc((void**)&d_a,sizeof(int)*N);
    cudaMalloc((void**)&d_b,sizeof(int)*N);
    cudaMemcpy(d_a,h_a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*sizeof(int),cudaMemcpyHostToDevice);
    dim3 grid_size(1);
    dim3 block_size(N);
    vector_addition<<<grid_size, block_size>>>(d_a,d_b,d_c,N);
    cudaMemcpy(h_c,d_c,sizeof(int)*N,cudaMemcpyDeviceToHost);
    for(int i=0; i<10; i++)
    {
        std::cout<<h_c[i];
    }

}
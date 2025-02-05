#include <iostream>
#include <cuda_runtime.h>

__global__ void increment_gpu(int *a, int N) {
    int i = threadIdx.x; 
    if (i < N) {
        a[i] = a[i] + 1;  
    }
}

int main(void) {
    int N = 5;  
    int* h_a = (int*)malloc(N * sizeof(int));  
    for (int i = 0; i < N; i++) {
        h_a[i] = 0; 
    }
    int* d_a;  
    cudaMalloc((void**)&d_a, N * sizeof(int)); 
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid_size(1);
    dim3 block_size(N);  
    increment_gpu<<<grid_size, block_size>>>(d_a, N);  
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);  
    std::cout << "Array after incrementing: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;
    return 0;  
}

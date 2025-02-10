#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  

__global__ void conv1D(float *A, float *F, float *C, int w, int f) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (col < (w - f + 1)) {  
        float sum = 0.0f;
        for (int j = 0; j < f; j++) {  
            sum += A[col + j] * F[j];
        }
        C[col] = sum;  
    }
}

int main() {
    int w = 1024;
    int f = 5;

    float *h_A = (float*)malloc(w * sizeof(float));
    float *h_F = (float*)malloc(f * sizeof(float));
    float *h_C = (float*)malloc((w - f + 1) * sizeof(float));

    for (int i = 0; i < w; i++) h_A[i] = 1.0f;  
    for (int i = 0; i < f; i++) h_F[i] = 0.2f;  

    float *d_A, *d_F, *d_C;
    cudaMalloc(&d_A, w * sizeof(float));
    cudaMalloc(&d_F, f * sizeof(float));
    cudaMalloc(&d_C, (w - f + 1) * sizeof(float));

    cudaMemcpy(d_A, h_A, w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, f * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (w - f + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1D<<<gridSize, BLOCK_SIZE>>>(d_A, d_F, d_C, w, f);
    
    cudaMemcpy(h_C, d_C, (w - f + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    free(h_A);
    free(h_F);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_F);
    cudaFree(d_C);

    return 0;
}

#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Tile size for shared memory

__global__ void matrix_multiplication_tiled(int *A, int *B, int *C, int m, int n, int q) {
    __shared__ int tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int sum = 0;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && (t * TILE_SIZE + threadIdx.x) < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (col < q && (t * TILE_SIZE + threadIdx.y) < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * q + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < q) {
        C[row * q + col] = sum;
    }
}

int main() {
    int m, n, p, q;
    std::cout << "Enter dimensions of matrix A (m n): ";
    std::cin >> m >> n;
    std::cout << "Enter dimensions of matrix B (p q): ";
    std::cin >> p >> q;

    if (n != p) {
        std::cerr << "Matrix multiplication not possible. Columns of A must match rows of B.\n";
        return -1;
    }

    int *h_A = new int[m * n];
    int *h_B = new int[p * q];
    int *h_C = new int[m * q];
    
    std::cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < m * n; i++) {
        std::cin >> h_A[i];
    }
    std::cout << "Enter elements of matrix B:\n";
    for (int i = 0; i < p * q; i++) {
        std::cin >> h_B[i];
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(int) * m * n);
    cudaMalloc(&d_B, sizeof(int) * p * q);
    cudaMalloc(&d_C, sizeof(int) * m * q);

    cudaMemcpy(d_A, h_A, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * p * q, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((q + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    
    matrix_multiplication_tiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, q);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, sizeof(int) * m * q, cudaMemcpyDeviceToHost);
    
    std::cout << "Resultant Matrix C:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            std::cout << h_C[i * q + j] << " ";
        }
        std::cout << "\n";
    }
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

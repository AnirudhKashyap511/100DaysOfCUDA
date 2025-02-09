#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16


__global__ void matrixMulTiledTransposed(const int* __restrict__ A, 
                                         const int* __restrict__ B, 
                                         int* __restrict__ C, 
                                         int m, int n, int q)
{
   
    __shared__ int sA[TILE_SIZE][TILE_SIZE];
 
    __shared__ int sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;


    int sum = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        int A_col = t * TILE_SIZE + threadIdx.x; 
        int B_row = t * TILE_SIZE + threadIdx.y;

        if (row < m && A_col < n) {
            sA[threadIdx.y][threadIdx.x] = A[row * n + A_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < q && B_row < n) {
            sB[threadIdx.x][threadIdx.y] = B[B_row * q + col];
        } else {
            sB[threadIdx.x][threadIdx.y] = 0;
        }


        __syncthreads();


        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }


    if (row < m && col < q) {
        C[row * q + col] = sum;
    }
}

int main()
{
    int m, n, p, q;
    std::cout << "Enter dimensions of matrix A (m n): ";
    std::cin >> m >> n;
    std::cout << "Enter dimensions of matrix B (p q): ";
    std::cin >> p >> q;

    if (n != p) {
        std::cerr << "Matrix multiplication not possible. "
                     "Columns of A must match rows of B.\n";
        return -1;
    }


    int* h_A = new int[m * n];
    int* h_B = new int[p * q];
    int* h_C = new int[m * q];

    std::cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < m * n; i++) {
        std::cin >> h_A[i];
    }
    std::cout << "Enter elements of matrix B:\n";
    for (int i = 0; i < p * q; i++) {
        std::cin >> h_B[i];
    }
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * m * n);
    cudaMalloc((void**)&d_B, sizeof(int) * p * q);
    cudaMalloc((void**)&d_C, sizeof(int) * m * q);

    cudaMemcpy(d_A, h_A, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * p * q, cudaMemcpyHostToDevice);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((q + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulTiledTransposed<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, q);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeof(int) * m * q, cudaMemcpyDeviceToHost);
    std::cout << "Resultant Matrix C:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            std::cout << h_C[i * q + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

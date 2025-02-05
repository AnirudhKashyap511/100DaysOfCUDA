#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_multiplication(int *a, int *b, int *c, int m, int n, int p, int q) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < q) {  
        int sum = 0;
        for (int k = 0; k < n; k++) {  
            sum += a[row * n + k] * b[k * q + col]; 
        }
        c[row * q + col] = sum;
    }
}

int main(void)
{   
    int m, n, p, q;
    std::cout << "Enter dimensions of matrix A (m n): ";
    std::cin >> m >> n;
    std::cout << "Enter dimensions of matrix B (p q): ";
    std::cin >> p >> q;


    if (n != p) {
        std::cerr << "Matrix multiplication not possible. Number of columns of A must be equal to rows of B.\n";
        return -1;
    }

    int* mat_a = (int*)malloc(sizeof(int) * m * n);
    int* mat_b = (int*)malloc(sizeof(int) * p * q);
    int* mat_c = (int*)malloc(sizeof(int) * m * q);


    std::cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < m * n; i++) {
        std::cin >> mat_a[i];
    }
    std::cout << "Enter elements of matrix B:\n";
    for (int i = 0; i < p * q; i++) {
        std::cin >> mat_b[i];
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(int) * m * n);
    cudaMalloc((void **)&d_b, sizeof(int) * p * q);
    cudaMalloc((void **)&d_c, sizeof(int) * m * q);
    cudaMemcpy(d_a, mat_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mat_b, sizeof(int) * p * q, cudaMemcpyHostToDevice);
    dim3 threadsBlock(16, 16);
    dim3 numberOfBlocks((q + 15) / 16, (m + 15) / 16); 
    matrix_multiplication<<<numberOfBlocks, threadsBlock>>>(d_a, d_b, d_c, m, n, p, q);
    cudaDeviceSynchronize();
    cudaMemcpy(mat_c, d_c, sizeof(int) * m * q, cudaMemcpyDeviceToHost);


    std::cout << "Resultant Matrix C:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            std::cout << mat_c[i * q + j] << " ";
        }
        std::cout << "\n";
    }

    free(mat_a);
    free(mat_b);
    free(mat_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

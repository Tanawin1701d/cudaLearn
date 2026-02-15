//
// Created by tanawin on 15/2/26.
//

#include <stdio.h>

#define TILE 16

__global__ void matmul_tiled(int* A, int* B, int* C, int N)
{
    ////// each block represents tile
    ////// each thread represents each cell of the tile
    __shared__ int A_tile[TILE][TILE];
    __shared__ int B_tile[TILE][TILE];

    int mem_pivot_row = blockIdx.y * TILE;
    int mem_pivot_col = blockIdx.x * TILE;

    int tile_thd_row = threadIdx.y;
    int tile_thd_col = threadIdx.x;

    int sum = 0; ///// for each destination cell in tile

    ///////// do block multiplication
    for (int iter = 0; iter < (N/TILE); iter++)
    {
        ////// AAAAAAAAAAAAAA col change but row equal
        int mem_itered_row_a = mem_pivot_row;
        int mem_itered_col_a = iter * TILE;
        ////// BBBBBBBBBBBBBB row change but col equal
        int mem_itered_row_b = iter * TILE;
        int mem_itered_col_b = mem_pivot_col;

        ///////  load data
        A_tile[tile_thd_row][tile_thd_col] = A[(mem_itered_row_a + tile_thd_row) * N + (mem_itered_col_a + tile_thd_col)];
        B_tile[tile_thd_row][tile_thd_col] = B[(mem_itered_row_b + tile_thd_row) * N + (mem_itered_col_b + tile_thd_col)];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += A_tile[tile_thd_row][k] * B_tile[k][tile_thd_col];
        }
        __syncthreads();
    }
    C[((mem_pivot_row + tile_thd_row) * N) + mem_pivot_col + tile_thd_col] = sum;

}


int main()
{

    const int N = 48;
    const size_t matrixSizeBytes = N * N * sizeof(int);

    //////////// CPU allocation
    int *h_A     = (int*)malloc(matrixSizeBytes);
    int *h_B     = (int*)malloc(matrixSizeBytes);
    int *h_C     = (int*)malloc(matrixSizeBytes);
    int *h_C_ref = (int*)malloc(matrixSizeBytes);

    ////////// initialize variable
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i * N + j;
            h_B[i * N + j] = i * N + j;
        }
    }

    // CPU reference calculation
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    //////////// GPU allocation
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSizeBytes);
    cudaMalloc(&d_B, matrixSizeBytes);
    cudaMalloc(&d_C, matrixSizeBytes);


    //////////// start computation
    dim3 blockDim(TILE, TILE);   ////each block has TILE x TILE  threads
    dim3 gridDim (N/TILE, N/TILE); ////each block has N/TILE x N/TILE blocks

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, matrixSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSizeBytes, cudaMemcpyHostToDevice);

    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result back to CPU
    cudaMemcpy(h_C, d_C, matrixSizeBytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < N * N; i++)
    {
        if (h_C[i] != h_C_ref[i])
        {
            correct = false;
            break;
        }
    }
    printf("Matrix multiplication is %s\n", correct ? "correct" : "incorrect");

    for (int i = 0; i < N*N; i++)
    {
        if ((i % N == 0) && (i != 0)) printf("\n");
        printf("%d ", h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
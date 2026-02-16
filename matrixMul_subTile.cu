//
// Created by tanawin on 16/2/26.
//

#include <stdio.h>

#define TILE 16
#define SUBTILE 8

__global__ void matmul_tiled(
    const int*  __restrict__ A,    //// __restrict__ is the promise that the all argument will not overlap
    const int* __restrict__ B,
    const int* __restrict__ C, int N)
{
    ////// each block represents tile
    ////// each thread represents each cell of the sub tile
    __shared__ int A_tile[TILE][TILE];
    __shared__ int B_tile[TILE][TILE];

    int mem_pivot_row = blockIdx.y * TILE;
    int mem_pivot_col = blockIdx.x * TILE;

    int cell_row = threadIdx.y; /// cell of the SUBTILE
    int cell_col = threadIdx.x; /// cell of the SUBTILE

    int ans_buf[TILE/SUBTILE][TILE/SUBTILE] = {};

    for (int subTile_row = 0; subTile_row < (TILE / SUBTILE); subTile_row++)
    {
        for (int subTile_Col = 0; subTile_Col < (TILE / SUBTILE); subTile_Col++)
        {
            ans_buf[subTile_row][subTile_Col] = 0;
        }
    }

    for (int iter = 0; iter < N/TILE; iter++)   //// TILE LEVEL ITERATION
    {
        /////// LOAD DATA AT TILE LEVEL
        for (int subTile_row = 0; subTile_row < (TILE / SUBTILE); subTile_row++){
            for (int subTile_Col = 0; subTile_Col < (TILE / SUBTILE); subTile_Col++){
                ////// cast back to tie level
                int tile_row = subTile_row * SUBTILE + cell_row;
                int tile_col = subTile_Col * SUBTILE + cell_col;
                ////// cast back to memory level
                int mem_row_a = mem_pivot_row               + tile_row;
                int mem_col_a = 0             + iter * TILE + tile_col;

                int mem_row_b = 0             + iter * TILE + tile_row;
                int mem_col_b = mem_pivot_col               + tile_col;

                // int mem_row_fix  = mem_pivot_row               + tile_row;
                // int mem_col_fix  = mem_pivot_col               + tile_col;
                // int mem_row_iter = mem_pivot_row + iter * TILE + tile_row;
                // int mem_col_iter = mem_pivot_col + iter * TILE + tile_col;
                ////// A: row is fixed but col is itering
                A_tile[tile_row][tile_col] = A[mem_row_a * N + mem_col_a];
                ////// B: col is fixed but row is itering
                B_tile[tile_row][tile_col] = B[mem_row_b * N + mem_col_b];
            }
        }
        __syncthreads();

        /////// MULTIPLY AT TILE LEVEL
        for(int sub_iter = 0; sub_iter < (TILE / SUBTILE); sub_iter++)
        {
            for (int subTile_row = 0; subTile_row < TILE / SUBTILE; subTile_row++){
                for (int subTile_Col = 0; subTile_Col < TILE / SUBTILE; subTile_Col++){

                    ///////
                    int tile_row_a = subTile_row * SUBTILE + cell_row;
                    int tile_col_a = sub_iter    * SUBTILE           ; //// depend on iter

                    int tile_row_b = sub_iter    * SUBTILE           ; //// depend on iter
                    int tile_col_b = subTile_Col * SUBTILE + cell_col;

                    for (int i = 0; i < SUBTILE; i++){
                        ans_buf[subTile_row][subTile_Col] += A_tile[tile_row_a][tile_col_a + i] * B_tile[tile_row_b + i][tile_col_b];
                    }
                }
            }
        }
        __syncthreads();
    }

    //////// PUT THE RESULT TILE TO THE MEMORY
    for (int subTile_row = 0; subTile_row < TILE / SUBTILE; subTile_row++){
        for (int subTile_Col = 0; subTile_Col < TILE / SUBTILE; subTile_Col++){
            ////// cast back to tie level
            int tile_row = subTile_row * SUBTILE + cell_row;
            int tile_col = subTile_Col * SUBTILE + cell_col;
            ////// cast back to memory level
            int mem_row_fix  = mem_pivot_row + tile_row;
            int mem_col_fix  = mem_pivot_col + tile_col;

            C[mem_row_fix * N + mem_col_fix] = ans_buf[subTile_row][subTile_Col];
        }
    }
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
    dim3 blockDim(SUBTILE, SUBTILE);   ////each block has TILE x TILE  threads
    dim3 gridDim (N/TILE , N/TILE); ////each block has N/TILE x N/TILE blocks

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
        printf("[%d, %d]", h_C[i], h_C_ref[i]);
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
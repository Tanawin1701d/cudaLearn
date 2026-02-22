//
// Created by tanawin on 16/2/26.
//

#include <stdio.h>

#define TILE 16
#define SUBTILE 8 /// subtile must divisible to TILE

constexpr int MAX_SUBTILE_ROW = TILE          / SUBTILE;
constexpr int MAX_SUBTILE_COL = TILE          / SUBTILE;
constexpr int MAX_SUBITER     = TILE          / SUBTILE;


__device__ void load_tile_AB(int A_tile[2][TILE][TILE], int B_tile[2][TILE][TILE],
                             const int* __restrict__ A,    //// __restrict__ is the promise that the all argument will not overlap
                             const int* __restrict__ B,
                             const int cell_row,
                             const int cell_col,
                             const int iter,
                             int tile_mem_srca_row_pivot_start,
                             int tile_mem_srca_col_pivot_start,
                             int tile_mem_srcb_row_pivot_start,
                             int tile_mem_srcb_col_pivot_start,
                             const int M, const int N, const int K
                             )
{
    int amt_thread = SUBTILE * SUBTILE;
    int loop_time_per_thread = ((TILE * TILE) + amt_thread - 1) / amt_thread;

    for (int stride = 0; stride < loop_time_per_thread; stride++)
    {
        int flatten_2_idx    = (stride * amt_thread) + (cell_row * SUBTILE) + cell_col;

        int tile_row = flatten_2_idx / TILE;
        int tile_col = flatten_2_idx % TILE;

        int  tile_mem_row_a = tile_mem_srca_row_pivot_start                 + tile_row;
        int  tile_mem_col_a = tile_mem_srca_col_pivot_start + (iter * TILE) + tile_col;
        bool isAExceed = tile_mem_row_a >= M || tile_mem_col_a >= N;

        int  tile_mem_row_b = tile_mem_srcb_row_pivot_start + (iter * TILE) + tile_row;
        int  tile_mem_col_b = tile_mem_srcb_col_pivot_start +                 tile_col;
        bool isBExceed = (tile_mem_row_b >= N) || (tile_mem_col_b >= K);

        A_tile[iter & 1][tile_row][tile_col] = isAExceed ? 0: A[tile_mem_row_a * N + tile_mem_col_a] ;
        B_tile[iter & 1][tile_row][tile_col] = isBExceed ? 0: B[tile_mem_row_b * K + tile_mem_col_b];

    }

}

__global__ void matmul_tiled(
    const int*  __restrict__ A,    //// __restrict__ is the promise that the all argument will not overlap
    const int* __restrict__ B,
          int* __restrict__ C,
          const int M,
          const int N,
          const int K
          )
{
    ////// each block represents tile
    ////// each thread represents each cell of the sub tile
    __shared__ int A_tile[2][TILE][TILE];
    __shared__ int B_tile[2][TILE][TILE];

    int tile_idx_row_pivot = blockIdx.y;
    int tile_idx_col_pivot = blockIdx.x;

    ////// calculate the target mem
    int tile_mem_des_row_pivot = tile_idx_row_pivot * TILE;
    int tile_mem_des_col_pivot = tile_idx_col_pivot * TILE;

    ////// calculate the source mem

    ////////// A
    int tile_mem_srca_row_pivot_start = tile_idx_row_pivot * TILE;
    int tile_mem_srca_col_pivot_start = 0;
    ////////// B
    int tile_mem_srcb_row_pivot_start = 0;
    int tile_mem_srcb_col_pivot_start = tile_idx_col_pivot * TILE;

    const int MAX_ITER  = (N + TILE - 1)/ TILE;


    ////////// cell is a element inside the subblock
    int cell_row = threadIdx.y;
    int cell_col = threadIdx.x;

    //////////


    int ans_buf[MAX_SUBTILE_ROW][MAX_SUBTILE_COL] = {};

    load_tile_AB(A_tile, B_tile, A, B, cell_row, cell_col, 0,
                     tile_mem_srca_row_pivot_start,
                     tile_mem_srca_col_pivot_start,
                     tile_mem_srcb_row_pivot_start,
                     tile_mem_srcb_col_pivot_start,
                     M, N, K);

    __syncthreads();

    for (int iter = 0; iter < MAX_ITER; iter++){   //// TILE LEVEL ITERATION
        /////// LOAD DATA AT TILE LEVEL
        if ( (iter + 1) < MAX_ITER){
            load_tile_AB(A_tile, B_tile, A, B, cell_row, cell_col, iter + 1 ,
                         tile_mem_srca_row_pivot_start,
                         tile_mem_srca_col_pivot_start,
                         tile_mem_srcb_row_pivot_start,
                         tile_mem_srcb_col_pivot_start,
                         M, N, K);
        }
        /////// MULTIPLY AT TILE LEVEL
        for(int sub_iter = 0; sub_iter < MAX_SUBITER; sub_iter++)
        {
            for (int subTile_row = 0; subTile_row < TILE / SUBTILE; subTile_row++){
                for (int subTile_Col = 0; subTile_Col < TILE / SUBTILE; subTile_Col++){

                    ///////
                    int tile_row_a = subTile_row * SUBTILE + cell_row;
                    int tile_col_a = sub_iter    * SUBTILE           ; //// depend on iter

                    int tile_row_b = sub_iter    * SUBTILE           ; //// depend on iter
                    int tile_col_b = subTile_Col * SUBTILE + cell_col;

                    for (int i = 0; i < SUBTILE; i++){
                        ans_buf[subTile_row][subTile_Col] += A_tile[iter & 1][tile_row_a][tile_col_a + i] *
                                                             B_tile[iter & 1][tile_row_b + i][tile_col_b];
                    }
                }
            }
        }
        __syncthreads();
    }

    //////// PUT THE RESULT TILE TO THE MEMORY
    for (int subTile_row = 0; subTile_row < MAX_SUBTILE_ROW; subTile_row++){
        for (int subTile_Col = 0; subTile_Col < MAX_SUBTILE_COL; subTile_Col++){
            //////TILE: cast back to tie level
            int tile_row = subTile_row * SUBTILE + cell_row;
            int tile_col = subTile_Col * SUBTILE + cell_col;
            //////C: cast back to memory level
            int tile_mem_des_row = tile_mem_des_row_pivot + tile_row;
            int tile_mem_des_col = tile_mem_des_col_pivot + tile_col;

            if (tile_mem_des_row < M && tile_mem_des_col < K){
                C[tile_mem_des_row * K + tile_mem_des_col] = ans_buf[subTile_row][subTile_Col];
            }
        }
    }
}


int main(){
    const int N = 50;  // inner dimension
    const int M = 60;  // rows of A, C
    const int K = 70;  // cols of B, C

    const size_t bytesA = M * N * sizeof(int);
    const size_t bytesB = N * K * sizeof(int);
    const size_t bytesC = M * K * sizeof(int);

    //////////// CPU allocation
    int *h_A     = (int*)malloc(bytesA);
    int *h_B     = (int*)malloc(bytesB);
    int *h_C     = (int*)malloc(bytesC);
    int *h_C_ref = (int*)malloc(bytesC);

    ////////// initialize variable
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i * N + j;
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            h_B[i * K + j] = i * K + j;
        }
    }

    // CPU reference calculation: C(MxK) = A(MxN) * B(NxK)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += h_A[i * N + k] * h_B[k * K + j];
            }
            h_C_ref[i * K + j] = sum;
        }
    }

    //////////// GPU allocation
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    //////////// start computation
    dim3 blockDim(SUBTILE, SUBTILE);   //// each block has SUBTILE x SUBTILE threads
    dim3 gridDim(
        (K + TILE - 1) / TILE,  //// number of tiles along columns (K)
        (M + TILE - 1) / TILE   //// number of tiles along rows (M)
    );

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to CPU
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Verify results
    bool correct = true;
    for (int i = 0; i < M * K; i++)
    {
        if (h_C[i] != h_C_ref[i])
        {
            correct = false;
            break;
        }
    }
    printf("Matrix multiplication is %s\n", correct ? "correct" : "incorrect");

    for (int i = 0; i < M * K; i++)
    {
        if ((i % K == 0) && (i != 0)) printf("\n");
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

//
// Created by tanawin on 10/2/26.
//

#include <iostream>
#include <cuda_runtime.h>

__device__ int warpReduceSum(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduc(const int* input, int* output, int n) {
    //std::cout << "warp size: " << warpSize << std::endl;
    printf("warp size: %d\n", warpSize);


    int tid = threadIdx.x; /// the thread number in the block
    int idx = blockIdx.x * blockDim.x + tid; /// for access main memory

    ////// the entire wrap is reducted into the val
    int val = 0; /// initialize the value
    if (idx < n){val = input[idx];} //// load the data from main memory

    val = warpReduceSum(input[idx]); //// do warp level reduction

    __shared__ int shared[32];
    int lane   = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    val = (tid < blockDim.x / warpSize) ? shared[lane] : 0;

    if (warpId == 0){val = warpReduceSum(val);}

    if (tid == 0){
        output[blockIdx.x] = val;
    }

}


int main()
{
    const int n = 256+32;

    ///// mem CPU allocate
    int* h_inp = new int[n];
    int* h_out = new int[1];

    for(int i = 0; i < n; i++) {
        h_inp[i] = i;
    }

    ///// device allocate
    int *d_input, *d_output;
    cudaMalloc(&d_input , n * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    ///// copy data to device
    cudaMemcpy(d_input, h_inp, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256 + 32;
    int block = 1;

    reduc<<<block, threadsPerBlock>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    ///// copy result back
    cudaMemcpy(h_out, d_output, sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "sum res using Wrap  " << h_out[0] << std::endl;
    std::cout << "sum res using cpu   " << n * (n-1) / 2 << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_inp;
    delete[] h_out;


}
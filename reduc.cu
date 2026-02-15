//
// Created by tanawin on 10/2/26.
//

#include <iostream>
#include <cuda_runtime.h>


__global__ void reduc(const int* input, int* output, int n) {
    extern __shared__ int sData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sData[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sData[tid] += sData[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sData[0];
}


int main()
{
    const int n = 256;

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

    int threadsPerBlock = 256;
    int block = 1;

    reduc<<<1, 256, 256 * sizeof(int)>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    ///// copy result back
    cudaMemcpy(h_out, d_output, sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "sum res " << h_out[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_inp;
    delete[] h_out;


}
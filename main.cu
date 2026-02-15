#include <iostream>
#include <cuda_runtime.h>

__global__ void add_kernel(const int* a, const int* b, int* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }


}


int main()
{
    const int n = 1024;
    const size_t bytes = n * sizeof(int);

    ///// mem CPU allocate
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    for (int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = 2*i;
    }

    ///// device allocate
    int *d_a, *d_b, *d_c;
    std::cout << "d_a :" << d_a << std::endl;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    ////// copy CPU 2 GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    ///// execute
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);


    ///// copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("h_c[10] = %d\n", h_c[10]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
cmake_minimum_required(VERSION 4.0)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
project(cudaLearn CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cudaLearn
        #main.cu
        #reduc.cu
        #reducWrap.cu
        #matrixMul.cu
        #matrixMul_subTile.cu
        #p6_matrixMul_general.cu
        #p7_matrixMul_doubleBuffering.cu
        p8_matrixMul_load_loop_reorder.cu
)

set_target_properties(cudaLearn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

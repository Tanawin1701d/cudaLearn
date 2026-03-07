//
// Created by tanawin on 16/2/26.
//

#include <filesystem>
#include <stdio.h>
#include <bits/exception_ptr.h>


__global__ void bfs(
    ///// src parameter
    const int* __restrict__ srcNodePtr,     ///// index is NODE_ID -- value is the start query index in desNodePtr
    const int* __restrict__ desNodePtr,     ///// the value is the destination of node
    ///// des parameter
          int* __restrict__ frontNode,
          int* __restrict__ nextFrontNode,
    ///// check parameter
          int* __restrict__ visitedNode,
    const int               AMT_NODE
)
{
    __shared__ int pending_size_a(1);
    __shared__ int pending_size_b(0);

    if (blockIdx.x == 0 && threadIdx.x == 0){
        frontNode[0] = 0;
    }

    int* frontNodeSize     = & pending_size_a;
    int* nextFrontNodeSize = & pending_size_b;

    ////// clear the visitedNode to 0
    for (int uniqueSearchIdx = blockIdx.x * blockDim.x + threadIdx.x;
             uniqueSearchIdx < AMT_NODE;
             uniqueSearchIdx += blockDim.x * gridDim.x)
    {
        visitedNode[uniqueSearchIdx] = 0;
    }
    __syncthreads();


    while (*frontNodeSize != 0){
        ///// pick the source node
        int uniqueSearchIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (; uniqueSearchIdx < *frontNodeSize; uniqueSearchIdx += blockDim.x * gridDim.x){
            int frontNodeId = frontNode[uniqueSearchIdx];
            ////// pick the destination node
            for (int edgeIdx = srcNodePtr[frontNodeId]; edgeIdx < srcNodePtr[frontNodeId + 1]; edgeIdx++){
                int desNodeId = desNodePtr[edgeIdx];
                if (visitedNode[desNodeId] == 0){
                    visitedNode[desNodeId] = 1;
                    int savedIndex = atomicAdd(nextFrontNodeSize, 1);
                    nextFrontNode[savedIndex] = desNodeId;
                }
            }
        }
        /////// swap the front and repeat it again
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0){
            ////// swap fron
            std::swap(frontNode, nextFrontNode);
            std::swap(frontNodeSize, nextFrontNodeSize);
        }
        __syncthreads();
    }

}

int main(){

    ///// build the constant number
    const int LAYER_NUM = 6;
    const int AMT_NODE  = (1<<LAYER_NUM) - 1;
    const int AMT_NODE_EXCEPT_LAST = AMT_NODE - (1 << (LAYER_NUM-1)); //// all node - last layer
    const int AMT_EDGE  = 2 * AMT_NODE_EXCEPT_LAST; /// build the tree


    ///// build the tree
    const int SRC_ARR_IN_BYTES = sizeof(int) * (AMT_NODE + 1);
    const int DES_ARR_IN_BYTES = sizeof(int) * AMT_EDGE;
    int* srcNodePtr            = (int*)malloc(SRC_ARR_IN_BYTES);
    int* desNodePtr            = (int*)malloc(DES_ARR_IN_BYTES);
    int* visitNodePtr          = (int*)malloc(SRC_ARR_IN_BYTES);

    int lastPlacedDesNode = 0;
    int startRowIdx = 0;
    for (int layerId = 0; layerId < LAYER_NUM; layerId++)
    {
        for(int colIdx = 0; colIdx < (1 << layerId); colIdx++)
        {
            int curNodeIdx = startRowIdx + colIdx;
            srcNodePtr[curNodeIdx] = lastPlacedDesNode;
            for (int i = 1; i <= 2; i++){
                int nextNodeIdx = 2 * curNodeIdx + i;
                desNodePtr[lastPlacedDesNode++] = nextNodeIdx;
            }
        }
        startRowIdx += (1 << layerId);
    }
    srcNodePtr[AMT_NODE] = lastPlacedDesNode;

    // allocate GPU memory
    int *srcNodePtrDevice   , *desNodePtrDevice;
    int *frontNodeDevice    , *nextFrontNodeDevice, *visitedNodeDevice;

    cudaMalloc(&srcNodePtrDevice       , SRC_ARR_IN_BYTES  );
    cudaMalloc(&desNodePtrDevice       , DES_ARR_IN_BYTES  );
    cudaMalloc(&frontNodeDevice        , 2*SRC_ARR_IN_BYTES);
    cudaMalloc(&nextFrontNodeDevice    , 2*SRC_ARR_IN_BYTES);
    cudaMalloc(&visitedNodeDevice      , SRC_ARR_IN_BYTES  );

    // Copy data to GPU
    cudaMemcpy(srcNodePtrDevice, srcNodePtr, SRC_ARR_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(desNodePtrDevice, desNodePtr, DES_ARR_IN_BYTES, cudaMemcpyHostToDevice);

    // start computation
    int threadSize = 8;
    int blockSize  = 1;

    bfs<<<threadSize, blockSize>>>(srcNodePtrDevice,
                                   desNodePtrDevice,
                                   frontNodeDevice,
                                   nextFrontNodeDevice,
                                   visitedNodeDevice,
                                   AMT_NODE);

    // Copy result back to CPU
    cudaMemcpy(visitedNodeDevice, visitNodePtr, SRC_ARR_IN_BYTES, cudaMemcpyDeviceToHost);


    for (int i = 0; i < AMT_NODE; i++)
    {
        if (visitedNodeDevice[i] != 1){
            printf("error node %d is not reached", i);
        }

    }


    delete srcNodePtrDevice   ;
    delete desNodePtrDevice   ;
    delete frontNodeDevice    ;
    delete nextFrontNodeDevice;
    delete visitedNodeDevice  ;

    cudaFree(&srcNodePtrDevice    );
    cudaFree(&desNodePtrDevice    );
    cudaFree(&frontNodeDevice     );
    cudaFree(&nextFrontNodeDevice );
    cudaFree(&visitedNodeDevice   );

    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utilities.h"
using std::cout;
using std::endl;

template <typename T>
__global__ void sumArrays_gpu(T* a, T* b, T* ans, unsigned int size){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size)
        ans[tid] = a[tid] + b[tid];
}

template <typename T>
__global__ void initArr_gpu(T *data, unsigned int size){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size){
        data[tid] = 1;
    }
}

int main(int argc, char** argv){
    // create arr on host
    constexpr int size = 1 << 14; // 2 ^ 14 = 16KB

    // config for cuda
    initDevice(0);
    int blockSize_x = 1024; 
    if(argc > 1) blockSize_x = atoi(argv[1]);
    dim3 blockSize(blockSize_x, 1);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);


    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    cudaEventQuery(start2);
    

    for(int i = 0; i < 10; ++i){

        // without UVA
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

        int *h_a = new int[size];
        int *h_b = new int[size];
        int *h_ans = new int[size];
        initArr(h_a, size);
        initArr(h_b, size);
        
        int *d_a=nullptr, *d_b=nullptr, *d_ans=nullptr;
        cudaMalloc((void**)(&d_a), size*sizeof(int));
        cudaMalloc((void**)(&d_b), size*sizeof(int));
        cudaMalloc((void**)(&d_ans), size*sizeof(int));
        cudaMemcpy(d_a, h_a, size*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size*sizeof(int), cudaMemcpyHostToDevice);
        
        sumArrays_gpu<int> <<<gridSize, blockSize>>> (d_a, d_b, d_ans, size);

        cudaMemcpy(h_ans, d_ans, size*sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time_without_UVA = 0;
        cudaEventElapsedTime(&time_without_UVA, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cout<<"time without UVA:"<<time_without_UVA<<endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ans);
        delete[] h_a;
        delete[] h_b;
        delete[] h_ans;

        cudaDeviceSynchronize();

        // with UVA
        cudaEvent_t start2, stop2;
        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2);
        cudaEventQuery(start2);

        int *a = nullptr, *b = nullptr, *ans = nullptr;
        // cudaMalloc((void**)(&a), size*sizeof(int));
        // cudaMalloc((void**)(&b), size*sizeof(int));
        // cudaMalloc((void**)(&ans), size*sizeof(int));
        CHECK(cudaMallocManaged((void**)(&a), size*sizeof(int), cudaMemAttachGlobal)); // a on Device
        // cudaMemAdvise(a, size*sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        // CHECK(cudaMemPrefetchAsync(a, size, cudaCpuDeviceId));
        CHECK(cudaMallocManaged((void**)(&b), size*sizeof(int), cudaMemAttachGlobal)); // b on Device
        // cudaMemAdvise(b, size*sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        // CHECK(cudaMemPrefetchAsync(b, size, cudaCpuDeviceId));
        CHECK(cudaMallocManaged((void**)(&ans), size*sizeof(int), cudaMemAttachGlobal)); // ans on Device
        // cudaMemAdvise(ans, size*sizeof(int), cudaMemAdviseSetPreferredLocation, 0);
        // initArr_gpu<int> <<<gridSize, blockSize>>> (a, size);
        // initArr_gpu<int> <<<gridSize, blockSize>>> (b, size);
        initArr(a, size);
        // CHECK(cudaMemPrefetchAsync(a, size, 0));
        initArr(b, size);
        // CHECK(cudaMemPrefetchAsync(b, size, 0));
        // cudaDeviceSynchronize();
        sumArrays_gpu<int> <<<gridSize, blockSize>>> (a, b, ans, size); 

        // CHECK(cudaMemPrefetchAsync(ans, size, cudaCpuDeviceId)); // ans copy to Host

        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float time_with_UVA = 0;
        cudaEventElapsedTime(&time_with_UVA, start2, stop2);
        cudaEventDestroy(start2);
        cudaEventDestroy(stop2);

        cout<<"time with UVA:"<<time_with_UVA<<endl;
        cudaFree(a);
        cudaFree(b);
        cudaFree(ans);

        cudaDeviceSynchronize();
    }

    return 0;
}
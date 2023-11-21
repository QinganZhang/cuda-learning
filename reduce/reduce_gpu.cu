#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utilities.h"
#include <nvToolsExt.h>
using std::cout;
using std::endl;

#define warpSize 32

template <typename T>
__global__ void reduce_global1(T* data1, T* data2, int size){ // 交错配对方式
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;

    T* data = data1 + blockIdx.x * blockDim.x;
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            data[tid] += data[tid + offset];
        }
        __syncthreads(); // 保证一个线程块中线程是同步的，否则可能出现数据竞争
    }
    if(tid == 0)
        data2[blockIdx.x] = data[0];
}

template <typename T>
__global__ void reduce_global2(T* data1, T* data2, int size){ // 相邻配对方式
    const int tid = threadIdx.x;
    T *data = data1 + blockIdx.x * blockDim.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if(tid % (2 * stride) == 0){
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0)
        data2[blockIdx.x] = data[0];
}

template <typename T, unsigned int blockSize> // blockSize即为blockDim.x
__global__ void reduce_global_unroll(T* data1, T* data2, int size, int n=4){
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * (n*blockDim.x);
    if(idx >= size) return ;

    T* data = data1 + blockIdx.x * (n*blockDim.x); // 每个线程块处理n*blockDim.x长度的数组
    for(int i = 1; i < n; ++i){
        int tmp_idx = idx + i * blockDim.x;
        if(tmp_idx < size){
            // data1[idx] += data1[tmp_idx];
            atomicAdd(data1 + idx, data1[tmp_idx]);
        }
    }
    // printf("data1[%d]: %d\n", idx, data1[idx]);

    __syncthreads();
	
    // 循环展开，注意原来for循环中同样有判断，可能产生分支
    // 使用非模板参数，在编译阶段就可以进行判断
    if(blockSize >= 1024){
        if(tid < 512)
            data[tid]+=data[tid+512];
        __syncthreads();
    }
    if(blockSize >= 512){
        if(tid < 256)
            data[tid]+=data[tid+256];
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128)
            data[tid]+=data[tid+128];
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64)
            data[tid]+=data[tid+64];
        __syncthreads();
    }        
	
	if(tid<32) // 还剩下最前面64个数
	{
        // volatile int类型变量是控制变量结果写回到内存，而不是存在共享内存，或者缓存中
        // 同时，因为此时warp不会产生分支，同一个warp内部严格SIMT，即严格按照顺序执行（不会产生数据冲突），完美折半累加
        // 效果是节省了__syncthreads()
		volatile int *vmem = data; 
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];
	}
    /*
    // 如果将数组累加到最前面32个数：
    if(blockSize>=64 && tid < 32)
        data[tid] += data[tid+32];
    __syncthreads();
    if(tid == 0){ // 一个线程内部需要将data[0]~data[31]累加
        // sum(data[0:32])
    }
    */
	if (tid == 0)
		data2[blockIdx.x] = data[0];
}


template <typename T, unsigned int blockSize>
__global__ void reduce_shared(T* data1, T* data2, int size){ // 交错配对方式
    __shared__ T s_data[blockSize];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;
    s_data[tid] = data1[idx];
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){ // 每个线程块中最多只有一个warp产生分支（当offset是32的倍数时，不会产生warp）
            s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
        data2[blockIdx.x] = s_data[0];
}

template <typename T, unsigned int blockSize>
__global__ void reduce_shared2(T* data1, T* data2, int size){ // 相邻配对方式
    __shared__ T s_data[blockSize];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;
    s_data[tid] = data1[idx];
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        int s_idx = 2*tid * stride; // s_idx: stride的偶数倍
        if(s_idx < blockDim.x){ // 每个线程块中最多只有一个warp产生分支（当blockDim.x是32*stride的倍数时，不会产生warp）
            s_data[s_idx] += s_data[s_idx + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
        data2[blockIdx.x] = s_data[0];
}

template <typename T, unsigned int blockSize>
__global__ void reduce_shared_shrink(T* data1, T* data2, int size, int n = 4){
    // 默认n=4,每个线程块处理4*blockDim.x长度的数据
    // n表示每个线程块折半累加n个数据块
    __shared__ T s_data[blockSize];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * (n*blockDim.x);
    if(idx >= size) return;

    s_data[threadIdx.x] = 0;
    
    for(int i = 0; i < n; ++i){
        int tmp_idx = idx + i * blockDim.x;
        if(tmp_idx < size){
            atomicAdd(s_data + threadIdx.x, data1[tmp_idx]);
        }
    }

    // s_data[threadIdx.x] = // 交错配对方式，每个线程块处理相邻的两块数据
    //     data1[threadIdx.x + blockIdx.x * (2*blockDim.x)] + 
    //     data1[threadIdx.x + blockIdx.x * (2*blockDim.x) + blockDim.x];
    // s_data[threadIdx.x] = // 相邻匹配方式，每个线程块处理相邻的两块数据
    //     data1[threadIdx.x * 2 + blockIdx.x * blockDim.x] + 
    //     data1[threadIdx.x * 2 + 1 + blockIdx.x * blockDim.x];
    
    __syncthreads();
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){ // 每个线程块中最多只有一个warp产生分支（当offset是32的倍数时，不会产生warp）
            s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
        data2[blockIdx.x] = s_data[0];    
}

template <typename T, unsigned int blockSize>
__global__ void reduce_shared_warp(T* data1, T* data2, int size, int n = 4){
    __shared__ T s_data[blockSize];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * (n*blockDim.x);
    if(idx >= size) return;

    s_data[threadIdx.x] = 0;
    
    for(int i = 0; i < n; ++i){
        int tmp_idx = idx + i * blockDim.x;
        if(tmp_idx < size){
            atomicAdd(s_data + threadIdx.x, data1[tmp_idx]);
        }
    }    

    __syncthreads();
    for(int offset = blockDim.x / 2; offset > 32; offset /= 2){
        if(tid < offset){ // 每个线程块中最多只有一个warp产生分支（当offset是32的倍数时，不会产生warp）
            s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
    }    

    // 此时offset = 32(应该是默认数组size是2的幂次), tid < offset说明此时线程都在一个warp中, 可以优化掉__syncthreads()
    if(tid < 32){ // 一个warp中严格SIMT
        s_data[tid] += s_data[tid + 32];__syncwarp();
        s_data[tid] += s_data[tid + 16];__syncwarp();
        s_data[tid] += s_data[tid + 8];__syncwarp();
        s_data[tid] += s_data[tid + 4];__syncwarp();
        s_data[tid] += s_data[tid + 2];__syncwarp();
        s_data[tid] += s_data[tid + 1];
    }
    if(tid == 0)
        data2[blockIdx.x] = s_data[0];  
}

template <typename T, unsigned int blockSize>
__global__ void reduce_shared_unroll(T* data1, T* data2, int size, int n = 4){
    __shared__ T s_data[blockSize];
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * (n*blockDim.x);
    if(idx >= size) return;

    s_data[threadIdx.x] = 0;
    for(int i = 0; i < n; ++i){
        int tmp_idx = idx + i * blockDim.x;
        if(tmp_idx < size){
            atomicAdd(s_data + threadIdx.x, data1[tmp_idx]);
        }
    }     

    __syncthreads();

    if(blockSize >= 1024){
        if(tid < 512)
            s_data[tid]+=s_data[tid+512];
        __syncthreads();
    }
    if(blockSize >= 512){
        if(tid < 256)
            s_data[tid]+=s_data[tid+256];
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128)
            s_data[tid]+=s_data[tid+128];
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64)
            s_data[tid]+=s_data[tid+64];
        __syncthreads();
    }        

    if(tid < 32){ // 一个warp中严格SIMT
        if(blockSize >= 64)
            s_data[tid] += s_data[tid + 32];__syncwarp();
        if(blockSize >= 32)
            s_data[tid] += s_data[tid + 16];__syncwarp();
        if(blockSize >= 16)
            s_data[tid] += s_data[tid + 8];__syncwarp();
        if(blockSize >= 8)
            s_data[tid] += s_data[tid + 4];__syncwarp();
        if(blockSize >= 4)
            s_data[tid] += s_data[tid + 2];__syncwarp();
        if(blockSize >= 2)
            s_data[tid] += s_data[tid + 1];
    }
    if(tid == 0)
        data2[blockIdx.x] = s_data[0];  
}


template <typename T, unsigned int blockSize>
__device__ __forceinline__ T warpReduceSum(T value){
    if(blockSize >= 32)value += __shfl_down_sync(0xffffffff,value,16);
    if(blockSize >= 16)value += __shfl_down_sync(0xffffffff,value,8);
    if(blockSize >= 8)value += __shfl_down_sync(0xffffffff,value,4);
    if(blockSize >= 4)value += __shfl_down_sync(0xffffffff,value,2);
    if(blockSize >= 2)value += __shfl_down_sync(0xffffffff,value,1);
    return value;    
}

template <typename T, unsigned int blockSize>
__global__ void reduce_shared_shuffle(T* data1, T* data2, int size, int n = 4){
    // shuffle指令可以使得同一个warp中的线程访问彼此的寄存器，避免了中间通过共享内存进行通信
    // 一个线程块中线程数量blockSize<=32*32=1024
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * (n*blockDim.x);
    
    if(idx >= size) return;

    T value = 0;
    for(int i = 0; i < n; ++i){
        int tmp_idx = idx + i * blockDim.x;
        if(tmp_idx >= size) break;
        // atomicAdd(&value, data1[tmp_idx]); // value是寄存器变量，无需Atomic
        value += data1[tmp_idx];
    }  
    __syncthreads();

    // warpSize = 32;
    const int warpId = threadIdx.x / warpSize; // warp的索引
    const int laneId = threadIdx.x - warpId * warpSize; // warp内部32个线程的索引
    constexpr int warpNum = blockSize / warpSize; // 一个线程块中warp的数量
    // __shared__ T warpLevelSums[warpNum]; 
    __shared__ T warpLevelSums[warpSize];  // 本来应该是warpNum，但是为了凑够一个warp
    
    T warpSum = warpReduceSum<T, blockSize> (value); 
    // warpSum是每个线程的私有变量，warpReduceSum将一个warp中的value reduce到第一个线程并返回

    if(laneId == 0) warpLevelSums[warpId] = warpSum;
    __syncthreads();

    warpSum = (threadIdx.x < warpNum) ? warpLevelSums[laneId]: 0;

    // if(warpId == 0) warpSum = warpReduceSum<T, warpNum> (warpSum);
    if(warpId == 0) warpSum = warpReduceSum<T, warpSize> (warpSum);

    if(tid == 0) data2[blockIdx.x] = warpSum;
    
}


template <typename T>
__global__ void init_gpu(T* arr, T val, int size){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;
    arr[idx] = val;
}

template <typename T>
__global__ void val_init_gpu(T* arr, T val){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0)
        *arr = val;
}

template <typename T>
__global__ void show_gpu(T* arr, int size){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return ;
    printf("arr[%d]:%d\n", idx, arr[idx]);
}

int main(int argc, char** argv){
    constexpr int in_size = 1 << 26; // 1 << x
    int k = 3; // 循环三次
    int *data1=nullptr, *data2=nullptr, *data3=nullptr, *result=nullptr, result_host=0;

    { // reduce_global1
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 512; 
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1;
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 512;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2;
        
        constexpr unsigned int data3_size = gridSize2;
        constexpr unsigned int blockSize3 = gridSize2;
        constexpr unsigned int gridSize3 = 1;
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&data3), data3_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2, blockSize2>>> (data2, 0, data2_size);
            init_gpu<int> <<<gridSize3, blockSize3>>> (data3, 0, data3_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_global1");
                reduce_global1<int> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_global1<int> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_global1<int> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
                nvtxRangePop();
            }
            else{
                reduce_global1<int> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_global1<int> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_global1<int> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
            }


            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_global1 result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(data3);
        cudaFree(result);
    }

    { // reduce_global2
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 512;
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1; 
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 512;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2; 
        
        constexpr unsigned int data3_size = gridSize2;
        constexpr unsigned int blockSize3 = gridSize2;
        constexpr unsigned int gridSize3 = 1;
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&data3), data3_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2, blockSize2>>> (data2, 0, data2_size);
            init_gpu<int> <<<gridSize3, blockSize3>>> (data3, 0, data3_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_global2");
                reduce_global2<int> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_global2<int> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_global2<int> <<<gridSize3, blockSize3>>> (data3, result, data3_size);                
                nvtxRangePop();
            }
            else{
                reduce_global2<int> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_global2<int> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_global2<int> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_global2 result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(data3);
        cudaFree(result);
    }

    { // reduce_global_unroll
        constexpr int n = 8;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 1024; // 1 << 10
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 10 - 3) = 1<<13
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 1024;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (13 - 10 - 3) = 1
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            // 极易出错，后面reduce过程中每个线程处理多个数据块，但是这里初始化每个线程初始化一个数据，因此线程配置不同
            // 否则分配的线程数量不够，导致有的数据没有初始化
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);
            
            if(i == k-1){
                nvtxRangePushA("last reduce_global_unroll");
                reduce_global_unroll<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_global_unroll<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
                nvtxRangePop();
            }
            else{
                reduce_global_unroll<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_global_unroll<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
            }



            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_global_unroll result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 512; 
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1; // 1 << (26 - 9) = 1<<17
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 512;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2; // 1 << (17 - 9) = 1<<8
        
        constexpr unsigned int data3_size = gridSize2;
        constexpr unsigned int blockSize3 = gridSize2;
        constexpr unsigned int gridSize3 = 1;

        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&data3), data3_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2, blockSize2>>> (data2, 0, data2_size);
            init_gpu<int> <<<gridSize3, blockSize3>>> (data3, 0, data3_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared");
                reduce_shared<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_shared<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_shared<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
                nvtxRangePop();
            }
            else{
                reduce_shared<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_shared<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_shared<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared2
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 512;
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1; // 1 << (26 - 10) = 1<<16
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 512;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2; // 1 << (16 - 10) = 1<<6
        
        constexpr unsigned int data3_size = gridSize2;
        constexpr unsigned int blockSize3 = gridSize2;
        constexpr unsigned int gridSize3 = 1;

        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&data3), data3_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2, blockSize2>>> (data2, 0, data2_size);
            init_gpu<int> <<<gridSize3, blockSize3>>> (data3, 0, data3_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared2");
                reduce_shared2<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_shared2<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_shared2<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
                nvtxRangePop();
            }
            else{
                reduce_shared2<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size);
                reduce_shared2<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size);
                reduce_shared2<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared2 result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared_shrink
        constexpr int n = 8;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 1024; // 1 << 10
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 10 - 3) = 1<<13
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 1024;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (13 - 10 - 3) = 1
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared_shrink");
                reduce_shared_shrink<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shrink<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);                
                nvtxRangePop();
            }
            else{
                reduce_shared_shrink<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shrink<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
            }


            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared_shrink result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared_warp
        constexpr int n = 8;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 1024; // 1 << 10
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 10 - 3) = 1<<13
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 1024;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (13 - 10 - 3) = 1
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared_warp");
                reduce_shared_warp<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_warp<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
                nvtxRangePop();
            }
            else{
                reduce_shared_warp<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_warp<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared_warp result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared_unroll
        constexpr int n = 8;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 1024; // 1 << 10
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 10 - 3) = 1<<13
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 1024;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (13 - 10 - 3) = 1
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared_unroll");
                reduce_shared_unroll<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_unroll<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
                nvtxRangePop();
            }
            else{
                reduce_shared_unroll<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_unroll<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared_unroll result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared_shuffle
        constexpr int n = 8;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 1024; // 1 << 10
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 10 - 3) = 1<<13
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 1024;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (13 - 10 - 3) = 1
        
        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;
        
        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared_shuffle");
                reduce_shared_shuffle<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shuffle<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
                nvtxRangePop();
            }
            else{
                reduce_shared_shuffle<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shuffle<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, result, data2_size, n);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared_shuffle result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    { // reduce_shared_shuffle
        constexpr int n = 2;
        constexpr unsigned int data1_size = in_size;
        constexpr unsigned int blockSize1 = 512; 
        constexpr unsigned int gridSize1 = (data1_size + blockSize1 - 1) / blockSize1 / n; // 1 << (26 - 9 - 1) = 1<<16
        
        constexpr unsigned int data2_size = gridSize1;
        constexpr unsigned int blockSize2 = 512;
        constexpr unsigned int gridSize2 = (data2_size + blockSize2 -1 ) / blockSize2 / n; // 1 << (16 - 9 - 1) = 1<<6
        
        constexpr unsigned int data3_size = gridSize2;
        constexpr unsigned int blockSize3 = gridSize2;
        constexpr unsigned int gridSize3 = 1;

        cudaMalloc((void**)(&data1), data1_size * sizeof(int));
        cudaMalloc((void**)(&data2), data2_size * sizeof(int));
        cudaMalloc((void**)(&data3), data3_size * sizeof(int));
        cudaMalloc((void**)(&result), 1 * sizeof(int));
        cudaEvent_t start, stop;

        for(int i = 0; i < k; ++i){
            init_gpu<int> <<<gridSize1 * n, blockSize1>>> (data1, 1, data1_size);
            init_gpu<int> <<<gridSize2 * n, blockSize2>>> (data2, 0, data2_size);
            init_gpu<int> <<<gridSize3, blockSize3>>> (data3, 0, data3_size);
            val_init_gpu<int> <<<1, 1>>>(result, 0);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaEventQuery(start);

            if(i == k-1){
                nvtxRangePushA("last reduce_shared_shuffle");
                reduce_shared_shuffle<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shuffle<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size, n);
                // reduce_shared_shuffle<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size, n);
                reduce_shared<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
                nvtxRangePop();
            }
            else{
                reduce_shared_shuffle<int, blockSize1> <<<gridSize1, blockSize1>>> (data1, data2, data1_size, n);
                reduce_shared_shuffle<int, blockSize2> <<<gridSize2, blockSize2>>> (data2, data3, data2_size, n);
                // reduce_shared_shuffle<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size, n);                
                reduce_shared<int, blockSize3> <<<gridSize3, blockSize3>>> (data3, result, data3_size);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);        

            cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(i == k-1)
                cout<<"reduce_shared_shuffle result:"<< result_host <<" time usage:"<<time<< endl;
        }

        cudaFree(data1);
        cudaFree(data2);
        cudaFree(result);
    }

    return 0;
}
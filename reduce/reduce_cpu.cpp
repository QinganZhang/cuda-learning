#include <cstdio>
#include <iostream>
using namespace std;

#define lowbit(x) ((x) & (-x))

template <typename T>
T recursiveReduce_cpu1(T *data, const int size){ // 交错配对
    if(size == 1) return data[0];
    
    const int stride = size / 2;
    for(int i = 0; i < stride; ++i){
        data[i] += data[i + stride];
    }
    if(size % 2 == 1) data[0] += data[size-1];

    return recursiveReduce_cpu1(data, stride);
}

template <typename T>
T reduce_cpu(T *data, const int size){ // 交错配对
    if(size == 1) return data[0];
    
    for(int stride = size / 2, len = size; stride != 0 ; stride /= 2){
        for(int i = 0; i < stride; ++i){
            data[i] += data[i + stride];
        }
        if(len % 2 == 1) data[0] += data[len - 1];
        len = stride;
    }
    return data[0];
}

template <typename T>
void recursiveReduce_cpu2_func(T *data, const int size, int stride = 1){ // 相邻配对
    if(size == 1) return ;
    if(2*stride > size) return ;
    
    for(int i = 0; i < size; i += stride*2){
        data[i] += data[i + stride];
    }
    
    recursiveReduce_cpu2_func(data, size, stride * 2);
}

template <typename T>
T reduce_cpu2_func(T *data, const int size){ // 相邻配对
    if(size == 1) return data[0];

    for(int stride = 1; 2 * stride <= size; stride *= 2){
        for(int i = 0; i < size; i += stride*2){
            data[i] += data[i + stride];
        }
    }
    return data[0];
}

template <typename T>
T recursiveReduce_cpu2(T* data, const int size){
    if(size == 1) return data[0];

    int ans = 0;
    T* start = data;
    for(int rest = size, len = 0; rest > 0; rest -= len){
        len = lowbit(rest); 
        recursiveReduce_cpu2_func(start, len); // 使用递归的相邻配对
        ans += start[0]; 
        start += len;
    }
    return ans;
}

template <typename T>
T reduce_cpu2(T* data, const int size){
    if(size == 1) return data[0];

    int ans = 0;
    T* start = data;
    for(int rest = size, len = 0; rest > 0; rest -= len){
        len = lowbit(rest); 
        ans += reduce_cpu2_func(start, len); // 使用递推的相邻配对
        start += len;
    }
    return ans;
}


template <typename T>
void initArr(T* arr, const int size){
    for(int i = 0 ; i < size; ++i){
        if(typeid(T) == typeid(int)){
            arr[i] = i;
        }
    }
}


int main(){
    constexpr int n = 10;
    int arr[n];
    
    initArr<int>(arr, n);
    cout<<recursiveReduce_cpu1(arr, n)<<endl;

    initArr<int>(arr, n);
    cout<<reduce_cpu(arr, n)<<endl;

    initArr<int>(arr, n);
    cout<<recursiveReduce_cpu2(arr, n)<<endl;

    initArr<int>(arr, n);
    cout<<reduce_cpu2(arr, n)<<endl;

}
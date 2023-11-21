
- recursiveReduce_cpu: on cpu

- reduce_global1: 交错配对方式，后半部分累加到前半部分
  - 在全局内存上进行累加，每个thread block负责一块数据的reduce，使用线程块级同步
    ![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-17-22:10:47.png)

- reduce_global2: 相邻配对方式，每隔stride进行累加
  - 在全局内存上进行累加，奇数倍stride位置累加到偶数倍stride位置上
    ![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-17-22:21:02.png)

- reduce_global_unroll:
  - 首先每个线程块，使用交错配对方式，处理n*blockDim.x长度的数组（绿色部分）
  - 在一个线程块内部，采用交错配对方式，将blockDim.x长度的数组累加到最前面64个数中
  - 然后在最前面一个warp中，完美进行折半累加（紫色和黑色部分）
  - 注意由于每个线程处理多个数据，核函数配置与通常不同（gridSize相应缩小为2^n倍），但是初始化等函数还需要保持原有核函数配置
  ![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-21:30:42.png)


- reduce_shared: 使用共享内存，采用交错配对方式
  - 不会有bank冲突：（比如blockSize=256）
    - 当offset=256/2时：![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-22:12:05.png)
    - 当offset=128/4时：![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-22:12:26.png)
    - 当offset=16时：![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-22:13:09.png)

- reduce_shared2: 使用共享内存，采用相邻配对方式
  - 会有bank冲突：
    - 当stride=1时：![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-22:06:33.png)
    - 当stride=2时：![](https://cdn.jsdelivr.net/gh/QinganZhang/ImageHosting/img/2023-11-20-22:08:42.png)

- reduce_shared_shrink: 
  - 之前的reduce都是每折叠累加后，工作的线程数量减半，可以第一步每个线程块累加n块数据，保证所有线程至少都进行过运算
  - 首先每个线程块折叠累加n=4个数据块（同reduce_global_unroll第一步）
  - 然后reduce_shared

- reduce_shared_warp:
  - reduce_shared_shrink
  - 最后reduce_global_unroll的第三步，一个warp中进行展开

- reduce_shared_unroll: 使用共享内存的reduce_unroll

- reduce_shared_shuffle:
  - 直接将reduce_shared_unroll最后一个warp中SIMT的处理改为shuffle操作，不是好办法，因为shuffle的本意就在于减少共享内存的处理
  - 首先进行reduce_shared_shrink的第一步，每个线程块累加n=8个数据块
    - 与之前区别是，之前是累加到共享内存中，现在累加到每个线程的寄存器私有变量中
  - 然后warpReduceSum使用shuffle操作，将每个warp中每个线程的私有变量累加到第一个线程中（warpSum）
  - 将每个warp的第一个线程的warpSum，复制到同一个warp中
    - 将每个warp中第一个线程的warpSum（私有变量）拷贝到共享内存， 
    - 再从共享内存中进行转运
  - 此时该warp中每个线程中的warpSum都是原来一个warp的和，现在再次进行warpReduceSum

- 继续优化：
  - 合理设置gridSize, blockSize, n

- 测试结果：(不是很公平，因为使用的核函数配置有所不同)
  | index | method | time(s) |
  | - | - | - |
  | 1 | reduce_global1 | 0.878592 |
  | 2 | reduce_global2 | 1.94355 | 
  | 3 | reduce_global_unroll | 0.53248 | 
  | 4 | reduce_shared | 1.12333 | 
  | 5 | reduce_shared2 | 1.152 |
  | 6 | reduce_shared_shrink | 0.529408 |
  | 7 | reduce_shared_warp | 0.459776 | 
  | 8 | reduce_shared_unroll | 0.448512 |
  | 9 | reduce_shared_shuffle | **0.396288** |


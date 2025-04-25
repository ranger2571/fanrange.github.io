参考文章：[GPU shared local memory bank 冲突 - 知乎](https://zhuanlan.zhihu.com/p/668474624)

这个文章已经写的很全面了：[CUDA ---- Shared Memory - 苹果妖 - 博客园](https://www.cnblogs.com/1024incn/p/4605502.html)

## 一、基础概念

先简单复习下相关概念

**GPU调度执行流程：**

- SM调度单位为一个[warp](https://zhida.zhihu.com/search?content_id=234604547&content_type=Article&match_order=1&q=warp&zhida_source=entity)（一个warp内32个Thread）
- [shared_memory](https://zhida.zhihu.com/search?content_id=234604547&content_type=Article&match_order=1&q=shared_memory&zhida_source=entity) 可以 被一个warp中的所有（32个）线程进行访问（**提问，什么叫进行访问，访问是否有顺序，访问应该**）

**GPU中的Shared Memory：**

先看段[cuda-c-programming-guide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%3Fhighlight%3Dbank%23shared-memory-5-x)中关于shared memory及bank的介绍：

> Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle.

即：

- shared_memory 映射到大小相等的32个Bank上，Bank的数据读取带宽为32bit / cycle；

**Shared Memory到Bank的映射方式：**

- 连续映射（即原文的 **successive 32-bit words map to successive banks**），按4Byte或者8Byte 映射到同一个Bank，可通过[cudaDeviceSetSharedMemConfig](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html%23group__CUDART__DEVICE_1ga4f3f8a422968f9524012f43ba852058)配置（cudaSharedMemBankSizeFourByte 或者 cudaSharedMemBankSizeEightByte ），也可以通过[cudaDeviceGetSharedMemConfig](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html%23group__CUDART__DEVICE_1g318e21528985458de8613d87da832b42)返回当前配置

**PS：这里BankSize并不是指某个Bank的实际大小，指的是连续BankSize数据映射到同一个Bank上**

举例：对shared memory访问addr的逻辑地址，实际映射到BankIndex为：

BankIndex=（addr/BankSize）%BankNum（32 Banks）

所以，Bank中的数据是分层组织的，借用[CUDA Shared Memory](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/1024incn/p/4605502.html)这篇博客中的图来做个示意（图中BankSize = 4Byte/Bank），在这种情况下，**Bank0的实际大小是 4Byte * 层数**

关于访问shared memory中Bank的介绍，[《Using Shared Memory in CUDA C/C++》](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/using-shared-memory-cuda-cc/)里还有一段：

> To achieve high memory bandwidth for concurrent accesses, shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously. Therefore, any memory load or store of n addresses that spans b distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is b times as high as the bandwidth of a single bank.  
> --- 《Using Shared Memory in CUDA C/C++》

有了上述的背景概念后，我们可以对读写过程先算下理论时间复杂度：

**假设读写shared memory次数为** N **， 一次读写的时间复杂度为** O(1) **，那么读写**N**次所需时间复杂度为** O(N)

**假设shared_memory被分成**B**块Bank，并且可以被进行同时访问，那么理想情况下，读写**N**次所需的时间复杂度为** O(N/B) **，**

## 二、Bank Conflict

这里先贴张图，来自[cuda-c-programming-guide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23shared-memory-5-x)，下图中**左边没有Bank Conflict | 中间存在Bank Conflict，称为2-way Bank Conflict | 右边没有Bank Conflict**

这里有个问题，当不同线程读写同一个Bank中的数据时，会发生什么？

回到[《Using Shared Memory in CUDA C/C++》](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/using-shared-memory-cuda-cc/)：

> However, if multiple threads’ requested addresses map to the same memory bank, the accesses are serialized. The hardware splits a conflicting memory request into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of colliding memory requests. An exception is the case where all threads in a warp address the same shared memory address, resulting in a broadcast. Devices of compute capability 2.0 and higher have the additional ability to multicast shared memory accesses, meaning that multiple accesses to the same location by any number of threads within a warp are served simultaneously.  
> --- 《Using Shared Memory in CUDA C/C++》

上面主要有两点：
（应该描述的都是一个warp中的内容）
- 当多个线程读写同一个Bank中的数据时，会由硬件把内存读写请求，拆分成 **conflict-free requests**，进行顺序读写
- 特别地，当一个warp中的**所有线程**读写**同一个地址**时，会触发**broadcast**机制，此时不会退化成顺序读写

注：上面提到触发broadcast机制的条件是**all** threads acess **same address**，但在翻阅cuda-c-programming-guide以及最新版本的[NVProfGuide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)时，发现只要是**多个**thread **读写**就会触发broadcast（不需要All）

另外关于读写同一地址时的行为，在[NVProfGuide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)里，给出了更明确的流程：

> When multiple threads make the same read access, one thread receives the data and then broadcasts it to the other threads. When multiple threads write to the same location, only one thread succeeds in the write; which thread that succeeds is undefined.

即，

- 多个线程读同一个数据时，仅有一个线程读，然后broadcast到其他线程
- 多个线程写同一个数据时，仅会有一个线程写成功（不过这里没有提及是否会将写操作执行多次（即a. 多个线程写入，最后一个线程随机写完; or b. 随机挑选一个线程执行写入），具体流程存疑）（问问兴哥）


如[cuda-c-programming-guide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23shared-memory-5-x) 中给了BroadCast示意图：**左边模拟随机访问 | 中间Thread 3,4,6,7,9访问Bank5中同一个地址 | 右边多个Thread访问 Bank12, Bank20 触发广播机制**

依据Bank Conflict 的定义以及BroadCast的触发条件 来看，该图中的左/中/右三种访问形式，**均没有“Bank Conflict”情况**

所以，这里用一句话解释什么是Bank Conflict：

**在访问shared memory时，因多个线程读写同一个Bank中的不同数据地址时，导致shared memory 并发读写 退化 成顺序读写的现象叫做Bank Conflict；**

特别地，**当同一个Bank的内存访问请求数为** M **时，叫做M-way Bank Conflict；**

回到开始读写N次的理论时间复杂度 O(N/B) ， **我们可以看到，当存在M-way Bank Conflict时，时间复杂度变成** O(M∗N/B) **（退化了M倍）**；

## 三、如何发现存在Bank Conflict？

关于检测 Bank Conflict ， 目前[NVProf工具](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)已经可以检测出某段Kernel函数存在Bank Conflict）

> Updates in 2023.2  
> ...  
> Added support for rules to highlight individual source lines. Lines with global/local memory access with high excessive sector counts and shared accesses with many **bank conflicts** are automatically detected and highlighted.  
> ...

另关于如何避免Bank Conflict的解法（如在[CUDA Best Practices里提到的增加Padding等](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html%3Fhighlight%3Dbank%2520conflict%23shared-memory-and-memory-banks)）

## 参考资料

感兴趣的读者，可以参考下其他人对bank conflict的定义

- stackoverflow：什么是bank conflict？：[https://stackoverflow.com/quest](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming)


# 示例
![[Pasted image 20250322164715.png]]是[【CUDA调优指南】Shared Memory与Bank Conflicts_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AicNemE9m/?spm_id_from=333.1387.upload.video_card.click&vd_source=e7488b73976f312b16ed6db284f782cd)视频中的例子，但是显然作者没有搞明白，或者没有说明白这个bank conflict是什么

似乎他被自己的图搞糊涂了，我也被搞糊涂了，可以问问兴哥了

gmem是$16*16=256$个数

我的理解是一个warp是32个线程，（先忽略————如果一个线程要只处理一个数，那么要有8个warp）

bank conflict 只存在于一个warp 内

对gmem的转置，应该是先整个warp，32个线程同时读入，所以会读入gmem的前两行。

写的顺序是，如果按行写入，那么前两行的数据，应该分别写入32个不同的bank。按行写入的时候，满足32个bank后（每个bank，存入一个float32的数）就换行，从头开始）。

但是这里是按列写入，所以32个bank中，只有4个bank会被访问，但是这4个bank会被访问8次。

也就是图片中的8way conflict

那为什么总的bank conflict是$56=7*8$呢？

[一文读懂cuda stream与cuda event - 知乎](https://zhuanlan.zhihu.com/p/699754357)

cuda编程里最重要的特性就是异步：CPU提交任务到GPU中异步执行。为了控制异步之间的并发顺序，cuda引入了stream和event的概念。本文尝试分析理解stream和event的含义，充分理解并使用stream、event，是正确、高效利用GPU的必要条件。


## 阶段性总结

至此，我们了解了cuda编程中的stream的概念。一个stream就是一个kernel execution队列，我们可以用event来记录队列状态、等待队列执行完毕、查询任务完成的时间。

## 关于default stream 与 legacy stream

stream的概念非常清晰，使用起来也非常简单易懂。是的，如果世界上每个使用cuda的人，在调用每个cuda函数的时候，都写明自己要用到哪个stream，那么世界将变得非常美好。作为一位现代cuda使用者，我们应该尽可能做到：

- kernel launch的时候设计好stream参数，这样可以更好地利用GPU的硬件资源，做到多个kernel并发执行
- 尽可能使用[带stream参数版本的显存相关函数](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)，例如`cudaMallocAsync`

可惜，现实并不是这样。即使我们自己做到了这一点，也难免会遇到其它人的代码不符合这些要求：

- 很多人调用cuda kernel的时候，简单粗暴地直接`kernel<<<gridDim, blockDim>>>()`，不指定stream参数
- 有一些cuda函数，本身也没有stream参数（例如`cudaMalloc`）

此时，它们的stream参数，就是default stream。而这个default stream，是可以在编译期间控制的：

- 用`nvcc`编译时加上`--default-stream per-thread`，那么每个CPU thread会有一个自己的default stream
- 默认选项为`--default-stream legacy`，整个CPU进程共享同一个特殊的stream，称为null stream。

这个null stream特别烦人，所有使用`cudaStreamCreate`构造的stream，都和这个null stream有隐含的依赖关系：

- null stream的任务完成之前，所有stream都得等着
- null stream加入任务之前，也得等所有的stream里面的任务都完成

为了解决这个问题，最简单的做法就是，创建stream的时候，尽可能用non-blocking stream，这样的stream与null stream之间就没有依赖关系了。
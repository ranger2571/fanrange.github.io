




## **开源了什么以及为什么开源**

自 Mooncake 架构公开以来受到了非常广泛的关注。在和大家讨论的过程中我们也逐渐明确虽然上层的推理框架和模型各有不同，但在 KVCache 的存储、管理和传输一层面其实可以作为一个独立的公共组件抽象出来被大家所共享。

甚至在大模型时代下这一类面向高速内存语义设计的存储有**可能可以成为继块存储、文件存储、对象存储以来一类新的通用标准存储系统的门类**。因此很多 Infra 厂商也非常希望能够将这一块的最佳实践尽可能地总结共享出来，为后续进一步的发展打一个基础。

不过由于原本的 Mooncake 是一个非常定制化且耦合的方案，一方面不太方便完整的开源，另一方面即便完整的开源其它单位其实也没法用。所以我们本次我们采用的是一种逐步逐层重新造一遍轮子，然后一边替换现有线上实现，一边分批开源的模式。

最后形成的架构如上图所示，其中传输引擎 Transfer Engine 部分现在已经在 GitHub 开源。除了 Transfer Engine 主体外，我们还提供了一个基于 Transfer Engine 实现的 P2P Store 库，支持在集群中的节点之间共享临时对象（例如 checkpoint 文件），同时避免了单台机器上的带宽瓶颈。此外，我们还修改了vLLM 并集成 Transfer Engine，通过高效利用 [RDMA](https://zhida.zhihu.com/search?content_id=250915184&content_type=Article&match_order=1&q=RDMA&zhida_source=entity) 设备，使得多机多卡 prefill-decode disaggregation 更加高效。

未来，我们计划发布一个基于 Transfer Engine 实现的池化的 KVCache 系统，以实现更灵活的P/D解耦，敬请期待！


## **本次开源的 Transfer Engine 提供了什么？**

如 [Mooncake (1)](https://zhuanlan.zhihu.com/p/705754254) 讲的那样，Mooncake 是非常典型的分离式架构，不同节点上的 Distributed KVCache Pool 需要通过高速网络进行连接，这样才能在 P/D 分离的场景下实现 TTFT 的降低同时避免 GPU 空转导致的浪费。

为了实现这一高速传输的目标，RDMA 是目前最主流的版本答案，但是直接使用 RDMA Verbs 写程序相当繁琐且有不少暗坑容易踩进去，而 nccl 在动态的环境中又不够灵活高效。

**因此， Mooncake 设计了专门的性能传输引擎 Transfer Engine，用于整个系统数据平面的传输。

在设计上 Transfer Engine 支持通过 TCP、RDMA、基于 NVIDIA [GPUDirect](https://zhida.zhihu.com/search?content_id=250915184&content_type=Article&match_order=1&q=GPUDirect&zhida_source=entity) 的 RDMA 以及 NVMe over Fabric（NVMe-of）协议进行快速、可靠和灵活的数据传输。主要特点如下：

• 支持多种介质：在大模型推理期间，KVCache 数据可能位于本地 DRAM、本地 GPU VRAM、远程 DRAM、远程 GPU VRAM，甚至还可以是远程 NVMe 设备上的某个文件（作为二级存储）。对于不同介质，Transfer Engine 内部必然采用不同的传输方式（这里还包括链路选择等），从而取得最好的性能。比如，如果一次传输是本地 DRAM 到本地 DRAM，显然 memcpy() 才是最高效的方式；大块数据的跨设备传输会尽量让多张网卡同时工作，提高聚合带宽；根据介质的硬件拓扑信息选择恰当的传输链路，减小 NUMA 对传输带宽的影响等。

• 提供统一抽象：Transfer Engine 提供 Segment 和 BatchTransfer 两个核心抽象对用户屏蔽了传输的相关细节（以 RDMA 为例，用户不再需要关心 QP 元数据交换等乱七八糟的细节了）。Segment 代表一段可被远程读写的连续地址空间（可以是 DRAM 或 VRAM 提供的非持久化存储，也可以是 NVMeof 提供的持久化存储）。BatchTransfer 封装了操作请求，负责将一个 Segment 中非连续的一组数据空间的数据和另外一组 Segment 的对应空间进行数据同步，支持 Read/Write 两种方向。可以参考 transfer_engine_bench.cpp 实现一个自己的应用。

• 性能表现佳：与分布式 PyTorch 使用的gloo和TCP相比，Transfer Engine 具有最低的I/O延迟。在 4×200 Gbps 和 8×400 Gbps RoCE 网络中，Transfer Engine 处理相当于 LLaMA3-70B 模型 128k tokens 生成的KVCache 大小的 40GB 数据时，能够提供高达 87 GB/s 和 190 GB/s 的带宽，相比 TCP 协议快约 2.4 倍和 4.6 倍。
最经典的点，就是Paged attention

存储优化是大模型推理中非常重要的一个环节。我在 [聊聊大模型推理服务中的优化问题](https://zhuanlan.zhihu.com/p/677650022) 中对一部分论文进行了解读，主要包括 Efficient Memory Management for Large Language Model Serving with PagedAttention（也就是vllm）等方法，最近正好和几篇热点论文再拓展阅读一下，例如月之暗面 KIMI chat 的 Mooncake 。

存储管理，包括计算芯片缓存、显存、内存甚至磁盘，都可以算在内，因为不管是对离线的参数、缓存，还是对在线的数据搬运、数据复用，都属于存储管理。

这篇笔记会更加侧重于如何优化当前大模型自回归特点下的 KVCache 在内存和显存中的摆放位置、管理、检索、传输、预测等等，目标则是为了从整体上提高服务的服务质量，减小延时，提高吞吐。

### 存储管理概述：

我们先总结一下当前大模型推理下内存管理中主要面对的新问题，

首先，Weights 和 KVCache 的存储占据了内存使用的主导地位，尤其是当上下文的场景时，由于生成长度是不确定的，可长可短，也不能提前无脑分配内存空间来存储这些缓存，所以存储空间如何分配是极具有挑战性的。

最 Naive 方法可以根据每个请求的最大长度提前分配存储空间，这种方法显然会造成存储资源的大量浪费。稍微高级一点的比如 [s3](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2306.06000)、FasterTransformer，会预测一个长度，可以减少浪费，但是显然没法应付更加复杂的情况。

所以就有了，前面我们分析过的 PagedAttention (vLLM) ，就借鉴了操作系统中的分页（Page）方式进行存储，这样就可以不连续的方式将生成的 KVCache 动态地映射到预先分配的物理块上，这样可以减少大量存储碎片，并且实测有效。

LightLLM 在 vLLM 的基础上，使用了更细粒度（以 Token 为单位）的 KVCache 存储，减少了不规则边界的浪费。（**现在vllm v1是这样的吗？**）这里面又会带来新的问题，比如内存访问不连续的问题，就又出现了 FlashInfer 这样的工作，为 KVCache 编排了多样化的数据布局。

还有类似 arxiv2405 的 vAttention 这种虚拟内存方法来优化这个问题。

最近还有不少序列并行方法会把 KVCache 分布化，再叠加在较大批次的推理任务调度场景，也就是大 batch 下，内存管理会更加复杂。可以看到，当前这里的研究问题非常具有挑战性。

而今天要聚焦的就是通过分布式计算的思路解决存储问题的一大类方法，也就是 Disaggregated inference，直译的话就是分散推理，或者叫分离式推理。

### Disaggregated inference：

几篇题目中都有共同的 Disaggregated inference，直译的话就是分散推理，或者叫分离式推理。[DistServe](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2401.09670) 和 [TetriInfer](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2401.11181) （两篇的 arxiv 时间都是 2401，论文中 DistServe 引用了 TetriInfer，我们再后边会简单过一下 TetriInfer 的大体思路）都提到了 disaggregating prefill and decoding 的概念，简单说就是将 prefill 和 decode 计算分配给不同的 GPU，这和更早的 Splitwise （arxiv 2311）是一样的解决思路。

同时，这几天铺天盖地的 kimi chat 的 mooncake 也是这样的分散推理方法，且是以 KVCache 为中心的分离式推理架构。这些架构的动机是 Prefill 和 Decode 阶段表现出的特点不同，一个容易组大 batch 是计算密集的，一个是自回归一个字一个字蹦的是访存密集性。

Splitwise 论文出发点，把不同时期的 GPU 放在不同的分区去用，老显卡和新显卡的计算访存比不同，可以安排不同的计算任务上去，详见下面的七个 insight。然后传递中间的 embedding，其实**需要传递的中间 embedding ，就是 prefill 生成的所有 KVCache** 。从设计理念上说，将 Splitwise 看成是一个**流水线并行**，就是传递中间结果，但是如果是把 KVCache 当做内存管理，那就是**统一的分布式系统视角**，decode 机器不是去 prefill 机器取数据，prefill 机器也不是给 decode 送数据，不是个单纯的流水线的逻辑了，而是**两者都是往 pool 里面写数据，decode 用的时候也是去 pool 里取数据**（prefill 也会去取数据）。

按 [佳瑞解读](https://zhuanlan.zhihu.com/p/706097807) 中提到的融合派和分离派的分歧方法来说，我可以管它叫做**流水视角**和**统一视图**的区别，这两者当然都是分离派，但视角不同时，我们看待问题的思路会不同，统一视图下，整个问题更像传统分布式系统了，流水视角下，总归欠点意思不够全面。

所以我们发现我们已经把分离式系统看做是一个传统的分布式系统了，分布式系统中面临的调度和优化方法都可以往里招呼了。当然，大家最担心的问题，也就是两个部分之间通信的问题，如果设备间通信带宽较小延迟过大。当前的实验也都是需要在 NVLink 的加持下才能完成如此大规模的同步和异步数据传输。

### Splitwise 中的 insight：

或者也可以叫 decode 七宗罪：

- 不同的推理服务可能具有截然不同的提示（prompt）和解码（decode）分布。BurstGPT 中也有更为直接的数据统计。
- 在混合连续批处理（mixed continuous batching）的大部分时间内，都花费在 decode 阶段，**因为这个阶段的 batch 太小了。**(这个部分的内容怎么理解？？？)
- 对于大多数请求来说，端到端（E2E，用户请求总时间）的大部分时间都花在 decode 阶段，这个同洞察 II。
- prefill 阶段的批量大小如果想要确保良好的性能，需要限制批处理大小（因为是计算密集型，容易达到roofline吗？）。相比之下， decode 阶段需要更大的 batch，可以在没有任何缺点的情况下获得高吞吐量。（访存密集型就需要大量的增加patch大小）
- prefill 阶段的批量处理是 compute-bound，decode 阶段则是 memory-bound。
- 从功率角度说，prefill 阶段可以有效利用 GPU 的功率，但 decode 阶段并没有。
- decode 阶段完全可以在更便宜、最大性能较低的硬件上运行，以实现更好的单位功率性能 perf per walt（Perf/W）和单位花销性能 perf per dollar (Perf/$)。

最核心的其实就是，prefill是计算密集型，decode是访存密集型，所以才有PD分离，才有增加decode的batch大小

### MemServe ：（没细看）

MemServe 看来是 memory server 的意思，实际上还是个简称 Memory-Enhanced Model Serving，主要是在分散推理的 setting 下引入了 MemPool，也就是一个弹性内存池，用于管理**分布式内存**和跨服务实例的 KVCache。

当 KVCache 越来越大的时候，内存池中的布局就是一个非常大的问题，所以这里引入一个索引的机制，有了索引内存的检索就靠索引了，其实也就是给 pagedattention 中的 page 加一个 id，可以通过其他的检索机制来优化这个检索过程。本文中给出的就是 global prompt trees 的结构机制来进行优化。

因为是统一架构，所以这里涉及了几种 instances 类型，P-only、D-only、PD-colocated 机器类型（这里的 PD 就是 prefill-decode 的意思），同时在运算时就分为 (1) PD-colocated（只用第三种 instances 的）, (2) PD-colocated with caching, (3) PD-disaggregated （用前两种 instances 的、或者混着一起用）, (4) PD-disaggregated with caching 。

先看结果，这里有几个评价指标：time-to-first-token (TTFT), job completion time (JCT), and time-per-output-token (TPOT)。

从 ShareGPT workload 的结果 JCT 来说，基于 4）PD-D + Mem Pool 比 1）PD-C 提高了 42%，4）PD-D + Mem Pool + Context Cacheing 可以再提升 29%。从 LooGLE dataset 的结果 JCT 来说，前面两个对应的数字是 10.8% 和 26.9%。

我们再来看一下这里 MemPool 具体是个什么机制，首先作者提出一个 PD-Basic 也就是 naive/vanilla 的机制，其实就是 DistServe 和 Splitwise ，作为优化的 baseline 了。然后我们看一下 PD-Caching-1/2/3 ，是三个不同 level 的缓存设计机制，PD-Caching-1 只在 P-only instances 启用缓存，主要缺点是在多轮长上下文聊天场景中，P-only 节点需要重复转发 KVCache 到 D-only 节点，严重浪费带宽；PD-Caching-2 在 D-only 也就是 decoder 节点进行缓存，同时，再结合 PD-Caching-3 允许，P-only 和 D-only 的机器可以互相传输病维护两边的索引。这里还有一个优化方法即邻近原则调度进行一定的优化，我们后边再介绍。我们其实可以把这里的 Caching Pool 理解为一个不完全的同步所有数据的异步共享机制，各种类型的节点间即要维护自己的缓存也要按需推送和拉取部分远程数据。

MemServe 的全局调度器一方面负责整个框架和调度，另一方面还维护了 global prompt trees 全局提示树 和 locality-aware scheduling policy 本地感知调度等策略来优化内存管理，优化 KVCache 管理。每个节点上会有对全局树型缓存进行分布式维护。调度的时候，请求的提示词会通过对所有类型的树查询完成对全局的查询，就可以通过策略模块，针对分布式负载情况选择具有最长公共前缀（即最大保留历史KVCache）的实例，以达到最优的检索和访问效率了。

### MoonCake：

来自月之暗面 KIMI chat，和清华。

我们再来看一下 Mooncake，这里几篇博客写的都挺好（[方佳瑞：Mooncake阅读笔记](https://zhuanlan.zhihu.com/p/706097807)，[许欣然：关于 Mooncake 的碎碎念](https://zhuanlan.zhihu.com/p/705910725)，[章明星：在月之暗面做月饼](https://zhuanlan.zhihu.com/p/705754254)），包括两篇原作者的，这里我就不展开 Mooncake 的细节了，着重比一比和 MemServe 的区别吧。

月饼的本质也是以 KVCache 为中心的调度器，或者叫解耦架构，架构中也有一个全局调度器。目标是达到延时最优，同时平衡最大化整体有效吞吐量，两者是一样的。

不过 KIMI 现在的访问量很大，是有高访问量的工业实践项目，所以它是面临高度过载场景的挑战的，并且 MemServe 也是没有太多涉及多请求服务的内容。MemServe 中对异构硬件的讨论也是有的，但是针对性设计不多，而 Mooncake 利用了 GPU 集群中未充分利用的 CPU、DRAM 和 SSD 资源希望可以得到充分利用。同时，还有个小设计，叫做基于预测的早期拒绝策略，其实说不好听的就是这一单这次不给你服务了呵呵。

在某些模拟场景中，Mooncake 在遵守 SLOs 的同时吞吐量可以增加高达525%，在真实工作负载下，Mooncake的创新架构使 Kimi 能够处理 75% 以上的请求，争取少拒绝服务。

最近还有一个概念，就是**模型即服务（MaaS）**提供商，其实我个人更像把这个服务叫做 **LLM/VLM/MLM Infer as a Service** 哈哈，其实还是**模型推理即服务**。

#### Mooncake 有几个大的特点：
1）在分块/层完成 prefill 阶段，也就是 block/layer-wise 的设计，或者可以理解为对 pagedattention 中的 page 的一种细化，它也拥有一个统一视角的 pool ，来统一维护 KVCache 的存储。2）传输方式采用流式，3）和 continuous batch 进行结合，就是可以处理多请求服务系统。

说起来简单，做起来难，
所以作者就提出了核心目标和主要洞见：
0）**KVCache 的容量会长期保持高位**。这是挺有洞见的结论，**长上下文技术需求不变，这个结论就不会变**。
1）尽可能多地重用 KVCache 以减少所需的计算资源，以及在分布式环节中转移到目标节点的耗时，重用缓存过程中还使用了 Prefix Cache，这和 MemServe 中的 prompt tree 比较相似；
2）最大化每个批次中的令牌数量以提高模型 FLOPs 利用率。
3）从远程位置调用 KVCache 会延长 TTFT（time-to-first-token），而大批量大小会导致更大的 TBT （time-between-tokens），（这里采用的 TBT 而没有采用 MemServe 中的 Time Per Output Token (TPOT)，这两者的区别在 [作者解读](https://zhuanlan.zhihu.com/p/705754254) 中详细解答了），在 Mooncake 中则采用了平衡的策略。
4）存储在较低层级存储（比如相比显存，内存就低，相比内存，SSD 就更低）上的KVCache 会带来更多问题。
5）加上了预测功能，使用了冷热温度来衡量块的常用程度，最热的块就复制到多个节点而最冷的块应该被交换出去。（类似于LRU）

这几个点在 MemServe 中还是相对提到较少，但是以上这些问题都是分布式系统优化中非常常见的问题，在 KIMI 的实践中也验证了分布式计算系统的方法论，随着人工智能大模型等新技术的高发展和推动下，会保持常青常绿，还在不断进化以满足新的需求和挑战。
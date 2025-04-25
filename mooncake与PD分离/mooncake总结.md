
## Introduction

1. Mooncake 相比于开源框架而言，能利用的资源更多，但是需要考虑的限制也更复杂。就我所知，当前的 open source serving engine 很难考虑到如何 fully utilize [DRAM SSD](https://zhida.zhihu.com/search?content_id=249364565&content_type=Article&match_order=1&q=DRAM+SSD&zhida_source=entity) RDMA 这样的硬件条件。然而，open source engine 可能也不太用考虑 SLO（service level objective 也即用户优先等级）的问题？
2. Mooncake 将计算操作和硬件资源都进行了彻底的解构。对于后者而言，所有的硬件都被解耦而后重新建立了对应的资源池。每种资源池（比如 CPU 池、DRAM 池、SSD 池）各自有独立的优化目标，以期望整体达到最大优化。
3. P D 彻底分节点进行导致设计者必须考虑 [KV cache](https://zhida.zhihu.com/search?content_id=249364565&content_type=Article&match_order=1&q=KV+cache&zhida_source=entity) 在 P D 节点之间传输的问题。如下图所示，KV Cache 需要考虑 P D 之间所有的传输需求。

4. KV Cache 传输的调度几乎是 Mooncake 的核心。直观来说，有两个优化方案：首先是尽可能复用 KV cache；第二是尽可能加大单个 batch 内的 token 数目，用以最大化 MFU（Model FLOPs Utilization）。然而，复用 KV cache 涉及到了 KV cache 远距离传输的问题，也可能这些 KV cache 需要从更底层的存储设备读取上来，这会加大 TTFT（time to first token，产生第一个 token 的时间）；高强度传输 KV cache 也可能导致网络卡顿；而更大的 batch size 往往会导致 TPOT （time per output token，解码每个 token 的时间，这在原文中称为 time between tokens）增大。因此，这些直观的【通量最大】优化可能降低延迟，违背了 SLO。

5. 基于这些复杂的 trade off，参考上图，Mooncake 设计了 global scheduler（conductor）。对于每个 request，Conductor 需要为其选择一组用于 prefiil 和 decode 的设备，然后进行调度。首先迁移尽可能多的 KV cache 到 prefill 设备上；接着在 prefill 设备上通过 chunked and layer-wise prefill 的方法连续地 stream prefill 所得的 KV cache 到 decode 设备上；最后在 decode 设备上加载 KV cache，将此 request 加入到 continuous batching 中，完成 decode。

6. 这样的调度听上去非常自然，但是策略的限制非常复杂。如前所述，在 prefill 阶段，四处传输 KV cache 可能会增大 TTFT。为此，conductor 也需要预测某块 KV cache 可以复用的可能，对高强度使用的 KV cache block 进行交换以及复制。最火热的 block，比如 system prompt 的 block 几乎每个设备都该本地缓存一份；而冷门的 block，比如某个用户上传的邪门文档，可能就该被早日擦除。基于这些考虑，大部分的 DRAM 都会被用于维持 global KV cache pool，这样又给 scheduler 能利用的 DRAM 上了压力。

7. 与此相对，decode 阶段的优化目标显著不同。但是，如前文所述，尽可能增大 batch 内的 tokens 数目会增大 TPOT。

8. 更加糟糕的是，工业级部署的 serving engine 时常要面对极其不均匀的服务压力，在 overload 和 far from enough 之间反复横跳。这里 Mooncake 实现了一个【拒绝策略】（老实说我觉得这是现实的需求，但是我的服务要是经常被某个产品拒绝，我可能就不用这个产品了）

9. 为了实现这样的拒绝策略，设计者希望能够预测下一个阶段的服务器负载程度并且拒绝【某些】可能服务不过来了的请求。这一策略接下来讨论。

10. 采用 [chunked pipeline parallelism](https://zhida.zhihu.com/search?content_id=249364565&content_type=Article&match_order=1&q=chunked+pipeline+parallelism&zhida_source=entity)（CPP）的方式将单个 request 拆分为多个 chunk，这些 chunks 可以按照顺序在不同的 prefill nodes 上完成计算。这有助于减小长请求的 TTFT（【老实说】我不太理解如何是减小，在我看来这会让序列的 TTFT 更加稳定可控）。相比于传统的 sequence parallelism（SP）而言，CPP 减轻了网络压力且 simplifies the reliance on frequent elastic scaling.【这句我没读懂 】
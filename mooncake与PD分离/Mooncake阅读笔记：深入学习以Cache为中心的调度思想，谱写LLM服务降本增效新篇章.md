[Mooncake阅读笔记：深入学习以Cache为中心的调度思想，谱写LLM服务降本增效新篇章 - 知乎](https://zhuanlan.zhihu.com/p/706097807)

Mooncake，采用分离式设计，将[Prefill](https://zhida.zhihu.com/search?content_id=245041616&content_type=Article&match_order=1&q=Prefill&zhida_source=entity)和[Decode](https://zhida.zhihu.com/search?content_id=245041616&content_type=Article&match_order=1&q=Decode&zhida_source=entity)两阶段解耦，构建了一个全局[KVCache Pool](https://zhida.zhihu.com/search?content_id=245041616&content_type=Article&match_order=1&q=KVCache+Pool&zhida_source=entity)，实现以Cache为中心的调度。

（将单个同构 GPU 集群的资源打散并重新组织成三个可以独立弹性伸缩的资源池。其中 [Prefill Pool](https://zhida.zhihu.com/search?content_id=244965348&content_type=Article&match_order=1&q=Prefill+Pool&zhida_source=entity) 处理用户输入，主要对 Time To First Token (TTFT) 负责。同时因为 Prefill 相对计算密集，这一部分也承担着抬高整体资源利用率的任务。Prefill 处理完之后对应的 KVCache 会被送到 [Decode Pool](https://zhida.zhihu.com/search?content_id=244965348&content_type=Article&match_order=1&q=Decode+Pool&zhida_source=entity) 进行 autoregression 式的流式输出。虽然我们希望尽可能攒大的 batch 以提升 MFU，但这一部分主要需要对 Time Between Tokens (TBT) 负责。）

Mooncake分离式架构动机是Prefill和Decode阶段性质不同，Prefill是计算密集，受限算力带宽用不满，Decode是访存密集性，受限带宽算力用不满，所以用同一种硬件部署两阶段往往顾此失彼，不是最有性价比。

因此，最近很多工作对二者进行拆分，和Mooncake最相似的是今年5月份发布的微软和华盛顿大学的工作[Splitwise](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2311.18677)，它里面列出了Prefill和Decode不同的七个Insights值得大家仔细读一下。因为Mooncake开发也需要一段时间，它和Splitwise应该是不谋而合的同期工作。

拆分Prefill/Decode之后，LLM推理系统就更像一个分布式内存系统+流处理系统，这就是传统计算机系统研究者最擅长的领域。某大佬和我讲的sys三板斧，batch， cache，调度都可以招呼上。比如，Decode可以进一步拆成Attention和非Attention算子分离调度，也是章明星团队近期的一个工作叫[Attention Offloading](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2405.01814)。

（**和Ktransformer 项目的关系？？？）


## 3 Overview of Mooncake’s Disaggregated Archtecture

如下面Figure 1所示，Mooncake采用了**分离式**架构。这里分离有两层含义：

一是，将Prefill与Decode计算资源分开，这与前人工作无异，如splitwise和distserve等；Prefill阶段优化目标是利用request间存在共同前缀的机会，尽可能复用KVCache，同时满足TTFT（Time To First Token） SLO，最大化MFU（论文中似乎笔误成minimum）和KVCache小于CPU内存限制。Decode优化目标为最大化吞吐，满足TBT（Time between tokens ，Decode阶段两个Token之间的时间）SLO和KVCache小于GPU显存限制。

二是，将KVCache和计算分离开，它将GPU集群的CPU、DRAM、SSD和RDMA资源分组组成Distributed KVCache Pool，KVCache也是分块以Paged方式管理，KVCache Blocks如何在Pool中调度请求和复用KVCache乃本文精髓。

#### Mooncake处理一个request的流程
我们跟随论文的安排，跟Figure 4先走马观花过一遍Mooncake处理一个request的流程。一个request到达（tokenized之后的），调度程序会选择一对Prefill Instance和Decode Instance，模型参数要在两个Instance都有副本，并启动包含四个步骤的工作流程：

s1）**KVCache Reuse**：Prefill Instance将request分成token blocks，考虑request之间存在共同前缀，**需要尽可能将token block调度到Prefix KVCache最长的节点处理（这一句就没懂什么意思，什么叫调度到最长节点）**，来resuse KVCache。为此作者提出一种以KVCache为中心的调度，将在章节5 KVCache-centric Scheduling中详细介绍。

s2）**Incremental Prefill**：使用Prefix Cache，Prefill Instance只需要计算Prefix Cache没覆盖的部分。长序列计算需要多卡并行，用[TeraPipe](https://link.zhihu.com/?target=https%3A//proceedings.mlr.press/v139/li21y.html)方式做流水并行。将在章节5 Implementation of the Prefill Pool中详细介绍。

s3）**KVCache Transfer**：和Splitwise一样，分离是设计需要将KVCache从Prefill Instance搬运到Decode Instance。Mooncake通过异步传输，与上述Incremental Prefill步骤重叠，将每个模型层生成的KVCache流式传输到目标Decode Instance的CPU内存，以减少等待时间。

s4）**Decoding**：在Decoding Instance的CPU DRAM中接收到所有KVCache后，请求Continous Batching处理。

工作流中，s3, s4平平无奇，前人设计如Splitwise、DistServe和vLLM等已经覆盖。s1和s2比较精彩，本文接下来的章节4和章节5来详细介绍。

## 4 Implementation of the Prefill Pool

章明星说原本章叫“Prefill: To Seperate or Not. Is it a Question?” 。这说明分离式设计并不是共识。

Prefill和Decode的计算性质不同，前者吃计算，后者吃带宽。这会带来很多次生影响，比如Batching方式就不同，Prefill不需要batching，Decode需要大Batching。处理Prefill和Decode有**融合派**和**分离派**两大流派。

**融合派：** 将Prefill放到Decode step间隙或者和某个Decode step一起推理。

2022年OSDI的LLM Continous Batching 开山之作Orca将Prefill和Decode Step融合在一个Batching Step做forward，Orca时代还没有Paged Attention，还需要Selective Batching来将Attention解Batching。

2023年vLLM做Batching时，prefill和decoding则是独立forward的，一个Batching step要么处理decoding要么处理prefill。prefill直接调用xformers处理计算密集的prefill attn计算；decoding手写CUDA PA处理访存密集的attn计算。

后来，以[Sarathi](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2403.02310)和[FastGen](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2401.08671)为代表，将prefill序列拆成chunk，chunk prefilling可以插入到decode phase中，甚至和decode step融合。

比如，flash attention的_flash_attn_with_kvcache_函数支持q 部分有kvcache（decode）部分没有（prefill）。章明星也提到，对于*短*的prefill chunk和decode step融合和纯做decode step延迟可能相同_，这相当于prefill白嫖decode没用满的算力。对这段Continous Batching发展历史感兴趣可以读：[方佳瑞：大模型推理核心技术之Continuous Batching和我的WXG往事](https://zhuanlan.zhihu.com/p/676109470)。

融合派的缺点是，Prefill和Decode在相同设备上，二者并行度被绑定。如果prompt长度有限，prefill阶段占比很小基本不到10%，所以忍一忍也无所谓。不过，对于长序列当Prefill比例升高，其Prefill并行度不够灵活的缺陷就暴露出来。

**分离派：**

考虑Prefill/Decode性质差异，人们开始尝试把Prefill和Decode放到不同的设备来分别处理，比如，[Splitwise](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2311.18677)和[DistServe](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2401.09670)，Mooncake也是对他们的继承和发展。

分离派可以给Prefill和Decode设置不同的并行度，二者有更大的灵活性。

许欣然提到分离派*允许整个系统往 “算力/\$” 和 “带宽/\$” 的两个方向独立发展，对硬件优化也是更友好的。

在机房里，分离派可以用不同GPU混部来降本，比如H800做Prefill，H20做Decode，二者用RDMA互联。

分离派遇到的**最大挑战是如何在不同设备之间传输KVCache，集群需要高速网络来互联，而网络成本不容小觑**。所以，分离派的硬件往往需求很高端，目前得是是A/H卡，硬件成本高且无法Scale也为人诟病。

分离派一个优势就是Prefill并行度灵活，为了降低长序列Prefill的TTFT，可以分配多卡甚至多机GPU并行处理Prefill，比如长序列我们就多分配一些GPU，短序列少点。

**如何做Prefill并行？** 长序列Prefill的batch size是1，没法用数据并行。张量并行性能不满足，它通信量很大无法扩展出节点，而且GQA的head number也限制了它并行度。那是否可以用序列并行（SP）？Ulysses通信量远低于TP，Ring可以和计算重叠，这里也感恩Mooncake引用了我们的最近的工作[USP](https://zhuanlan.zhihu.com/p/698031151)。但是SP推理需要在每个卡上replicate模型参数的，对大模型不利，如果用ZeRO方式shard参数，通信量都增加了很多；而且，SP每层都要通信，占用宝贵的网络带宽，网络带宽还得留着给KVCache送Decode Instance用。

Mooncake为Prefill阶段设计了Chunked Pipeline Parallelism (CPP) ，其实就是[TeraPipe](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2102.07988)。TeraPipe正是vLLM核心作者Zhuohan Li的2021年的工作，当时是用在流水线训练里，这里只用它的forward过程即可。TeraPipe将模型沿着Layer切分成N个stage，放在不同设备上，将输入沿着序列维度切分成若干chunk。这样就可以在一个推理任务里，不同chunk流水在不同stage上的计算就可以流水起来起来。如下图所示，切分方式看左图，流水方式看右图一个Forward过程即可。

TeraPipe用于训练时不能均分序列，一个token要和之前所有token做计算（因为Causal Attention引起的），位置越靠后计算量越大，均分序列的话Attention计算会负载不均衡，所以越靠后sequence chunk越小才好，所以TeraPipe论文用动态规划来找最佳划分策略。在Mooncake中，章明星说TeraPipe*只有 forward 没有 backword 的话不需要啥动态规划来平衡每个 Chunk 的计算量，只要给每个 Chunk 设置一个最小的值就行*。

这点我是有些疑惑的，我觉得forward也需要和训练类似的负载均衡策略，如下图上半部分。对这个问题，我新开了一个文章讨论：


#### TeraPipe做推理好处 
1）它仅在每个stage的边界处需要跨节点通信，这可以很容易地与计算重叠。这导致更好的MFU并且KVCache传输的网络资源争用更少。

2）短和长上下文均适用。原文给了解释原因是bringing no significant overhead for short context prefill and avoiding frequent dynamic adjustment of node partitioning。我理解是：layer分布在不同设备不需要改变，长短上下文都TeraPipe，长文相当于用上了多卡资源，尽管有些气泡，相当于并行加速了，可以满足TTFT。短文本来也不用并行TTFT单GPU也可以满足，因为TeraPipe通信少，所以和一个设备做Prefill时间一样，因此和_单个设备做Prefill比_没有明显开销。

我觉得这里还有一些问题可以深入讨论一下，第一，是流水并行的气泡问题，可以放一些Prefill阶段不同GPU的扩展性。第二，TeraPipe可以和SP组成混合并行，更容易去扩展到多机。第三，TeraPipe方式切分参数会导致Prefill并行度没法变化，切分成8个stage就必须一直做PP=8的并行了，因此，不能弹性改变Prefill的计算资源。当然，Mooncake可能在集群里放置很多Prefill Instance，每个Instance的并行度不同，然后在Instance之间做request-level的load balance。

这里安利一下我们团队的[DiT扩散模型](https://zhida.zhihu.com/search?content_id=245041616&content_type=Article&match_order=1&q=DiT%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B&zhida_source=entity)的并行推理工作[PipeFusion](https://link.zhihu.com/?target=https%3A//github.com/PipeFusion/PipeFusion)也用了TeraPipe式的token-level切分的流水并行，因为DiT不是Causal Attention所以更适合TeraPipe方式推理。

### 4.2 Layer-wise Prefill

这一节比较晦涩，我理解是在Prefill阶段对KVCache做CPU Offloading（或者传输到Decode Instance？）。这些操作都是异步的，而且是Layer-wise的，也就是一层Prefill一个Layer的KVCache Block算出来，就立刻transfer（发给decode Instance）或dump（cpu offload）来最小化GPU显存占用。

为何着急忙慌赶人家KVCache走呢？我理解是因为Prefill机器比Decode少，因为Prefill负载占LLM推理的比例低，但是Prefill产生的KVCache和Decode消耗KVCache一样多，所以Decode那边为了把硬件榨干，需要让KVCache刚好用满GPU显存，那Prefill显存肯定不够，必须Offload。这也是Mooncake论文Figure 1中之所以写成“Prefill阶段KVCache < DRAM”，“Decode阶段KVCache < VRAM”的原因。

Mooncake论文Figure 5试图证明Layer-wise有效果。这个图画的草率了，全文没有提到Serialized是什么意思。

我理解他是想说Splitwise论文中Fig. 11（下图），也就是KVCache从Prefill Instance通过Layer-wise方式传输给Decode Instance，这个可以和Prefill计算重叠起来，甚至和Decode第一个step部分计算重叠起来。Splitwise采用异步流式来和Prefill计算重叠，我觉得Mooncake也是类似。

## 5 KVCache-centric Scheduling

KVCache Pool用Paged方式管理KVCache（本章简称Cache）。

如下图所示，黄色是已经有的Prefix Cache Blocks，其他request算了，本request可以复用。粉色是本request自己算的Incremental Cache Blocks。这Pool也会用一些Cache策略来新陈代谢，比如LRU、LFU之类的。

Prefill节点接收到一个request，它根据prefix cache ID将前prefix cache从远程CPU内存加载到本地GPU内存，以启动request的prefill计算。如果不存在prefix cache，则需要自己Prefill计算了。这种选择平衡了三个目标：尽可能多地重用KVCache（三板斧中Caching）、平衡不同预填充节点的工作负载，并保证TTFT SLO（三板斧中Scheduling）。

### 5.1 Prefill Global Scheduling

本节介绍如何给Prefill计算做reuse kvcache blocks和load balance。在Mooncake它将一个request切分成token blocks处理，类似FastGen和Sarathin中的Chunk。路由这些token blocks到哪台机器，要考虑因素很多。

MoonShot用户很多，request之间有重叠部分，可以reuse共同前缀，prefix cache hit length越长计算越少。我猜测这个就是[Kimi Context Caching](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/anZhObqWPLWZrNQCFv64Ag)能力的来源。但是，都路由到prefix cache hit length最长的机器，机器之间会**负载不均衡**。为了load balance，还需要考虑distribution of reusable KVCache blocks。本节也callback了KVCache为中心的宗旨，是让request主动去找KVCache。

#### 先说怎么找到max prefix cache来尽可能reuse kvcache blocks。

一个token block经过prefill计算产生一个KVCache block，而且一个token block prefill计算需要所有Prefix token blocks的Prefix KVCache blocks。这些KVCache blocks都是在Distributed KVCache Pool中，不一定在本地内存，怎么快速检索到众多前缀KVCache blocks呢？需要建立一个token block -> KVCache block的映射关系，根据映射关系去检索prefix token blocks的KVCache blocks。这个映射关系是一个Hash Table。

为了快速找到一个token block最大前缀max prefix token blocks，Hash设计有讲究。每个Block的Hash Key是基于前一个Block的Hash Key计算得到，图中B=Hash(A+b)。如果两个Block Hash Key相同，不仅该token block的KVCache block，那么之前所有prefix KVCache也都可以复用。如果不这样设计，你可能要反复遍历所有KVCache Hash，而用这种方式只需要遍历一次，检索代价从多项式降低到线性。这个技巧很巧妙，**来自vllm。**（可以在你的项目中提到，虽然你没做什么工作，但是你对vllm的研究很深入也是加分项）

#### 再说怎么做load balance。

借助上面算出来的max prefix cache length信息，注意每个机器都不一样。可以用request长度+此机器的prefix长度+队列中的等待时间来估计TTFT。将request分配给估计最短TTFT的机器，并相应更新该机器的缓存和队列时间。如果无法实现SLO，直接返回HTTP 429 Too Many Requests。

### 5.2 Cache Load Balancing

本节介绍如何KVCache负载根据使用频率做再平衡。

在Mooncake集群中，每个Prefill机器管理其自己本地的prefix caches。这些caches的使用频率差异很大。例如，系统提示几乎被每个请求访问，而存储来自本地长文档内容的cache可能只被一个用户使用。我们希望不同机器cache使用频率相近，因此要调整cache在分布式集群的位置。

因为很难预测一个cache未来使用频率。Mooncake提出了一种基于启发式的自动热点迁移方案。

上一节所述，request可能并不总是被定向到具有最长prefix cache length的Prefill机器上。**在这种情况下，如果估计的prefill时间短于cache传输时间**，因为要排队等待，**cache位置和request转发到另一个机器**。该机器主动检索KVCache并将其从远端机器拉取到本地。另外，如果远程机器的最佳prefix cache length不长，还可以**重计算**。（这个指的是什么？？？）

这两种策略合在一起，不仅减少了prefill时间，还促进了热点缓存的自动复制，使其能够更广泛地分布在多台机器上。

## 6 Overload-oriented Scheduling

现有的LLM服务通常假设所有请求都会被处理，并根据请求吞吐量、TTFT和TBT进行优化。然而，处理每个请求既不经济也不现实，尤其是在请求激增时，集群资源增长跟不上请求增长，导致过载。为了平衡成本和用户体验，系统应在负载达到阈值前尽可能处理请求，之后拒绝或推迟剩余请求。

本节描述了为Moonshot设计的early rejection policy，应该就是下图氪金之后和Kimi一起登月的背后原理。我没花时间看，就不班门弄斧分析了，但是这一环节对线上服务很重要。当然，作为用户还是希望Kimi不要拒绝服务。

## 总结

本文是阅读Mooncake技术报告的学习笔记。通过Mooncake还是学到了很多干货，这里也感谢作者团队能够分享技术。 短短一年内，创业团队能做出Mooncake这种完整的系统工作，并在线上服务海量用户，实打实节约成本，并给社区一些方向层面的输出，是非常了不起的成就。
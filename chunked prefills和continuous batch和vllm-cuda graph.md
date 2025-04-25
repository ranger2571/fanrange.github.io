
### vllm的cuda graph
cuda graph是在decode阶段使用，vllm固定了一些shape的 graph，decode的shape是[batch,1,channel]，所以可以通过设定一些batch的graph，来进行使用。（减少kernel launch的时间，因为decode阶段是访存密集型，对减小kernel launch的时间的需求比较大）

而prefill的情况是seq的长度不确定，那么就很难设定graph来使用。（可以使用padding，但是padding会占用显存，不一定是好的方法）

### continuous batch的工程实现
>静态/动态/连续批处理对比
>- **静态批处理**：客户端将多个Prompt打包进一个请求中，并在批次中所有序列完成后返回响应。通常，多数推理服务支持这种方法，但并不要求这样做
>- **动态批处理**：多个请求的Prompt在服务端内部动态打包进一个批次处理。通常，这种方法的表现不如静态批处理，但如果响应短或长度一致，可以接近最优。当请求具有不同参数时，这种方法效果不佳。
>- **连续批处理**：将请求在到达时一起批量处理，它不是等待批次中所有序列完成，而是在迭代推理层级将序列组合在一起。它可以实现比静态批处理高10倍到20倍的吞吐量，目前是最先进的方法。

#### 最基础最basic的概念：

区别于static batch，static batch是等到request 满足一个batch后，才进行一个batch输入 scheduler中，一起进行prefill，然后一起进行decode，然后一起输出。（重点，batch，一起）

continuous batch是，将之前的调度思路从“batch输入，prefill，decode”的串行思路，切换成动态思路。具体来说，分两部分：
1. 从满足batch大小才输入scheduler，变成只要有空间就将请求输入scheduler中。
2. 从之前先一起进行prefill，再一起进行decode；切换成：每进行一次prefill或者decode，就进行一次判断，看能否增大batch的大小，把新的request输入scheduler中。而且scheduler优先进行prefill阶段，目的是在进行decode时，增大batch大小，提升效率。
	1. （decode完成应该才能增加新request，prefill完成应该是将之前sawped的请求重新进行decode）
	2. （v1中没有swaped队列了，但是使用了hash kv cache来进行调度，把请求重新放回waitting队列，如果cache没用被覆盖，就可以调用，逻辑更简单了）
> 为什么优先进行prefill呢？
> 虽然 prefill 的计算量远大于一次 generation，但是得益于大模型强大的并行计算能力，一次 prefill 和 generation 所需的时间可以认为是相等的，即 T1，T2，T3，T4 四个阶段的时间可以认为是相等的。

#### 从文章和算法的角度来理解
问题是：已经结束的request，没有及时释放内存占用，导致内存浪费而且降低了tpot

两个算法结合解决上述问题：

1.迭代式调度

每次迭代（一次prefill或者一次decode）作为一次调度请求，计算出一个token后即返回结果

2.batch执行算法更新

同一batch中输入的长度不同，不能由一个cuda kernel一把算完。Transformer分为若干个阶段，由输入生成qkv，qkv进行注意力计算，layer norm，ffn，其中除了qkv进行注意力计算其他都是token-wise的多个请求可以拼成一个进行计算，qkv注意力计算因为不加载模型参数，所以分开计算和合在一起计算相差不大，基于上述特性，实现一个batch一起的计算可以提升效率

#### vllm的continuous batch的实现
[vLLM调度器解密（上）：Continuous Batch 是如何工作的？ - 知乎](https://zhuanlan.zhihu.com/p/1117099341)
在开始之前给大家抛出几个面试高频题：

1. 调度期间来的请求会被立即处理吗？如果不会，则在什么时机会被处理呢？
2. Schedule的running队列会同时存在prefill阶段的seq_group和decode阶段的seq_group么？
3. Scheduler是如何平衡prefill请求与decode请求的执行的？会不会出现饥饿的情况，怎么解决？

针对Schedule的调度有两种方式Iteration-Level调度和chunked_prefill调度方式。
上图的\_schedule_default主要就是Iteration-Level调度的内容；\_schedule_chunked_prefill就是 Chunked Prefill调度的内容。

Continuous Batch 也叫 batching with iteration-level scheduling。相比于传统的naive batch 的一次申请未来用到的最大空间，Continuous Batch 会在推理的每次迭代（prefill or decode）时进行一次batch，从而降低了内存碎片，提高了并行度和吞吐量。

（其实是和paged attention的技术进行结合使用？是的，看下面

>和ORCA不同之处在于，vLLM Batching时候prefill和decoding是分开的，一个Batching step要么处理decoding要么处理prefill。这样实现比OCRA更简单了，prefill直接调用xformers处理计算密集的prefill attn计算；decoding手写CUDA PA处理访存密集的attn计算。
>
>[大模型推理核心技术之Continuous Batching和我的WXG往事 - 知乎](https://zhuanlan.zhihu.com/p/676109470)



##### Continuous Batching 核心逻辑（下面的内容太繁琐了没必要看）
1. 调度器优先调度prefill阶段请求（waiting队列中的请求），其次会处理running队列的请求，如果调度running队列时没有发生抢占，则会调度swap队列的请求。
2. 调度prefill阶段请求，会有一些准入条件，包括：
	1. swap队列为空
	2. 距离上次调度prefill请求满足一定的时间间隔
3. 调度器还支持了优先级调度的能力（_schedule_priority_preemption函数实现），支持按照seq_group的优先级进行任务调度。

##### prefill阶段调度
Prefill 阶段核心逻辑实现在\_schedule_prefills 函数里，它核心是给waiting队列中的seq_group分配物理空间，直到物理空间不足或者waiting队列空了。  
其中有个判断调度waiting队列的时间点值得关注下：  
**_passed_delay函数的逻辑**  
在_passed_delay函数中最重要的是以下三个参数的相关逻辑：

- `self.prev_time`：上一次调度发起的时间点，初始化为0。每执行1次推理阶段前，调度器都要做一次调度，这个变量存放的就是上次调度发起的时间点。
- `self.prev_prompt`：为bool类型，初始化为False。若上一次调度时，调度器有从waiting队列中取出seq_group做推理，即为True，否则为False。
- `self.last_prompt_latency`：记录 当前调度时刻（now） - 最后一次有从waiting队列中取数做推理的那个调度时刻 的差值（并不是每一次调度时，调度器一定都会从waiting队列中取seq_group，它可能依旧继续对running队列中的数据做推理），初始化为0。

##### decode阶段调度

_schedule_running函数逻辑
_schedule_running函数中，最主要有个抢占的逻辑。

- 在running队列中的任务没有足够的物理空间执行任务的时候，会执行抢占的逻辑  
- 如果running队列中存在其他的任务，会将最后进入队列中的任务的空间抢占了
- 如果没有其他的任务则暂不执行当前的任务，等到下一轮调度再执行。
- 每执行完一次抢占之后，都会判断当前的空间是否满足seq_group的执行，如果足够，则继续进行下一个任务的分配。直到将running队列中的seq_group都处理完。

其中抢占的分为两种，下面进行进一步分析。  
**_preempt函数逻辑（抢占逻辑）**  
当一个seq_group被抢占时，对它的处理有两种方式：

1. Swap模式：如果该seq_group剩余生命周期中并行运行的最大seq数量 > 1（此时seq数量多，重新计算的成本相对较高），此时会采取swap策略，即把seq_group下所有seq的KV block从gpu上卸载到cpu上。
2. Recompute模式：如果该seq_group剩余生命周期中并行运行的最大seq数量 = 1（seq数量少，重新计算成本不高），此时会采取Recompute策略，即把该seq_group相关的物理块都释放掉，然后将它重新放回waiting队列中(放在最前面)。等下次它被选中推理时，就是从prefill阶段开始重新推理了，因此被称为“重计算”。  
    

- _schedule_swapped函数逻辑

#### 总结一下：  

1. 本文简单介绍了一下schedule是如何被engine调用的。
2. Schedule主要结构包括了waiting、running、swapped队列
	1. waiting队列存放尚未开始推理的seq_group
	2. running队列存放的是上1个推理阶段被送去做推理的seq_group
	3. swapped队列：用于存放被抢占的seq_group

3. vLLM 调度策略有两种：Iteration-Level调度和 [Chunked Prefill 调度](https://zhida.zhihu.com/search?content_id=249242567&content_type=Article&match_order=1&q=Chunked+Prefill+%E8%B0%83%E5%BA%A6&zhida_source=entity)。本篇文章主要介绍了Iteration-Level调度策略。  	
	1. waiting、running、swapped队列有着优先级的区别：waiting（prefill） > running/swapped（decode）。但是不是每次调度都能调度waiting的队列的。调度waiting的队列的seq_group是有准入条件的：swapped队列是空的 和 距离上次调度waiting队列满足一定的时间间隔。 通过有条件的调度，实现了prefill请求和decode请求处理之间的平衡
	2. 同时针对running队列的调度，由于资源是有限的，存在抢占的策略
	3. vLLM 的Iteration-Level调度还支持了基于优先级的调度策略，整体思路是每次调度前，将waiting队列跟running队列中的seq_group进行基于优先级的重新排序，针对waiting队列的seq_group进入running队列，会执行抢占running队列的seq_group的空间

## chunked prefill
##### PD分离的意义：
LLM的推理过程可以分为prefill阶段和decode阶段。而prefill阶段是compute-bound（计算密集型）的，占用的显存较少，但是GPU的利用率较高。decode阶段是memory-bound（存储密集型的）的，GPU的利用率较低，而占用的显存比较多。
而针对prefill阶段的优化能降低TTFT；针对decode的优化能降低TPOT。TTFT和TPOT是LLM中非常重要的两个指标。因此需要对prefill阶段与decode阶段进行相关的优化

>TTFT和TPOT 是什么呢？TTFT和TPOT是LLM推理中非常重要的两个指标。
>TTFT（Time To First Time）： 首token延迟，就是从输入到输出第一个token的延迟。这个指标决定了LLM的用户体验是在线stream应用中最重要的指标。人与人之间对话的延迟容忍度通常为200毫秒。所以TTFT最好也是需要在200ms左右的时间范围内。
>TPOT（Time Per Output Time）：每个输出token的延迟（不含首个token）。该指标是直接决定了整个推理过程的时间。
>另外，Throughput（TPS 吞吐量，即每秒针对所有请求生成的token数）、Latency（延迟，即从输入到输出最后一个token的延迟）
>QPS（Queries Per Second）表示服务器或模型每秒钟能处理的独立查询请求数量。 这个指标更多用来衡量模型在在线部署场景下的并发处理能力。一个完整的查询可能包含输入预处理、模型推理、后处理等，而 QPS 则从请求数量角度体现系统负载能力。

由于prefill阶段与decode阶段的特性不一样，一般采用分治思想，朝着独立的方向进行优化。这样就不用在TTFT和TPOT之间做trade off。
##### chunked prefill这种调度模式存在的合理性
但思考一下下面的问题：

如果我能采取一种方法，使得处于prefill阶段的请求和处于decode阶段的请求能组成一个batch同时计算，而在组建这样的batch的过程中，我又充分考虑了最大化GPU计算单元利用率、最小化IO读写次数（简而言之，怎么能榨干一块gpu我就怎么来）。
那么这时，我是不是在不解耦的情况下，同样也能同时保全TTFT和TPOT呢？

其实上述思考就是chunked prefill产生的思想了。

### vllm实现chunked prefill
chunked prefill与Continuous Batch主要的不同点是：

1. chunked prefill调度逻辑中running队列的seq_group优先级是高于waiting队列

2. 整体的优先级为：running队列中decode请求（decode请求） > running队列中prefill请求（之前被调度的未完成的prefill请求） > swapped队列中的请求（之前被抢占的请求） > waiting队列中的请求（新的prefill请求）

3. chunked prefill调度不支持按照seq_group优先级调度
4. chunked prefill每轮调度的seq_group有可能即存在prefill阶段的seq_group，又存在decode阶段的seq_group


# KVcache
为什么需要，加速decode

为什么可以加速decode，是attention计算的自回归机制决定的。

即对于每一个batch的每一个head，最后一个token的Q_head_size要和前面的每一个token的K_head_size进行gemv计算，虽然理论的计算是[batch,head_id,1,head_size]\*[batch,head_id,seq_len_now,head_size]
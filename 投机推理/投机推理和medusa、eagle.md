[[LLM 投机推理] 超越 Medusa 的投机采样—— EAGLE 1/2 论文解读 - 知乎](https://zhuanlan.zhihu.com/p/716344354)
# 也需要考虑很多技术，是否可以融合？是否可以拼接
### 有了VLLM这些成熟的框架，推测解码如EAGLE的主要应用场景在哪里？
[(38 封私信 / 42 条消息) 有了VLLM这些成熟的框架，推测解码如EAGLE的主要应用场景在哪里？ - 知乎](https://www.zhihu.com/question/663793097)

>问题：page attention和dynamic batch都尽可能的利用好GPU的算力，增大吞吐量，对某个样本的实际解码时间基本没有变化；但是像EAGLE1/2这种推测解码，真正的从算法结构本身上来做加速，其核心思想是利用GPU闲置的算力来尽可能的提升单个样本的实际解码时间，也就是极大概率会降低吞吐量。这这个角度来讲，VLLM和推测解码像是两种相异的方法，很难糅合到一起。但是在VLLM框架或者生态已经发展的比较成熟的情况下，像推测解码这类算法的应用场景或者受众可能会变得狭窄一些，可能更适合于那些用户并发量不太高的场景或者边缘终端场景?

解答：
#### 1
[vllm](https://zhida.zhihu.com/search?content_id=720405694&content_type=Answer&match_order=1&q=vllm&zhida_source=entity)具体情况不太了解，sglang那边已经把eagle合入主线了。这些框架的目标就是要集成各种彼此正交的优化手段，最大化系统吞吐。

题主认为，paged attention和dynamic batch已经尽可能最大化利用GPU算力，其实不是的。paged attention是显存管理手段，和利用算力关系不大，和提高吞吐有关。dynamic batching确实是尽可能的提高batch size，从而提高算力的利用。

但是要考虑到，llm推理的[decode](https://zhida.zhihu.com/search?content_id=720405694&content_type=Answer&match_order=1&q=decode&zhida_source=entity)阶段是memory bound，基本卡在内存访问上，算力并没有被充分利用。而[prefill](https://zhida.zhihu.com/search?content_id=720405694&content_type=Answer&match_order=1&q=prefill&zhida_source=entity)阶段是compute bound，计算强度比较高，算力利用比较充分。如何理解这一点呢？我们在和大模型对话时，即使我们输入的token很多，也可以迅速的拿到首token，但后面的输出往外吐的比较慢。也就是说，GPU每秒能prefill的token，要比能decode的token多很多。

投机推理能实现加速，也恰恰是利用了decode阶段的memory bound的特点。投机推理的验证相当于是跑一遍prefill，当然比跑n遍decode更快
#### 2
说简洁点儿吧，主要三方面：

1. 端侧，batch size = 1，此时计算核肯定打不满；
2. 长文本，这点 MagicDec[[1]](https://www.zhihu.com/question/663793097#ref_1) 讲的很明白：长文本场景下部署投机解码可以增加吞吐；
3. [Long CoT](https://zhida.zhihu.com/search?content_id=720690676&content_type=Answer&match_order=1&q=Long+CoT&zhida_source=entity) 这类对降低 latency 的需求很迫切的场景。此类场景中，可以忍受为了降低 latency 而浪费部分算力。
参考
4. [^](https://www.zhihu.com/question/663793097#ref_1_0)MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding [https://openreview.net/forum?id=CS2JWaziYr](https://openreview.net/forum?id=CS2JWaziYr)

# 关键参数
在整个过程中，有以下几个环节影响推理耗时：

1. draft模型生成draft token的速度

2. 目标模型计算草稿token概率的速度

3. 草稿token被接受的比率

可以使用的草稿模型，可以是大模型的轻量化蒸馏版本，也可以是n gram模型，或者是其他的小型化模型

## 选择模型的关键是
平衡

- draft模型生成draft token的速度（生成速度）
- 草稿token被接受的比率（结果准确率）

使用小尺寸大模型来实现的投机算法，草稿模型的推理开销依然很大，并且市面上大模型的尺寸都是固定的，可选范围较少（如果目标模型本身就是7B的，未必能找到更小的如1B尺寸的同类模型）；

而lookahead方法虽然效率更高但草稿的接受率较低，总是需要重新采样，加速效果不理想




# 细节的MEDUSA和EAGLE的论文研究背景
- Token level和Feature level的特征表达关系
	- 在自回归生成中，使用 Token 的隐藏层特征（Feature Level）比直接使用 Token Embedding 预测 Next Token 效果更好。
	- 这是因为 Token Embedding只是文本的一个简单转化，没有经过深层网络抽取特征，表达能力不足，在使用轻量 Draft 模型时预测效果会有折损；而 Token 的隐藏层特征指最后一层 TransformerLayer 的输出，LM Head 的输入。隐藏层特征经过深层网络的计算，其表达能力要强于 Token Embedding，也更适用于采样。
	- 常规的投机推理就是 Token Level 自回归解码方式，接收率受限；

# Medusa
### 特点
- feature level的特征输入。
- 不使用独立的draft模型，而是在原有的模型基础上进行添加medusa head模块
- 使用tree attention来一次性验证的draft tokens的序列的概率。极大地提高了验证草稿token的速度：

### 路径验证的内容
**最笨的验证方法是串行验证每条路径，然后输出最长匹配的路径。但是这样需要反复调用原始 LLM 推理，其调用次数为∑i=1N∏j=1iCi，最多接收 N 个 Token。对比接收 N 个 Token 仅需调用 N 次原始 LLM 推理的自回归采样，性能会大幅下降。这决定了 Medusa 不能使用串行的方式验证所有路径，必须使用并行方式同时验证所有路径。**
![[Pasted image 20250410151252.png]]
上图展示 Tree Attention 的过程。图中有两个 Medusa Heads，Medusa Head 1 生成 2 个 Draft Tokens，分别为 “It” 和 “I”；Medusa Head 2 生成 3 个 Draft Tokens，分别为 “is”、“'” 和 “the”。第一步需要构造验证序列。以 Medusa Head 1 的 Draft Token 为终点的路径一共有 2 条，这是树的第一层；以 Medusa Head 2 的 Draft Token 为终点的路径一共有 6 条，这是树的第二层。将所有路径糅合在一起，得到一个序列长度为 8 的验证序列。前两个 Token 为树第一层的两个节点，对应两条长度为 1 的路径。这两个 Token 只看到自己，所以上图矩阵中(0, 0)节点到(1, 1)节点的小矩阵的对角线上的 Mask 设置为 1；第 3 到第 5 个 Token，只看到自己以及前置的 Token “It”，所以在自己及 “It” 位置上的 Mask 值设置为 1；第 6 到第 8 个 Token 同理。再完成 Tree Mask 构造后，就可以通过 Tree Attention 完成一次原始 LLM 推理调用，批量验证所有 Draft 路径。

（**提问：如何通过 Tree Attention 完成一次原始 LLM 推理调用，批量验证所有 Draft 路径？？？）
（答：Tree Attention 将所有路径拼起来，变成一个输入的序列。经过一次原始 LLM 推理调用，得到一个形状为 [S, V] 的输出概率，输入序列中每个token对应一个next token的概率，求argmax后就得到期望的 next token。然后验证这个期望的 next token 和输入时Draft路径中的的next token是否一致）




- 加入的模型结构是medusa heads，来估计next token
- 对draft model的next token的预测结构进行评估和选择，基础的是使用tree attention对每一个路径进行评估，（**优化措施是：对每一次，todo！！）
# 对MEDUSA和speculative decoding的观察总结
Token-Level 自回归解码方式的不足：在自回归生成中，使用 Token 的隐藏层特征（Feature Level）比直接使用 Token Embedding 预测 Next Token 效果更好。这是因为 Token Embedding只是文本的一个简单转化，没有经过深层网络抽取特征，表达能力不足，在使用轻量 Draft 模型时预测效果会有折损；而 Token 的隐藏层特征指最后一层 TransformerLayer 的输出，LM Head 的输入。隐藏层特征经过深层网络的计算，其表达能力要强于 Token Embedding，也更适用于采样。常规的投机推理就是 Token Level 自回归解码方式，接收率受限；


采样算法的不确定性会限制生成 Next Token 的隐藏层特征的效果：当前 Token 的隐藏层特征经过 LM Head 后使用采样算法随机生成 Next Token。而在 Next Token 生成 Next Next Token 过程中，需要生成 Next Token 的隐藏层特征，该特征完全由 Next Token 决定，间接由采样算法决定。 采样算法具有随机性，有一定概率选择不同的 Next Token，这样使得 Next Token 的隐藏层特征也具有随机性，会影响 Next Next Token 的生成效果。

Medusa 也有类似的随机问题。虽然 Medusa 采样时使用隐藏层特征，但是在生成每一个 Draft Token 时，都缺乏上一个 Token 的信息，导致生成效果也不稳定。**所以为了提高 Token 生成的稳定性，在生成当前 Token的隐藏层特征时，可以融合当前 Token 的 Embedding 以及上一个 Token 的隐藏层特征，保证 Draft Token生成时具备足够的信息，增强 Token 生成效果。**

# EAGLE

### 特点
1. EAGLE的模型的设计，是一个**同构的**轻量LLM，embedding和LM head层是复用大模型，中间中transformer block使用一个One Auto-regression Head替代，中间的 One Auto-regression Head(简称 AR Head) 为由一层 FC 层以及一层 Transformer Layer 组成。AR Head 是唯一需要微调的网络层，训练成本也是极低。
2. EAGLE 使用 Draft 模型通过自回归采样方式生成 Draft Tokens，并且为了提升接收率，在运行 AR Head 前会融合上一个 Token 的隐状态与当前 Token 的 Embedding层，并通过 AR Head 的 FC 层，将特征向量进行融合（形状从 [seq_len,hidden_size∗2] 变为 [seq_len,hidden_size]）。
3. 在 Draft 阶段，自回归采样的第一轮输入是原始 LLM 模型生成的 Token，通过 TopK 采样生成第一轮的 Top K1 个 Draft Token。该 K1 个 Draft Token 作为第二轮输入，对应生成 K1组 Draft Tokens，每组通过采样生成 K2 个 Draft Token。下一轮会从当前轮中选择若干个 Draft Token 的输出作为输入，继续自回归采样，直至到达预设的运行轮次。（也就是K1\*K2\*K2\*K2......）
4. 通过上述方式的 Draft 采样过程，就得到一颗 Draft 树（第一层是K1个节点，下面每一层应该是上面的节点数\*K2）。 
5. **与 Medusa 不同的是，Draft 树并不是不同轮次的 Draft Tokens 通过笛卡尔积方式生成的，而是根据不同的 Draft Token，生成不同的Draft Token 子节点**。这样的好处是：天然地去掉 Draft 树中无用的分支，使 Verify 阶段的验证序列长度大大缩小；使路径中每个节点之间的相关性更高，进一步提高接收率。（这个部分的内容怎么理解？）
6. 在生成 Draft 树后，Verify 阶段使用 Medusa 的 Tree Attention 进行验证，批量验证所有 Draft 路径。与 Medusa 类似，EAGLE 也使用先验的方式，提供一个静态的 Draft 树。对于任意输入，Draft 阶段都将按照图中所示的 Draft 树生成 Draft Tokens。
### 结论
EAGLE 方法融合 Speculative Decoding 和 Medusa 的优点，使用 Token-Level & Feature-Level 方式进行 Draft Token 的采样，不仅提高了每个 Draft Token 的准确率（与Speculative Decoding 对比），还提高了每个节点之间的相关性（与 Medusa 对比），使接收率大大提高；并且自回归的采样方式也避免了通过笛卡尔积方式生成 Draft 树，使得验证序列的长度大大缩小，也减少了 Verify 阶段的 overhead，与 Medusa 相比提高了 Compute-Bound 的 batch size 临界点，可以在更大的流量下提升推理速度。

# EAGLE2
- 重点：预测的token的接受率，不仅与token所在的位置有关（也就是预测的token所处的层数），也和当前token的预测上文有关，

> 接受率除了与 Token 所在位置相关以外（在树中所处的位置），还和上文相关（树中的祖宗节点）。作者在 Alpaca 数据集上测试了 Vicuna 7B，记录了不同的 Draft Token 的接收概率，如下图所示。下图左侧表示 Draft 树的结构，一共有 6 个节点，分别是 P1 至 P6 节点。右侧表示不同位置的接收率。通过接收率可以观察到，树中的左上角部份接收率更高，右下角的接收率更低。P3、P4 和 P5、P6 虽然都是同一层的节点（即同一个 Step 的 Draft Tokens），但接收率上 P3、P4 普遍高于P5、P6 节点，一个重要的原因是 P3、P4 的父节点为 P1，其概率高于 P5、P6 节点的父节点 P2。P3、P4的概率甚至普遍高于 P2，**这更加说明在生成 Draft 树的时候，采用静态 Draft 树并不是一个最优选择，更应该选择动态 Draft 树，选择接收率高的节点继续发展子节点。**

-  原始 LLM 自回归生成的 Token 概率分布表示 Token 接收概率。Eagle 的 Draft 模型生成的 Draft Tokens 概率分布与 Token 接收率分布接近。下图展示了 Draft Tokens 生成概率和 Token 接收率的分布图，可以看出分布很接近，**可以通过 Draft Tokens 生成概率预估 Token 的接收率。**

EAGLE 2 提出上文感知的动态 Draft Tokens 树结构，通过 Token 的联合概率预估接收率，利用接收率动态展开 Draft 树，在保证树节点个树基本不变的前提下（即verify开销基本不变），提高 Draft Tokens 的接受长度。这是一种二阶段方法，包括 Expand 阶段和 Rerank 阶段。

- Expand 阶段：生成 Draft 树。每个节点包含路径的概率信息，通过将路径上边的权重相乘得到。Expand 过程中，当展开的节点的概率值小于阈值时，则停止该节点继续展开，否则继续展开。如下图 Expand 阶段所示，阈值为 0.1，K 为 2，每个节点取前 2 个子节点进行 Expand。当节点概率小于 0.1 时，不再对该 Token 采样下一个 Token。
- ReRank 阶段：保留概率最高的 K 个节点，其余节点删除。当不同的节点概率相等时，保留浅层节点，目的是保证树的结构。如下图所示，K 为 8，取树中概率最高的前 8 个 Token 组成新的 Draft 树。

### 结论

EAGLE 2 在 EAGLE 1 的基础上，加入了动态Draft 树生成，在保证 Draft 树节点个数基本不变的前提下（即 Verify 阶段的 overhead 基本不变），提高 Draft Tokens 的接受长度。

# EAGLE3
EAGLE3主要有以下几点改进：

1. 尽管next feature预测任务可能比较容易，但要求草稿模型去拟合目标模型的隐藏层是一种限制，导致在训练数据扩增时，草稿模型的表达能力没能得到相应提升，所以又改回了next token预测；

2. 此后，草稿模型的隐藏层将不再与目标模型的隐藏层同分布，于是从第二步开始，模型的输入出现了偏差



# 总结论

投机采样的核心优化方向是 Draft Token 的接收率及投机采样的 overhead。接收率上的优化主要依赖模型架构的魔改，overhead 的优化主要依赖压缩 Draft 树规模。EAGLE 提出了 Token-Level & Feature-Level 融合采样算法和动态 Draft 树生成算法，提升 Draft Token 的接收率，并且通过剔除大量无用的 Draft 树分支，提升了验证效率。

# # sglang 笔记: Eagle 投机采样实现
[sglang 笔记: Eagle 投机采样解读 - 知乎](https://zhuanlan.zhihu.com/p/1888664178388088605)



# Deepseek的MTP似乎也是类似的投机采样技术
[DeepSeek-V3 MTP 工程实现思考 - 知乎](https://zhuanlan.zhihu.com/p/29082207943)

MTP方法，核心思想：**通过解码阶段的优化，将1-token的生成，转变成multi-token的生成，从而提升训练和推理的性能。具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段通过一次生成多个token，实现成倍的推理加速来提升推理性能。**


#### MTP 与 EAGLE 不同的点
如上图所示，除了多做了一次 Norm 这种细节之外，主要是多步推理的时候的串行。EAGLE 在多步推理时，只使用到了一个草稿模型做自回归推理；**MTP 在多步推理时，其实是多个草稿模型进行串行推理**。
## **MTP 实现**

### **1. MTP 加载**

虽然很多框架都支持了 EAGLE，但一般的实现，都只支持 1 个草稿模型。而 MTP 从设计上，需要加载多个草稿模型，每一个 MTP 层，都是一个草稿模型。

在推理的时候，要根据不同的 step，选不同的模型进行推理。这就使得 MTP 草稿模型的加载和推理的调度比其它投机编码要复杂。

但如果 MTP 的步长等于 1，那就相当于 1 个草稿模型，实现会简单很多。

### **2. MTP Prefill**
从上图可以看出，第 i 个 MTP Module 的输入 token，是第 i+1 个 token 到第 n 个 token，n 是当前生成的总长度。而它不仅需要 token 的 embedding，还需要 token 在前一个模型计算得到的 hidden states。

比如 MTP Module 1 的输入，是 token 2 到 5 的 embedding 和 main model 最后一层输出的 token 2 到 5 的 hidden states。

这也就意味着，在完成 DeepSeek-V3 的 prefill 时，需要输出最后一层的 hidden states，才能进行第 1 个 MTP 的 prefill；第一个 MTP 输出最后一层的 hidden states，才能进行第 2 个 MTP 的 prefill，以此类推。

可以注意到：**多个 MTP 的多次 prefill 计算是串行的**。这意味着每增加 1 个 MTP Module，每次推理的时候就要多一轮串行的 prefill，并且多一份 kv cache。一个主模型加 N 个小模型的推理，可能会严重影响计算调度的效率，可能这也是**为什么 DeepSeek-V3 只输出了 1 个 MTP Module 的原因**。大概他们也认为，**仅使用 1 个 MTP Module 性价比最高**。

### **3.MTP PD 分离**

我在之前[一篇博客](https://link.zhihu.com/?target=https%3A//yangwenbo.com/articles/reflections-on-prefilling-decoding-disaggregation-architecture.html)[4]中列举了 PD 分离背后面临的很多架构选择，MTP 会让 PD 分离变得更复杂。框架有两种选择：

选择一：**Prefill 节点做 MTP Prefill**：如下图所示，P 节点做完 DeepSeek-V3 Prefill 以后，保留最后一层所有 token（除了第 1 个，即index 0）的 hidden states，采样生成的第一个 token，获得 tokenid，然后将这些输入到 MTP Module 1 做 Prefill。最后将 1) DeepSeek-V3 61 层的 KV Cache; 2) DeepSeek-V3 MTP 的 KV Cache; 3) DeepSeek-V3 生成的第一个 tokenid；4) DeepSeek-V3 MTP 生成的第一个草稿 tokenid 和概率；这 4 部分传给 D 节点。
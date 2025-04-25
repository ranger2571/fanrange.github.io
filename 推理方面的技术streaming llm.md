[LLM推理技术之StreamingLLM：如何拥有无限长生成能力 - 知乎](https://zhuanlan.zhihu.com/p/659875511)

核心是softmax的设计bug，导致必须要对一些token进行注意力关注？

然后有了两个思路——softmax的分母+1，(从数学上不是很懂)

> 个人理解：在Attention机制中，Softmax的输出代表了key/query的匹配程度的概率。因此，如果softmax在某个位置的值非常大，那么在反向传播时，这个位置的权重就会被大幅度地更新。然而，**有时候attention机制并不能确定哪个位置更值得关注，但由于Softmax需要所有位置的值的总和为1，因此必须“表态”给某些位置较大的权重**，这就可能导致错误的权重更新，而这个错误在后续的过程中很难被纠正。如下是Miller的原话：
> 
> The problem with using softmax is that it forces each attention head to make an annotation, even if it has no information to add to the output vector
> 
> 于是乎，他改进了一下Softmax，也就是把softmax的分母加了个1仅此而已，这样所有位置值可以加和不为1，这样Attention就有了可以不对任何位置“表态”的权利。
> 
> StreamingLLM的作者采用了类似的观点解释attention sink现象。SoftMax函数的性质使得所有经过attention结构的激活张量不能全部为零，虽然有些位置其实不需要给啥注意力。因此，**模型倾向于将不必要的注意力值转嫁给特定的token，作者发现就是initial tokens**。在量化异常值的领域也有类似的观察（这里引用了一些Song Han组的文职），导致Miller大佬提出了SoftMax-Off-by-One作为可能的解决方案。
> 
> 有了这个洞见，作者设计Window Attention的改进版。思路也是很直接，在当前滑动窗口方法基础上，重新引入了一些initial tokens的KV在注意力计算中使用。StreamingLLM中的KV缓存可以概念上分为两部分，如图4所示：（1）attention sink是4个initial tokens，稳定了注意力计算；（2）Rolling KV缓存保留了最近的token，这个窗口值是固定的，图中为3。
## 背景

本文作者着力解决的核心问题是：能否在不牺牲效率和性能的情况下，部署一个能处理无限输入的LLM？以实现流式LLM应用部署的效果。也就是可以不受长度限制不停地输出，具体效果可参考StreamingLLM的主页视频。这个需要说明，**无限长输入和无限长上下文还不同，前者不需要对所有输入有记忆**。

[https://user-images.githubusercontent.com/40906949/272380427-2bd1cda4-a0bd-47d1-a023-fbf7779b8358.mp4](https://link.zhihu.com/?target=https%3A//user-images.githubusercontent.com/40906949/272380427-2bd1cda4-a0bd-47d1-a023-fbf7779b8358.mp4)

解决这个问题有几个显著的挑战：

1. 在解码阶段，由于[KV Cache](https://zhida.zhihu.com/search?content_id=234767425&content_type=Article&match_order=1&q=KV+Cache&zhida_source=entity)存在导致内存使用或延迟增加，内存上线和推理服务SLA存在，导致KV Cache不能无限大，这是性能瓶颈。不太了解KV Cache概念的请移步这个[知乎问题](https://www.zhihu.com/question/596900067)。

2. 现有模型的外推（extrapolation）能力有限，也就是说当序列长度超过pretraining时设定的注意力窗口大小时，它们的表现会下降，这是模型能力的瓶颈。如下图1所示，Dense Attention具有O(T^2)的时间和内存复杂度。当文本长度超过预训练文本长度时，其运行的性能会下降。

目前主流地增加输入文本长度的方法有如下两大类方法：

**1. 长度外推（Length Extrapolation）**：该方法让训练在较短文本上的LLM能够在推理时处理较长的文本。比如，大家经常听到的编码方法[RoPE](https://zhida.zhihu.com/search?content_id=234767425&content_type=Article&match_order=1&q=RoPE&zhida_source=entity)，[ALiBi](https://zhida.zhihu.com/search?content_id=234767425&content_type=Article&match_order=1&q=ALiBi&zhida_source=entity)等都归于此类。然而，目前尚未有方法实现无限长度的外推，还无法满足作者流式应用的需求。关于外推性可以阅读苏剑林的如下博客。

[Transformer升级之路：7、长度外推性与局部注意力](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/9431)

**2.上下文窗口扩展（Context Window Extension）**：该方法实打实地去扩大LLM的上下文窗口长度，也就是序列长度。因为Attention的计算量和内存需求都随着序列长度增加而成平方增长，所以增加序列长度很难，一些实现方法包括：训练时用[FlashAttention](https://zhida.zhihu.com/search?content_id=234767425&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity)等工程优化，以打破内存墙的限制，或者一些approximate attention方法，比如[Longformer](https://zhida.zhihu.com/search?content_id=234767425&content_type=Article&match_order=1&q=Longformer&zhida_source=entity)这种Window Attention方法。如图1所示，Window Attention缓存最近的L个token的KV。虽然在推理过程的效率高，但一旦开头的token的KV被驱逐出Cache，模型推理的表现就会急剧下降（PPL约高模型表现越差）。在图2中橘色PPL曲线在token数目超过KVCache Size后出现跃升。一个降低内存需求的优化是，让Window Attention重新计算从每个新令牌的L个最近令牌中重建KVCache。虽然它在长文本上表现良好，但由于上下文重新计算中的二次注意力导致的O(T*L^2)复杂性，使其相当慢。
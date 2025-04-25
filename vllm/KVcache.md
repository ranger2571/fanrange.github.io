在 Transformer 的 Encoder-base 的模型（如 [BERT系列](https://zhida.zhihu.com/search?content_id=234743998&content_type=Article&match_order=1&q=BERT%E7%B3%BB%E5%88%97&zhida_source=entity)）中，推理和训练过程保持了高度的统一性（差异仅仅在于是否存在反向过程）。而在 Decoder-base 的生成式模型（如 [GPT系列](https://zhida.zhihu.com/search?content_id=234743998&content_type=Article&match_order=1&q=GPT%E7%B3%BB%E5%88%97&zhida_source=entity)）中，推理和训练存在相当大的差异性，主要体现在推理过程具有以下3点特征：

- **[自回归](https://zhida.zhihu.com/search?content_id=234743998&content_type=Article&match_order=1&q=%E8%87%AA%E5%9B%9E%E5%BD%92&zhida_source=entity)**
- **两阶段**（prefill+decode）
- **KV cache**

以上三点实际上也是相辅相成、不可分割的，其中**自回归**的生成模式是根本原因，**两阶段**是外在的体现形式，**KV cache** 是优化手段。




（我的理解：自回归是由attention的设计思路定义的，所以不可能有核心的变化。目前对attention的修改也是两个思路的——1. MQA-GQA-MLA，通过选择某些算法上的设计优化、折损思路，对attention进行加速；2. flash attention技术为代表的，对attention计算时的QK相乘的结果矩阵S、通过softmax的结果P，不存储，而是使用类似与online softmax的技术存储每次的计算的最值和元素和，然后降低对cache的占用）

下面将通过梳理整个推理过程，来理解 KV cache 的作用及优化方法。

### **KV cache的显存占用分析**

假设输入序列的长度为 s ，输出序列的长度为 n ，以 FP16 来保存KV cache，那么**KV cache的峰值显存占用大小为** b(s+n)h∗l∗2∗2=4blh(s+n) 。这里第一个2表示K/V cache，第二个2表示 FP16 占2个bytes。

以 GPT3 (175B) 为例，对比 KV cache 与模型参数占用显存的大小。GPT3 模型weight占用显存大小为350GB (FP16)，层数 l 为96，维度 h 为12888。

|batch size|s+n|KV cache(GB)|KV cache/weight|
|---|---|---|---|
|4|4096|75.5|0.22|
|16|4096|302|0.86|
|64|4096|1208|3.45|

可见随着 batch size 和 长度的增大，KV cache 占用的显存开销快速增大，甚至会超过模型本身。

而 LLM 的窗口长度也在不断增大，因此就出现一组主要矛盾，即：**对不断增长的 LLM 的窗口长度的需要与有限的 GPU 显存之间的矛盾。**因此优化 KV cache 就显得非常必要。


## LLM缓存压缩方案
在 LLM 推理过程中，[KV cache](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=KV+cache&zhida_source=entity) 是最基础、应用最广泛的优化方案，其核心思想即“以空间换时间”。这样就会导致另一个问题：即随着序列长度的增加，KV cache 占用的显存会显著增加，进而制约序列长度的增加和 MFU 的提高。

为此也出现了一些优化方法，笔者将其归类如下：

- **有损压缩派**：即不改变模型结构，而通过有损的方式减少 KV cache，具体实现有如下方式：

	- [量化稀疏](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E9%87%8F%E5%8C%96%E7%A8%80%E7%96%8F&zhida_source=entity)：即对 KV cache 进行 8 bits，4 bits 甚至 2 bits 量化以减少显存占用，典型方案如 [KIVI](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.02750)，[QServe](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2405.04532)
	- [窗口优化](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E7%AA%97%E5%8F%A3%E4%BC%98%E5%8C%96&zhida_source=entity)：即不计算 dense attention, 只保留部分 cache 进行存储和计算，典型方案如 [H2O](https://link.zhihu.com/?target=https%3A//browse.arxiv.org/pdf/2306.14048.pdf)，[StreamingLLM](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2309.17453.pdf), [FastKV](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2502.01068)

- **[缓存卸载派](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E7%BC%93%E5%AD%98%E5%8D%B8%E8%BD%BD%E6%B4%BE&zhida_source=entity)**：即将 GPU 上的缓存 offload 到 CPU 上，典型方案如 [OffloadedCache](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/transformers/main/en/internal/generation_utils%23transformers.OffloadedCache)， [Prefix Caching](https://link.zhihu.com/?target=https%3A//docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- **[信息共享派](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E4%BF%A1%E6%81%AF%E5%85%B1%E4%BA%AB%E6%B4%BE&zhida_source=entity)**：即改变多头机制中 Q, K, V 的一一对应关系，而采用 Q 与 KV 多对一的方式以共享 KV，这也是当前主流预训练模型所采用的方式, 典型方案如 GQA, MQA
- **[分解映射派](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E5%88%86%E8%A7%A3%E6%98%A0%E5%B0%84%E6%B4%BE&zhida_source=entity)**：即在 attention 机制的计算框架下，通过低秩分解、因子分解等方法仅保留少量的缓存信息，然后通过计算（近似）等价变换为 KV , 典型方案如 [MLA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2405.04434), [TPA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.06425), [MFA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.19255)，本篇将主要讨论该方法
- **[线性改造派](https://zhida.zhihu.com/search?content_id=252955094&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E6%94%B9%E9%80%A0%E6%B4%BE&zhida_source=entity)**：即部分或完全放弃 softmax attention, 而将 attention 机制进行线性化处理，以使缓存量不随序列长度增加而显著增加，典型方案如 RWKV, RetNet, Mamba, MiniMax-01

从 transformer 的角度出发，以上方式的改造的激进程度依次递增，在笔者之前的文章中已比较详细讨论过有损压缩派、信息共享派和线性改造派的内容，在此不予赘述。本篇的内容将主要围绕分解映射派展开。



# KV cache稀疏化的思路：（但是你要说这个，就要详细的说 streami llm或者量化的内容）

一个是KVcache的稀疏化，就是不保存所有的kvcahce 而是只保存一部分

典型的就是streami llm

除了最开始的几层，剩下的层数都是把注意力关注到了最开始的token

一个就是量化，把fp8的kv cache 量化成int4，
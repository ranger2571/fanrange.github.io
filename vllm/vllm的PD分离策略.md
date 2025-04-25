[vLLM PD分离方案浅析 - 知乎](https://zhuanlan.zhihu.com/p/1889243870430201414)

在LLM推理计算中Prefill和Decode两个阶段的计算/显存/带宽需求不一样，通常Prefill是算力密集，Decode是访存密集。一些场景中P和D两者分开计算可提升性能。[vLLM](https://zhida.zhihu.com/search?content_id=255722900&content_type=Article&match_order=1&q=vLLM&zhida_source=entity)是一种主流的推理框架，本文主要围绕其PD分离场景做讨论。

PD分离方法出现的简单推演:

先是有整体部署的方案:一个推理服务实例里面起一个模型，适合数据长度固定场景，推理服务可通过增加batch size提升吞吐/性能。但这种方式对于[Transformer架构](https://zhida.zhihu.com/search?content_id=255722900&content_type=Article&match_order=1&q=Transformer%E6%9E%B6%E6%9E%84&zhida_source=entity)（里面包含自回归计算过程）效率却并不高。为了提升资源效率，在大语言模型中设计出了[kv cache](https://zhida.zhihu.com/search?content_id=255722900&content_type=Article&match_order=1&q=kv+cache&zhida_source=entity)减少重复计算，模型计算过程能拆成两步:prefill（一次运算，算力消耗大）、decode（自回归多次迭代，存储密集）；这两个过程如果放一个实例（硬件资源固定）中会出现P阶段显存利用率低、D阶段算力使用不高的问题，为了进一步提升系统效率又衍生出了P和D分开计算的解决方法。

## 1 vLLM PD分离方案现状

开源的vLLM0.8.x版本的PD分离功能依靠[kv transfer](https://zhida.zhihu.com/search?content_id=255722900&content_type=Article&match_order=1&q=kv+transfer&zhida_source=entity)来完成（代码[PR](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/pull/10502)），能够支持1P1D场景的运行。

其工作关键:
	顺序执行P和D计算；
	用一个kv transfer线程交换kv cache信息;
	通过proxy控制交互过程。

软件流程图如下所示。

这种方式是一种生产者（producer）和消费者（consumer）模式，P将运算完成的kv值以非阻塞的方式插入buffer中，D通过阻塞模式去获取buffer中的kv值。buffer是双端队列（LookupBuffer），buffer之间的数据传递需要通过pipe完成，主要是解决远端数据传递（[Remote transfer](https://zhida.zhihu.com/search?content_id=255722900&content_type=Article&match_order=1&q=Remote+transfer&zhida_source=entity) ），pipe可选择pynccl或者mooncacke store。


# 月球大叔的视频
## PD分离的关键问题
- 怎么传输KV cache
	- 两种模式，一种是pooling model，mooncake（要求是p和d对pool的传输速度都很快，而且pool对外也有很高的带宽）；
	- 第二种是p2p model，sender和receiver要互相确定，然后直接建立connection，进行传输
- 怎么从vllm中抽取KV cache，或者为vllm注入kv cache 
	- connector api 
		- call model runner 
			- model forward 推理前，try去receive kv cache ，也就是injec kv cache 到paged attention中
			- model forward
			- model forward 推理后，extract kv cache 
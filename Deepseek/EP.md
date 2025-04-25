[LLM(33)：MoE 的算法理论与 EP 的工程化问题 - 知乎](https://zhuanlan.zhihu.com/p/28558622452)

Mixture-of-Experts (MoE) 并非新的技术，在大模型时代，随着 [Mixtral](https://zhida.zhihu.com/search?content_id=254734227&content_type=Article&match_order=1&q=Mixtral&zhida_source=entity), DeepSeek 的广泛采用和推动，而逐步成为大模型的标配之一。
而与 MoE 算法相伴而生的，则是 [Expert Parallelism](https://zhida.zhihu.com/search?content_id=254734227&content_type=Article&match_order=1&q=Expert+Parallelism&zhida_source=entity) (EP) 的工程化技术，二者互为表里、相辅相成。因此本篇将结合笔者自身的经验和思考，系统整理和讨论 MoE 和 EP 相关问题，主要将围绕以下问题展开：

- MoE 的算法理论，包括其设计初衷、其表征能力与平衡性问题
- MoE 与计算单元(主要是GPU)的适应性与计算效率问题及 EP 的出现
- 训练及推理过程中 EP 的计算与通信过程的分析及优化


## 一、MoE 的算法理论

众所周知，当前的大模型面临一个不可能三角（并行训练、模型表现、低成本推理），而以 Transformer 为基础的LLM 则因其二次复杂度而导致推理成本居高不下。另一方面，受 scaling law 支配的 LLM 则不断追求更大的模型规模，这就导致了更高的推理成本。

由此，一个自然而然的思路是，实现一个整体参数较大，而推理时参数规模较小的模型，于是 MoE 应运而生。
### 1.1 MoE 与条件计算

在相同计算预算下，由于 MoE 的稀疏性，可以训练更大的模型规模或者更多的数据，这样的话，计算效率就显著提高了，那么这种方式的依据是什么呢？条件计算也许能提供一个有效的解释。


条件计算，即网络的部分在各自基础上激活，以不成比例增加计算的情况下显著提高模型容量的方法。条件计算被提出作为一种方法，通过按需激活某些参数，基于每个样本进行计算，在不增加所需计算量的情况下提高深度神经网络的容量。

在 Transformer 的 FFN 中，有一个重要的观察是，其计算过程中的神经元激活是非常稀疏的，在一次计算中只有 90%的输入激活不到 5%的神经元，因此省略计算未激活的神经元就可以大大降低冗余计算。因此，通过训练可以将相关神经元有效组织在一起，这样就形成了 MoE 整体框架。

### 1.2 MoE ：稀疏与并行同在

为了实现上述目标，就不得不提到 MoE 的两个核心概念：

**Experts**

MoE 层被用于替代密集的前馈网络（FFN）层，每个 MoE 层包含一定数量的 expert，每个 expert 都是一个神经网络，每次只有其中部分 expert 参与计算。从这个角度看，整体上的计算是**稀疏**的，而被选中的 experts 的计算又是**并行**的。

  

| 模型             | expert 总数 | 推理expert数 | shared expert 数 |
| -------------- | --------- | --------- | --------------- |
| Mixtral-8x7B   | 8         | 2         | 0               |
| Qwen2-57B-A14B | 64        | 8         | 1               |
| DBRX           | 16        | 4         | 0               |
| Grok 1         | 8         | 2         | 0               |
| DeepSeek V3    | 256       | 8         | 1               |


**Router**

router（或门控网络）也是一个前馈神经网络（FFN），用于根据特定输入选择 expert。它输出概率，并使用这些概率来选择最佳匹配的expert。 其公式表示如下：y=∑i=1NG(x)iEi(x)其中 N 表示 expert 数量， Ei(x) 表示第 i 个 expert 的输出， G(x)i 表示 router 对第 i 个 expert 的门控输出， 如果 G(x)i=0 即表示不需要计算该expert。

最简单的 router 算法会对token向量应用线性变换，形成一个大小为`N`的向量（即 expert 的数量）。然后，我们可以应用 softmax 函数，在 token 的 expert 集合上形成一个概率分布。我们可以使用这个分布来选择我们的标记应该选择的expert，只需简单地选择分布中 topK 的expert即可：


### 1.3 负载均衡问题

因为 MoE 的结构由多个 expert 构成，每次只选择部分执行，如果Expert的分配不均衡，就可能出现如下局面：某些 Expert（Dead Expert）几乎一直闲置，浪费算力；某些Expert要处理的Token太多，根本忙不过来，只能Token Drop（即放弃处理部分Token）。这样的话，既浪费了显存，又使模型没有达到预期的参数量。

因此负载均衡就显得非常重要，MoE 的负载均衡问题主要体现在两个层面：

- expert 的负载均衡，主要体现在训练阶段的loss设计上 (本节讨论)
- GPU 的计算均衡，主要体现在推理阶段 EP 的使用上 (将在第二部分讨论)

## 二、EP 及其工程化问题

**MoE 的通信时间往往超过了计算时间，因此关注于通信问题及其优化就成了 MoE 工程化的核心问题。**

随着 MoE 结构的流行，其模型规模也在逐步增大，需要进一步考虑其分布式问题，仅以 MoE 层为例，将 expert 分散到不同的 GPU 上就成了最自然的想法，这就是 Expert Parallelism(EP)。

EP 的工程化问题是一个热点的方向，仅 DeepSeek 已开源的项目中就有 [DeepEP](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP)，[EPLB](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/EPLB)，[ESFT](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/ESFT) 与此直接相关。接下来开始吧。

### 2.1 EP 的计算与通信过程

首先我们来梳理一下 MoE EP 的基本过程，考虑最常见的DP+EP（且DP=EP）的情形：

- router 为每一个 token 决定其目标的 expert
- dispatch 阶段
	- 布局转换：针对同一目标专家的标记被分组在连续的内存缓冲区中
	- all2all: 将 token 分配给相应的 expert
- expert 分别计算分配到的 tokens
- combine 阶段
	- all2all: 将处理过的 token 重新组合到它们的GPU上
	- 布局转换：将 token 恢复到其原始位置

为了更加深刻理解上述过程，我们不妨 DeepSeek 实现的最基础的 torch 版本，代码如下：

```python
    def moe_ep(self, x, topk_ids):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.n_routed_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // self.num_experts_per_tok]
        if self.ep_size > 1:
            tokens_per_expert_group = torch.empty_like(tokens_per_expert)
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.ep_group)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(dim=1).cpu().tolist()
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1).cpu().tolist()
            gathered_tokens = All2All.apply(sorted_tokens, output_splits, input_splits, self.ep_group)
            gatherd_idxs = idxs.new_empty(gathered_tokens.shape[0], device="cpu")
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.to(idxs.device).argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_group.view(self.ep_size, -1).sum(dim=0)
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            outputs.append(expert(sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            sorted_tokens = torch.empty_like(outs)
            sorted_tokens[gatherd_idxs] = outs
            gathered_tokens = All2All.apply(sorted_tokens, input_splits, output_splits, self.ep_group)
            outs = gathered_tokens

        y = torch.empty_like(outs)
        y[idxs] = outs
        return y
```





### 2.2 EP 的性能优化与负载均衡

除了上文提及的通信延时问题（1）外，MoE EP 还面临另外两个问题，分别是：

- 推理过程 GPU 的负载均衡问题（2）
- expert 较多，显存占用多而计算少的问题（3）

接下来将以 2-3-1 的顺序分别讨论以下问题。

**2.2.1 GPU 负载均衡问题**

尽管在训练过程中已尽可能推动 experts 之间的均衡，但是在推理时，面对不同的语言、任务、数据类型时，不同层的不同 expert 被分配的负载仍然存在非常大的差异，因此当进行 EP 时，不同 expert 的差异就会反映到 GPU 的负载上，这样就会导致算力的不平衡不充分。

在常见的 EP 的实现中对每一层都是按照相同的序号顺序进行分配，如下的 vllm 的实现，这种实现非常简单明了，也足够通用，但是没有结合模型本身的特点，因此在特定模型上性能可能会会受影响。

```text
# Create a tensor of size num_experts filled with -1
self.expert_map = torch.full((self.num_experts, ), -1, dtype=torch.int32)
# Create a expert map for the local experts
local_num_experts = num_experts // self.ep_size
ep_rank = get_tensor_model_parallel_rank()
if ep_rank < (self.ep_size - 1):
    # Each non-last rank gets local_num_experts experts.
    self.expert_map[ep_rank * local_num_experts:(ep_rank + 1) * local_num_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
else:
    # All remaining experts are assigned to the last rank.
    local_num_experts = num_experts - ep_rank * local_num_experts
    self.expert_map[-local_num_experts:] = torch.arange(0, local_num_experts, dtype=torch.int32)
```

为了解决以上问题 DeepSeek 在其开源周中给我们提供了一个 [EPLB](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/EPLB) 的工具，在此介绍其核心思想及过程：

- 尽可能选择接近目标场景的数据集，离线测试目标模型并统计每层 expert 的负载权重；
- 使用 EPLB 算法先进行分层的负载均衡计算，再进行全局的负载均衡计算
- 按照计算后得到的 expert map 将对应的 expert 加载到对应的 GPU

**2.2.2 experts 的显存占用问题**

尽管 MoE 的结构可以在保持模型总参数量较大的情况下使用较少的参数量进行推理，大大加快推理速度，但另一方面，较大的参数量仍然需要占用大量的 GPU 显存，而显存又是GPU的主要瓶颈。因此如果显存占用较多，而计算量小时，整个系统的利用率仍然较低。(**也就是中小公司内部进行Deepseek moe模型部署的时候，出现的问题：请求量小，但是机器运行的成本高，导致根本没有云厂商的api的价格便宜)

从统计结果来看，Expert 参数占总参数的 90% 以上，而在一次推理过程中，被激活的参数通常只有 20% ～30%（DeepSeek V3/R1 的比例更低，约 5% ），为了解决这一矛盾，expert offload 技术便应运而生。

当前采用 expert offload 技术的方案也有许多，由于篇幅限制，本文将选取其中比较典型的几种对其核心思想加以介绍。

- [MoE-Infinity](https://link.zhihu.com/?target=https%3A//arxiv.org/html/2401.14361v2)

MoE-Infinity 通过一种感知稀疏性的expert缓存，细致地跟踪推理过程中的 expert 激活模式。通过分析这些模式，系统优化了缓存中 expert 的替换和预取。尽管预取可以在 expert 加载与 GPU 计算之间重叠，但这些预测方法提供的效益有限，因为 expert 加载成本通常远大于基于 MoE 的LLMs推理过程中的 GPU 计算成本。

- [HOBBIT](https://link.zhihu.com/?target=https%3A//arxiv.org/html/2411.01433v2)

HOBBIT 一个混合精度 expert 卸载系统，以实现灵活高效的 MoE 推理。其核心思想是通过动态替换不那么关键的缓存缺失 expert 为低精度版本，可以显著降低专家加载延迟，同时保持模型精度。HOBBIT 引入了三种创新技术，以映射 MoE 计算的天然层次结构：（1）一个基于token的动态 expert 加载机制，（2）一个基于层的自适应expert 预取技术，（3）一个基于序列的多维 expert 缓存策略。

- [Ktransformers](https://link.zhihu.com/?target=https%3A//github.com/kvcache-ai/ktransformers)

Ktransformers 是一种 GPU/CPU 混合计算的方法，以 DeepSeek V3/R1 为例，其在GPU上完成 MLA/KVCache 的复杂计算，而在 CPU（Intel AMX）上完成 expert 计算，通过这种方式可以在单张 4090 和大内存CPU上实现 671B 模型的推理计算。
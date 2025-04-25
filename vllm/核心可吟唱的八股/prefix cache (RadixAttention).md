# 背景
在LLM推理中，我们经常会面临具有长system prompt的场景以及多轮对话的场景。

长system prompt的场景，system prompt在不同的请求中但是相同的，KV Cache的计算也是相同的；而多轮对话场景中，每一轮对话需要依赖所有历史轮次对话的上下文，历史轮次中的KV Cache在后续每一轮中都要被重新计算。

这两种情况下，如果能把system prompt和历史轮次中的KV Cache保存下来，留给后续的请求复用，将会极大地降低首Token的耗时。如果Prefix Cache和Generated KV Cache都可以缓存，在多轮对话的应用中，忽略边界情况，基本上可以认为其消除了历史轮次中生成对话的recompute。**（那么代价是什么？KV cache的大小被极大的扩充了？）

如下图所示，这种情况下，每轮对话中，都只有当前轮的prompt需要在prefill阶段进行计算。历史轮次中的Prefix + Generated KV Cache都会被缓存命中。![[Pasted image 20250323103127.png]]
# 提出技术 RadixAttention （和SGlang扯上关系）
RadixAttention想要解决的正是这个问题。RadixAttention是在SGLang的论文（《Efficiently Programming Large Language Models using SGLang》）中被提出的，其目的是为了实现Automatic KV Cache Reuse。

RadixAttention使用radix tree（基数树，说实话，这棵树我也不太熟悉）而不是prefix tree。Radix Tree最大的特点就是，它的node，不仅可以是一个单独的元素，也可以是一个变长的序列。具体体现在，在必要的时候，一个已经在Tree中的大node可以动态地分裂成小node，以满足动态shared prefix的需求。**（对这个东西完全不了解，看看下面的图能不能清晰的解释）

![[Pasted image 20250323103336.png]]

图中（1）~（9）表示Radix Tree的动态变化。每条树边都带有一个标签，表示子字符串或token序列。节点用颜色编码以反映不同的状态: 绿色表示新添加的节点，蓝色表示缓存的节点，红色表示已被驱逐的节点。这里讲解一下前面的5个步骤。

**步骤(1)**，Radix Tree最初是空的。**步骤(2)**，服务器处理传入的用户消息: “Hello”，并使用LLM输出: “Hi”，进行响应。system prompt为: "You are a helpful assistant"、用户的prompt为: “Hello!”，LLM回复为: “Hi!”；整个对话被合并为一个大node，保存到Radix Tree的节点a。**步骤(3)**，同一个用户输入了一个新的prompt，服务器从Radix Tree中找到的历史对话前缀(即对话的第一个回合)并复用它的KV Cache。新回合作为新节点追加到树中。**步骤(4)**，开始一个新的聊天会话。Radix Tree将(3)中的大节点“b”，动态地拆分为两个节点，以允许两个聊天会话共享system prompt。**步骤(5)**，第二个聊天继续。然而，由于内存限制，(4)中的节点“c”必须被清除。新的对话被缓存到新的节点"d"。

直观上来说，Radix Tree与Prefix Tree有许多相似之处。值得注意的是，在RadixAttention中，无论是Prefix还是Generate阶段产生的KV Cache，都会被缓存。这可以最大程度增加KV Cache被新请求复用的几率。

## vLLM的 Automatic Prefix Caching: Hash RadixAttention

vLLM中Prefix Caching使用的是RadixAttention算法，但是使用hash码作为物理KV Block的唯一标识，在工程上感觉更加简单。暂且把这种实现称为: **Hash RadixAttention**。

vLLM中通过BlockSpaceManagerV1类来专门管理block分配。以下是BlockSpaceManagerV1的allocate方法，分析代码之前，先解释一下SequenceGroup数据结构。SequenceGroup在vLLM中用于辅助sampling的实现，group中的所有seq都具有相同的prompt，可以理解成相同的prompt产生的不同采样结果。针对最简单的greedy search，我们可以认为group中只有一个seq。

接下来，我们来看下prefix caching，我们可以看到，代码走的是enable_caching分支，调用了gpu_allocator.allocate来分配block；这个gpu_allocator.allocate需要传入当前block的hash码以及已经被hash处理过的tokens数量。
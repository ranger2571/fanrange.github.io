[大模型推理加速-MEDUSA - 知乎](https://zhuanlan.zhihu.com/p/703461293)
# 背景
LLM推理的主要瓶颈是**自回归解码的顺序性**

每一个生成token都依赖之前的token（顺序性）。

而且在每次forward计算时，都需要将完整的模型参数从高带宽内存中移动到加速器缓存中（似乎就是显存）。（参数量大，而且需要频繁的移动，导致了低效）
# 投机推理的问题
理想的能兼顾模型参数小、计算快和准确度高的模型很难找（小参数的LLM和n gram模型的对比）

一个推理系统中，设置两个推理模型，增加系统复杂度

**采样效率低下**：在使用推测解码进行采样时，需要使用重要性采样方案。这会引入额外的生成开销，尤其是在更高的采样温度下。

# MEDUSA
- medusa heads来进行模型的结构修改，修改量少，需要少量微调就能实现推理。（添加位置，LM head的输入之前，将transformer block的输出的hidden states，输入medusa heads，来进行推测）
	- medusa heads的模型结构——带有残差连接的前馈神经网络
- 从token level的推理，变成了hidden states level的推理
- medusa heads是多个头同时推理，降低了解码的步骤数
- 路径计算方式不同（和eagle的区别，eagle是），medusa是每个heads会计算head_i个结果，那么总路径就是所有的head_i相乘。每一个路径代表一个可能性。
- tree attention 来进行可行的路径验证，实现一次计算，验证多条路径，有效减少了计算的步骤数，将多次运行压缩到一次。（使用mask，使得前一个token只能看到本token对应的下一个可能token）
	- 树结构是通过**取得每个medusa head的前预测的笛卡尔积**来构建
	- 使用mask，保证每个token只能和它的前驱节点有关系

问题
- medusa heads是采用多个头，分别同时推理next token、next next token...，头与头的推理的过程之间是没有关系的，所以导致推理的可靠性低
- hidden state不是特别有效？（忘记了）
### 如何优化树的构建，最大化预期接受的长度
加入上一个节点的概率，通过概率进行节点的threshold的判断

### 如何训练
- medusa-1 在主主干模型冻结的情况下微调美杜莎头。损失函数是美杜莎头的预测和真实标记之间的交叉熵损失。
- medusa-2 涉及联合训练美杜莎头和主干模型。为了保持主干模型的下一个标记预测能力和输出质量，我们使用组合损失（主模型与heads的损失叠加）、不同的学习率（相差四倍）和针对头部的预热策略。

### 接受策略
#### 这里是一个隐藏的point

投机解码随着temperature的提升，命中率会降低。因为temperature提升，draft model所选择的候选token的多样性就增大，也就降低了命中原模型token，从而被接受的概率。（这里的投机解码是最基础的那个，使用数学进行推导的）

但是这种特性并不合理。通常更高的temperature参数一般对应更强的creativity特性，因此合理的情况应该是随着温度提高，候选序列有更大的概率被接受。这和投机解码的情况是相反的。

另外，MEDUSA认为候选序列的分布没有必要完全match原模型的分布。我们要做的应该是选出typical的候选，也就是只要候选序列不是极不可能的结果，就可以被接受。

给定context x1,x2,⋯,xn，候选序列 (xn+1,xn+2,⋯,xn+K+1)，我们按以下这个条件来接受候选token

$$p_{original}(x_{n+k}|x_1,x_2,⋯,x_{n+k−1})>min(ϵ,δexp⁡(−H(p_{original}(⋅|x_1,x_2,⋯,x_{n+k−1})))))$$
其中 H(⋅) 表示entropy function，ϵ,δ 分别是hard threshold和entropy-dependent threshold。

两个threshold的解释：（1）ϵ 保证所选的token的概率不能低于特定值，保证不选出可能性很低的结果（2）当一个位置的多个候选token的entropy较高时，表示多个候选都是reasonable的，那么 δ 和exp(entropy)的乘积会更小，各个token都有更大的机会被接受。

当temperatrue为0的时候，相当于贪心解码，这个时候只有概率最大那个token有非0概率。随着温度提升，其他token的概率也提升，因此它们也有一定的机会被接受。随着温度提升，这些token被接受的概率会增大。

最后选择被接受的解码长度最长的候选序列作为最终结果。
## medusa的问题
**Medusa的局限所在：它通常适用于 bs 较小的场景。参考下面的内容，是不是medusa适合单batch或者小batch的场景

tree decode 的时候一次性过了多个token，然后因为他每次不确定能出几个token，只能batch size = 1 作为推理，naive transformers 推理方案也许有提升，按照现在公开的一些比如lightllm 这种动态组batch，计算利用效率比较高的推理框架，这玩意是否能提升需要打一个大大的问号

这个速度提升感觉有点够呛，我们内部实现了一下，2 个medusa head 为例，默认的choices 应该是39 个，所以每次 tree decoding 需要输入 （1 39），然后选择最优的一个路径，当然这里面平均可能出2个token。问题在于 这个1 x 39 的推理速度 是不是小于 2 次 1个token 的单独infer。单个batch 测试其实速度还行，但是随着batch 的增加 速度就显著降低，当然我们没有全部接入，只是简单的模拟了一下 ，10x39 的infer时间和 2 次 10x1 的infer 时间

计算效率：
假如2个head，相当于一次tree decoding迭代最多可以出来1个正常的token+2个美杜莎token。同时还知道了下次的正常头的token以及美杜莎头的token，相当于从第二次迭代开始，一次tree decoding，最多可以出现1 + 2 + 1 = 4个token 这样按你描述的相当于评估1 x 39 和 3次单独的infer对比？ 这里我没有具体对比过。换个角度看，如果忽略39个额外的token长度 = 1个token长度，相当于美杜莎头最多可以带来4倍的提升（因为本来一次推理出来1个token，现在最多可以出来4个），而实际官方宣传的是2倍的速度提升。是不是可以理解成额外的token编码以及头计算相当于牺牲了2倍的速度提升，这样看是不是合理一点？至于批量的推理，这个我也没有经验，我刚看了下官方的repo，似乎暂时也主要针对单条的推理，现在更新了sparse tree ，性能比之前更快一些。


推理输出token的估计：
2个head 的话 一次tree decoding 最多可以出 1 + 2 个token，最后一次预测是用不到的， posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()。

accept_length 应该不是不会超过2

我感觉他这个方法的主要的点在于，不同的输入下，llm 的推理时间 消耗在 io 还是计算上，小输入时 大部分时间都在io 层面，也就是增加输入infer 时间波动不大，大输入时 大部分时间花费在计算层面，计算资源成为了主要瓶颈。repo 里面因为测试的都是单batch 然后 tree decoding 长度还在 60 以下，基本还属于io 瓶颈部分，所以有一些提升，随着batch 的增加，增加到150 以上，计算资源成为了主要瓶颈，这时候infer 时间就会成倍的增加，这时候就没有太多优势了


# vllm的medusa的实现的特点：
1. 仅支持从**top-1 token**生成候选提议。（也就是贪心的解码策略）
2. 提供了一个可选的`token_map`，通过将draft模型的词汇表缩减为**最常使用的token**，从而通过减少采样开销来获得额外提速。
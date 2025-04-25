主要看的文章链接：[[Decoding优化]🔥原理&图解FlashDecoding/FlashDecoding++ - 知乎](https://zhuanlan.zhihu.com/p/696075602)

### 0x01 FlashDecoding[[1]](https://zhuanlan.zhihu.com/p/696075602#ref_1)

一般情况下FlashAttention forward pass在Q的seqlen维度以及batch_size维度做并行。可以看到，对于当前的Q的分块Queries，forward pass会在thread block中，逐个遍历所有的K, V分块，计算逐个分块的局部Attention输出。每个局部的Attention输出，会在thread block内部遍历的过程中，随着每一次迭代，根据当前次迭代的值进行scale，一直到沿着K,V的迭代完成后，就获得了最终正确的Output。

这种方式，对于训练的forward是work的，因为训练时，seqlen或bs会比较大，GPU资源能够被有效地利用。但是在推理的Generation阶段，是逐token生成，在利用KV Cache的情况下，每次推理实际的queries token数为1，已经无法通过queries进行并行了，GPU资源无法得到有效的利用，特别是如果bs还比较小，那GPU资源浪费将会更加严重。于是针对这种情况，FlashAttention作者开发了FlashDecoding，对推理阶段的forward进行优化。基本的思路其实也很直观：既然，Q和BS无法进一步并行了，那么对K,V进行并行是不是就可以了呢？没错，这就是FlashDecoding的思路。

FlashDecoding的做法如下：

> 1. 首先，将K/V切分成更小的块，比如5块；  
> 2. 然后在这些K/V块上，使用标准FlashAttention进行计算，得到所有小块的局部结果  
> 3. 最后，使用一个额外的kernel做全局的reduce，得到正确输出

在128K context的情况下，FlashDecoding比标准FlashAttention快50倍。
### 0x02 FlashDecoding++[[2]](https://zhuanlan.zhihu.com/p/696075602#ref_2)（非官方）

FlashDecoding++最主要的创新点，在于提出了基于统一max值的异步softmax。我们知道，safe-softmax的计算公式中，需要先求每行x的最大值，然后减去这个max(x)之后，再做softmax以防止数值溢出。

$$softmax⁡(x)=[ex1−m(x),…,exd−m(x)]∑iexi−m(x)=[ex1−ϕ,…,exd−ϕ]∑iexi−ϕ,∀ϕ∈R$$FlashDecoding++认为，这个max值，不一定需要online计算max(x)，而是可以是一个合理的先验值 ϕ 。我们对上边的公式分子分母提取公因式，可以得到：

$$softmax⁡(x)=e−m(x)[ex1,…,exd]e−m(x)∑iexi=e−ϕ[ex1,…,exd]e−ϕ∑iexi,∀ϕ∈R=[ex1,…,exd]∑iexi,∀ϕ∈R$$可以发现，使用先验值 ϕ 与直接计算max(x)，最终softmax的结果，在数学上是等价的。问题在于如何确定这个先验值 ϕ 以防止数值异常，比如对于一个很小的x，这时如果使用一个非常大的先验值 ϕ，就可能导致概率值异常。FlashDecoding++认为一个合理的先验值 ϕ，可以直接从数据集中进行统计获得。对于不同的模型，这个先验值也是不一样的。
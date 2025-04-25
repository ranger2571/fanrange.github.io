[从Coding视角出发推导Ring Attention和FlashAttentionV2前向过程 - 知乎](https://zhuanlan.zhihu.com/p/701183864)
[大模型训练加速之FlashAttention系列：爆款工作背后的产品观 - 知乎](https://zhuanlan.zhihu.com/p/664061672)

# V2比V1的提升点
### 主要优化
我认为就是修改了QKV的循环逻辑
###  FA V2其他优化

今年7月发布的V2版本是在V1基础上融合了社区的一些改进，实现了FLOPs利用率的飞跃。

我觉得V2最重要的提升点是参考Phil Tillet的Tirton版本，更改了Tiling循环的顺序，也就是笔者本文图1的效果。

V1版本循环顺序设计是outer和inner loop和图1反过来，在outer loop扫描时做softmax的规约，这导致outer loop必须在一个thread block里才能共享softmax计算中间结果的信息，从而只能对batch * head维度上以thread block为粒度并行切分。V2中调换了循环顺序，使outer loop每个迭代计算没有依赖，可以发送给不同的thread block并行执行，也就是可以对batch* head* sequence三层循环以thread block为粒度并行切分，从而显著增加GPU的吞吐。反向遵循同样的原理，不要把inner loop放在softmax规约的维度，因此正向反向的循环顺序是不同的。

说实话，我看V1论文中原始图1就觉得循环顺序很不合理，我盯着循环执行的zigzag顺序图看了一下午，百思不得其解。现在FA github readme里还不把这个图改过来，有点说不过去。

**另外，个人推测V2性能最大的提升就是来源是来自这个循环顺序的调换，

Tri Dao大佬把OpenAI Triton的作者Phil Tillet挂在coauthor位置一点也不过分。所以，看到很多报道FA V2的新闻都用“_斯坦福博士一己之力让Attention提速9倍！XXX_“的标题，细心的读者会觉得比较扎心。

随着循环顺序调换之后，一个thread block内warps粒度的划分也需要改进，作者把V1版本沿着K切分共享Q，改为沿着Q切分共享K。

另外作者做了提取公因式数学变换，减少了一些non-matmul FLOPs，调优了thread block size。最终更达到了最优情况下72% A100 FLOPs利用率的效果。

写到这，文章有点长了，笔者已经写不动了，对于推理的Flash Decoding优化我今后再单开一篇文章解读。

# RingAttention和FlashAttentionV2
对比来看RingAttention和FlashAttentionV2本质上是等价的，FlashAttentionV2是分子和分母分开计算，分别迭代更新，而RingAttention是分子分母一起计算，更像是FlashAttention V1的计算方式。其计算量综合来看是高于FlashAttetionV2。

补充说明一下，这里说的FlashAttentionV1的计算方式，并不是说FLashAttentionV1是用LSE符号表示来计算。

FlashAttentionV1同样用的是 l 和 m 符号，只不过是它每一步更新都会除以 l 来矫正，其实是没有必要的，迭代计算过程中上一步的 l ，和当前的 l 会消除掉，所以只需要计算最终的 l 即可，这也是FlashAttentionV2的优化。

但是从公式角度上看FlashAttentionV1即使没有用LSE的符号表示，它的计算量和LSE的方式相同，本质上是一样的。
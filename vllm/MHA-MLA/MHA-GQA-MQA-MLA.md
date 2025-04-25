MHA 多头注意力机制

MHA使用KV cache加速

但是KVcache过大，占用显存

（补充：在GPU上部署模型的原则是：能一张卡部署的，就不要跨多张卡；能一台机部署的，就不要跨多台机。这是因为“卡内通信带宽 > 卡间通信带宽 > 机间通信带宽”，由于“木桶效应”，模型部署时跨的设备越多，受设备间通信带宽的的“拖累”就越大，事实上即便是单卡H100内SRAM与HBM的带宽已经达到了3TB/s，但对于Short Context来说这个速度依然还是推理的瓶颈，更不用说更慢的卡间、机间通信了。）

减少KV Cache的目的就是要实现在更少的设备上推理更长的Context，或者在相同的Context长度下让推理的batch size更大，从而实现更快的推理速度或者更大的吞吐总量。当然，最终目的都是为了实现更低的推理成本。

实现方法-多种多样 不知道flash attention算不算

MQA  **M**ulti-**Q**uery **A**ttention
直接让所有Attention Head共享同一个K、V，

MHA与MQA之间的过渡版本GQA（**G**rouped-**Q**uery **A**ttention）
GQA的思想也很朴素，它就是将所有Head分为g个组（g可以整除h），每组共享同一对K、V，


MLA（**M**ulti-head **L**atent **A**ttention）
（因为我在简历中写了自己使用cuda完成MLA算子，所以这部分的理论和实现，需要重点的进行学习）
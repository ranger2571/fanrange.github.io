
KV cache的存在，本来是为了避免在推理阶段对前置序列的重复计算的。但是，**随着前置序列的长度变长**（我们记为kv_len），需要读取的**KV cache也将越来越大**，数据的传输成本增加，这就使得**attn计算逐渐变成memory bound**。**我们采取了一些策略来缓解KV cache过大的问题，其中2种就是大家熟知的MQA和GQA**。

- **在MQA的情况下，一个token所有的heads都共享同一个k和v**。这样在降低param weights大小的同时，还让原本需要保存num_heads份的kv cache降低为只需保存1份。
- 但是，MQA可能造成模型效果上的损失，毕竟原来对于1个token，它的每个head都有各自的k、v信息的，现在却被压缩为一份。**所以GQA作为一种折衷的方案出现了**，即将1个token的head分成num_group组，每个group内共享同一个k，v信息，使得信息压缩不像GQA那样严重。

**但是，不管是MQA还是GQA，对于1个token来说，总是存在heads上k、v信息被压缩的情况。那么是否有一种办法，能在尽量不压缩head上k，v信息的情况下，节省kv cache，提高整体推理速度呢**


文章来源：[vLLM官方文档Paged Attention解读 - 知乎](https://zhuanlan.zhihu.com/p/712664813)
### 输入和输出

```cuda
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale)
```

- `scalar_t`表示query，key，value的数据类型，例如FP16.
- `HEAD_SIZE`表示每个head中的element数；
- `BLOCK_SIZE`表示每个block中的token 数；
- `PARTITION_SIZE`表示张量并行的gpu数目。

**参数：**

- out：[num_seqs, num_heads, head_size]
- q：[num_seqs, num_heads, head_size]
- k_cache：[num_blocks, num_kv_heads, head_size/x, block_size, x] x表示一个向量化的大小，如float16 -> 16 / sizeof(float16) = 8
- v_cache：[num_blocks, num_kv_heads, head_size, block_size]
- head_mapping：[num_heads] 用于MQA, GQA，确定用的KV_head
- block_tables：[num_seqs, max_num_blocks_per_seq] block_tables映射表，表示每个sequence映射到哪几个block上
- context_lens：[num_seqs] 当前seq，已经推理出来key cache，value cache的长度

### **Kernel的一些常量定义**

- **Sequence**: 表示一个请求的句子。例如 `q` 的shape 是`[num_seqs,``num_heads,``head_size]`. 一共有个 `num_seqs` 个query sequence 通过 `q`指针来表示。由于这里我们考虑的是decoding阶段的single query kernel，所以每个sequence只有一个token，这里的 `num_seqs`就等于当前batch中token的总数。
- **Context**: 包括生成token的文本，例如`["What",``"is",``"your"]` 是 context tokens, input query token 是 `"name"`。模型生成的token是 `"?"`.
- **Vec**: 一起读取和计算的elements list，也就是向量化读取的vector。

	- 对于 query 和 key，vec size (`VEC_SIZE`) 取决于每个**[thread group](https://zhida.zhihu.com/search?content_id=246500574&content_type=Article&match_order=1&q=thread+group&zhida_source=entity)**能够一次读取16 bytes数据的data数量。例如，对于FP16（2 bytes），and THREAD_GROUP_SIZE = 2，那么 VEC_SIZE=16/2bytes/2threadgroup = 4，也就是一个thread一次读取4个query 和key data。
	- 对于value，vec size (`V_VEC_SIZE`) 取决于每个**thread**能够一次读取16 bytes数据的data数量。FP16数据，`V_VEC_SIZE`= 16/2bytes = 8。

- **Thread group**:
	- 是一组thread，其大小通过`THREAD_GROUP_SIZE`来设定。
	- 定义了一次可以读取和计算一个query token和一个key token的单元。每个thread只处理一个token data的部分数据。每个thread group处理的元素数量定义为 `x`。
	- 例如，一个thread group包含了2个thread，head size=8，thread 0 处理index为 0, 2, 4, 6 的head， thread 1处理 index 1, 3, 5, 7的head。
	- （之所以存在，就是一个q的token的数据量都很大了，一个head，一个token，hiddendim=128，那么一个token就需要128/4=32个vec才能一次性读取结束。而且）

- **Block**: kv cache的存储单元。对于一个head，每个 block 存储固定大小为`BLOCK_SIZE`数量的tokens。每个block可能只会包含整个context的部分token。例如，block size = 16，head size = 128, 对于一个head，一个 block 存储了16 * 128 = 2048个元素。
- **Warp**: 在一个流多处理器（SM）上，同时可以调度的一组thread数量，也就是`WARP_SIZE`，为32。每个[warp](https://zhida.zhihu.com/search?content_id=246500574&content_type=Article&match_order=1&q=warp&zhida_source=entity)一次迭代处理一个query token和一个block的key token之间的计算，如果是多轮迭代，就可以处理多个block。例如，如果一个context由4个warp和6个block，warp 0 处理 block0，block4，warp 1 处理 block1, block5；warp 2 处理block2，warp 3 处理block3。
- **Thread block**: 硬件上的block。是一组`NUM_THREADS`可以访问相同[shared memory](https://zhida.zhihu.com/search?content_id=246500574&content_type=Article&match_order=1&q=shared+memory&zhida_source=entity)的thread。每个[thread block](https://zhida.zhihu.com/search?content_id=246500574&content_type=Article&match_order=1&q=thread+block&zhida_source=entity)包含了`NUM_WARPS`个warp，每个thread block处理一个query token和整个context key之间的计算。
- **Grid**: kernel启动的所有线程。其shape定义为 `(num_heads,``num_seqs,``max_num_partitions)`.


**为什么key要被加载到寄存器，而query却是被加载到shared memory？**

首先看shared memory和register的scope：shared memory的scope是整个thread block，也就是说thread block内的每一条线程都可以拿到shared mem里面的东西，在这里就是q_vecs；但是寄存器的scope是thread，也就是说在key加载的时候，NUM_TOKENS_PER_THREAD个vecs仅被那条负责加载的线程看到。

再来看我们定义的问题处理的规模：每个thread block处理一个query token和整个context key之间的计算。也就是说，所有key token是和同一个query token做点积运算的，不同的key token被分配给了同一个thread block的不同的thread group，这些thread group要拿到同一个query token的值进行点积计算得出注意力分数，所以query必须对整个thread block内的不同thread group可见。假设我把query加载到寄存器里面，那就只有负责加载的线程能获取它自己负责的vecs，那些跟这条加载线程在不同的thread group的线程加载了某个key token，当这个key想要query来做点积的时候，根本就拿不到那条加载线程负责的query vecs，计算也就无法进行。

### **load key**

并行设计：三层循环。注意，这里K向量的数据是直接加载到每个线程的寄存器的。

1. 最外层循环，就是用**一个warp处理一个block**, 对一个SM来说，通常都是4个Wraps， 可以同时并行处理四个block。

2. 中间层循环，就是**一个thread group处理一个token**。因为每个block有多个token，所以一次iteration处理一个token。

如果warp_size > block size，那么一个thread group有多个thread，处理一个token；

如果warp_size < block size，那么一个thread group有一个thread，处理多个token。

3. 内层循环，就是**一次循环加载一个K向量的一部分, size为 x** 。注意，这里对VEC_SIZE做了一个变换，最后用的是x作为加载的size，不是VEC_SIZE。


**key的内存布局：**

K Cache的布局为[num_blocks, num_kv_heads, **head_size/x**, block_size, **x**]，这是为了优化写入shared memory的操作。

在Q和K矩阵的同一行元素被读入寄存器并进行点乘运算后，结果需要被存入shared memory。如果一个warp中所有线程都计算Q、K同一行数据，会导致写入shared memory的同一个位置，这将造成warp内不同线程顺序地写入。因此，为了优化，**warp的线程最好计算Q和K的不同行数据。**

因此，在设计K Cache布局时，我们将block_size放在比head_size更低的维度。由于warp size大于block_size，我们需要将head_size拆分为head_size/x和x两个维度，借x到最低维度，以确保每个线程读入的数据量和计算量都足够大。最后，每个线程组派一个线程去写入shared memory（也就是qk的结果logits在shared memory—），这样一个warp有blk_size个线程并行写入shared memory，从而增加了shared memory的访问带宽。这种设计策略是为了实现高效的并行计算和内存访问，以提高整体的计算性能。

  

如果warp_size > block size，那么一个thread group有多个thread，处理一个token；

t0，t1... 为thread 编号，相同颜色的方块表示同一个thread group，每次iteration处理一个token
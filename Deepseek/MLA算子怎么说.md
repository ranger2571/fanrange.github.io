# 将 MLA 展开成 MHA 进行的计算
V2论文的开源实现，

上图为论文中给出的 MLA 完整公式，

以下对应公式分析实际的开源实现 [modeling_deepseek.py](https://link.zhihu.com/?target=https%3A//huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py) 中非 flash attn 版本的 DeepseekV2Attention 算子实现。


### q向量

输入向量h，进行Q降维矩阵乘，对应上述公式中的 $W^{DQ}$ ，得到了。

对应的是q_a_proj， q_a_proj 大小为 [hidden_size, q_lora_rank] = [5120, 1536]。

对Q进行升维矩阵乘的时候，选择了合并矩阵乘的方式进行加速。

(提问，为什么q_b_proj的维度是这样的？？？答，是将两个矩阵乘，使用concat操作，合并为一个，来提速。计算好后再spilit)

q_b_proj 大小为 [q_lora_rank, num_heads * q_head_dim] = [q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)] = [1536, 128*(128+64)] = [1536, 24576] 对应上述公式中的 $W^{UQ}$ 和 $W^{QR}$ 合并后的大矩阵



### kv向量
输入向量h，进行KV降维矩阵乘，对应上述公式中的$W^{DKV}W^{KR}$，得到了 $c_t^{KV}$。

其中 kv_a_proj_with_mqa 大小为 [hidden_size， kv_lora_rank + qk_rope_head_dim] = [5120, 512 + 64] = [5120, 576]，对应上述公式中的 WDKV 和 WKR 。

kv_b_proj 大小为 [kv_lora_rank， num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)] = [512, 128*((128+64)-64+128)] = [512, 32768]，对应上述公式中的 WUK 和 WUV。由于 WUK 只涉及 non rope 的部分所以维度中把 qk_rope_head_dim 去掉了。

### 部分总结
到这里，其实思路很清晰，就是mla这种设计，公式不变，然后合并一些简单的可以合并的矩阵乘

### 下一部分，关于展开成MHA和继续优化的情况
不过从代码中我们也可以看到在开源实现中是展开成 MHA 存储的 KVCache，所以没有拿到这部分的收益。

## 优化实现

优化上述实现的核心在于实现论文中简单提了一句的”矩阵吸收“

、
![](https://pic3.zhimg.com/v2-ff3f90fd1314afa63a4a72e17f7ec912_1440w.jpg)

### $W^{UK}$ 吸收

比如对于 WUK 矩阵我们有

$atten_weights=q_t^⊤k_t=(W^{UQ}c_t^Q)^⊤W^{UK⊤}c_t^{KV}=c_t^{Q⊤}W^{UQ} W^{UK}c_t^{KV}=(c_t^{Q⊤}W^{UQ} W^{UK})c_t^{KV}$

也就是说我们事实上不需要将低维的ctKV展开再计算，而是直接将 WUK 通过结合律先和左边做乘法。

对应的实现代码如下



# MLA算子提速
为了充分发挥 MLA 的优势，本文首先详细分析了现有的开源实现，并探索了一种简单易改的“矩阵吸收”技巧的实现方法。测试结果显示优化后的 [DeepseekV2Attention](https://zhida.zhihu.com/search?content_id=243733609&content_type=Article&match_order=1&q=DeepseekV2Attention&zhida_source=entity) 算子实现可以实现单算子十倍以上的提速。


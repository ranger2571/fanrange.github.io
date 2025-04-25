但是前面说的都是训练的部分，推理事实上不一定需要MLA，Deepseek的推理就是MHA和MQA结合

[深度解析FlashMLA: 一文读懂大模型加速新利器 - 知乎](https://zhuanlan.zhihu.com/p/27976368445)
[flashMLA 深度解析 - 知乎](https://zhuanlan.zhihu.com/p/26080342823)
FlashMLA是一种在变长序列场景下的加速版MLA（[Multi-Head Linear Attention](https://zhida.zhihu.com/search?content_id=254617878&content_type=Article&match_order=1&q=Multi-Head+Linear+Attention&zhida_source=entity)），针对decoding阶段优化。目前deepseek已将其开源：[FlashMLA](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/FlashMLA)，其主关键内容：

**性能**（H800 SXM5 CUDA12.8）：**3000 GB/s HBM带宽；580 TFLOPS吞吐**

**特点**：

- 存算优化：双warp group计算流设计与应用（存算重叠、数据双缓存）；
- 分页缓存：KV page block管理提升显存利用率，更好适应变长序列；
- [SM负载均衡](https://zhida.zhihu.com/search?content_id=254617878&content_type=Article&match_order=1&q=SM%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1&zhida_source=entity)：动态调整block数据，充分利用GPU算力。

当前代码仓的限制条件：

- 针对[NVIDIA Hopper架构](https://zhida.zhihu.com/search?content_id=254617878&content_type=Article&match_order=1&q=NVIDIA+Hopper%E6%9E%B6%E6%9E%84&zhida_source=entity)GPU优化
- 数据类型：BF16/FP16
- 分页缓存管理粒度：64-block
- 仅包含推理场景


## 1 计算原理分析

### 1.1 计算公式

**FlashMLA这个库解决的是什么计算问题?**

MLA计算主要包含升/降秩线性计算和attention计算部分，FlashMLA完成MLA中MHA计算部分，不负责升/降秩的线性乘法操作。MLA的结构如下图所示：



计算MLA的MHA和通常的MHA计算存在差异，先来分析如下问题：

**问题1：FlashMLA的计算公式输入结构组成是什么样的？**

一般而言，计算MHA时需要Q/K/V三个输入值，而MLA由于引入升降秩操作，算MHA时输入值发生了变化。 MLA的公式如下，FlashMLA完成（公式46）的计算。

在deepseekV2中有提到矩阵W可以调整，具体是 WUK 转移到WUQ计算中，WUV转移到WO计算中。

### 1.2 Attention分块运算

输入、输出明确后需要对KQV进行分块计算（按照FlashAttention类型原理）， FlashMLA的分块逻辑如下：


大致步骤：

- 从Q取q_block单位，从K取k_block单位完成qk运算'、softmax运算得到p_block;

- 从V取v_block单位，然后分块成两份，分别与p_block计算得到o_block0和o_block1刷新到结果O上；

- 从V取v_block单位，然后分块成两份，分别与p_block计算得到o_block0和o_block1刷新到结果O上；

## 2 计算流程分析

**怎么利用Hopper架构提速分块MLA的计算过程，使其达到“Flash”的标准？**

回答这个问题需要先了解一下hopper架构的一个例子（[cutlass库](https://zhida.zhihu.com/search?content_id=254617878&content_type=Article&match_order=1&q=cutlass%E5%BA%93&zhida_source=entity)）：**[Ping-Pong](https://link.zhihu.com/?target=https%3A//pytorch.ac.cn/blog/cutlass-ping-pong-gemm-kernel/)**计算方式。

> 技术上称为"sm90_gemm_tma_warpspecialized_pingpong"，采用异步流水线运行，利用 warp 专精化。与更经典的同构内核不同，"warp 组"承担专门的角色。请注意，一个 warp 组由 4 个 warp 组成，每个 warp 有 32 个线程，总共 128 个线程。 Warp是GPU中的基本执行单元,由32个线程组成，Warp group是多个warp的集合用于协同工作（理解warp原理参看：[GPU硬件分析](https://zhuanlan.zhihu.com/p/508862848)的第三节）

该操作采用生产者（Producer）、消费者（Consumer）模式。Cutlass的Ping-Pong例子中包含1个生产者、2个消费者，如下图所示，生产者专门负责搬运数据，消费者完成计算。采用这种模式能够更充分的利用TensorCore。
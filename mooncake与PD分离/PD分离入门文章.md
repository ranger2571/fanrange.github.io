[LLM推理优化 - Prefill-Decode分离式推理架构 - 知乎](https://zhuanlan.zhihu.com/p/9433793184)
## 什么是 [Prefill-Decode 分离](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=Prefill-Decode+%E5%88%86%E7%A6%BB&zhida_source=entity)？

在传统的 LLM 推理框架中，Prefill 和 Decode 阶段通常由同一块 GPU 执行。推理引擎的调度器会根据显存使用情况及请求队列状态，在 Prefill 和 Decode 之间切换，完成整个推理过程。

而在 Prefill-Decode 分离式架构（以下简称 PD 分离式架构）中，这两个阶段被拆分到不同的 GPU 实例上独立运行。如下图所示，这是 DistServe 提供的一张架构图：

在 PD 分离式架构中：

- Prefill Instance 专注于 Prefill 阶段的计算。
- Decode Instance 专注于 Decode 阶段的生成任务。

当 Prefill Instance 完成 [KV Cache](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=KV+Cache&zhida_source=entity) 的计算后，会将其传输给 Decode Instance，后者接续生成结果。这种架构独立优化了两个阶段的性能，因此又被简称为 PD 分离。

## 为什么需要 Prefill-Decode 分离？

在大模型推理中，常用以下两项指标评估性能：

- [TTFT](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=TTFT&zhida_source=entity)（Time-To-First-Token）：首 token 的生成时间，主要衡量 Prefill 阶段性能。
- [TPOT](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=TPOT&zhida_source=entity)（Time-Per-Output-Token）：生成每个 token 的时间，主要衡量 Decode 阶段性能。

当 Prefill 和 Decode 在同一块 GPU 上运行时，由于两阶段的计算特性差异（Prefill 是计算密集型，而 Decode 是存储密集型），资源争抢会导致 TTFT 和 TPOT 之间的权衡。例如：

- 若优先处理 Prefill 阶段以降低 TTFT，Decode 阶段的性能（TPOT）可能下降。
- 若尽量提升 TPOT，则会增加 Prefill 请求的等待时间，导致 TTFT 上升。

PD 分离式架构的提出正是为了打破这一矛盾。通过将 Prefill 和 Decode 分离运行，可以针对不同阶段的特性独立优化资源分配，从而在降低首 token 延迟的同时提高整体吞吐量。

## 分离式推理架构的优化方向

### 1. 算力与存储的独立优化

在 PD 分离架构中，Prefill 和 Decode 阶段的资源需求不同，分别体现为：

- Prefill 阶段：计算密集型（compute-bound）。在流量较大或用户提示长度较长时，Prefill 的计算压力更大。完成 KV Cache 的生成后，Prefill 阶段本身无需继续保留这些缓存。
- Decode 阶段：存储密集型（memory-bound）。由于逐 token 生成的特性，Decode 阶段需频繁访问 KV Cache，因此需要尽可能多地保留缓存数据以保障推理效率。

因此，在 PD 分离架构下，可以分别针对计算和存储瓶颈进行优化。

### 2. [Batching 策略](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=Batching+%E7%AD%96%E7%95%A5&zhida_source=entity)的独立优化

在 DistServe 的实验中，Batching 策略对两阶段的性能影响显著，但趋势相反：

- Prefill 阶段：吞吐量随 batch size 增加逐渐趋于平稳。这是因为 Prefill 的计算受限特性（compute-bound），当 batch 中的总 token 数超过某个阈值时，计算资源成为瓶颈。
- Decode 阶段：吞吐量随 batch size 增加显著提升。由于 Decode 阶段的存储受限特性（memory-bound），增大 batch size 可提高计算效率，从而显著增加吞吐量。

下图展示了两阶段吞吐量随 batch size 变化的趋势：

### 3. 并行策略优化

在 PD 合并架构中，Prefill 和 Decode 阶段共享相同的并行策略（如数据并行 DP、[张量并行](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C&zhida_source=entity) TP 或[流水线并行](https://zhida.zhihu.com/search?content_id=250909492&content_type=Article&match_order=1&q=%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C&zhida_source=entity) PP）。但在 PD 分离架构中，可分别为两个阶段选择最优的并行策略。

DistServe 的实验结果显示：

- Prefill 阶段：
- 在请求率较小时，更适合张量并行（TP）。  
    
- 在请求率较大时，更适合流水线并行（PP）。  
    
- Decode 阶段：
- GPU 数量增加时，PP 可显著提高吞吐量（因为其处理方式是流水线化的）。  
    
- TP 则可降低延迟（减少单个请求的处理时间）。
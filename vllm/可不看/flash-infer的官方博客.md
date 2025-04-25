[Accelerating Self-Attentions for LLM Serving with FlashInfer | FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
# flash-infer 0.1
## Attentions in LLM Serving

There are three generic stages in LLM serving: _prefill_, _decode_ and _append_. During the prefill stage, attention computation occurs between the KV-Cache and all queries. In the decode stage, the model generates tokens one at a time, computing attention only between the KV-Cache and a single query. In the append stage, attention is computed between the KV-Cache and queries of the appended tokens. _append_ attention is also useful in [speculative decoding](https://arxiv.org/abs/2211.17192): the draft model suggests a sequence of tokens and the larger model decides whether to accept these suggestions. During the attention stage, proposed tokens are added to the KV-Cache, and the large model calculates attention between the KV-Cache and the proposed tokens.
在 LLM 服务中，有三个通用阶段：_预填充_、_解码_ 和 _追加_。在预填充阶段，注意力计算发生在键值缓存和所有查询之间。在解码阶段，模型逐个生成标记，仅在键值缓存和单个查询之间计算注意力。在追加阶段，注意力计算发生在键值缓存和追加标记的查询之间。_追加_ 注意力在推测性解码中也很有用：草稿模型提出一系列标记，而大型模型决定是否接受这些建议。在注意力阶段，建议的标记被添加到键值缓存中，大型模型计算键值缓存和建议标记之间的注意力。
The crucial factor affecting the efficiency of attention computation is the length of the query ($l_q$), determining whether the operation is compute-bound or IO-bound. The operational intensity (number of operations per byte of memory traffic) for attention computation is expressed as $O\left(\frac{1}{1/l_q + 1/l_{kv}} \right)$, where $l_{kv}$ represents the length of the KV-Cache. During the decode stage, where $l_q$ is consistently 1, the operational intensity is close to $O(1)$, making the operator entirely IO-bound. In the append/prefill stages, the attention operational intensity is approximately $O(l_q)$, leading to compute-bound scenarios when $l_q$ is substantial.
影响注意力计算效率的关键因素是查询的长度，这决定了操作是计算密集型还是 IO 密集型。注意力计算的操作强度（每字节内存流量的操作数）表示为$O\left(\frac{1}{1/l_q + 1/l_{kv}} \right)$, where $l_{kv}$ 代表键值缓存的长度。在解码阶段，由于 $l_q$始终为 1，操作强度接近$O(1)$, 使操作完全受 IO 限制。在追加/预填充阶段，注意力操作强度约为 $O(l_q)$, 当 $l_q$ 较大时，场景变为计算密集型。

下图展示了预填充、追加和解码阶段的注意力计算过程：

[https://flashinfer.ai/assets/imgs/llm-attentions.png](https://flashinfer.ai/assets/imgs/llm-attentions.png)
图 1：解码注意力逐行填充注意力图，预填充注意力在因果掩码下填充整个注意力图，而追加注意力填充梯形区域。

下图展示了三个阶段注意力计算的roofline模型。解码注意力性能始终低于峰值带宽上限（受 GPU 峰值内存带宽限制），因此是 IO 密集型。预填充注意力具有高操作强度，处于峰值计算性能上限之下（受峰值浮点性能限制）。追加注意力在查询长度较小时为 IO 密集型，查询长度较大时为计算密集型。
图 2：LLM 服务中注意力操作符的屋顶模型，数据来自 A100 PCIe 80GB。
### 单请求与批量处理

服务 LLM 模型的两种常见方式是批量处理和单请求。 批量处理将多个用户请求分组在一起并并行处理，以提高吞吐量。然而，注意力内核的操作强度与批量大小无关，批量解码注意力的操作强度仍为O(1)。

## FlashInfer 概览

FlashAttention 提出将多头注意力融合到一个内核中，通过将在线 softmax 技巧推广到自注意力，从而避免在 GPU 全局内存中显式存储注意力矩阵的开销。FlashAttention2 通过采用更合理的分块策略和减少非张量操作的数量进一步提升性能，以缓解 A100/H100 非张量核心性能较低的问题。vLLM 提出 PageAttention，将键值缓存组织为页表，以缓解 LLM 服务中的内存碎片问题。

FlashInfer 在多种键值缓存格式（如不规则张量和页表）上实现了单请求和批量版本的 FlashAttention，涵盖预填充、追加和解码的所有三个阶段。对于单请求解码/预填充和批量解码内核，FlashInfer 实现了单请求解码/预填充和批量解码内核的行业领先性能。此外，FlashInfer 还实现了 _分页键值缓存的预填充/追加内核_，这是现有库尚未完成的工作，并可用于服务推测性解码设置中的模型。

许多近期工作提出了键值缓存压缩技术以减少内存流量。为此，FlashInfer 优化了 _分组查询注意力_、_融合 RoPE 注意力_ 和 _量化注意力_ 的内核，以高效服务压缩的键值缓存：

- **分组查询注意力**：分组查询注意力使用较少的键和值头，从而节省内存流量。分组查询注意力的操作强度从 增长到 ，其中 是查询头的数量， 是键和值头的数量。像 A100/H100 这样的 GPU 非张量核心性能较低，因此传统的分组查询注意力实现是计算密集型的。FlashInfer 提出在分组查询注意力的解码阶段使用预填充（多查询）注意力内核（利用张量核心），与 vLLM 实现相比，速度提升高达 2-3 倍。
    
- **融合 RoPE 注意力**：RoPE（旋转位置嵌入）已成为 Transformer 的标准组件，大多数现有的服务系统在键值缓存中存储应用旋转嵌入后的键（post-RoPE 键）。然而，像 StreamingLLM 这样的近期工作会修剪键值缓存中的标记，修剪后标记的位置会发生变化，因此键值缓存中的 post-RoPE 键变得无意义。在这种情况下，FlashInfer 提出在键值缓存中存储 pre-RoPE 键，并将 RoPE 融合到注意力内核中。在各种平台和设置的实验表明，FlashInfer 的融合 RoPE 注意力内核可以即时应用 RoPE，几乎无开销。
    
- **量化注意力**：另一种压缩键值缓存的方法是通过剪枝，FlexGen 和 Atom 表明可以将键值缓存剪枝到 4 位，且准确率损失可忽略不计。FlashInfer 实现了低精度注意力内核，从而几乎可以实现与压缩比率成线性的加速（4 位约为 4 倍，8 位约为 2 倍）。
    

像 LightLLM 和 sglang 这样的近期工作使用了一种特殊的 PageAttention 形式，其中页大小等于一，以便在结构化生成等复杂服务场景中轻松管理键值缓存。FlashInfer 通过在 GPU 共享内存中预取页索引来优化 PageAttention 内核，从而使内核性能不受页大小的影响。

在后续部分中，我们将深入探讨 FlashInfer 实现的详细优化和基准测试结果。

# FlashInfer 0.2 - 高效且可定制的 LLM 推理服务内核

经过四个月的开发，我们很高兴地宣布 **FlashInfer 0.2** 的发布。此次重大更新带来了性能提升、增强的灵活性和关键的错误修复。此次发布的亮点包括：

- 使用 **FlashAttention-3 模板** 实现更快的稀疏（分页）注意力
- 注意力变体的 **JIT 编译**
- 支持 **多头潜在注意力（MLA）解码**

## FlashAttention-3 模板与块/向量稀疏性

FlashAttention-3 通过巧妙地重叠 softmax 和矩阵乘法，为 Hopper GPU 带来了突破性的优化。FlashInfer 0.2 集成了 FA-3 模板，在 Hopper 架构上显著提升了预填充注意力性能。

### 灵活的块稀疏性和向量稀疏性

FlashInfer 的突出特点是其高度灵活的块稀疏 FlashAttention 实现，支持 **任意块大小配置**。我们的 PageAttention 操作符实现为 **块稀疏注意力内核**，其中 `page_size` 指定块的列数。在最细粒度上，FlashInfer 支持 **向量稀疏性** 1（`page_size=1`），允许精确的内存管理（在 sglang 中使用）和高效的 KV-Cache 令牌修剪。

通过利用 CuTe 的 `CustomStride` 和 `ComposedLayout` 抽象，我们将向量稀疏性扩展到 FlashAttention-3。受 CUTLASS 的 gather/scatter 卷积的启发，这通过简单修改生产者的内存加载模块实现。

### 性能基准测试

我们比较了两种注意力实现：`page_size=1` 的 PageAttention（使用向量稀疏注意力实现）和可变长度的密集注意力，在 FA-2（v0.1.*）和 FA-3（v0.2）后端下，对相同的问题大小进行基准测试。基准测试使用 `head_dim=128`，`causal=True`，变化的批量大小 `(B)` 和序列长度 `(L)`，输入 Q/K/V 张量采用高斯初始化。

![img](https://flashinfer.ai/assets/imgs/fa3-template.png)

在 H100 SXM5 上，FA-2 和 FA-3 模板下密集/向量稀疏注意力的性能比较，使用 CUDA 12.4 编译。y 轴：不同设置，x 轴：实现的 TFLOPs/s。

**结果**：在相同条件下，向量稀疏注意力实现了密集注意力吞吐量的 90%。FA-3 后端始终优于 FA-2。得益于 FlashInfer 稳定的 API，从 FA-2 升级到 FA-3 无需更改代码——只需安装 FlashInfer 0.2。重现这些结果的参考基准测试脚本可在此处找到。

## 注意力定制的 JIT 编译

受 FlexAttention 的启发，FlashInfer 0.2 引入了可定制的编程接口以编译不同的注意力变体。我们在 CUDA/Cutlass 中设计了一个模块化的注意力模板。用户可以通过在注意力变体类中指定 `LogitsTransform`/`QueryTransform` 等函数来定义自定义注意力变体。类字符串将专门化我们预定义的 Jinja 模板，FlashInfer 使用 PyTorch 的 JIT 加载函数来编译和缓存这些内核。像 FlashSigmoid 这样的新变体可以用最少的代码实现。更多案例请参阅我们的 JIT 示例。

![img](https://flashinfer.ai/assets/imgs/jit.png)

左侧：FlashInfer 中的 JIT 工作流程。右侧：编译新的注意力变体。

除了支持新的注意力变体外，FlashInfer 支持 JIT 的其他好处包括：

- **减小轮子大小**：由于我们预编译了所有注意力变体的组合，FlashInfer 的二进制大小在最近的版本中呈指数增长。为了使轮子大小可控，我们不得不减少专门化，这会损害内核性能（如在 #602 中观察到的，FlashInfer v0.1.6 的预填充性能甚至比 FlashInfer v0.1.1 更差，因为我们把编译时参数移到了运行时，损害了性能）。FlashInfer v0.2 通过仅预编译一部分 **核心** 内核来解决这个问题，而大多数注意力变体则通过 JIT 编译。
- **轻松开发**：对于小的 CUDA 更改，无需重新安装 FlashInfer，只需以 JIT 模式安装 FlashInfer。

我们通过最小化头文件依赖和利用分拆编译优化了 JIT 编译的速度。因此，所有 Llama 模型的内核都可以在服务器级 CPU 上在 **15 秒内** 完成 JIT 编译。更多详细信息，请参阅我们的 JIT 预热脚本。

## 融合多头潜在注意力（MLA）解码内核

**多头潜在注意力（MLA）** 在 Deepseek v2 中引入，通过将 KV-Cache 投影到低秩矩阵来压缩它。由于缺乏优化的内核，实现 MLA 的高吞吐量具有挑战性。FlashInfer 社区最近通过 **矩阵吸收技巧** 实现了一个融合内核，提高了内存效率。详细解释请参阅 #551。

![img](https://flashinfer.ai/assets/imgs/mla.png)

FlashInfer 中 MLA 解码内核的工作流程

未来计划包括通过张量核心加速 MLA 解码，从而惠及推测性解码。

## 对可变长度输入的 CUDAGraph 兼容性

FlashInfer 0.2 修复了在捕获和回放阶段查询长度不同时，预填充注意力与 CUDAGraph 的不兼容问题，通过准确估计上资源界限。现在可以使用 CUDAGraph 加速使用 FlashInfer 内核的推测性解码和分块预填充工作负载。

## 与 torch.compile 兼容

FlashInfer 0.2 遵循 PyTorch 自定义操作符标准，确保与 **torch.compile** 兼容。

## 打包和 CI/CD

我们现在提供每日构建，以便用户无需等待稳定版本即可测试最新功能。

## 其他值得注意的改进

#### FusedAddRMSNorm 修复

修复了 `FusedAddRMSNorm` 中的数值问题，这可能导致某些模型输出不良。

#### 集成 Cutlass 的 SM90 分组 GEMM

我们将 **Cutlass 3.5 SM90 分组 GEMM** 集成到我们的 SegmentGEMM API 中，加速了 LoRA 和 MoE 服务。

#### 支持非连续 KV-Cache

KV-Cache 现在可以利用非连续存储布局，改善了对卸载的支持。

#### 更快的 `plan` 函数

`plan` 函数现在使用非阻塞的主机到设备内存传输，提高了性能。在 FlashInfer v0.2 之后，建议在 `plan` 函数中传递 **主机张量** 而不是设备张量，以减少同步。

#### KV-Cache 追加优化

通过将每个元素并行化而不是每个请求并行化，提高了小批量大小的 KV-Cache 追加吞吐量。一个新的 API，get_batch_indices_positions，支持这一点。注意，我们对这个 API 做了一些破坏性更改，以适应不同的并行化模式。有关新 API 使用的基准测试，请参阅我们的基准测试。

#### 标准化 RoPE 接口

我们标准化了 RoPE 接口，以与其他框架保持一致。FlashInfer 采用了 **fp32 sin/cos** 计算，以避免数值问题。

## 路线图

我们感谢社区的热爱和支持。为了提高透明度，我们发布了开发路线图，您可以在其中提供反馈并影响 FlashInfer 的未来。

自 v0.1.6 以来，贡献者人数从 41 人增加到 52 人。我们感谢以下开发者的贡献：

- @yzh119：JIT、FA-3 模板等
- @abcdabcd987：torch.compile 支持、打包
- @nandor：可变长度 CUDAGraph 支持
- @ur4t：打包、CI/CD
- @zhyncs：每日构建、CI/CD
- @tsu-bin：MLA 解码
- @xslingcn：Cutlass 分组 GEMM
- @yuxianq：JIT、错误修复
- @LinHeLurking：非连续 KV-Cache
- @Abatom：FusedAddRMSNorm 修复
- @jeejeelee：分组 GEMM 错误修复
- @mvpatel：更快的 `plan` 函数
- @Ubospica：预提交设置
- @dc3671：改进的单元测试
- @Pzzzzz5142：JIT 编译修复
- @reyoung：错误修复
- @xiezhq-hermann：ARM 编译修复
- @Bruce-Lee-LY：性能优化
- @francheez：拼写错误修复

![img](/assets/imgs/FlashInfer-white-background.png)![img](/assets/imgs/flashinfer-v02.jpg)
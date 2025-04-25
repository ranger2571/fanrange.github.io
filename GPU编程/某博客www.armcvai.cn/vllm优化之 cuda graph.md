## 一 vllm 中的 cuda graph

在 cpu 进行 cuda kernel launch（启动）的 eager mode 时，其存在可较大的 kernel launch 开销，尤其是对于 batch较小/模型较小的情况，kernel 启动开销过大！由此，作者基于 torch 使用 cuda graph 来直接减少 kernel launch 的开销。

> 这里解释下 **cuda kernel launch，其实际是 CPU 向 GPU 发送指令的过程，即算子下发过程**！

CUDA Graph 是 NVIDIA 在 CUDA 10 中引入的一种新特性，旨在优化 GPU 上的任务提交和执行流程。通过将一系列 CUDA 操作（如内核启动、内存拷贝等）表示为**一个图结构**，并在 GPU 上执行该图，可以显著减少 CPU 与 GPU 之间的通信开销，提高整体性能。

因为 cuda graph 必须限制输入输出的的形状，而在 llm 推理中，prefill 阶段输入的 batch，seq_len，这两个维度都是动态变化的，因此 cuda graph 只能应用在 `decode` 阶段，同时需要提前预设一批 `batch`，针对每个不同 `batch` `capture` 不同的 `cuda graph`，运行时根据输入的 `batch` 找到匹配的 cuda_graph_runner 即可。当然，这也会带来一个问题，就是预设的 batch 数越多，使用 cuda graph 优化后带来的额外显存消耗也增加。

> 值得一提的是，如果想看作者实现的 cuda graph 的思路来源，可以参考文章 [Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning) 和 llama 仓库代码 [llama-cuda-graph-example-masked_attn](https://github.com/fw-ai/llama-cuda-graph-example/blob/masked_attn/llama/generation.py#L123)。

在代码实现上，主要是基于 `GPUModelRunnerBase`、`ModelRunner` 和 `CUDAGraphRunner` 两个类，github 完整代码在[这里](https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py#L1730)，作者 commit 提交历史记录关键的 commit 在[这里](https://github.com/vllm-project/vllm/pull/1926/commits/87e565488aac528f6dd2161e9355d950fa74bfd1)。
### 1.1 总结

`cuda graph` 解决了可能存在的所有 CPU 开销的来源：**如用户编写的逻辑、PyTorch 调度逻辑、内存分配开销以及 GPU 驱动/内核开销**（静态图优势）。

## 二 vllm 的 cuda graph 源码剖析

下面是我针对 `vllm` 仓库上应用 cuda graph 的代码分析。

1，`ModelRunner` 类实际是对之前 `Model` 类的包装，封装了模型的前向传播逻辑，并管理与推理相关的资源。其中推理执行函数 `execute_model` 是核心，其负责接收输入数据，执行模型的前向传播，输出结果是 `sample` 采样后的结果，而不是模型推理结果 `logits`。`ModelRunner` 类的初始化如下所示：

```python
class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    带有采样步骤的 GPU 模型运行器。

    继承自 GPUModelRunnerBase，使用 ModelInputForGPUWithSamplingMetadata 作为模型输入类型。
    """
    # 指定模型输入的类型
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = ModelInputForGPUWithSamplingMetadata
    _builder_cls: Type = ModelInputForGPUBuilder  # 假设存在ModelInputForGPUBuilder 类
```

`ModelRunner` 类的继承比较复杂，但这对于支持众多模型和推理模式的 vllm 来说是无可厚非的，如果是自己想要测试下 cuda graph，初始化很简单，只要如下所示：

```
class ModelRunner():
    def __init__(self, model):
        self.model = model
        self.graph_runners = {}  # (int, CUDAGraphRunner)
```

`execute_model` 函数执行模型的前向传播和采样步骤，其中和 cuda graph 相关的代码如下所示:

```
# 当前 CUDA 图仅支持解码阶段
assert model_input.attn_metadata is not None
prefill_meta = model_input.attn_metadata.prefill_metadata
decode_meta = model_input.attn_metadata.decode_metadata
# 如果需要 CUDA 图，并且没有填充元数据，使用预构建的模型执行器
virtual_engine = model_input.virtual_engine
if prefill_meta is None and decode_meta.use_cuda_graph:
	assert model_input.input_tokens is not None
	graph_batch_size = model_input.input_tokens.shape[0]
	model_executable = self.graph_runners[virtual_engine][graph_batch_size]
else:
	model_executable = self.model
```

其中 `prefill_metadata` 和 `decode_metadata` 都是一个抽象属性，需要在子类中实现此方法。代表的意义分别是指应返回预填充注意力阶段所需的元数据，和应返回解码注意力阶段所需的元数据。

> 在 vllm 中，实现了多种 “元数据” 用来描述注意力机制在预填充和解码阶段各方面的辅助信息。这些信息对于管理令牌的处理和存储至关重要，尤其是在处理多模态数据或需要精确控制令牌位置的复杂模型中，具体代码在 `vllm/attention/backends/abstract.py` 中。

到这里，还有一个问题就是 `self.graph_runners` 是在哪里完成初始化的，按照学习 cuda graph 的经验，`self.graph_runners` 一般是在 `capture_model` 函数中完成初始化。且以往简单模型推理场景我们习惯把 `capture_model` 函数定义在 `ModelRunner` 类中，但在 vllm 中，`capture_model` 是定义在 `GPUModelRunnerBase` 中。

2，`GPUModelRunnerBase` 类是一个**基类**，它提供了模型加载、初始化等核心功能，为子类如 `ModelRunner` 和 `CUDAGraphRunner` 提供了通用的操作接口。和 cuda graph 优化相关的是 `capture_model` 函数，其核心实现逻辑代码如下所示:

```
# 代码有所省略
_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [_BATCH_SIZE_ALIGNMENT * i for i in range(1, 1025)

class GPUModelRunnerBase():
	def __init__():
		self.max_batchsize_to_capture = _get_max_graph_batch_size(self.scheduler_config.max_num_seqs)
		# 为了方便理解代码，去除了外层的 virtual_engine，直接以 batch_size 为键
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}

	@torch.inference_mode()
	def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
		# 创建全零的输入标记张量，形状为 (max_batch_size,)
		input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
		# 创建全零的输入位置张量，形状为 (max_batch_size,)
		input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
		
        # 获取要捕获的批量大小列表，确保批量大小不超过最大批量大小
		graph_batch_size = self.max_batchsize_to_capture
		batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size]
        # 使用注意力状态的图捕获上下文和通用的图捕获上下文
        with self.attn_state.graph_capture(max_batch_size), graph_capture() as graph_capture_context:
			# 反向遍历批量大小列表，从大到小
            for batch_size in reversed(batch_size_capture_list):
				# 获取指定批量大小的注意力元数据
                attn_metadata = self.attn_state.graph_capture_get_metadata_for_batch(
                    batch_size,
                    is_encoder_decoder_model=self.model_config.is_encoder_decoder
                )
				# 创建 CUDA 图运行器
                graph_runner = CUDAGraphRunner(
                    self.model,
                    self.attn_backend.get_name(),
                    self.attn_state.graph_clone(batch_size),
                    self.model_config.is_encoder_decoder
                )
				# 准备捕获输入的字典
                capture_inputs = {
                    "input_ids": input_tokens[:batch_size],
                    "positions": input_positions[..., :batch_size],
                    "intermediate_inputs": intermediate_inputs[:batch_size] if intermediate_inputs is not None else None,
                    "kv_caches": kv_caches[virtual_engine],
                    "attn_metadata": attn_metadata,
                    "memory_pool": self.graph_memory_pool,
                    "stream": graph_capture_context.stream
                }

				# 在前向传播上下文中，捕获模型的执行过程
                with set_forward_context(attn_metadata):
                    graph_runner.capture(**capture_inputs)

                # 更新图内存池
                self.graph_memory_pool = graph_runner.graph.pool()
                # 将捕获的图运行器存储到 graph_runners 中
        		self.graph_runners[batch_size] = graph_runner
```

`SchedulerConfig` 是调度配置类，`max_num_seqs` 是 `SchedulerConfig` 类的初始化参数之一，表示单次迭代中可处理的最大序列数，可以理解为传统意义上的 `max_batch_size`。`batch_size_capture_list` 是一个 batch_size 列表，前三个元素是 1 2 4 后面的元素值间隔 8 并小于调度类中的 `max_num_seqs` 参数值。

`capture_model` 函数流程可总结如下:

1. 创建虚拟输入数据。
2. 确定要捕获的批量大小列表。
3. 开始 `CUDA Graph` 捕获上下文，并循环处理不同的批量大小，捕获 `CUDA Graph`:
    - 创建 `CUDA Graph Runner`：创建 CUDAGraphRunner 对象 `graph_runner`;
    - 准备用于捕获的输入：将之前创建的虚拟输入和其他必要的参数组合成一个字典 `capture_inputs`;
    - 捕获 CUDA Graph：在前向传播上下文中，使用 `graph_runner.capture(**capture_inputs)` 捕获模型的执行过程;
    - 存储捕获的 `CUDA Graph`: self.graph_runners[batch_size] = graph_runner。
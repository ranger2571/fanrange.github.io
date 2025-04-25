## 一、架构优化

### 1. 整体架构优化

V1的整体模块化设计如下图所示，和V0比较最大的区别是：

1. 将preprocess和postprocess的逻辑给模块化了。随着对各种不同类型的模型支持，不同模型表现出的前后处理差异化越来越大，尤其是多模态模型，那么将这些逻辑抽象出来单独处理可以进一步增强可扩展性。
2. 添加EngineCore来执行主循环，专注于调度器和模型执行器，核心推理部分没有什么设计上的变化，仍旧是沿用Executor->Worker->ModelRunner->Model的路线。


### 2. 并行推理架构优化

在vLLM V0中，调度器和Worker 0位于同一进程，这样做是为了在向各个工作进程广播输入数据时减少进程间的通信开销，但这种设计导致了不对称架构，增加了系统复杂性。

vLLM V1舍弃了这种方式，通过在Worker侧缓存请求状态并仅传输增量更新，实现了对称架构，简化了分布式逻辑，使Worker在单GPU和多GPU环境中以相同方式运行，如下图所示。（这个应该就是多机多卡的情况下，）

调用流程如下所示：

1. `Executor`将请求信息通过`collective_rpc`添加到消息队列`rpc_broadcast_mq`中。
2. `WorkerProc`在loop中不断从`rpc_broadcast_mq`队列中获取请求信息，并交由`Worker`->`ModelRunner`处理。
3. `WorkerProc`处理完成后将请求消息添加到消息队列`worker_response_mq`中。
4. `Executor`从消息队列`worker_response_mq`中取出结果信息进行后续操作。


## 二、性能优化

### 1. CPU overhead剥离（和ktransformer的将moe的）

这个优化是从SGLang中借鉴的，目前已成为各大推理框架的标配。

为了减少昂贵的GPU等待时间，将**CPU的操作和GPU的操作拆分到两个进程中并行执行，使用[ZeroMQ](https://zhida.zhihu.com/search?content_id=253989199&content_type=Article&match_order=1&q=ZeroMQ&zhida_source=entity)来进行进程间通信**。这种设计使得CPU密集型任务，如分词、多模态输入处理、分词输出和请求流式传输，能够与核心执行循环并行运行，从而最大化模型的吞吐量。

执行流程如下图所示：

1. 进程0主要负责前后处理和流式传输，通过`input_socket`将请求发送给进程1。
2. 进程1在接收到请求后将其放入输入队列中。
3. `engine_core`的主循环会不断从输入队列中取出请求，处理完成后将其放入输出队列中。
4. 进程1再从输出队列中取结果通过`output_socket`将其发送回进程0，完成后处理后即可返回用户。


### **2. 调度器简化**

调度器最大的变化来源于应用chunked-prefill功能后削弱了prefill和decode阶段的区别，使得vllm不再区分两者的差异而用同样的方式来进行调度。调度决策以一个简单的字典形式表示，例如{请求ID: token数量}，这使得调度器能够支持chunked-prefill、prefix-cache和speculative decode等功能。例如，chunked-prefill可以动态分配每个请求的token数量，提升处理效率。

如下图所示，step0把R1、R2的完整prefill token和R3的部分prefill token组batch计算，step1和step2把R1、R2的decode和R3的部分prefill token 组batch计算， step3把R1、R2和R3的detoken部分组batch计算，这样每次计算的token可能是不同request的prefill阶段和decode阶段组合。


由于不需要考虑prefill和decode阶段的差异性，scheduler的整体逻辑也简化了很多，整体代码只有600+行，流程如下图所示。

- 请求到来调度器需要把它加入到waiting队列中等待调度，同时加入到全局的请求记录Map中。
- 调度的过程中，每个请求只记录已处理的token数量和需要处理的token数量，调度器尝试在每次调度时让已处理token数量追上需要处理的token数量。
- 每次调度时，先调度running队列中的请求，如无抢占行为再调度waiting队列中的请求。

- 对running队列中的每个请求尝试在`kv_cache_manager`中分配token需要的slots，如果不足失败则开启抢占模式，释放低优先级请求占用的空间。
- 对waiting队列中的每个请求尝试在`kv_cache_manager`中分配token需要的slots，如果不足则继续在waiting队列等待。

### **3. 分段 CUDA graphs**

vLLM 在V0时已经开始利用PyTorch的`torch.compile`功能自动优化模型，减少对自定义内核的需求。但是V0是对整图进行编译，好处是保持最小的kernel launch CPU开销，坏处是灵活性比较差(比如必须是静态shape)。而在V1中通过引入分段CUDA graphs，缓解CUDA graph的限制，提升了并行计算效率。

如下图所示，将transformer的decoderLayer拆分成三个部分，其中QKV_PROJ部分由于都是Linear层的基础操作，可以编译成CUDA graph进行加速，第二部分Attention实现业界有非常多的加速版本，如[Flashattention V3](https://zhida.zhihu.com/search?content_id=253989199&content_type=Article&match_order=1&q=Flashattention+V3&zhida_source=entity)、[Xformers](https://zhida.zhihu.com/search?content_id=253989199&content_type=Article&match_order=1&q=Xformers&zhida_source=entity)等，可以灵活地引入三方库的方式进行加速，最后的MLP部分可以和下一层的QKV_PROJ部分融合编译成一张CUDA graph进行加速。

这种方式使得灵活性得到大大增强，不过会引入一些CPU上的overhead，vLLM Team表示在H100上测试8B模型的overhead可以忽略不计。
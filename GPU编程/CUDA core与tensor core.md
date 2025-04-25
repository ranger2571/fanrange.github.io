## CUDA core？**

在讨论 Tensor Core 的架构和实用性之前，我们首先需要谈论 CUDA 核心。 **CUDA（Compute Unified Device Architecture）是 NVIDIA 的专有并行处理平台和 GPU 的 API，而 CUDA 核心是 NVIDIA 图形卡中的标准浮点单位。** 

每个 CUDA 核心能够执行计算，每个 CUDA 核心可以在一个时钟周期内执行一个乘加操作。虽然 CUDA 单核性能不及 CPU 核心，但在深度学习中，许多 CUDA 核心可以通过 **并行执行** 进程来加速计算。

在 Tensor Core 发布之前，CUDA 核心是加速深度学习的关键硬件。因为它们只能在单个计算上进行操作，所以受 CUDA 核心性能限制的 GPU 也受可用 CUDA 核心数量和每个核心的时钟速度的限制。为了克服这一限制，NVIDIA 开发了 Tensor Core。

## Tensor Core？**

Tensor Core 是专门的核心，可以实现混合精度训练。第一代这些专门核心通过融合乘加计算实现。这允许两个 4 x 4 FP16 矩阵相乘并添加到一个 4 x 4 FP16 或 FP32 矩阵中。

## 混合精度计算
混合精度计算之所以被命名为如此，是因为虽然输入矩阵可以是低精度 FP16，但最终输出将是 FP32，输出精度只有最小损失。实际上，这在极小的负面影响模型最终有效性的情况下快速加速计算。
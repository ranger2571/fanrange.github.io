文章：[LLM(31)：序列并行的典型方案与实现细节 - 知乎](https://zhuanlan.zhihu.com/p/14665512019)
这篇文章中的并行加速类，中也提到了ring attention[聊聊大模型推理服务之长上下文 - 知乎](https://zhuanlan.zhihu.com/p/698138500)
序列并行（sequence parallelism, context parallelism）是 3D 并行之后针对长文本场景的在序列维度进行切分的并行方法。从实现方法上来说，当前最主流有 [Ring-Attention](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.01889) 和 [DeepSpeed Ulysess](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.14509)，关于这些方法的原理及实现，之前的众多大佬已经论述备至，因此本篇的目标并不在于提出某种新的方法，而在整体梳理这些方法的核心思想、实现细节、优劣势等，仅以此作为方案解读和学习笔记。


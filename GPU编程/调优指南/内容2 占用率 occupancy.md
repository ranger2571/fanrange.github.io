
占用率是指每个多处理器上活跃线程束的数量与可能的最大活跃线程束数量之比。

另一种看待占用率的方式是，将其视为硬件处理线程束的能力中正在被积极使用的那部分所占的百分比。

然而，较高的占用率并不总是能带来更高的性能，不过，较低的占用率总会降低隐藏延迟的能力，进而导致整体性能下降。在执行过程中，理论占用率和实际达到的占用率之间存在较大差异，通常意味着工作负载高度不均衡。


可以通过shared memory、寄存器的大小、l1 cache的大小等因素计算一个warp的最大的
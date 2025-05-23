## 一 前言

影响 CUDA 内核性能的因素

1. **block大小**: 块大小（每个块中的线程数）直接影响 GPU 的并行度和资源利用。常见的块大小是 128、256 或 512 个线程。
2. **grid大小**: 网格大小（块的数量）决定了内核执行的总线程数。网格大小应足够大，以充分利用 GPU 的并行处理能力。
3. **线程束（Warp）**: CUDA GPU 中的线程束（Warp）是 32 个线程的组，这些线程同时执行指令。因此，块大小应为 32 的倍数，以最大化资源利用。
4. **内存访问模式**：高效的内存访问模式（如合并访问）可以显著提高内核性能。避免内存访问冲突和未对齐访问，以减少内存带宽的浪费。（合并访问，一次读取最少32个连续字节的数据）
5. **共享内存和寄存器使用**：合理利用共享内存和寄存器可以加快数据访问速度。然而，过多的共享内存或寄存器使用可能限制每个块的并发数，从而降低整体性能。
6. **线程并发数和占用率**：线程并发数（active warps per SM）和占用率（occupancy）影响 GPU 的资源利用率。高占用率通常意味着更高的性能，但并非总是如此，具体取决于内核的特性。


## 二 CUDA 内核优化策略

### 2.1 优化内存层次结构

CUDA GPU提供了多种内存类型，每种类型的访问速度和用途不同。有效利用这些内存类型可以显著提升内核性能。

1，全局内存优化

**合并内存访问**：合并内存访问指的是**多个线程在同一时钟周期内访问连续且对齐的内存地址**，从而让GPU能够以最少的内存事务处理更多的数据。这种访问模式能够充分利用GPU的内存带宽，显著提升内核性能。

2，共享内存优化

共享内存的访问速度远快于全局内存，**适用于重复访问的数据**。通过将常用数据加载到共享内存，减少全局内存访问次数（内存复用）。另外，确保不同线程访问不同的fbank，避免访问冲突。

### 2.2 提高并行度

最直接和简单的就是**选择合适的块大小**可以提高并行度，确保 GPU 的计算资源被充分利用。通常选择为 `32`（线程束大小 warp size）的倍数，如128、256、512。

#### 2.2.1 避免分支化

分支化指的是同一线程束（`warp`）内的线程因为**条件语句**（如 if-else、switch 等）执行不同路径，导致这些线程需要分别执行不同的指令，这会降低并行效率，导致内核性能变差。
（因为需要先等一部分线程，执行指令A，再等一部分线程执行指令B，导致时间从T变成2T-NT）

#### 2.2.2 展开循环

循环展开通过**将循环体中的代码“展开”成多个相似的操作，从而减少循环的执行次数和控制开销**。
（需要考虑的是，展开循环对寄存器的占用，展开过多，就会导致寄存器不够用）

高级语言层面上来看，循环展开使性能提高的原因可能不是显而易见的，这种提升来其实是来自于编译器执行循环展开时低级指令的改进和优化。在 GPU 编程中，循环展开的目的是为了优化流水线处理和增加并发操作来提高计算性能。
#### 2.2.3 动态并行

CUDA 的动态并行允许在 GPU 端直接创建和同步新的 GPU 内核。

在动态并行中，内核执行分为两种类型：父母和孩子。父线程、父线程块或父网格启动一个新的网格，即子网格。子线程、子线程块或子网格被父母启动。子网格必须在父线程、父线程块或父网格完成之前完成，只有在所有的子网格都完成之后，父母才会完成。下图说明了父网格和子网格的适用范围。


### 2.3 指令级优化

- **使用内联函数**：减少函数调用开销。
- **利用快速数学运算**：如`__fmaf_rn`（浮点乘加）。
- **避免不必要的计算**：预计算不变表达式，减少内核中的计算量。

### 2.4 利用流和并发执行

通过 CUDA 流（`Streams`）实现**内核和内存传输的并行执行，提升整体吞吐量**。即使用多个流，将数据传输和计算重叠。

### 2.5 Warp 级别优化

利用 Warp 级别的指令和操作，提高并行执行效率。如使用 Warp Shuffle（`__shfl`）在线程之间高效传递数据，并尽量减少需要跨 Warp 同步的操作。

### 2.6 高效的同步与分支管理

- **最小化同步点**：减少内核中的`__syncthreads()`调用，以降低同步开销。
- **优化分支**：尽量避免在内核中使用复杂的条件分支，减少线程束内的分歧。
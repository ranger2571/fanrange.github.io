[CUDA ---- Shared Memory - 苹果妖 - 博客园](https://www.cnblogs.com/1024incn/p/4605502.html)
shared memory在之前的博文有些介绍，这部分会专门讲解其内容。

在global Memory部分，数据对齐和连续是很重要的话题，当使用L1的时候，对齐问题可以忽略，但是非连续的获取内存依然会降低性能。

依赖于算法本质，某些情况下，非连续访问是不可避免的。使用shared memory是另一种提高性能的方式。

GPU上的memory有两种：

- On-board memory

- On-chip memory

global memory就是一块很大的on-board memory，并且有很高的latency。而shared memory正好相反，是一块很小，低延迟的on-chip memory，比global memory拥有高得多的带宽。我们可以把他当做可编程的cache，其主要作用有：

· An intra-block thread communication channel 线程间交流通道（**所以是block内访问的**）

· A program-managed cache for global memory data可编程cache

· Scratch pad memory for transforming data to improve global memory access patterns

本文主要涉及两个例子作解释：reduction kernel，matrix transpose kernel。

shared memory（SMEM）是GPU的重要组成之一。

物理上，每个SM包含一个当前正在执行的block中所有thread共享的低延迟的内存池。SMEM使得同一个block中的thread能够相互合作，重用on-chip数据，并且能够显著减少kernel需要的global memory带宽。

由于APP可以直接显式的操作SMEM的内容，所以又被称为可编程缓存。

由于shared memory和L1要比L2和global memory更接近SM，shared memory的延迟比global memory低20到30倍，带宽大约高10倍。

当一个block开始执行时，GPU会分配其一定数量的shared memory，这个shared memory的地址空间会由block中的所有thread 共享。shared memory是划分给SM中驻留的所有block的，也是GPU的稀缺资源。所以，使用越多的shared memory，能够并行的active就越少。

关于Program-Managed Cache：在C语言编程里，循环（loop transformation）一般都使用cache来优化。在循环遍历的时候使用重新排列的迭代顺序可以很好利用cache局部性。在算法层面上，我们需要手动调节循环来达到令人满意的空间局部性，同时还要考虑cache size。cache对于程序员来说是透明的，编译器会处理所有的数据移动，我们没有能力控制cache的行为。shared memory则是一个可编程可操作的cache，程序员可以完全控制其行为。

## Shared Memory Allocation

我们可以动态或者静态的分配shared Memory，其声明即可以在kernel内部也可以作为全局变量。

其标识符为：__shared__。

下面这句话静态的声明了一个2D的浮点型数组：

__shared__ float tile$[size_y][size_x];$

如果在kernel中声明的话，其作用域就是kernel内，否则是对所有kernel有效。

如果shared Memory的大小在编译器未知的话，可以使用extern关键字修饰，例如下面声明一个未知大小的1D数组：

extern __shared__ int tile[];

由于其大小在编译器未知，我们需要在每个kernel调用时，动态的分配其shared memory，也就是最开始提及的第三个参数：

kernel<<<grid, block, isize * sizeof(int)>>>(...)

应该注意到，只有1D数组才能这样动态使用。

## Shared Memory Banks and Access Mode

之前博文对latency和bandwidth有了充足的研究，而shared memory能够用来隐藏由于latency和bandwidth对性能的影响。下面将解释shared memory的组织方式，以便研究其对性能的影响。

## Memory Banks

为了获得高带宽，shared Memory被分成32（对应warp中的thread）个相等大小的内存块，他们可以被同时访问。不同的CC版本，shared memory以不同的模式映射到不同的块（稍后详解）。如果warp访问shared Memory，对于每个bank只访问不多于一个内存地址，那么只需要一次内存传输就可以了，否则需要多次传输，因此会降低内存带宽的使用。

## Bank Conflict

当多个地址请求落在同一个bank中就会发生bank conflict，从而导致请求多次执行。硬件会把这类请求分散到尽可能多的没有conflict的那些传输操作 里面，降低有效带宽的因素是被分散到的传输操作个数。

warp有三种典型的获取shared memory的模式：

· Parallel access：多个地址分散在多个bank。

· Serial access：多个地址落在同一个bank。

· Broadcast access：一个地址读操作落在一个bank。

Parallel access是最通常的模式，这个模式一般暗示，一些（也可能是全部）地址请求能够被一次传输解决。理想情况是，获取无conflict的shared memory的时，每个地址都在落在不同的bank中。

Serial access是最坏的模式，如果warp中的32个thread都访问了同一个bank中的不同位置，那就是32次单独的请求，而不是同时访问了。

Broadcast access也是只执行一次传输，然后传输结果会广播给所有发出请求的thread。这样的话就会导致带宽利用率低。

某些thread访问到同一个bank的情况，这种情况有两种行为：

· Conflict-free broadcast access if threads access the same address within a bank

· Bank conflict access if threads access different addresses within a bank
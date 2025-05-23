GPU计算的特色——并行计算（并行性）
# 并行计算
写并行程序主要是分解任务，我们一般把一个程序看成是指令和数据的组合，当然并行也可以分为这两种：

- 指令并行
- 数据并行

我们的任务更加关注数据并行，所以我们的主要任务是分析数据的相关性，哪些可以并行，哪些不能并行。

CUDA非常适合数据并行  
数据并行程序设计，第一步就是把数据依据线程进行划分。

不同的数据划分严重影响程序性能，所以针对不同的问题和不同计算机结构，我们要通过和理论和试验共同来决定最终最优的数据划分。

为了提高并行的计算能力，我们要从架构上实现下面这些性能提升：

- 降低延迟
- 提高带宽
- 提高吞吐量

延迟是指操作从开始到结束所需要的时间，一般用微秒计算，延迟越低越好。  
带宽是单位时间内处理的数据量，一般用MB/s或者GB/s表示。  
吞吐量是单位时间内成功处理的运算数量，一般用gflops来表示（十亿次浮点计算），吞吐量和延迟有一定关系，都是反应计算速度的，一个是时间除以运算次数，得到的是单位次数用的时间–延迟，一个是运算次数除以时间，得到的是单位时间执行次数–吞吐量。

# 异构计算
异构计算，首先必须了解什么是异构，不同的计算机架构就是异构，

按照指令集划分或者按照内存结构划分，但是我觉得只要两片CPU型号不一样就应该叫异构

CPU我们可以把它看做一个指挥者，主机端，host，而完成大量计算的GPU是我们的计算设备，device。

CPU和GPU之间通过PCIe总线连接，用于传递指令和数据，这部分也是后面要讨论的性能瓶颈之一。  

一个异构应用包含两种以上架构，所以代码也包括不止一部分：

- 主机代码
- 设备代码

衡量GPU计算能力的主要靠下面两种**_容量_**特征：

- CUDA核心数量（越多越好）
- 内存大小（越大越好）

相应的也有计算能力的**_性能_**指标:

- 峰值计算能力
- 内存带宽

nvidia自己有一套描述GPU计算能力的代码，其名字就是“计算能力”，主要区分不同的架构，早其架构的计算能力不一定比新架构的计算能力强


### CPU和GPU线程的区别：

1. CPU线程是重量级实体，操作系统交替执行线程，线程上下文切换花销很大
2. GPU线程是轻量级的，GPU应用一般包含成千上万的线程，多数在排队状态，线程之间切换基本没有开销。
3. CPU的核被设计用来尽可能减少一个或两个线程运行时间的延迟，而GPU核则是大量线程，最大幅度提高吞吐量

### CUDA：一种异构计算平台

CUDA平台不是单单指软件或者硬件，而是建立在Nvidia GPU上的一整套平台，并扩展出多语言支持

一个CUDA应用通常可以分解为两部分，

- CPU 主机端代码
- GPU 设备端代码

CUDA nvcc编译器会自动分离你代码里面的不同部分，如图中主机代码用C写成，使用本地的C语言编译器编译，设备端代码，也就是核函数，用CUDA C编写，通过nvcc编译，链接阶段，在内核程序调用或者明显的GPU设备操作时，添加运行时库。

**注意：核函数是我们后面主要接触的一段代码，就是设备上执行的程序段**

#### 一般CUDA程序分成下面这些步骤：

1. 分配GPU内存
2. 拷贝内存到设备
3. 调用CUDA内核函数来执行计算
4. 把计算完成数据拷贝回主机端
5. 内存销毁


# CUDA编程模型
CUDA编程模型为应用和硬件设备之间的桥梁，所以CUDA C是编译型语言，不是解释型语言，OpenCL就有点类似于解释型语言，通过编译器和链接，给操作系统执行（操作系统包括GPU在内的系统），下面的结构图片能形象的表现他们之间的关系：

其中Communication Abstraction是编程模型和编译器，库函数之间的分界线。  
可能大家还不太明白编程模型是啥，编程模型可以理解为，我们要用到的语法，内存结构，线程结构等这些我们写程序时我们自己控制的部分，这些部分控制了异构计算设备的工作模式，都是属于编程模型。  

GPU中大致可以分为：

- 核函数
- 内存管理
- 线程管理
- 流
等几个关键部分。  

以上这些理论同时也适用于其他非CPU+GPU异构的组合。  

下面我们会说两个我们GPU架构下特有几个功能：

- 通过组织层次结构在GPU上组织线程的方法
- 通过组织层次结构在GPU上组织内存的方法

也就是对内存和线程的控制将伴随我们写完前十几篇。  

从宏观上我们可以从以下几个环节完成CUDA应用开发：

1. 领域层
2. 逻辑层
3. 硬件层

第一步就是在领域层（也就是你所要解决问题的条件）分析数据和函数，以便在并行运行环境中能正确，高效地解决问题。  

当分析设计完程序就进入了编程阶段，我们关注点应转向如何组织并发进程，这个阶段要从逻辑层面思考。  

CUDA模型主要的一个功能就是线程层结构抽象的概念，以允许控制线程行为。这个抽象为并行变成提供了良好的可扩展性（这个扩展性后面有提到，就是一个CUDA程序可以在不同的GPU机器上运行，即使计算能力不同）。  

在硬件层上，通过理解线程如何映射到机器上，能充分帮助我们提高性能。

## CUDA编程结构

一个异构环境，通常有多个CPU多个GPU，他们都通过PCIe总线相互通信，也是通过PCIe总线分隔开的。所以我们要区分一下两种设备的内存：

- 主机：CPU及其内存
- 设备：GPU及其内存

注意这两个内存从硬件到软件都是隔离的（CUDA6.0 以后支持统一寻址），我们目前先不研究统一寻址，我们现在还是用内存来回拷贝的方法来编写调试程序，以巩固大家对两个内存隔离这个事实的理解。
![[Pasted image 20250329142054.png]]
从host的串行到调用核函数（核函数被调用后控制马上归还主机线程，也就是在第一个并行代码执行时，很有可能第二段host代码已经开始同步执行了）。

我们接下来的研究层次是：

- 内存
- 线程
- 核函数
    - 启动核函数
    - 编写核函数
    - 验证核函数
- 错误处理

## 内存管理

内存管理在传统串行程序是非常常见的，寄存器空间，栈空间内的内存由机器自己管理，堆空间由用户控制分配和释放，CUDA程序同样，只是CUDA提供的API可以分配管理设备上的内存，当然也可以用CDUA管理主机上的内存，主机上的传统标准库也能完成主机内存管理。  

我们先研究最关键的一步，这一步要走总线的

cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,  <br>  cudaMemcpyKind kind)|

这个函数是内存拷贝过程，可以完成以下几种过程（cudaMemcpyKind kind）
- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice

这四个过程的方向可以清楚的从字面上看出来，这里就不废话了，如果函数执行成功，则会返回 cudaSuccess 否则返回 cudaErrorMemoryAllocation

## 线程管理
我们必须明确，一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程，这种分层的组织结构使得我们的并行过程更加自如灵活

一个线程块block中的线程可以完成下述协作：

- 同步
- 共享内存

#### 不同块内线程不能相互影响！他们是物理隔离的！

### 编写核函数
声明核函数有一个比较模板化的方法：


# CUDA执行模型

CUDA执行模型揭示了GPU并行架构的抽象视图，

GPU架构是围绕一个流式多处理器（SM）的扩展阵列搭建的。通过复制这种结构来实现GPU的硬件并行

### SM（从机器的角度，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行）

GPU中每个SM都能支持数百个线程并发执行，每个GPU通常有多个SM，当一个核函数的网格被启动的时候，多个block会被同时分配给可用的SM上执行。

**注意:** 当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。

在SM上同一个块内的多个线程进行线程级别并行，而同一线程内，指令利用指令级并行将单个线程处理成流水线。


### 线程束

CUDA 采用单指令多线程SIMT架构管理执行线程，不同设备有不同的线程束大小，但是到目前为止基本所有设备都是维持在32，也就是说每个SM上有多个block，一个block有多个线程（可以是几百个，但不会超过某个最大值），但是从机器的角度，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行，线程束中的每个线程执行同一条指令，包括有分支的部分，
### CUDA编程的组件与逻辑

下图从逻辑角度和硬件角度描述了CUDA编程模型对应的组件。

SM中共享内存，和寄存器是关键的资源，线程块中线程通过共享内存和寄存器相互通信协调。  
寄存器和共享内存的分配可以严重影响性能！
![[Pasted image 20250329143854.png]]
因为SM有限，虽然我们的编程模型层面看所有线程都是并行执行的，但是在微观上看，所有线程块也是分批次的在物理层面的机器上执行，线程块里不同的线程可能进度都不一样，但是**同一个线程束内的线程拥有相同的进度。**

并行就会引起竞争，多线程以未定义的顺序访问同一个数据，就导致了不可预测的行为，CUDA只提供了一种块内同步的方式，块之间没办法同步！  

同一个SM上可以有不止一个常驻的线程束，有些在执行，有些在等待，他们之间状态的转换是不需要开销的。

# 详细说明线程束运行的本质

从外表来看，CUDA执行所有的线程，并行的，没有先后次序的，但实际上硬件资源是有限的，不可能同时执行百万个线程，所以从硬件角度来看，物理层面上执行的也只是线程的一部分，而每次执行的这一部分，就是我们前面提到的线程束。

## 线程束和线程块

线程束是SM中基本的执行单元，当一个网格被启动（网格被启动，等价于一个内核被启动，每个内核对应于自己的网格），网格中包含线程块，线程块被分配到某一个SM上以后，将分为多个线程束，每个线程束一般是32个线程（目前的GPU都是32个线程，但不保证未来还是32个）在一个线程束中，所有线程按照单指令多线程SIMT的方式执行，每一步执行相同的指令，但是处理的数据为私有的数据，

线程束和线程块，一个是硬件层面的线程集合，一个是逻辑层面的线程集合，我们编程时为了程序正确，必须从逻辑层面计算清楚，但是为了得到更快的程序，硬件层面是我们应该注意的。
## 线程束分化

线程束被执行的时候会被分配给相同的指令，处理各自私有的数据，

条件分支越多，并行性削弱越严重。 （ 当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。）

注意线程束分化研究的是一个线程束中的线程，不同线程束中的分支互不影响。

因为线程束分化导致的性能下降就应该用线程束的方法解决，根本思路是避免同一个线程束内的线程分化，而让我们能控制线程束内线程行为的原因是线程块中线程分配到线程束是有规律的而不是随机的。


## 资源分配

我们前面提到过，每个SM上执行的基本单位是线程束，也就是说，单指令通过指令调度器广播给某线程束的全部线程，这些线程同一时刻执行同一命令。

一个SM中可能有多个warp，这些warp中，一个时间点，只能有一个在执行。

那么对别的不在执行的warp进行分类，，
一类是已经激活的，也就是说这类线程束其实已经在SM上准备就绪了，只是没轮到他执行，这时候他的状态叫做阻塞，
还有一类可能分配到SM了，但是还没上到片上，这类我称之为未激活线程束。

而每个SM上有多少个线程束处于激活状态，取决于以下资源：

- 程序计数器
- 寄存器
- 共享内存

线程束一旦被激活来到片上，那么他就不会再离开SM直到执行结束。  
每个SM都有32位的寄存器组，每个架构寄存器的数量不一样，其存储于寄存器文件中，为每个线程进行分配，同时，固定数量的共享内存，在线程块之间分配。  
一个SM上被分配多少个线程块和线程束取决于SM中可用的寄存器和共享内存，以及内核需要的寄存器和共享内存大小。

这是一个平衡问题，就像一个固定大小的坑，能放多少萝卜取决于坑的大小和萝卜的大小，相比于一个大坑，小坑内可能放十个小萝卜，或者两个大萝卜，SM上资源也是，当kernel占用的资源较少，那么更多的线程（这是线程越多线程束也就越多）处于活跃状态，相反则线程越少。

上面讲的主要是线程束，如果从逻辑上来看线程块的话，可用资源的分配也会影响常驻线程块的数量。  
特别是当SM内的资源没办法处理一个完整块，那么程序将无法启动，这个是我们应该找找自己的毛病，你得把内核写的多大，或者一个块有多少线程，才能出现这种情况。

当寄存器和共享内存分配给了线程块，这个线程块处于活跃状态，所包含的线程束称为活跃线程束。  
活跃的线程束又分为三类：

- 选定的线程束
- 阻塞的线程束
- 符合条件的线程束

当SM要执行某个线程束的时候，执行的这个线程束叫做选定的线程束，准备要执行的叫符合条件的线程束，如果线程束不符合条件还没准备好就是阻塞的线程束。  
满足下面的要求，线程束才算是符合条件的：

- 32个CUDA核心可以用于执行
- 执行所需要的资源全部就位

Kepler活跃的线程束数量从开始到结束不得大于64，可以等于。  
任何周期选定的线程束小于等于4。  （warp scheduler 的大小是4）
由于计算资源是在线程束之间分配的，且线程束的整个生命周期都在片上，所以线程束的上下文切换是非常快速的，。

# 循环展开

GPU没有分支预测能力，所有每一个分支他都是执行的，所以在内核里尽量别写分支，分支包括啥，包括if当然还有for之类的循环语句。  
如果你不知道到为啥for算分支语句我给你写个简单到不能运行的例子：
```
for (itn i=0;i<tid;i++)  
{  
    // to do something   
}
```

如果上面这段代码出现在内核中，就会有分支，因为一个线程束第一个线程和最后一个线程tid相差32（如果线程束大小是32的话） 那么每个线程执行的时候，for终止时完成的计算量都不同，这就有人要等待，这也就产生了分支。

> 循环展开是一个尝试通过减少分支出现的频率和循环维护指令来优化循环的技术。  
> 上面这句属于书上的官方说法，我们来看看例子，不止并行算法可以展开，传统串行代码展开后效率也能一定程度的提高，因为省去了判断和分支预测失败所带来的迟滞。  
> 先来个c++ 入门循环

```
for (itn i=0;i<tid;i++)  
{  
    a[i]=b[i]+c[i]; // to do something   
}
```
这个是最传统的写法，这个写法在各种c++教材上都能看到，不做解释，如果我们进行循环展开呢？
```
for (int i=0;i<100;i+=4)  
{  
    a[i+0]=b[i+0]+c[i+0];  
    a[i+1]=b[i+1]+c[i+1];  
    a[i+2]=b[i+2]+c[i+2];  
    a[i+3]=b[i+3]+c[i+3];  
}
```

没错，是不是很简单，修改循环体的内容，把本来循环自己搞定的东西，我们自己列出来了，这样做的好处，从串行较多来看是减少了条件判断的次数。  
但是如果你把这段代码拿到机器上跑，其实看不出来啥效果，因为现代编译器把上述两个不同的写法，编译成了类似的机器语言，也就是，我们这不循环展开，编译器也会帮我们做。  
不过值得注意的是：**_目前CUDA的编译器还不能帮我们做这种优化，人为的展开核函数内的循环，能够非常大的提升内核性能_**  
在CUDA中展开循环的目的还是那两个：

1. 减少指令消耗
2. 增加更多的独立调度指令  
    来提高性能  
    如果这种指令
	a[i+0]=b[i+0]+c[i+0];  
    a[i+1]=b[i+1]+c[i+1];  
    a[i+2]=b[i+2]+c[i+2];  
    a[i+3]=b[i+3]+c[i+3];  
    
被添加到CUDA流水线上，是非常受欢迎的，因为其能最大限度的提高指令和内存带宽。  
下面我们就在前面归约的例子上继续挖掘性能，看看是否能得到更高的效率。










# 内存架构
## CUDA内存模型

对于程序员来说，分类内存的方法有很多中，但是对于我们来说最一般的分法是：

- 可编程内存
- 不可编程内存

对于可编程内存，如字面意思，你可以用你的代码来控制这组内存的行为；相反的，不可编程内存是不对用户开放的，也就是说其行为在出厂后就已经固化了，对于不可编程内存，我们能做的就是了解其原理，尽可能的利用规则来加速程序，但对于通过调整代码提升速度来说，效果很一般。  

CPU内存结构中，一级二级缓存都是不可编程（完全不可控制）的存储设备。  
另一方面，CUDA内存模型相对于CPU来说那是相当丰富了，GPU上的内存设备有：

- 寄存器
- 共享内存
- 本地内存
- 常量内存
- 纹理内存
- 全局内存

上述各种都有自己的作用域，生命周期和缓存行为。

CUDA中每个线程都有自己的私有的本地内存；线程块有自己的共享内存，对线程块内所有线程可见；所有线程都能访问读取常量内存和纹理内存，但是不能写，因为他们是只读的；全局内存，常量内存和纹理内存空间有不同的用途。对于一个应用来说，全局内存，常量内存和纹理内存有相同的生命周期。















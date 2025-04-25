reduce可以分成两部分，多个block进行reduce到一个block内，一个block内规约到一个数字，



kernel接口为：

```text
__global__ void reduce(T *input, T* output)
```

其中，input代表输入的数组，即一个长度为N的数组，output代表输出数组，即第一阶段的结果，即长度为M的数组。

在[CUDA编程](https://zhida.zhihu.com/search?content_id=183014041&content_type=Article&match_order=1&q=CUDA%E7%BC%96%E7%A8%8B&zhida_source=entity)中，我们首先需要设置三个参数:

1. **BlockNum**：即开启的block数量，即上面所说的M，代表需要将数组切分为几份。
2. **Thread_per_block**:每个block中开启的线程数，一般而言，取128，256，512，1024这几个参数会比较多。
3. **Num_per_block**:每个block需要进行reduce操作的长度。

其中，BlockNum* Num_per_block=N

# -1 baseline
不使用shared memory，直接对global memory进行处理
``` 
不看grid，只看block内

比如，一个block处理2048个数据，有4*32个线程，4个warp
不使用shared memory
先跨warp规约到一个block内，然后再调用折半规约

int idx=threadidx.x;
int tid=threadidx.x;(0-127)
int sum=0;
for(int i=idx;i<num_data;i+=blockdimx.x){
	sum+=input[i];
}
input[tid]=sum;
__syncthreads()
offest=64;
for(int i=offest;i>0;i=i/2){
if(tid<i)input[tid]+=input[tid+i];
}
__syncthreads()
if(tid==0) out=input[tid];

```
# baseline
reduce的**叠加规约**+shared memory，（一个线程对应一个元素）

对于一个block内的思路就是，block的分配的线程，进行跨线程处理？
``` 
//这里其实就默认了数据和线程数是相等的，不然的话sha就不够用了，因为sha在这里是用于导入数据，所以需要复用，或者与数据量对应
__shared__ float sha[blockdimx.x];
int idx=blockidx.x*blockdimx.x+threadidx.x;
int tid=threadidx.x;
sha[tid]=input[idx];//一个block的数据全部存到smem中
for(int offest=blockdimx.x/2;offest>0;offest=offest/2)
	if(tid<offest)sha[tid]+=sha[tid+offest];
return sha[0];
```
**（没写叠加，感觉完全没必要写这个）
# baseline+1
reduce的 **叠加规约的一个变形，（没看懂）** +shared memory，（一个线程对应一个元素）
``` 
//这里其实就默认了数据和线程数是相等的，不然的话sha就不够用了，因为sha在这里是用于导入数据，所以需要复用，或者与数据量对应
__shared__ float sha[blockdimx.x];
int idx=blockidx.x*blockdimx.x+threadidx.x;
int tid=threadidx.x;
sha[tid]=input[idx];//一个block的数据全部存到smem中
for(int offest=blockdimx.x/2;offest>0;offest=offest/2)
	if(tid<offest)sha[tid]+=sha[tid+offest];
return sha[0];
```

# baseline+2
如何优化bank conflict

哈哈哈，方法就是折半规约，
因为折半规约在一个warp的32个线程处理大于32个数字的时候，32个线程会连续的读取前32个数，和后面需要处理的数，所以前面的会连续读入，不存在conflict

# baseline+3.0
解决线程浪费

``` 
//这里其实就默认了数据和线程数是相等的，不然的话sha就不够用了，因为sha在这里是用于导入数据，所以需要复用，或者与数据量对应

假设一个block处理512个数据，但是只有256个线程，要么就扩大sha的大小，
1.__shared__ float sha[4*blockdimx.x];
要么就要对输入sha的数据也做一次规约。(这里作者选择使用直接相加存入sha的方法)
2.__shared__ float sha[blockdimx.x];
//作者的代码
	unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();
//作者的代码

int idx=blockidx.x*blockdimx.x+threadidx.x;
int tid=threadidx.x;

for(int offest=blockdimx.x/2;offest>0;offest=offest/2)
	if(tid<offest)sha[tid]+=sha[tid+offest];
return sha[0];
```
# baseline+3.1
删除无用的同步指令，在一个warp内的时候，不需要同步，warp自身就已经实现同步了

```cuda
__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

# 完全展开操作
这个就不可能手撕了，了解一下就行


# 修改block的形状和大小和数量



# 使用shuffle指令
Shuffle指令是一组针对warp的指令。

Shuffle指令最重要的特性就是**warp内的寄存器可以相互访问**。

在没有shuffle指令的时候，各个线程在进行通信时只能通过shared memory来访问彼此的寄存器。

而采用了shuffle指令之后，warp内的线程可以直接对其他线程的寄存器进行访存。

通过这种方式可以减少访存的延时。除此之外，带来的最大好处就是可编程性提高了，在某些场景下，就不用shared memory了。


# 作者的内容已经结束了，但是我感觉还能做的地方有很多，作者最多只考虑了一个线程处理两个数据的情况，但是在softmax的算子中，似乎一个线程会安排远大于这个数量的计算？？

# baseline+3.2
解决线程浪费

``` 
//这里其实就默认了数据和线程数是相等的，不然的话sha就不够用了，因为sha在这里是用于导入数据，所以需要复用，或者与数据量对应

假设一个block处理512个数据，但是只有128个线程，要么就扩大sha的大小，
1.__shared__ float sha[4*blockdimx.x];
要么就要对输入sha的数据也做一次规约。
2.__shared__ float sha[blockdimx.x];

__shared__ float sha[blockdimx.x];
int idx=blockidx.x*blockdimx.x+threadidx.x;
int tid=threadidx.x;
for(int i=tid;i<num_data;i+=blockdimx.x)
	sha[tid]+=input[i];//一个block的数据全部存到smem中

for(int offest=blockdimx.x/2;offest>0;offest=offest/2)
	if(tid<offest)sha[tid]+=sha[tid+offest];
return sha[0];
```

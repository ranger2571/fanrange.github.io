KV Cache的实际利用率只有20%～40%，大部分的KV Cache里的显存都被浪费了。这种浪费有三种原因：
（1）大模型生成时，并不知道要生成多少个token，所以总是按照生成参数里设置的最大的token数来预分配KV cache，比如最大的token数为1000，但是模型可能只生成到第100个token的时候就输出了终止符结束了，那么预分配的那900个token的KV cache就被浪费了
问题一（**没有使用的空间上的浪费）

（2）假如一个样本真的可以输出到1000个token，但是在它刚开始输出第一个token的时候，剩下的token都还没有用到，但是显存已经被预分配占用了。这时其他的请求也无法被响应，而未被响应的请求有可能只需要输出10个token就结束了，本来可以和正在进行输出的样本并行处理。
问题二（**空间上没有浪费，但是时间上降低了并发的大小，也是浪费）

（连续的物理显存的缺点：用page attention解决）  
（3）显存之间的间隔碎片，即使最大的生成长度一致，但是因为prompt的长度不同，每次预分配的KV cache大小也不同，当一个请求生成完毕，释放缓存，但是下一个请求的prompt的长度 大于 释放的这个请求的prompt的长度，所以无法放入被释放的缓存中，这种无法被使用的缓存就是**碎片。**（物理缓存设计导致的浪费）

  

vllm就是解决了kv cache里的浪费问题，用更大的batch size来处理请求，从而提高了系统的吞吐量（Throughput）：原始系统里请求的batch size为8，经过vllm优化后，batch size可以增大到40。原来系统每秒可以响应300个token，通过vLLM优化后可以输出900个token

## vllm做了哪些优化呢？  
（1）page attention： 

类似于KV Cache的问题，操作系统里也遇到过，操作系统给每个应用分配内存，到底要不要给每个程序预分配内存？程序关闭后怎么回收内存？内存碎片怎么处理？怎么最大化地利用内存？  
  
操作系统是通过利用虚拟内存，和页管理技术来解决的，操作系统分配内存是按照最小单元页来分配，物理内存被划分为很多页，每个进程（Process）要用到的内存被映射到不同的页上。  
  
Page attention把显存也划分为 KV block，显存按照KV block来管理KV cache，每个请求request需要的kv cache被划分到显存 里 不同的KV Block里，
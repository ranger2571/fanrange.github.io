v1作为default的方式进行推理的

问题是什么？

存在一个chunk size， 指chunk size大小的prefill的token数+1个decode token，进行拼batch

chunk size怎么设？太大的话，decode的速度太慢；太小的话，generation speed会增加（因为decode的对应量增加了），但是gpu utilization会下降会（prefill的量太小不能充分利用算力），

对应了怎么设定chunk size的大小才合适的问题


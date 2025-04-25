## 从global memory到shared memory

假设有矩阵A,B，需要计算矩阵A和B的乘，即矩阵C。A、B、C三个矩阵的维度分别为，，m∗k，k∗n，m∗n ，且三个矩阵中的数据都是单精度浮点数。对于C中每一个元素，C[i][j]，可以看作是A的一行和B的一列进行一次归约操作。采用最naive的GEMM算法，在GPU中，一共开启m∗n 个线程，每个线程需要读取矩阵A的一行与矩阵B的一列，而后将计算结果写回至矩阵C中。因而，完成计算一共需要从global memory中进行2mnk 次读操作和m*n次写操作。大量的访存操作使得GEMM效率难以提高，因而考虑global memory中进行分块，并将矩阵块放置到shared memory中。其示意图如下：

## 从shared memory到register

随后，我们进一步考虑从shared memory到register的过程。在这里，只分析**一个block**中的计算。当进行K轮迭代中某一轮迭代时，GPU将维度为，bm∗bk，bk∗bn 的小矩阵块存储到shared memory中，而后各个线程将shared memory中的数据存入register中进行计算。
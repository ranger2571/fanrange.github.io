[NLP(5)：Transformer及其attention机制 - 知乎](https://zhuanlan.zhihu.com/p/476585349)

**$QK^T$的含义**

我们知道，对于向量 xi 和 xj ，其内积 <xi,xj> **表征一个向量在另一个向量上的投影，即两个向量的相似关系。**那么对于矩阵 X ， $XX^T$ **是一个方阵，以行向量的角度理解，里面保存了每个向量与自己和其他向量进行内积运算的结果，即与其他向量的相似程度。**使用query和每个key进行计算，得到就是相应的attention。

**dk 与softmax**

dk 是 q 和 k 的维度，主要是因为随着维度增加，点乘的结果也会变大，结果太大也会对softmax的结果产生影响，所以需要除以一个系数。

下面简要予以证明：

假设向量 q 和 k 的各个分量是互相独立的随机变量，维度为 d_k ，均值是0，方差是1，即

$E[q_i]=E[k_i]=0$ $var⁡[q_i]=var⁡[k_i]=1$ 其中 $i∈[0,d_k]$

则有

$E[q⋅k]=E[∑_{i=1}^{d_k}q_ik_i]=∑_{i=1}^{d_k}E[q_i k_i]=∑_{i=1}^{d_k}E[q_i]E[k_i]=0$

$var⁡[q⋅k]=var[∑_{i=1}^{d_k}q_ik_i]=∑_{i=1}^{d_k}var⁡[q_i k_i]=∑_{i=1}^{d_k}var⁡[q_i]var⁡[k_i]=∑_{i=1}^{d_k}1=d_k$

则 $var(q⋅k/\sqrt(dk))=dk/(\sqrt(dk))2=1$

**将方差控制为1，也就有效地控制了softmax反向的梯度消失的问题**。

# 大模型的模块的计算量
[[LLM]预训练模型MFU计算器 - 知乎](https://zhuanlan.zhihu.com/p/20401860293)

## 1 基本模块flops计算

### 1.1 线性层的计算量
线性层的计算公式为 Y = wX + b 涉及到矩阵的乘法与加法运算。  

矩阵乘法与加法的flops的计算为：
**乘法计算量**：对于两个矩阵A和B的乘法C=AB，其中A是m×n矩阵，B是n×p矩阵，C是m×p矩阵。每个元素Cij需要进行 n 次乘法和n-1次加法，总共有mp个元素，因此总FLOPS为：  
mp(n+(n-1)) = 2mnp - mp。  
**加法/减法计算量**：对于两个矩阵A和B的加法C=A+B，其中A和B都是m×n矩阵，C也是m×n矩阵。每个元素Cij需要进行一次加法，总共有mn个元素，因此总FLOPS为mn。  

对于linear计算，里面涉及一个矩阵乘和一个矩阵加法，由于元素需要展平再运算，权重w的维度[m, n] 输入的维度是[1, n] 输出维度[1, m]，其计算量为  2mn  
不计算bias偏置矩阵的计算量为  2mn - m（就是带入前面的乘法计算量和加法计算量）  

对于transformer的线性层输入与输出一般用相同的大小，形状都为：[batch_size, seq_len, d_model],  

线性层的创建一般使用 nn.Linear(hidden_size, hidden_size, bias=False)  

**所以计算量为：**  
flops = 2 * batch_size * seq_len * hidden_size * hidden_size  
**如果不一致时：**  
flops = 2 * batch_size * seq_len * size_1 * size_2

### 1.2 Attention模块的计算

一般的MHA([MultiHeadAttention](https://zhida.zhihu.com/search?content_id=253102524&content_type=Article&match_order=1&q=MultiHeadAttention&zhida_source=entity))计算的构造如下：

主要运算：

- Q/K/V: 线性映射
- scores: QK乘法运算
- attention_qkv: V和attention_weights乘法运算
- out_linear: 线性度计算

次要运算：

- softmax计算
- masked_fill计算

对于主要运算中有个需要考虑点：

- Attention的变化：query attention中KV的heads数量与Q的heads数量不一致。
- 序列并行（context parallel/ring attention）: 考虑并行度。

次要运算在估算flops时通常可以忽略，这里列出其计算方式：

softmax的flops计算量： 输入的shape：(bs, heads, seq_len, seq_len)

元素计算涉及指数运算、加法运算、除法运算。计算量：

3 * bs * heads * seq_len * (seq_len - 1)

masked_fill是一个掩码操作，包含判断操作和赋值操作，假设是需要遍历整个矩阵，每个元素操作一次，而赋值操作仅对需要操作的元素赋值，输入矩阵的大小为[bs, heads, seq_len, seq_len], 操作的个数为X。所以计算量：

bs * heads * seq_len * seq_len + X

由于X操作相对来说较小, 公式简化为：

bs * heads * seq_len * seq_len

### 1.3 [LayerNorm](https://zhida.zhihu.com/search?content_id=253102524&content_type=Article&match_order=1&q=LayerNorm&zhida_source=entity)/RMSNorm

Layer_norm的计算内容一般如下：

```text
import numpy as np

def layer_normalization(x, epsilon=1e-8):
    mean = np.mean(x, axis=-1, keepdims=True) # 最后一个维度
    std = np.std(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / (std + epsilon)
    return normalized_x
```

假设数据的长度为L

包含平均值计算、标准差计算、偏移计算；

- **mean计算包含L加法和一次除法： L + 1
- **std计算，每个元素进行一个减法、一个乘法、一个加法。最后进行一个除法和一个乘法操作： 3*L + 2
- **标准化：每个元素一次减法、一次除法操作： 2*L

忽略单次运算，所以操作计算量：**6 * batch_size * seq_len * hidden_size

```text
def calcu_layer_norm_flops(batch_size, seq_len, hidden_size):
  return 6 * batch_size * seq_len * hidden_size
```

RMSNorm 常见的代码实现如下：

```python
# 参考Llama定义
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        '''
        LlamaRMSNorm is equivalent to T5LayerNorm
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

主要计算内容：

- 元素二次方、元素求平均（n-1）、一个rsqrt运算、一个求和运算
- 两个乘法操作

忽略单次运算，flops数等于：

4 * batch_size * seq_len * hidden_size

```python
def calcu_rmsnorm_flops(batch_size, seq_len, hidden_size):
  return 4 * batch_size * seq_len * hidden_size
```


### 1.4 MLP/FFN层的计算

MLP层的构建常见的方式如下：

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))
```

主要包含两个线性层操作和一个Relu计算。

输入/输出: [batch_size, seq_len, hidden_size]

dff值：ffn_hidden_size

计算量为两次线性层运算 + 一个relu操作，其flops操作数量如下：

2 * batch_size * seq_len * hidden_size * ffn_hidden_size + batch_size * seq_len * ffn_hidden_size

Llama的MLP有些改动，一般的计算包含三次线性运算（gate_proj、up_proj、down_proj, 参看hugging face的LlamaMLP定义）一个silu运算，一个元素乘法运算。



# DeepSeekV3模型的MFU
[DeepSeekV3 MFU计算工具与算式 - 知乎](https://zhuanlan.zhihu.com/p/26107304514)
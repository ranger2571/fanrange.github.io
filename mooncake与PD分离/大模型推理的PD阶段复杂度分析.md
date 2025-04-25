[大模型推理 PD 阶段复杂度分析 - 知乎](https://zhuanlan.zhihu.com/p/32054265636)
在大模型推理中，prefill 和 decode 阶段的复杂度分析需结合模型结构（如 Transformer）的具体实现。一般的 Transformer 模型结构的 PD 阶段的复杂度分析如下：

## Prefill 阶段

- Attention 部分

- 假设输入序列长度为 n ，每个 token 的维度为 d 。计算注意力时，Q、K、V 矩阵的维度均为 n×d 。
- Q 与 K 的乘积复杂度为 O(n2d) ，softmax 和与 V 的乘积同样为 O(n2d) 。
- **总时间复杂度**为 O(n2d) ，主导项为序列长度的平方与维度的乘积。

- [FFN](https://zhida.zhihu.com/search?content_id=255433325&content_type=Article&match_order=1&q=FFN&zhida_source=entity) 部分：

- 前馈网络对每个 token 进行两次线性变换，复杂度为 O(nd2) （假设中间维度为 4d ）。
- 对于 L 层模型，总复杂度为 O(L(n2d+nd2)) 。若 n≫d ，注意力部分主导；反之 FFN 可能更显著。

- Prefill 总复杂度

- 综上，总时间复杂度为 $O(L(n^2d+nd^2))$，注意力主导时近似为 O(n2d) 。

### Decode 阶段

- Attention 部分

- 假设每个 token 的维度为 d，每次生成一个 token 时，需计算新 token 的 Q 与缓存的 K（长度 n ）的点积，复杂度为 O(nd) 。
- 与 V 的乘积和输出映射同样为 O(nd) 。
- 每层每步骤的注意力计算复杂度为 O(nd) 。

- FFN 部分

- 每个 token 经过两次线性变换，复杂度为 O(d2) 。
- 每层每步骤的 FFN 复杂度为 O(d2) 。

- Decode 总复杂度

- 综上，每层每步骤的复杂度为 O(nd+d2) 。对于 L 层模型，总复杂度为 O(L(nd+d2)) 。
- 若 n 与 d 相当，或 d≫n ，则 FFN 的 O(d2) 可能主导；反之注意力部分主导。
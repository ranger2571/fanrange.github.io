[(35 封私信 / 32 条消息) 大模型推理框架，SGLang和vLLM有哪些区别？ - 知乎](https://www.zhihu.com/question/666943660/answer/98952571127)

# MLA理论
苏佬的总结很到位：在训练阶段，除了
1. 对Q多了一步低秩投影以及
2. 只在部分维度加RoPE 外，MLA与Q、K的Head Size由dk换成dk+dr的MHA基本无异

数据量由原来的 $h*(d_k+d_v)$ 变为 $d_c+d_r$ ，太妙了！（注意，论文中提到的 $(d_c+d_h^R)l$的$l$是layer层数，也就是transformer block的数量，后面的分析暂不考虑 $l$

MLA的KVCache与h无关，增大h只会增加计算量和提升模型性能，不会增加KVCache（KVCache是速度的瓶颈）

**MLA in V2->V3（完全一样）**

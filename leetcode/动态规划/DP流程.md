# 核心流程
- 拆解问题
- 定义dp[j]的物理含义
- 画dp流程图，找到dp的动态规划的逻辑
- 对应物理意义进行初始化，
- 解决异常的case

# 代码随想录的内容

**对于动态规划问题，代码随想录将拆解为如下五步曲，这五步都搞清楚了，才能说把动态规划真的掌握了！**

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组


# 背包问题
## 01背包
有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

这是标准的背包问题，以至于很多同学看了这个自然就会想到背包，甚至都不知道暴力的解法应该怎么解了。

这样其实是没有从底向上去思考，而是习惯性想到了背包，那么暴力的解法应该是怎么样的呢？

暴力解法应该是，使用一个类似于二叉树的结构？每层是一个物品，是否添加到背包中作为分叉，然后遍历所有不超过背包容量的情况，得到value最大的情况

### 01背包，先考虑二维数组的情况
因为有两个维度——重量和价值，所以使用二维更直观

见代码随想录的详细讲解

通过这个举例，我们来进一步明确dp数组的含义。

即**dp/[i]/[j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少**。

**要时刻记着这个dp数组的含义，下面的一些步骤都围绕这dp数组的含义进行的**，如果哪里看懵了，就来回顾一下i代表什么，j又代表什么。

### 例题
[416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)
思考的想法是——01背包是取哪些，算价值

那么等和分割，其实可以先算所有数的sum和num，然后抽取0.5$*$num(错了，不限制个数)个数，看能否满足0.5$*$sum


[1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)
01背包是重量、价值、个数、目标
个数是n，重量是stone_i，价值可能等于重量，目标

反过来想，就是拿进背包，背包的量是两个数中的较小的乘2

然后可以得到总的背包的重量，总重量减去背包重量，就是得到的结果？

那么dp i j 是代表什么呢？
### 学习
#### 1. 决策树
1. 我们希望找到与样例一致的决策树，并且规模尽可能小，但寻找极小一致树是一个很难的问题，要在$2^{2^n}$棵树中搜索，但可以通过简单启发法获得良好的近似解。
2. DTL算法，优先测试最重要的属性，最重要意味着对样例的分类具有最大的差异性，希望通过较少的测试就达到正确的分类。
- ![20200104193939.png](https://raw.githubusercontent.com/s974534426/img_for_notes/master/20200104193939.png)
- 度量分类前后数据集的纯度和不确定性
    1. 信息增益：
        - 信息熵：$H(x)=-\sum_{i=1}^{n}p_ilog_2p_i$，熵越小，不确定性越小
        - 信息增益：$g(D,A)=H(D)-H(D|A)$，信息增益越大，说明这个特征分类效果更明显
        - 缺点：信息增益偏向取值较多的特征
    2. 信息增益比：
        - $g_R(D,A)=\frac{H(D)-H(D|A)}{H(D)}$
    3. 基尼系数：
        - $Gini(p)=\sum_{i=1}^{n}p_i(1-p_i)=1-\sum_{i=1}^{n}p_k^2$

#### 2. 最近邻
- 作为分类器，渐进贝叶斯误差小于最优贝叶斯误差的两倍
- 很自然地适用于多分类任务
- 不需要训练
- 非线性决策边界
- 测试时间很慢
- 需要存储大量数据
- 对相似的函数十分敏感

#### 3. Naive Bayes 
1. $f(x)=argmaxP(y|x)=argmax\frac{P(xy)}{P(x)}=argmax\frac{P(x|y)P(y)}{P(x)}=argmaxP(x|y)P(y)$
2. 优缺点
    - 很快
    - 大多数情况下准确性高
    - 无参数
    - 输出概率
    - 适合处理多分类问题
    - 强的前提假设
    - 处理数值特征不方便

#### 4. 学习理论
1. 学习问题：存在未知的目标函数和样本集，我们能否通过学习算法在假设空间中找到一个假设，使得该假设近似等于目标函数
2. Bias-variance dilemma
    - larger hypothesis space$\rightarrow$lower bias but higher variance
    - smaller hypothesis space$\rightarrow$smaller variance but higher bias
3. PAC-learnable PAC可学习性
   1. 存在一个学习器，可以以任意高的概率输出一个错误率任意低的假设，且学习过程的时间最多以多项式时间增长

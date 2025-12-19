# **Holistic Features are almost Sufficient for Text-to-Video Retrieval**

TeachCLIP蒸馏目的：不是为了减少参数量，而是让学生**学到教师的帧级能力（细粒度）**，同时保持CLIP4Clip低推理开销

## TeachCLIP vs X-CLIP 参数量与 FLOPs 对比

| 模型                  | 参数大小(Parameters) | FLOPs (inference阶段)                  | 说明                                                         |
| --------------------- | -------------------- | -------------------------------------- | ------------------------------------------------------------ |
| **TeachCLIP**         | ≈ 200M               | 53.65G （12 帧输入）                   | 学生模型，基于 CLIP4Clip + AFA，保持轻量推理                 |
| **X-CLIP 等教师模型** | ≈ 220M               | 145G （8 帧输入） / 287G （16 帧输入） | 教师模型，引入多粒度对齐与交互(multi-grained contrastive similarity)，推理开销更大 |

## 此前模型

现有VTR模型有两类：
（1）轻量整体特征模型（CLIP4Clip）推理快，但缺少帧级细节，不准
（2）重型细粒度模型（X-CLIP，TS2-Net）利⽤帧间交互提升效果，但开销太⼤⽬标：轻量化的效率（基础）+细粒度对⻬能⼒（要学习的能⼒）

---

CLIP4Clip（2021）：

![image-20250916143625115](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250916143625115.png)

X-CLIP（2022）：

![image-20250916143549391](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250916143549391.png)

在视觉端和文本端，CLIP4Clip 与 X-CLIP 的底层编码方式（帧→patch→ViT、词→embedding→Transformer）几乎一样；区别在于上层如何使用这些特征：CLIP4Clip做池化只保留全局，X-CLIP保留细粒度并做多粒度对齐

| 模型          | 视频侧处理                  | 文本侧处理                      | 相似度计算                                                   |
| ------------- | --------------------------- | ------------------------------- | ------------------------------------------------------------ |
| **CLIP4Clip** | 帧特征 → 池化成全局视频向量 | token 向量 → 池化成全局句子向量 | 只做 **全局–全局** 对齐                                      |
| **X-CLIP**    | 保留帧特征 + 视频全局特征   | 保留 word 特征 + 句子全局特征   | 计算 **帧–句子、帧–词、视频–句子、视频–词**，再通过 AOSM 融合 |

## TeachCLIP

![image-20250916144008671](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250916144008671.png)

### student模型

- 基于 **CLIP4Clip**：视频采样成帧 → 每帧过 CLIP ViT → 得到帧特征 → 经过时序 Transformer 得到增强帧特征

- 将原本的 **mean pooling** 替换为 **AFA (Attentional frame-Feature Aggregation)**：

  - 产生帧权重 $$\{w_i\}$$ 对帧特征 $$\{\phi_i\}$$ 加权求和：
    $$
    \phi(x) = \sum_{i=1}^m w_i \cdot \phi_i
    $$

  - AFA 结构：Linear($$d \times d$$) → ReLU → Linear($$d \times 1$$) → Softmax。

- 这个AFA几乎不增加参数量

### teacher教师

- **视频级软标签 ($$y_c$$)**：教师提供视频–文本相关度分布，用于指导学生的排序。  
- **帧级软标签 ($$y_f$$)**：教师提供帧–文本相关度分布，用于指导学生的帧权重分配。

### 学习目标

训练时学生联合优化三类损失：

#### 1. 细粒度蒸馏 (Frame-level)

- 目标：与文本更相关的帧应获得更高权重；学习标准：AFA 权重分布 $$w$$ 要接近教师分布 $$y_f$$

- 学生侧：AFA模块的Softmax 输出层，代表AFA认为哪些帧更重要

- 教师侧：帧–文本相似度矩阵（AOSM中最细粒度的矩阵），代表教师认为每一帧和文本的匹配程度

- 损失函数（交叉熵）：
  $$
  \ell_{FgT} = -\frac{1}{b}\sum_{i=1}^{b}\sum_{k=1}^{m} y_f(f_{i,k},t_i)\,\log w_{i,k}
  $$

#### 2. 粗粒度蒸馏 (Video-level)

- 目标：把所有视频+文本分成mini batch，每次训练让teacher和student对于这个batch内的匹配接近

- 学生侧：视频 → AFA → 全局视频向量 $v_i$；文本 → CLIP 文本 Transformer → 全局句子向量$t_j$;在这个batch里算视频–文本相似度矩阵

- 教师侧：同样算一个视频-文本相似度矩阵，但teacher表示包含了多粒度对齐信息，更准

- 分别对于两个相似度矩阵按照行/列做softmax，用person距离匹配（强调排序一致性）

- 损失函数：
  $$
  \ell_{CgT} = \frac{1}{b}\sum_i d_p(\sigma(B_{i,\cdot}),\sigma(y_c(v_i,\cdot)))
  + \frac{1}{b}\sum_j d_p(\sigma(B_{\cdot,j}),\sigma(y_c(\cdot,t_j)))
  $$
  其中 $$\sigma$$ = softmax。

#### 3. 对比学习 (InfoNCE)

- 该阶段与teacher无关，只用学生自己的相似度矩阵  

- 正样本最大化相似度，负样本最小化相似度

- 损失函数：

  - **总损失（对称 InfoNCE）：**

  $$
  \ell_{IN} = \tfrac{1}{2}\Big[ \ell_{NCE}^{v \to t} + \ell_{NCE}^{t \to v} \Big]
  $$

  - **视频 → 文本方向：**

  $$
  \ell_{NCE}^{v \to t} = -\tfrac{1}{b}\sum_{i=1}^{b} 
  \log \frac{\exp(B_{ii}/\tau)}{\sum_{j=1}^{b} \exp(B_{ij}/\tau)}
  $$

  - **文本 → 视频方向：**

  $$
  \ell_{NCE}^{t \to v} = -\tfrac{1}{b}\sum_{i=1}^{b} 
  \log \frac{\exp(B_{ii}/\tau)}{\sum_{j=1}^{b} \exp(B_{ji}/\tau)}
  $$

#### 总损失

$$
\ell = \ell_{CgT} + \ell_{FgT} + \ell_{IN}
$$

### 推理阶段

- 不需要教师模型，也不保留帧特征
- 只用 AFA 加权得到全局视频向量，与文本做相似度计算  
- 推理成本与 CLIP4Clip 几乎相同

## TeachCLIP实验

![image-20250916152118841](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250916152118841.png)

1. **主实验（baseline）**
   - 在 MSR-VTT、MSVD、DiDeMo、ActivityNet 四个VTR数据集上，对比 TeachCLIP 和 CLIP4Clip、X-CLIP 等方法的性能
   
   尝试跑了1epoch后的结果对比，在MSRVTT-9k（9k训练，1k测试）配置下，
   
   | Models                                          | R@1  | SumR  |
   | ----------------------------------------------- | ---- | ----- |
   | CLIP4Clip （baseline）                          | 42.8 | 195.5 |
   | X-CLIP（teacher）                               | 45.3 | 200.8 |
   | TeachCLIP（论文中多教师蒸馏结果）               | 46.8 | 203.7 |
   | TeachCLIP（X-CLIP和TS2Net多教师，训练1个epoch） | 42.3 | 191.4 |
   
2. **FLOPs & 推理效率**
   - 统计不同模型在推理阶段的计算量，比较 TeachCLIP 与教师模型在 FLOPs 上的差异
3. **消融实验**
   - 分别去掉 AFA、细粒度蒸馏、粗粒度蒸馏，观察性能变化，验证各模块的重要性
4. **教师泛化性实验**
   - 换用不同教师模型（如 X-CLIP-B/16、CLIP4Clip-T），检验 TeachCLIP 是否都能有效学习

![image-20250917094913138](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250917094913138.png)

![image-20250917094922167](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250917094922167.png)

---

## 对于CLIP4clip AFA的轻量化替换

| 学生模型方案           | 参数量 (Params) | 计算量 (FLOPs) | 替换可能产生的问题                            | 可行性评估           |
| ---------------------- | --------------- | -------------- | --------------------------------------------- | -------------------- |
| **CLIPPING-AFA**       | 60-70M          | 50-60%         | MobileViT 表示能力下降，可能性能 ceiling 降低 | 高，适合大幅降耗     |
| **CenterCLIP-AFA**     | 80-85M          | 60-70%         | 聚类精度影响，重要信息丢失风险                | 中高，存储与效率折中 |
| **瘦身 CLIP4Clip-AFA** | 90-95M          | 75-80%         | 减层/降宽损失建模能力，性能较原ViT略低        | 最高，工程较简单     |

CLIPPING：

<img src="/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250924213705470.png" alt="image-20250924213705470" style="zoom:50%;" />

CenterCLIP：

<img src="/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250924215047778.png" alt="image-20250924215047778" style="zoom:50%;" />


# BiReTopK：双层可恢复性引导的Top-K层选择

> 更新时间：2026-08-04  
> 状态：D3IL原型验证；Top-3有闭环真值，Top-2只有稳定搜索结果、尚无闭环真值  
> 作用阶段：架构压缩/跨架构蒸馏的结构选择，不属于步数蒸馏

## 1. 研究目标与定位

长期目标是构建可迁移到FastWAM、DreamZero等World Action Model的demo-free压缩框架：先完成
架构压缩/跨架构蒸馏，再单独完成步数蒸馏，同时尽量保持Teacher的闭环成功率和条件多模态行为
分布。D3IL Avoiding的24个mode只用于科学评价，不能作为可迁移训练信号。

当前短期问题是改进LightDP-style层选择：静态SVD重要性和shared soft gate都可能选择剪枝瞬时
误差小、但独立Repair后并非最优的结构。我们的目标不是重做完整LightDP，而是用BiReTopK替换
其层选择模块。

建议规范名称：

- 英文：**BiReTopK: Bilevel Recoverability-guided Top-K Layer Selection**；
- 中文：**双层可恢复性引导的Top-K层选择**。

名称中的三部分分别表示：

1. `Bilevel`：Inner执行短Repair，Outer根据Repair后的held-out结果更新结构分数；
2. `Recoverability-guided`：评价剪枝结构经过恢复后的潜力，而不是剪枝瞬时误差；
3. `Top-K`：学习统一layer scores，对任意给定K产生严格K层硬子网。

## 2. 为什么需要新方法

### 2.1 D3IL闭环真值

FM-4x72-16 Teacher删除一层得到四个FM-3x72-16硬候选。相同Teacher-rollout Repair后的闭环结果
表明：

| 结构 | 保留Teacher层 | Standard-120 epoch-500 SR | 覆盖 | 熵 |
|---|---|---:|---:|---:|
| **keep013** | 0,1,3 | **91.7%** | **23/24** | **0.921** |
| keep012 | 0,1,2 | 75.0% | 21/24 | 0.911 |
| keep023 | 0,2,3 | 65.0% | 18/24 | 0.790 |
| keep123 | 1,2,3 | 27.5% | 8/24 | 0.604 |

keep013的正式Standard-480为`437/480=91.0%`、`24/24`、`H=0.929`。这说明正确问题是
“哪个硬结构最可恢复”，而不是“哪层静态权重最大”。

### 2.2 SVD先验与LightDP-style选择

LightDP-style截断SVD分数把静态Top-3指向keep012。SVD衡量权重低秩重构误差，不能直接表示某层
对noise-to-mode路由、闭环纠错或Repair可塑性的因果作用。SVD只初始化gate score，不负责初始化
模型权重；Student权重仍由Teacher对应层复制。

### 2.3 等概率shared gate仍然失败

为排除SVD混杂，四层logit全部初始化为0，并统一采用Hard Top-3前向与straight-through反向。
250 epochs结果如下：

| Outer目标 | 最终mask | 最终logits（层0/1/2/3） |
|---|---|---|
| mean Endpoint | keep012 | 0.159 / 0.368 / 0.419 / -0.073 |
| Endpoint-CVaR | keep012 | 0.124 / 0.318 / 0.361 / -0.050 |
| CVaR＋短轨迹 | keep012 | 0.069 / 0.372 / 0.358 / -0.112 |
| CVaR＋短轨迹＋双向集合距离 | keep012 | 0.070 / 0.387 / 0.398 / -0.122 |

因此错误不能再归因于SVD。机制问题包括：

1. shared-supernet使早期高频结构获得更多训练并自增强；
2. hard前向、soft反向评价的是实际不存在的“部分层”网络；
3. 交替更新没有计算结构对`w_after_repair`的影响，不是真正优化`J_after_repair`；
4. gate局部梯度不等价于独立硬子网的反事实可恢复性。

## 3. 无标签可恢复性代理

给定held-out Teacher-rollout状态`s`和相同噪声`z`，Teacher与Student分别产生轨迹。当前审计的
无标签代理包括：

```text
Endpoint-CVaR:
  e_i = mean_square(y_student_i - y_teacher_i)
  L_cvar = mean(largest 20% of e_i)

Short-trajectory:
  L_traj = mean_t MSE(x_student_t, x_teacher_t)
  t in {0.25, 0.50, 0.75}

Teacher -> Student coverage:
  L_cov = mean_i min_j distance(y_teacher_i, y_student_j)

Student -> Teacher precision:
  L_prec = mean_j min_i distance(y_student_j, y_teacher_i)
```

这些量不读取D3IL mode、环境奖励、原始demonstration或专家动作。mode只在训练结束后验证代理是否
预测闭环多样性。

在四个Top-3硬结构上，epoch-0和epoch-50的Endpoint、velocity、短轨迹和双向集合代理均正确给出
`keep013 < keep012 < keep023 < keep123`（误差越低越好）；单独Gram距离不稳定，已排除为主指标。

## 4. 短Repair校准

四个硬结构从相同Teacher映射独立初始化，不共享权重；使用全部Teacher rollout，不做成功标签筛选；
episode-id模10余2严格留作held-out评价。Endpoint-CVaR结果如下：

| Repair预算 | keep013 | keep012 | keep023 | keep123 |
|---:|---:|---:|---:|---:|
| 5 epochs | **0.194** | 0.341 | 0.771 | 3.602 |
| 10 epochs | **0.171** | 0.267 | 0.697 | 3.265 |
| 25 epochs | **0.148** | 0.175 | 0.572 | 2.967 |
| 50 epochs | **0.140** | 0.151 | 0.525 | 2.882 |

velocity-CVaR、短轨迹和双向集合距离在全部预算也给出相同排序。因此D3IL Top-3中5 epochs已经是
可用的`J_after_repair`近似。四候选完整计算只用于校准，不是最终可扩展算法。

## 5. BiReTopK方法

### 5.1 Layer scores与精确Top-K采样

对L个Teacher层学习连续分数：

```text
alpha = [alpha_1, ..., alpha_L]
alpha_i = 0 at initialization
```

给定目标层数K，训练期通过Gumbel-TopK采样严格K层硬子网：

```text
g_i ~ Gumbel(0,1)
S = TopK(alpha / temperature + g, K)
```

部署时不再采样：

```text
S_final = TopK(alpha, K)
```

因此前向从未运行soft混合网络，最终结构也天然满足`|S|=K`。

### 5.2 Paired one-swap反事实

从硬子集`S`中选一层`i`，从集合外选一层`j`，构造：

```text
S_swap = S - {i} + {j}
```

两候选只差一层，分别从相同Teacher映射初始化，使用相同minibatch、噪声、Flow时刻和短Repair
步数。随后在相同held-out状态与噪声上评价。这一配对回答：

> 在其余K-1层完全相同的上下文中，保留i还是保留j更有利于Repair后恢复？

### 5.3 双层目标

Inner loop：

```text
w'_S = ShortRepair(InitializeFromTeacher(S), D_train)
```

Outer评价：

```text
J(S) = Recoverability(w'_S, D_heldout)
```

如果`S`胜过`S_swap`，则交换出的`i`胜过交换入的`j`，用Bradley--Terry/RankNet式偏好损失更新：

```text
L_pref = softplus(-(alpha_i - alpha_j))
```

反之交换`i,j`。该更新不对hard mask或Repair过程使用straight-through梯度，而把Inner loop当作
真实黑盒反事实评价。

### 5.4 退火与分数极化

前期使用较高温度和显式探索，避免错误早锁；后期降低温度，并只在获得足够比较证据后增加Top-K
边界margin：

```text
margin = alpha_(Kth) - alpha_(K+1th)
L_margin = softplus(target_margin - margin)
```

真正需要的是稳定Top-K排序，而不是人为要求每个连续score数值恰好等于0或1。inclusion
probability可通过Gumbel-TopK Monte Carlo估计，部署gate本身严格为0/1。

### 5.5 伪代码

```text
initialize alpha_i = 0 for all L layers

for outer_round = 1 ... R:
    S = sample_exact_topk(alpha, K)
    choose i in S and j outside S
    S_swap = S - {i} + {j}

    w_S     = independent_short_repair(S, paired_train_schedule)
    w_swap  = independent_short_repair(S_swap, paired_train_schedule)

    score_S    = recoverability(w_S, paired_heldout_suite)
    score_swap = recoverability(w_swap, paired_heldout_suite)

    update alpha_i and alpha_j from paired preference
    anneal exploration/temperature only after stable evidence

return TopK(alpha, K)
```

## 6. 实验结果

### 6.1 Top-3：有闭环ground truth的正式算法验证

通用BiReTopK在`L=4,K=3`上运行seed 42--45：

| Seed | 最终结构 | Inclusion probability（层0/1/2/3） |
|---:|---|---|
| 42 | keep013 | 1.000 / 1.000 / 0.000 / 1.000 |
| 43 | keep013 | 1.000 / 1.000 / 0.00006 / 0.99994 |
| 44 | keep013 | 1.000 / 0.99872 / 0.00128 / 1.000 |
| 45 | keep013 | 1.000 / 1.000 / 0.00031 / 0.99969 |

4/4 seed选出闭环sweep中的最佳结构keep013，而SVD LightDP-style和四种等概率shared gate均选错
为keep012。这是当前最强正证据。

### 6.2 Top-2六组合校准

5-epoch校准中四个生成行为指标多数支持keep02，但velocity支持keep03。进一步执行
`keep02/keep03 × seed 42--45 × epoch 10/25`：

| 指标 | keep02获胜 | keep03获胜 | 结论 |
|---|---:|---:|---|
| Endpoint-CVaR | 8/8 | 0/8 | 稳定支持keep02 |
| Velocity-CVaR | 0/8 | 8/8 | 稳定支持keep03 |
| 短轨迹误差 | 6/8 | 2/8 | 倾向keep02 |
| Coverage | 4/8 | 4/8 | 不稳定 |
| Precision | 6/8 | 2/8 | 倾向keep02 |

这不是简单统计噪声，而是局部vector-field fidelity与最终生成行为fidelity的稳定目标冲突；同时
集合距离对held-out状态/噪声敏感，当前方差过大。

### 6.3 Top-2探索性搜索

在明确标注ground truth未确认的条件下，使用Endpoint、短轨迹、coverage、precision多数偏好运行
四seed BiReTopK。最终四组均选择keep02：

| Seed | 最终结构 | Inclusion probability |
|---:|---|---|
| 42 | keep02 | 1.000 / 0.000 / 1.000 / 0.000 |
| 43 | keep02 | 1.000 / 0.000 / 1.000 / 0.000 |
| 44 | keep02 | 1.000 / 0.000 / 0.999 / 0.001 |
| 45 | keep02 | 1.000 / 0.000 / 1.000 / 0.000 |

该结果证明搜索在当前reward下稳定，不证明keep02是闭环最优Top-2。必须通过统一长期Repair与闭环
评价建立Top-2真值。

## 7. 与相关工作的关系

### 7.1 LightDP

[LightDP](https://arxiv.org/abs/2508.00697)同时进行机器人Diffusion Policy网络压缩和步数
蒸馏，并强调pruning后的retraining/recoverability。任务最接近，但其层选择使用静态SVD先验和
可学习门控，训练可访问demonstration，也没有系统评价mode preservation。BiReTopK是对其架构
选择模块的替代，不应被描述为完整LightDP复现。

### 7.2 Differentiable Subset Pruning

[Differentiable Subset Pruning of Transformer Heads](https://aclanthology.org/2021.tacl-1.86/)
为Transformer head学习importance variables，并用Gumbel-TopK严格控制K。它是Top-K数学形式
的直接先例；区别是使用连续松弛和任务loss，没有独立短Repair或Teacher-relative可恢复性。

### 7.3 LeGR

[LeGR](https://openaccess.thecvf.com/content_CVPR_2020/html/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.html)
学习跨层filter global ranking，并用同一ranking产生不同预算的CNN。它是“学习统一压缩排序”最
接近的工作，但目标是分类准确率/延迟，不涉及生成策略、Teacher rollout或mode保持。

### 7.4 DARTS与shared-weight NAS

[DARTS](https://arxiv.org/abs/1806.09055)提供Inner权重、Outer架构参数的经典双层形式，但依赖
连续soft超网。[Few-Shot NAS](https://proceedings.mlr.press/v139/zhao21d.html)等工作进一步指出
weight-sharing架构评价存在失真。我们“独立代理正确、shared gate错误”的实验是该问题在机器人
生成策略层选择上的直接实例。

### 7.5 Pairwise-ranking NAS

[RankNAS](https://arxiv.org/abs/2109.07383)和
[BRP-NAS](https://proceedings.neurips.cc/paper/2020/hash/768e78024aa8fdb9b8fe87be86f64745-Abstract.html)
说明NAS可以只学习架构相对排序而非绝对性能。它们通常训练额外的architecture predictor；
BiReTopK直接通过同上下文one-swap短Repair产生偏好并更新layer scores。

### 7.6 AMC与MetaPruning

[AMC](https://openaccess.thecvf.com/content_ECCV_2018/html/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.html)
用RL黑盒验证回报搜索压缩策略；
[MetaPruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_MetaPruning_Meta_Learning_for_Automatic_Neural_Network_Channel_Pruning_ICCV_2019_paper.html)
训练PruningNet为候选结构快速生成权重，再进化搜索。前者支持不可微结构回报，后者提示未来可以
摊销FastWAM候选的短Repair成本。

## 8. 创新边界

不能声称：

- 首次学习layer score或global ranking；
- 首次使用Gumbel-TopK/严格K子集；
- 首次用双层优化搜索架构；
- 首次用pairwise ranking做NAS；
- 首次优化pruning后的recoverability，LightDP已明确涉及。

当前可能成立、但仍需FastWAM和更多任务验证的贡献是：

1. 证明LightDP/DARTS式shared soft gate不能可靠预测生成式机器人策略的post-repair可恢复性；
2. 以独立短Repair后的paired Teacher-relative偏好学习任意固定K的硬层子集；
3. 不访问原始demonstration、mode标签或环境奖励；
4. 将架构选择评价从平均任务fidelity扩展到条件轨迹和多模态行为保持；
5. 把架构压缩与步数蒸馏分阶段验收，并面向通用WAM迁移。

一句话定位：

> BiReTopK不是新的Top-K数学工具，而是把exact-K子集学习、黑盒双层优化和post-repair偏好评价
> 组合成面向demo-free多模态机器人生成策略的可恢复架构选择方法。

## 9. 有效性审计与结论边界

### 已完成

1. Top-3存在四候选完整闭环ground truth；
2. SVD、等概率shared gate与BiReTopK形成机制对照；
3. 代理在独立硬结构上预测正确；
4. 5/10/25/50预算排序稳定；
5. 通用Top-3实现4/4 seed选择keep013；
6. Top-2实现4/4 seed稳定选择keep02。

### 尚未完成

1. Top-2没有统一长期Repair＋闭环ground truth；
2. 旧Top-3完整Repair使用过成功rollout筛选，而新搜索使用全部rollout，最终公平协议仍需补齐；
3. 只在4层Teacher验证，尚未证明30层FastWAM上的计算效率和score稳定性；
4. Top-2集合距离方差高，通用reward尚未完全确定；
5. 单一layer score隐含近似稳定排序；强非传递层间交互可能需要额外建模。

当前最严格结论：

> BiReTopK已在具有完整闭环真值的Top-3任务上稳定纠正LightDP-style错误层选择；在Top-2上
> 表现出强搜索稳定性，但尚不能证明找到闭环最优结构，也尚不能宣称已验证任意K和大型WAM。

## 10. FastWAM迁移设计

对于L约30层的FastWAM，算法本身仍只维护`L`个layer scores，每轮评价两个只差一层的硬Top-K
候选，避免枚举`C(L,K)`。但必须重新校准：

1. 最短有效Repair预算，不能直接照搬D3IL的5 epochs；
2. Teacher/Student的动作轨迹、world latent与视频预测误差；
3. 同条件多噪声的Teacher-relative coverage/precision；
4. score在不同上下文中的swap胜率，审计是否存在非传递交互；
5. wall-clock、显存和查询成本；
6. selected hard model的真实闭环任务成功率与行为多样性。

若同一层在不同上下文频繁出现偏好反转，单一score假设失效。此时才考虑低秩pairwise交互：

```text
Score(S) = sum_{i in S} alpha_i + sum_{i,j in S} u_i^T u_j
```

不应在没有反转证据前增加复杂度。若希望一次支持多个K，可令score以压缩率为条件：

```text
alpha_i(K) = f_theta(layer_embedding_i, K/L)
```

## 11. 下一步实验顺序

1. `keep02`与`keep03`做同协议长期Repair和配对Standard-120，确定Endpoint还是Velocity更能预测
   闭环可恢复性；必要时扩展到六个Top-2组合；
2. 在all-rollout统一协议下复核Top-3 selected与LightDP keep012；
3. 汇总Top-3/Top-2的每层swap偏好矩阵和上下文反转率；
4. FastWAM先做小压缩率pilot，例如`30→29`与`30→27`；
5. 只有代理能预测FastWAM长Repair/下游表现后，才做更激进K和步数蒸馏；
6. 架构压缩通过后，再单独执行few-step/one-step distillation并审计mode变化。

## 12. 代码与产物

- 通用BiReTopK：`train_discrete_bilevel_depth_search.py`
- 独立硬结构Repair：`train_recoverable_depth_pruning.py`
- 无标签代理：`audit_depth_recoverability_proxies.py`
- 早期shared-gate负对照：`train_recoverability_guided_lightdp.py`
- Top-3/Top-2根目录：`logs/avoiding/biretopk_v1/`
- 原始categorical删除特例：`logs/avoiding/categorical_bilevel_depth_search_v1/`
- Shared-gate消融：`logs/avoiding/recoverability_guided_lightdp/`
- 离散短Repair校准：`logs/avoiding/discrete_repair_optimizer_v1/`
- 顺序验证脚本：`scripts/run_biretopk_sequential_validation.sh`
- Top-2冲突审计：`scripts/run_k2_disambiguation_and_search.sh`
- 深度压缩闭环真值：`logs/avoiding/tinysr_depth_recoverability/`

相关总览：`07_可恢复性引导的深度压缩实验.md`与
`09_LightDP论文分析与保模态对照.md`。

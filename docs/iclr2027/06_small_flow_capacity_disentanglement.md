# Small-FM-16：模型容量、求解步数与蒸馏退化的解耦实验

## 1. 背景

当前 Flow Matching 结果形成了两条明显不同的现象：

```text
Full teacher, 16-step:        95.8% success, 24 modes, H=0.945
Same-size distilled, 1-step: 97.3% success, 24 modes, H=0.895
7.24x small distilled, 1-step:
  500 epochs:                80.6% success, 12 modes, H=0.147
  2000 epochs:               16.0% success,  9 modes, H=0.309
```

同规模 16→1 蒸馏没有明显 mode collapse，而跨模型规模蒸馏出现显著退化。因此必须判断：退化来自小模型本身的表达能力、1-step 映射难度，还是跨规模蒸馏目标。

## 2. 为什么更长蒸馏可能使闭环更差

### 2.1 Teacher–student capacity gap

大 teacher 表达的是“状态与噪声到多种合理动作模式”的映射。37,982 参数的小 student 可能无法同时拟合所有模式；继续优化平均 shortcut/CFM loss 时，降低 loss 的容易路径可能是优先拟合高频模式，牺牲低频模式和闭环恢复行为。

知识蒸馏研究已经观察到：teacher 与 student 容量差距过大时，小 student 可能无法模仿大 teacher，更强或训练更充分的 teacher 不一定产生更好的 student。参考：

- Cho & Hariharan, [On the Efficacy of Knowledge Distillation](https://arxiv.org/abs/1910.01348), 2019.
- Guo et al., [Reducing the Teacher-Student Gap via Spherical Knowledge Distillation](https://arxiv.org/abs/2010.07485), 2020.

### 2.2 离线目标与闭环目标错位

训练状态来自专家数据分布，执行状态来自 student 自己诱导的分布。早期小误差会改变后续状态，并在闭环中累积：

```text
training: s ~ p_expert
rollout:  s ~ p_student
```

这与模仿学习中的 covariate shift 和 compounding error 一致。DAgger 通过在 student 实际访问的状态上查询 teacher 并聚合数据来缓解该问题：

- Ross et al., [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686), 2011.

当前实验中，2000-epoch 小模型的离线验证 loss 从约 0.296 降至 0.187，但闭环成功率从 80.6% 降到 16.0%，正说明离线 loss 不是可靠的闭环代理。

### 2.3 Pointwise loss 不完整描述多模态分布

Shortcut MSE 能评价每个训练样本是否接近 teacher target，但不直接评价 24-mode 覆盖、模式概率、轨迹闭环可恢复性和集合级分布。近期 Flow policy 蒸馏工作也将 averaged trajectories 和 distribution collapse 作为单步 student 的核心问题，并使用集合级匹配：

- Dong et al., [From Flow to One Step: Real-Time Multi-Modal Trajectory Policies via IMLE-based Distribution Distillation](https://arxiv.org/abs/2603.09415), 2026.
- Liu et al., [Short-to-Long Distillation](https://openreview.net/forum?id=lTTqwVEohr), 2025.

### 2.4 500 与 2000 epochs 不是简单续训

两组实验分别重设 cosine scheduler。500-epoch run 在第 500 轮已接近最低学习率；2000-epoch run 在第 500 轮仍处于较高学习率。因此现有结果证明“更长训练配置更差”，但尚不能证明“从同一个 500-epoch checkpoint 继续训练必然变差”。

## 3. 必要的解耦矩阵

| 实验 | 参数量 | 推理步数 | 训练方式 | 解答的问题 |
|---|---:|---:|---|---|
| A. Large-FM-16 | 274,826 | 16 | 原始数据完整 CFM 训练 | 大模型基准 |
| B. Small-FM-16 | 37,982 | 16 | 原始数据完整 CFM 训练 | 小模型自身容量是否足够 |
| C. Small-FM-1 solver | 37,982 | 1 | 使用 B checkpoint 直接一步求解 | 小模型内部的降步损失 |
| D. Small-Distilled-1 | 37,982 | 1 | 从 Large-FM-16 蒸馏 | 跨规模蒸馏的额外影响 |

差异可解释为：

```text
A → B：模型容量影响
B → C：求解步数影响
C → D：跨规模蒸馏目标与 teacher–student gap
```

判据：

- 如果 B 已经 mode collapse：小模型容量是主要原因。
- 如果 B 正常而 C collapse：小模型难以表达 1-step flow map。
- 如果 B、C 正常而 D collapse：跨规模蒸馏目标是主要原因。
- 如果 B 成功率低但模式仍多：容量主要影响控制精度。
- 如果 B 成功率高但模式少：小模型自身存在生成分布容量瓶颈。

## 4. Small-FM-16 正式配置

- 数据、预处理、历史窗口、seed 与 Large-FM-16 相同。
- 训练目标：原始线性 CFM，不使用 teacher 或蒸馏 loss。
- Transformer：2 层、embedding 36、3 heads。
- 参数量：37,982。
- 训练：5000 epochs。
- batch size：256。
- 验证间隔：50 epochs。
- EMA checkpoint：`eval_best_flow.pth`。
- 闭环评估：同一 checkpoint 分别使用 16-step 与 1-step，各 480 条。

## 5. 实验状态与结果

训练与两组480-rollout均已完成。

- 训练耗时：35分44秒。
- 最终记录的验证CFM loss：0.2204。
- 参数量：37,982。
- 16-step与1-step使用同一个最优EMA checkpoint，仅改变solver steps。

| 实验 | 参数量 | 步数 | 成功数 | 成功率 | 成功模式 | 归一化模式熵 |
|---|---:|---:|---:|---:|---:|---:|
| A. Large-FM-16 | 274,826 | 16 | 460/480 | 95.8% | 24/24 | 0.945 |
| B. Small-FM-16 | 37,982 | 16 | 319/480 | 66.5% | 24/24 | 0.942 |
| C. Small-FM-1 solver | 37,982 | 1 | 268/480 | 55.8% | 16/24 | 0.700 |
| D. Small-Distilled-1，500 ep | 37,982 | 1 | 387/480 | 80.6% | 12/24 | 0.147 |
| Same-size Distilled-1参考 | 274,826 | 1 | 467/480 | 97.3% | 24/24 | 0.895 |

轨迹图：

- Small-FM-16：`logs/avoiding/small_fm16/eval_step16_480/trajectory_comparison.png`
- Small-FM-1 solver：`logs/avoiding/small_fm16/eval_step1_480/trajectory_comparison.png`

### 5.1 A→B：模型容量主要降低控制成功率，没有导致模式塌缩

参数量缩小7.24倍后，16-step成功率从95.8%降到66.5%，说明容量对速度场精度和闭环控制能力有明显影响。但是Small-FM-16仍覆盖24/24模式，熵为0.942，几乎等于大模型的0.945。

因此，小模型结构本身能够表示完整多模态分布；“小模型容量不足必然导致mode collapse”不符合当前数据。

### 5.2 B→C：小模型内部降到1-step会损失模式

同一个Small-FM checkpoint从16-step改为1-step后：

- 成功率：66.5% → 55.8%；
- 模式覆盖：24 → 16；
- 模式熵：0.942 → 0.700。

这部分退化完全来自solver步数，因为网络和checkpoint均未改变。小模型学习到的连续速度场可以通过多步积分表达全部模式，但单次Euler更新不足以准确近似完整transport map。

### 5.3 C→D：跨规模蒸馏提高成功率，但强烈压缩模式分布

与Small-FM-1 solver相比，跨规模直接蒸馏student的成功率从55.8%提高到80.6%，说明teacher shortcut target有效提高了单步控制精度。但模式覆盖从16降到12，熵从0.700降到0.147，输出高度集中到少数高频路径。

因此，当前最准确的归因不是“小模型容量单独导致collapse”，而是：

```text
有限student容量
+ 极端1-step transport
+ pointwise跨规模蒸馏目标
→ 用模式集中换取较高闭环成功率
```

### 5.4 与同规模蒸馏对比

大模型同规模16→1蒸馏仍有24/24模式和0.895熵，说明1-step蒸馏本身并不必然collapse。显著塌缩只出现在小student的跨规模蒸馏中，支持teacher–student capacity gap与目标压缩共同作用的解释。

### 5.5 最终解耦结论

1. 小模型完整训练能够保持多模态，但成功率低于大模型。
2. 小模型直接使用1-step solver会同时损失成功率和模式。
3. 跨规模蒸馏恢复了大量成功率，却进一步牺牲模式均衡性。
4. 当前研究现象可表述为“跨规模单步蒸馏中的success–diversity trade-off”，而不是简单的“小模型容量导致mode collapse”。

### 5.6 当前关键结论与尚缺对照

当前A/B/C/D证据链支持以下受限结论：

> 在Avoiding任务中，Large-FM-16到7.24倍压缩Small-FM-1的pointwise shortcut蒸馏，在提高闭环成功率的同时显著压缩模式分布。

该结论不能扩展为“任何大模型蒸馏小模型都会mode collapse”。现有数据已排除两个过度简化的解释：

- Small-FM-16有24/24模式和0.942熵，因此小模型结构本身不必然collapse。
- 大模型同规模16→1蒸馏有24/24模式和0.895熵，因此1-step蒸馏本身也不必然collapse。

但仍缺少实验E：

| 实验 | Teacher | Student | 步数 | 目的 |
|---|---|---|---:|---|
| E. Small→Small Distilled-1 | Small-FM-16 | 同结构Small-FM | 1 | 隔离teacher–student规模差距 |

判断标准：

- 若E的熵明显高于Large→Small的0.147，则capacity gap是跨规模蒸馏collapse的重要来源。
- 若E同样接近0.147，则主要问题更可能是小模型承载1-step pointwise蒸馏映射的能力，而不是teacher规模。
- E还应与未蒸馏Small-FM-1 solver的55.8% / 16 modes / 0.700熵比较，判断蒸馏带来的success–diversity交换。

### 5.7 实验E：Small→Small同规模蒸馏结果

- Teacher：Small-FM-16，37,982参数。
- Student：同结构Small-FM，37,982参数。
- 蒸馏：16→1，500 epochs。
- 最优epoch：389。
- 最优验证loss：0.0427。

| 方法 | 成功数 | 成功率 | 成功模式 | 归一化模式熵 |
|---|---:|---:|---:|---:|
| Small-FM-16 | 319/480 | 66.5% | 24/24 | 0.942 |
| Small-FM-1 solver | 268/480 | 55.8% | 16/24 | 0.700 |
| Small→Small Distilled-1 | 287/480 | 59.8% | 23/24 | 0.885 |
| Large→Small Distilled-1 | 387/480 | 80.6% | 12/24 | 0.147 |

Small→Small蒸馏相对直接1-step solver，将成功率从55.8%提高到59.8%，模式从16恢复到23，熵从0.700恢复到0.885。它接近Small-FM-16的多样性，而没有出现Large→Small蒸馏中的严重模式集中。

因此，teacher–student规模差距是当前mode collapse的重要因素。Large teacher提供了小student难以同时表达的高精度、多模式target；pointwise目标使student通过集中到少数高成功率模式获得80.6%成功率。Small teacher的目标复杂度与student容量匹配，蒸馏能保留23种模式，但成功率提升有限。

当前最准确的结论是：

> 在本任务和蒸馏实现下，跨规模Large→Small单步蒸馏产生明显success–diversity trade-off；同规模Small→Small蒸馏基本保留多模态，因此collapse不能归因于小模型或一步蒸馏本身，而与teacher–student capacity gap密切相关。

轨迹图：`logs/avoiding/small_to_small_distill_500/eval480/trajectory_comparison.png`

## 6. 公平性与局限

- “相同数据规模”指相同数据、batch size 和 epoch/update 数；小模型仍可拥有适合自身的优化超参数，但本轮先保持与大模型相同设置建立最直接基线。
- 至少应报告 480-rollout；后续若差异接近，需要补 3 个训练 seed。
- 训练配置中的离线 train/test 数据是否严格独立受原项目配置限制，闭环 rollout 是主要证据。
- B 与 C 共用同一个 checkpoint，可干净隔离 solver 步数；D 使用不同训练目标，比较时必须明确。

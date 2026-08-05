# 步数蒸馏：一致性蒸馏与CTM实验

## 1. 文档边界

本文件只记录已经完成结构Repair的Flow student如何从多步压缩到低步数，重点为16→1。Repair后16步结果是CTM输入基线；CTM后1步结果是步数压缩结果。

## 2. 方法关系

### 2.1 Endpoint fidelity

对相同`state, noise`，直接让student一步输出匹配teacher多步终点。它是paired fidelity distillation，不是完整anytime consistency。

### 2.2 Consistency Distillation

训练student在同一teacher probability-flow轨迹上的不同时间点产生一致终点。标准CD含边界预条件，也可以扩展到多时间映射。

### 2.3 Flow-CTM

学习`G(x_t,t,s,state)`的anytime-to-anytime映射。当前Boundary-CTM不是与CD无关的新方法，而是带官方式边界预条件的Flow consistency/trajectory实现。

### 2.4 DSM辅助

使用denoising/flow matching目标稳定训练。它需要可用的数据或teacher生成的替代轨迹；不是天然必须依赖原始专家动作。

### 2.5 DDIL

在student诱导的Flow状态上查询teacher，改善分布偏移；已有结果显示它可提高SR，但也可能强化少数安全模式。

## 3. DDPM前期蒸馏结论

DDPM的8→4与4→2保持较高成功率和多数模式，2→1出现明显退化。DDPM蒸馏使用确定性DDIM轨迹作为常见teacher target，但模型本身仍源于DDPM训练；DDIM是采样/轨迹选择，不代表研究对象变成另一模型。

这些结果用于验证低步数蒸馏流程，当前论文主线仍为Flow Matching。

## 4. Flow同规模与跨规模基线

| 方法 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| `FM-4x72-16-Full` | 95.8% | 24 | 0.945 |
| `FM-4x72-1-Solver` | 84.6% | 15 | 0.667 |
| `FM-4x72-1-Distill-4x72` | 97.3% | 24 | 0.895 |
| `FM-3x48-16-Full` | 92.7% | 24 | 0.965 |
| `FM-3x48-1-Solver` | 82.1% | 14 | 0.569 |
| `FM-3x48-1-Distill-3x48`，Teacher-init | 92.1% | 23 | 0.868 |

同架构Teacher-init蒸馏不必然导致mode collapse；跨架构迁移与初始化误差才是关键前置因素。

## 5. Endpoint与Flow-CTM多seed结果

固定`FM-4x72-16-Full→FM-3x48-1`并使用Full-3x48 warm start：

| 方法 | Standard-480三seed均值SR | 平均覆盖 | 平均熵 |
|---|---:|---:|---:|
| Teacher endpoint | **91.0%** | **22.7** | **0.837** |
| Flow-CTM＋DSM | 84.2% | 12.3 | 0.431 |

当前证据表明，有限容量机器人策略中直接paired fidelity比自举trajectory consistency更能保护多模态分布。但这不是跨任务普遍结论。

## 6. Repair→CTM阶段定位

早期五损失repair后：

| 初始化 | Repair后16步 | CTM后1步 |
|---|---|---|
| Activation-aware | 87.5% / 3 / 0.034 | 99.2% / 1 / 0.000 |
| PCA | 70.0% / 4 / 0.100 | 98.3% / 1 / 0.000 |
| Early-layer | 75.8% / 6 / 0.146 | 99.2% / 1 / 0.000 |

Repair已经损失大部分模式，CTM在前10–50 epochs内进一步集中为单一路径。因此不能把最终collapse完全归因于CTM。

Corrected Dynamic checkpoint扫描：

| Repair epoch | CTM前16步 | CTM后1步 |
|---:|---|---|
| 25 | 4.2% / 2 / 0.157 | 24.2% / 5 / 0.368 |
| 50 | 5.0% / 3 / 0.318 | 42.5% / 6 / 0.338 |
| 100 | 18.3% / 5 / 0.345 | 60.8% / 5 / 0.374 |
| 300 | 55.8% / 7 / 0.233 | 69.2% / 5 / 0.300 |

CTM对Repair不足的student可同时提高SR和覆盖；当Repair较充分时才更明显地压缩覆盖。

## 7. 当前实验原则

1. CTM输入必须附带16步SR、Coverage和H；
2. CTM输出必须与同checkpoint的1-step solver-only比较；
3. 同时比较Endpoint、标准CD/Boundary-CTM及必要的DSM；
4. checkpoint选择不可只依赖离线consistency loss；
5. 候选方法补Standard-480、Paired-1000/JS和多seed。

## 8. MiniLMv2 Repair后接Flow-CTM

从1000-epoch完整MiniLMv2 checkpoint训练500-epoch Flow-CTM：

| 阶段 | 步数 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| MiniLMv2 Repair后 | 16 | 37.5% | 13/24 | 0.658 |
| Flow-CTM后 | 1 | **91.7%** | **5/24** | **0.174** |

CTM恢复54.2个SR百分点，但丢失8种模式，熵降低0.484。这说明Repair阶段获得的多模态关系没有被当前CTM链路保留；CTM将策略集中到少数闭环安全路径。该结果比只观察最终1步策略更清楚地定位了collapse发生位置。

轨迹：

- CTM前：`logs/avoiding/teacher_generated_structure_wave2_1000/minilmv2_relation/eval120/trajectory_comparison.png`
- CTM后：`logs/avoiding/teacher_generated_minilmv2_followup/minilm1000_ctm500/eval120/trajectory_comparison.png`

该对照目前是论文主线中最直接的mode-collapse实例：同一Repair checkpoint经过步数蒸馏后，闭环成功率显著提高，但随机噪声对应的多条可行路径被压缩为少数安全路径。下一步应比较300与2000轮MiniLMv2 checkpoint接相同CTM，判断CTM输出多样性是否受输入Repair分布影响，并在CTM中加入弱跨噪声关系保持。

## 9. 当前研究问题

最终目标不是单独寻找最高SR的CTM，而是回答：

> 在有限容量、teacher-only机器人策略蒸馏中，如何让结构Repair保留noise-to-mode函数，并使后续一步consistency distillation不将其压缩为少数安全路径？

## 10. 3500与6000 Repair起点的Standard-480验证

四卡同步对照设置：

| Repair起点 | 选择理由 | Repair评估 | CTM评估 |
|---|---|---|---|
| 3500 | Standard-120覆盖最高：20/24 | 16-step Standard-480 | 500-epoch CTM，1-step Standard-480 |
| 6000 | 续训阶段SR最高：65.0% | 16-step Standard-480 | 500-epoch CTM，1-step Standard-480 |

该实验用于判断CTM后的模式上限是否受输入Repair多样性决定，并量化高覆盖与高SR起点各自的SR增益、覆盖损失和熵下降。运行目录：`logs/avoiding/teacher_generated_minilmv2_repair_ctm_480/`。

### 10.1 最终Standard-480结果

| Repair起点 | 阶段 | SR | 覆盖 | 熵 |
|---:|---|---:|---:|---:|
| 3500 | Repair，16步 | 240/480，50.0% | 21/24 | 0.842 |
| 3500 | CTM，1步 | 299/480，62.3% | 11/24 | 0.429 |
| 6000 | Repair，16步 | 277/480，57.7% | 22/24 | 0.845 |
| 6000 | CTM，1步 | 318/480，66.3% | 12/24 | 0.451 |

### 10.2 Collapse幅度

| Repair起点 | SR变化 | 覆盖变化 | 熵变化 |
|---:|---:|---:|---:|
| 3500 | +12.3pp | −10 | −0.414 |
| 6000 | +8.5pp | −10 | −0.394 |

结论：

1. Standard-480确认MiniLMv2 Repair已恢复大部分teacher模式：21–22/24；
2. 相同Flow-CTM对两个起点都稳定丢失10种模式，collapse不是某个单一Repair checkpoint的偶然现象；
3. 6000起点在CTM前后均略优，说明更好的Repair输入能提高CTM输出上限，但不能解决collapse；
4. 当前下一优先级应从延长Repair训练转为`CTM＋弱跨噪声关系`、`CTM＋endpoint anchor`及二者组合；
5. 轨迹和metrics位于`logs/avoiding/teacher_generated_minilmv2_repair_ctm_480/epoch_{3500,6000}_{repair16,ctm1}/eval480/`。

## 11. Mode-preserving CTM TODO

在Repair四卡消融选出最佳16步checkpoint后，固定该checkpoint进行第二个四卡因子实验：

| GPU | CTM配置 | 初始权重 |
|---:|---|---|
| 0 | Flow-CTM＋DSM | `lambda_dsm=0.1` |
| 1 | CTM＋DSM＋teacher endpoint | `lambda_endpoint=0.1` |
| 2 | CTM＋DSM＋同状态K=4输出关系 | `lambda_mode=0.03` |
| 3 | CTM＋DSM＋endpoint＋K=4输出关系 | `lambda_endpoint=0.1, lambda_mode=0.03` |

同状态多噪声关系直接约束输出终点的中心化Gram，而不是只对齐内部attention：

```text
L_mode = MSE(
  Gram(center(student_endpoints)),
  Gram(center(teacher_endpoints))
)
```

Endpoint anchor保持逐噪声身份对应，Gram保持一组噪声结果的相对几何。训练500 epochs，保存100/250/500并做Standard-120。首轮目标为`SR≥65%、Coverage≥18、H≥0.70`；通过者补Standard-480、Paired-1000/JS和至少两个额外seed。

执行状态：

- [x] 以Velocity-250作为统一16步Repair起点；
- [x] Flow-CTM＋DSM基线；
- [x] CTM＋DSM＋弱teacher endpoint anchor；
- [x] CTM＋DSM＋同状态K=4输出Gram；
- [x] endpoint与Gram组合；
- [x] 各组100/250/500 checkpoint的Standard-120；
- [ ] 候选Standard-480；
- [ ] Paired-1000/JS与多seed。

注意：此前3500/6000 MiniLMv2起点的CTM实验已完成并定位了collapse，但不能替代上述实验；新的TODO要求固定当前最佳Velocity-250起点，验证约束能否在更高SR Repair基础上保留模式。

### 11.1 四因子首轮结果

Standard-120：

| 方法 | 最佳筛选epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Flow-CTM＋DSM | 250 | **77.5%** | 8/24 | 0.390 |
| CTM＋DSM＋Endpoint | 100 | 69.2% | 9/24 | 0.375 |
| CTM＋DSM＋Gram | 250 | 73.3% | 9/24 | 0.396 |
| CTM＋DSM＋Endpoint＋Gram | 100 | 70.8% | 8/24 | 0.394 |

若只看首轮多样性，Gram-100为`66.7%/10/H=0.485`。Gram最多挽回约2种模式，Endpoint没有改善当前CTM，组合也未显示互补性；四组都没有达到`Coverage≥18、H≥0.70`。因此弱输出几何正则不足以阻止一步CTM压缩已经由Repair恢复的多模态分布。

运行目录：`logs/avoiding/velocity250_mode_ctm/`。

## 12. Demonstration-assisted CTM oracle

### 12.1 目的与边界

该实验用于区分两种解释：

1. Teacher rollout buffer没有充分覆盖原始状态—动作分布；
2. CTM目标、有限容量表示或一步优化本身压缩模式。

两组均从相同Velocity-250 Repair checkpoint出发，使用Flow-CTM＋DSM=0.1、seed 42、500 epochs，并评估100/250/500 checkpoint。区别只在CTM训练数据：

- `demonstration_only`：100%原始示范状态与专家action chunk；
- `rollout_demo_50_50`：Teacher rollout和原始示范等权采样。

示范窗口使用Teacher部署包中的统计量归一化；CTM trajectory target仍由Teacher产生。该实验明确属于`Demonstration-assisted CTM oracle`：

```text
demonstration_free: false
uses_original_demonstrations: true
uses_expert_actions: true
```

它是数据上界和因果诊断，不是最终demonstration-free方法。

### 12.2 Standard-120结果

| CTM训练数据 | Epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Teacher rollout only基线 | 100 | 70.8% | 8/24 | 0.487 |
| Teacher rollout only基线 | 250 | 77.5% | 8/24 | 0.390 |
| 100% Demonstration | 100 | 78.3% | **12/24** | 0.533 |
| 100% Demonstration | 250 | **81.7%** | 11/24 | 0.350 |
| 100% Demonstration | 500 | 60.8% | 9/24 | 0.473 |
| 50% Rollout＋50% Demonstration | 100 | 80.0% | 11/24 | **0.538** |
| 50% Rollout＋50% Demonstration | 250 | 80.8% | 9/24 | 0.352 |
| 50% Rollout＋50% Demonstration | 500 | 65.8% | 7/24 | 0.418 |

运行目录：`logs/avoiding/demonstration_assisted_ctm_oracle/`。

评估目录中的通用`metrics.json`字段`uses_original_demonstrations:false`描述评估过程本身，不代表模型训练来源；训练来源以各模型目录的`model/metrics.json`为准，其中已明确记录`demonstration_free:false`。后续汇总工具应区分`training_data_provenance`与`evaluation_data_provenance`，避免误读。

### 12.3 结论

1. 原始示范确实缓解collapse。epoch 100时，纯Demo相对rollout-only实现`SR +7.5pp、覆盖 +4、H +0.046`；
2. Mixed-100达到更高SR和本组最高熵，但覆盖略低于纯Demo，说明真实示范状态和Teacher闭环状态具有一定互补性；
3. 训练到250/500后覆盖和熵重新下降，离线loss最佳epoch也不是闭环多样性最佳epoch，继续证明必须按SR—Coverage—H选择checkpoint；
4. 最佳结果仍只有12/24，远低于Velocity-250 Repair起点的22/24。数据覆盖不足是mode collapse的一个原因，但不是全部原因；
5. CTM自举目标、一步函数的优化偏置以及跨架构Student尚不完善的表示基底，仍会把概率质量集中到少数易拟合、闭环成功率高的路径。

本实验形成一个重要诊断：

> 原始示范能把跨规模一步CTM的多样性从8种提高到约11–12种，但不能恢复Repair阶段的22种模式；因此后续demonstration-free方案既要改善Teacher-query覆盖，也必须改变一步分布蒸馏目标。

## 13. 为什么同规模蒸馏容易保留mode，而跨规模容易collapse

### 13.1 当前对比不是单一“容量”变量

同规模/Full-student warm start：

- `FM-3x48-16-Full`先用完整原始示范独立训练；
- 它已经达到`92.7%/24/H=0.965`，具有适合3×48结构自身的闭环控制表示；
- 其noise→mode映射、低频模式和纠错行为在步数蒸馏前已经建立；
- 后续蒸馏主要解决16→1时间压缩和Teacher对齐。

跨规模Teacher-derived路线：

- 从4×72 Teacher经过层/宽度投影得到3×48；
- 投影后功能、内部坐标系和模式分隔均不完整；
- Repair只使用有限Teacher rollout，必须同时重建压缩表示、闭环控制和多模态noise→mode映射；
- CTM输入虽恢复到22种模式，但SR只有63.1%，说明许多模式仍位于脆弱、低裕量的决策区域；
- 一步CTM最容易通过增加少数安全路径的概率来降低平均损失和提高SR。

所以“同规模可以、跨规模不行”不能直接解释为3×48参数不足。同一个3×48若完整训练可以覆盖24种模式，说明容量足以表示这些模式；差别主要在可达的表示基底、监督覆盖和优化路径。

### 13.2 四个相互作用的原因

1. **表示基底失配**：Teacher的hidden channels、attention heads和层级功能不能无损投影到Student。Teacher中彼此分离的模式，在Student表示空间可能已经靠近或重叠；
2. **功能裕量不足**：跨规模Repair虽然恢复22种成功模式，但整体SR远低于Full-trained Student。低频模式可能只在少数噪声和状态下成功，CTM更新很容易将其抹除；
3. **数据覆盖不足**：固定Teacher rollout对低频状态和专家恢复动作覆盖有限。本次Demo oracle把覆盖由8提高到12，验证这一因素真实存在；
4. **平均损失与闭环目标不一致**：CTM/DSM按样本平均优化，高频、低误差和安全路径贡献更稳定。牺牲少量低频模式可能降低平均loss并提高总体SR，因此产生“高SR、低多样性”的局部最优。

可用以下因果链理解：

```text
跨架构投影
  → 模式在Student表示中分隔变弱
  → 有限rollout Repair只恢复脆弱的22-mode支持
  → CTM平均损失偏向高频/安全映射
  → 一步输出集中为8–12种模式
```

### 13.3 当前能够和不能够声明的结论

可以声明：

> 严重collapse不是有限Student参数量单独造成的，而是跨架构表示迁移误差、有限Teacher数据覆盖与一步CTM优化偏置共同造成；原始示范可以部分缓解，但不能消除该问题。

暂时不能声明：

- 所有跨规模蒸馏必然collapse；
- CTM理论上不能保持多模态；
- 完整训练Student与Teacher-derived Student的差异完全来自原始数据，因为二者同时还存在初始化和优化路径差异。

下一步最有信息量的验证是：在不访问原始示范的前提下扩大并主动均衡Teacher-query覆盖，同时将逐点/自举CTM改为条件样本集合的distribution matching；若在相同3×48和相同Repair起点上恢复模式，才说明提出的方法真正解决了跨规模、demonstration-free场景的问题。

## 14. BMD方向

《Behavioral Mode Discovery for Fine-tuning Multimodal Generative Policies》使用离散latent、noise steering和trajectory-level互信息发现预训练生成策略中的行为模式，并以互信息奖励缓解RL fine-tuning collapse。该方法与当前问题高度相关，但原版针对RLFT，不能直接等同于CTM约束。

本项目将优先研究Teacher-relative版本：在冻结Repair checkpoint上发现latent modes，构造无监督mode-balanced Teacher buffer，再加入Teacher-frozen mode classifier和per-mode conditional distribution matching。完整论文解读、方法改造、实验矩阵和TODO见[行为模式发现与保模态蒸馏](05_行为模式发现与保模态蒸馏.md)。

## 15. keep013深度压缩模型的16→1步蒸馏（2026-08-02）

### 15.1 目的与协议审计

本实验不再改变网络架构，专门检验TinySR启发的深度架构压缩成果能否继续完成一步蒸馏：

```text
FM-4x72-16 Full
  -> FM-3x72-16 keep013（架构压缩/跨架构蒸馏）
  -> FM-3x72-1（步数蒸馏）
```

一步Student与16步Teacher均从`FM-3x72-16 keep013` checkpoint出发。训练只使用Teacher
rollout状态，并对同一state/noise在线查询keep013 Teacher endpoint；不读取原始demonstration
或专家动作。两条方法共用结构、初始化、buffer、seed 42和500轮训练预算：

1. **Endpoint fidelity**：直接拟合`TeacherIntegrate16(noise,state)`，用一次
   `boundary_transition(0→1)`生成完整动作序列；
2. **Boundary-CTM＋DSM 0.1**：使用boundary-preconditioned anytime consistency目标，并加入
   弱去噪监督。

审计中修正了一个重要协议问题：Boundary-CTM训练的是`boundary_transition(0→1)`，不能用
普通一步ODE `integrate(steps=1)`替代评估。此次Standard-120明确使用boundary接口；旧结果如
使用普通integrate，需要按各自记录解释，不能直接混合。

### 15.2 Standard-120完整结果

| 方法 | Epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Endpoint | 100 | **113/120，94.2%** | 15/24 | 0.689 |
| **Endpoint** | **250** | **111/120，92.5%** | **21/24** | **0.807** |
| Endpoint | 500 | 108/120，90.0% | 19/24 | 0.762 |
| Boundary-CTM＋DSM | 100 | 79/120，65.8% | **12/24** | **0.465** |
| Boundary-CTM＋DSM | 250 | **98/120，81.7%** | 9/24 | 0.331 |
| Boundary-CTM＋DSM | 500 | **98/120，81.7%** | 8/24 | 0.348 |

Endpoint epoch-250是当前Pareto最佳点。它在Standard-120中保持92.5% SR和21种模式，明显
优于Boundary-CTM。Endpoint从100到250轮将覆盖由15恢复至21，但500轮又回落至19，说明
闭环最优checkpoint不由训练轮数或离线loss单调决定。

Endpoint-250随后补充Standard-480，达到`441/480=91.9%`、`23/24`、`H=0.807`。相比
keep013的`91.0%/24/H=.929`，SR的小幅差异不足以在单seed下宣称提升，但coverage少1种且熵
明显下降，表明一步化主要改变模式频率并损失一个尾部mode。仍需多seed、独立suite和solver-only
一步对照，才能正式量化蒸馏损失。

### 15.3 Latency对比

硬件与协议统一为V100、batch size 1、100次warmup、1000次CUDA Event计时；只测模型生成，
不包含环境物理仿真和数据搬运。Endpoint latency使用选中的epoch-250 checkpoint。

| 模型 | 结构/步数 | Mean | Median | P95 | 相对Full加速 |
|---|---|---:|---:|---:|---:|
| `FM-4x72-16 Full keep0123` | 4x72，16步 | **65.93 ms** | 65.29 ms | 69.57 ms | 1.0× |
| `FM-3x72-16 keep013` | 3x72，16步 | 51.19 ms | 51.16 ms | 51.39 ms | **1.29×** |
| `FM-3x72-1 Endpoint-250` | 3x72，1步 | **2.22 ms** | 2.19 ms | 2.38 ms | **29.7×** |

纯深度压缩提供约1.29倍加速，16→1步数压缩提供主要收益；联合后相对原始4x72-16 Full约
29.7倍。该数字是单样本纯生成kernel链路结果，不等价于机器人系统端到端wall-clock加速。

### 15.4 当前结论与边界

1. TinySR式架构压缩得到的高质量16步Student可以继续进行有效一步蒸馏；
2. 直接paired endpoint fidelity在当前有限深度Student上明显优于自举Boundary-CTM；
3. 一步Endpoint在Standard-480保持91.9% SR和23/24模式，但相对16步起点仍损失1种模式，
   且熵由.929降至.807；
4. 当前实验是3x72深度压缩路线的接口验证，不代表最终3x48小Student已经解决；
5. 下一步必须补一步solver-only、逐mode成功率和多seed；并判断是否需要在Endpoint上加入
   BMD/Natural Replay式分布约束，恢复最后1种模式并校正模式频率。

关键产物：`logs/avoiding/keep013_step_distillation/`。

## 16. Endpoint保模态2×2消融（2026-08-04）

### 16.1 目的与协议

针对Endpoint-250正式结果`91.9%/23/H=.807`相对16步keep013少1种模式且熵下降的问题，固定
3x72架构和Endpoint paired fidelity，分别加入同状态多噪声终点关系约束与短轨迹分布约束。辅助
noise均在当前state下在线查询Teacher，不使用原始示范、专家动作或24-mode标签。`K=4`，每batch
32个state group，`lambda_rel=lambda_traj=0.1`，轨迹时刻从0.25/0.5/0.75中随机选择。

### 16.2 完整Standard-120结果

| 方法 | e100 SR/Cov/H | e250 SR/Cov/H | e500 SR/Cov/H |
|---|---|---|---|
| Endpoint | 94.2% / 15 / .689 | **92.5% / 21 / .807** | 90.0% / 19 / .762 |
| ＋Relation | 88.3% / 14 / .691 | 90.0% / 21 / .813 | 90.8% / 20 / .809 |
| ＋Trajectory | 89.2% / 14 / .694 | **88.3% / 22 / .797** | **91.7% / 20 / .821** |
| ＋Both | 90.0% / 14 / .713 | 88.3% / 20 / .802 | 89.2% / 20 / .802 |

没有方案同时达到`SR>=90%`和24/24，因此不补多seed Standard-480。Relation未稳定扩大支持集；
Trajectory在e250将覆盖提高到22但牺牲4.2个百分点SR，继续训练后SR恢复而覆盖回落；Combined没有
叠加收益。当前证据指向短轨迹监督能够触及额外路径basin，但固定权重无法同时保持其闭环执行。
下一步只对Trajectory验证更低权重或延迟开启的curriculum，不继续训练Combined，也不以增加普通
epoch替代机制验证。

产物：`logs/avoiding/endpoint_mode_preservation_2x2/`；训练器新增参数位于
`train_deployed_flow_step_distillation.py`。

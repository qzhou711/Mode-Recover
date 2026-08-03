# 结构蒸馏（二）：结构迁移与Repair实验

## 1. 文档边界

结构蒸馏由“结构初始化＋功能Repair”组成。Repair指结构初始化之后、步数蒸馏之前的功能恢复；当前统一保持16步推理，目标是使压缩Student恢复Teacher的速度场、内部关系和`noise→mode`映射。当前Repair包含Teacher监督训练，因此不能简化描述为剪枝。

## 2. 数据假设

正式主线只使用teacher rollout buffer：

- 240条轨迹；
- teacher闭环成功率96.7%；
- 覆盖23/24 modes；
- 模式熵0.913；
- 不含原始demonstrations和专家动作。

旧repair直接使用训练状态和专家动作，且组合五个启发式损失，只保留为机制消融。

## 3. 方法谱系

| Repair方法 | 监督内容 |
|---|---|
| Dynamic/velocity | teacher生成`x_t`上的速度匹配 |
| Endpoint | 相同state/noise下的teacher终点匹配 |
| 简化Attention Relation | `QK attention + VV Gram + velocity` |
| 完整MiniLMv2 | `QK + QQ + KK + VV + velocity` |
| 同状态多噪声 | 固定state采样K个noise，匹配跨噪声关系 |
| Progressive | 分阶段修复student blocks |
| Teacher Assistant | `4×72→4×48→3×48`两阶段结构迁移 |
| DDIL | 在student诱导的中间Flow状态上查询teacher |

## 4. Corrected Dynamic与阶段定位

旧dynamic曾把位置参数错误绑定为`t→1`，正确实现应由teacher从`0→t`生成`x_t`。修正版Standard-120：

| 初始化/结构 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| Activation-aware `3×48` | 100% | 1/24 | 0.000 |
| PCA `3×48` | 90.0% | 1/24 | 0.000 |
| Early-layer `3×48` | 93.3% | 3/24 | 0.044 |
| Width-only `4×48` | 85.0% | 2/24 | 0.101 |

Dynamic可恢复SR，但容易集中到少数安全路径。

## 5. Dynamic checkpoint扫描

固定Early-layer＋多噪声endpoint `λ=0.03`：

| Repair epoch | CTM前16步：SR / 覆盖 / 熵 |
|---:|---|
| 25 | 4.2% / 2 / 0.157 |
| 50 | 5.0% / 3 / 0.318 |
| 100 | 18.3% / 5 / 0.345 |
| 300 | 55.8% / 7 / 0.233 |

Repair加深会恢复SR和覆盖，但约100轮后模式分布开始变得不均衡。checkpoint不能只按离线loss选择。

## 6. 关系迁移第二轮

统一`FM-4x72-16→FM-3x48-16`、Early初始化、teacher buffer、seed 42：

| 方法 | 训练预算 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Early，仅初始化 | 0 | 3.3% | 1 | 0.000 |
| 简化Attention Relation | 300 | 65.0% | 3 | 0.043 |
| 完整MiniLMv2 | 300 | **65.8%** | 6 | 0.221 |
| MiniLMv2＋K=4，权重1.0 | 300 | 56.7% | 6 | 0.287 |
| 完整MiniLMv2 | 1000 | 37.5% | 13 | 0.658 |
| 多噪声MiniLMv2 | 1000 | 46.7% | 8 | 0.375 |
| 完整MiniLMv2 | 2000 | 45.8% | 14 | 0.740 |
| 完整MiniLMv2 | 2500 | **51.7%** | 18 | 0.824 |
| 完整MiniLMv2 | 3000 | **51.7%** | 18 | 0.809 |
| 完整MiniLMv2 | 3500 | 48.3% | **20** | 0.829 |
| 完整MiniLMv2 | 4000 | 49.2% | 19 | **0.852** |
| 完整MiniLMv2，续训 | 5000 | 58.3% | 19 | 0.823 |
| 完整MiniLMv2，续训 | 6000 | **65.0%** | 18 | 0.805 |
| 完整MiniLMv2，续训 | 7000 | 57.5% | 19 | 0.815 |
| 完整MiniLMv2，续训 | 8000 | 62.5% | 17 | 0.798 |

完整MiniLMv2相对简化目标在300轮时不损失SR并将覆盖翻倍。训练到1000轮后多样性显著提高、SR下降；2500–4000轮进一步将覆盖提高到18–20、熵提高到0.809–0.852。4000→8000阶段SR重新提高，但覆盖降至17–19。关系目标形成非单调Pareto轨迹，不能用单一loss选择checkpoint；当前多样性峰值位于3500–4000附近。

## 7. Progressive与Teacher Assistant

| 方法 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| Progressive逐层解冻 | 16.7% | 4 | 0.353 |
| Teacher Assistant 150+150 | 25.0% | 7 | 0.538 |
| Teacher Assistant 500+500 | 34.2% | 11 | 0.694 |
| Teacher Assistant 700+300 | **35.0%** | **14** | **0.729** |

当前Progressive配方失败，但不代表所有模块替换方法失败。Teacher Assistant随第一阶段训练增强而同时改善SR和多样性，说明`4×48 assistant`质量曾是瓶颈之一；下面的双阶段评估进一步表明，当前更主要的剩余瓶颈位于第二阶段深度压缩。

双阶段评估进一步修正这一判断：

| Assistant Repair | `4×48`：SR / 覆盖 / 熵 | Student Repair | `3×48`：SR / 覆盖 / 熵 |
|---|---|---|---|
| 1000 epochs | 41.7% / 15 / 0.765 | 300 epochs | 27.5% / 11 / 0.672 |
| 700 epochs＋`K=4, λ=0.1` | **47.5% / 16 / 0.741** | 300 epochs | **33.3% / 12 / 0.726** |

第一阶段继续训练和弱多噪声约束确实能提高Assistant；但`4×48→3×48`后稳定损失约14个SR百分点和4种模式。当前主要瓶颈已经转移到第二阶段深度压缩。弱多噪声的收益没有被现有层映射完整继承。

## 8. 当前最佳判断

- 高SR筛选点：完整MiniLMv2 300 epochs；
- 高多样性筛选点：Teacher Assistant 700+300或完整MiniLMv2 1000 epochs；
- 多噪声权重1.0偏强，应扫描`0.03/0.1/0.3`；
- 正式进入CTM前，应选择多个Pareto checkpoint，而非单一最低loss checkpoint。

## 9. 下一步

1. 用相邻层合并或可学习adapter替代直接选择assistant层`0、2、3`；
2. 第二阶段继续保留弱多噪声关系，而不是只在Assistant阶段使用；
3. 对300/1000/2000 MiniLMv2 checkpoint绘制SR—覆盖—熵曲线并联合选点；
4. 将高SR的300轮checkpoint和高多样性的2000轮checkpoint分别送入同一CTM，比较collapse敏感性。

## 11. 4000轮checkpoint扫描说明

2500、3000、3500和4000均来自同一条4000轮训练轨迹，初始化、batch顺序和cosine学习率计划一致。评估使用已通过逐轨迹等价性验证的单GPU四worker Standard-120。产物位于`logs/avoiding/teacher_generated_minilmv2_4000_scan/`。

由于3500/4000仍只有120条样本，20与19模式的差异可能来自低频模式采样。进入论文结论前应对Pareto候选补Standard-480。

## 12. 4000→8000续训说明

由于第一阶段没有保存optimizer状态，5000–8000结果来自4000 checkpoint上的第二阶段训练：Adam状态重新初始化，初始学习率设为`3e-5`并重新执行4000轮cosine衰减。因此它是“checkpoint续训＋学习率重启”，不是原optimizer的无缝8000轮轨迹。

该阶段的主要趋势是SR恢复而多样性缓慢回退：5000–8000覆盖为19、18、19、17，均未超过3500轮的20；熵也未超过4000轮的0.852。继续单纯增加Repair轮数的优先级应降低。产物位于`logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/`。

## 13. Pareto checkpoint的Standard-480确认

| Repair checkpoint | SR | 覆盖 | 熵 |
|---:|---:|---:|---:|
| 3500 | 240/480，50.0% | 21/24 | 0.842 |
| 6000 | **277/480，57.7%** | **22/24** | **0.845** |

Standard-480改变了Standard-120中的细微排序：6000轮由18模式上升到22，说明120条会漏掉低频模式。两个checkpoint都已恢复大部分teacher模式，6000轮同时获得更高SR、覆盖和熵，是进入后续CTM的更强Repair起点。当前不需要继续延长Repair轮数；研究重心应转向CTM的mode preservation。

### 13.1 与Full-trained小模型路线的差距

| 路线 | 16步SR | 覆盖 | 熵 | 说明 |
|---|---:|---:|---:|---|
| `FM-4x72-16-Full` teacher | 95.8% | 24 | 0.945 | 大模型teacher |
| `FM-3x48-16-Full` | 92.7% | 24 | 0.965 | 原始数据训练的小模型 |
| Full-3x48 warm start跨规模1步蒸馏 | 88.8% | 23 | 0.832 | 已有任务能力后再蒸馏 |
| Teacher-only MiniLMv2 6000 Repair | 57.7% | 22 | 0.845 | 从结构投影恢复，无原始示范 |
| 上述Repair接Flow-CTM | 66.3% | 12 | 0.451 | 当前一步蒸馏发生collapse |

MiniLMv2已经较好恢复模式覆盖和熵，但闭环SR仍是主要缺口；随后CTM又损失10种模式。因此下一阶段应分别优化：

1. **Repair准确率**：增加teacher rollout状态覆盖、student-induced state查询和行为/endpoint锚定，使22模式不仅存在，而且能稳定成功；
2. **CTM模式保持**：在同状态多噪声组上约束CTM前后的endpoint/关系几何，防止一步映射集中到少数安全模式。

## 14. Repair准确率提升TODO

理论来源：MiniLMv2-style关系迁移负责跨结构表示保持；paired endpoint regression提供行为fidelity；DDIL/DAgger式student-induced states用于修复闭环covariate shift。所有实验从6000轮Repair checkpoint继续500 epochs，保持teacher-only、无原始示范数据。

统一损失：

```text
L = L_relation
  + lambda_v * L_velocity
  + lambda_e * L_endpoint
  + lambda_i * L_student_induced
```

四卡矩阵：

| GPU | 方法 | 权重 |
|---:|---|---|
| 0 | Relation continuation | `lambda_v=0.1, lambda_e=0, lambda_i=0` |
| 1 | Relation＋strong velocity | `lambda_v=1.0, lambda_e=0, lambda_i=0` |
| 2 | Relation＋velocity＋endpoint | `lambda_v=1.0, lambda_e=0.1, lambda_i=0` |
| 3 | Relation＋velocity＋endpoint＋DDIL | `lambda_v=1.0, lambda_e=0.1, lambda_i=0.25` |

训练500 epochs，保存100/250/500；先做Standard-120。首轮通过条件：`SR≥70%、Coverage≥20、H≥0.80`。最佳候选补Standard-480，并与冻结6000基线`57.7%/22/H=0.845`比较。

实现注意：

- Endpoint使用相同`state, noise`的teacher 16步终点与student可微16步终点；
- Student-induced状态由student从噪声积分到随机`t`产生，停止梯度后由teacher标注速度；
- Teacher-induced分支始终保留，防止DDIL只强化student已访问的少数模式；
- checkpoint按SR、覆盖和熵联合选择，不按单一离线loss。

### 14.1 第一轮结果

统一从MiniLMv2-6000 checkpoint继续500 epochs，并做Standard-120：

| 配置 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| MiniLMv2-6000冻结基线 | 65.0% | 18/24 | **0.805** |
| Relation续训 | 64.2% | 17/24 | 0.791 |
| Relation＋Velocity `lambda_v=1.0` | **68.3%** | **18/24** | 0.766 |
| Relation＋Velocity＋Endpoint | 63.3% | 15/24 | 0.719 |
| Relation＋Velocity＋Endpoint＋Student-induced | 65.8% | **18/24** | 0.701 |

Velocity是当前唯一提高SR且不减少覆盖的方法，但熵下降；Endpoint在Repair阶段同时损害SR与多样性，Student-induced只能部分挽回。因此下一轮优先优化Velocity，不继续增强Endpoint Repair。

## 15. 统一未完成TODO（2026-07-30）

以下列表合并此前制定但尚未全部完成的任务，后续按顺序执行。

### P0：Velocity Repair局部优化

- [x] 评估现有`lambda_v=1.0`的100、250、500 epoch checkpoint，定位SR与熵的非单调变化；
- [x] 扫描`lambda_v=0.25/0.5/1.0`；
- [x] 测试Velocity权重逐步ramp-up，避免早期破坏关系结构；
- [x] 测试mode-balanced teacher-buffer采样，避免Velocity偏向高频安全模式；
- [x] 按`SR/Coverage/H`联合选择Pareto checkpoint，不按训练loss或单一SR选择。
- [ ] 后续实现异步闭环选点：训练每50–100轮保存checkpoint，由空闲GPU在固定验证episode上评估`SR/Coverage/H`；满足覆盖与熵约束后按SR选best，并用独立Standard-480确认。

首轮验收仍为`SR≥70%、Coverage≥20、H≥0.80`。若没有配置同时通过，则保留“高SR候选”和“高多样性候选”各一个。

已完成的Velocity局部扫描（Standard-120）：

| 配置 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| `lambda_v=1.0`，100 epochs | 55.0% | 17/24 | 0.759 |
| `lambda_v=1.0`，250 epochs | **70.8%** | **20/24** | **0.810** |
| `lambda_v=1.0`，500轮训练的loss-best（约epoch 470） | 68.3% | 18/24 | 0.766 |
| `lambda_v=0.25`，500轮训练的loss-best（约epoch 470） | 63.3% | 17/24 | 0.805 |
| `lambda_v=0.5`，500轮训练的loss-best（约epoch 470） | 69.2% | 18/24 | 0.787 |

当前唯一同时通过三项首轮门槛的是`lambda_v=1.0, epoch=250`。后期loss-best结果反而损失2种模式并降低熵，说明训练过程可能非单调，后续必须以闭环Pareto指标选epoch。由于后期结果是约epoch 470的loss-best而非固定epoch-500，不能把该比较写成严格的“250优于500”。

### 15.1 Repair上限与下一轮四卡实验

理想目标是`FM-4x72-16-Full` teacher的`95.8%/24/H=0.945`；考虑student容量后，更现实的功能上限参考是原始数据完整训练的`FM-3x48-16-Full`：`92.7%/24/H=0.965`。当前Velocity-250仍有约22个SR百分点差距，不能认为Repair已经饱和。

下一轮固定teacher、MiniLMv2-6000起点、buffer、seed、学习率和250轮预算：

| GPU | 配置 | 目的 |
|---:|---|---|
| 0 | Velocity-250 Standard-480 | 严格确认首轮候选 |
| 1 | `lambda_v=1.0`，前100轮线性ramp | 减少早期关系结构破坏 |
| 2 | `lambda_v=1.0`，mode-balanced buffer | 提高低频模式监督 |
| 3 | ramp＋mode-balanced | 检查两者是否互补 |

模式均衡只改变teacher rollout buffer的采样权重，不引入原始示范；ramp在前100轮将Velocity权重由接近0线性提高到1.0。三组均直接评估固定epoch-250 checkpoint，避免使用不可比较的离线loss自动选择结果。

### 15.2 第二轮结果

固定250轮的Standard-120：

| 方法 | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| 原Velocity-250 | **70.8%** | **20/24** | 0.810 |
| 前100轮Velocity ramp | 64.2% | 17/24 | 0.757 |
| Mode-balanced buffer | 65.8% | 19/24 | **0.837** |
| Ramp＋Mode-balanced | 61.7% | 17/24 | 0.803 |

表面上Ramp没有改善当前起点，Mode-balanced提高熵但牺牲SR，组合也未显示互补收益。但此处存在学习率计划混淆：原Velocity-250是总计划500轮训练的中点，ramp/balanced是总计划250轮训练的终点；两者cosine学习率轨迹不同。因此这些消融是有效结果，但在统一scheduler复现前，只能作为初筛，不能把差异完全归因于ramp或balanced。

原Velocity-250的Standard-480严格确认：

| Repair checkpoint | SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| MiniLMv2-6000基线 | 277/480，57.7% | 22/24 | 0.845 |
| Velocity-250 | **303/480，63.1%** | **22/24** | 0.836 |

Velocity在不减少覆盖的前提下将SR提高5.4个百分点，熵仅下降0.009；Standard-120中的70.8%高估了绝对SR，但方法收益在更严格协议下仍成立。当前高SR主候选为Velocity-250，高熵候选为Mode-balanced-250；后者仍需Standard-480确认。

#### 与MiniLMv2-6000基线的完整比较

MiniLMv2-6000以内部Q/K/V关系迁移为主，仅使用较弱的Velocity监督；Velocity-250从该checkpoint继续250轮，将速度场监督提高到`lambda_v=1.0`，同时保留关系损失。两者使用相同teacher rollout buffer，均不访问原始示范。

| 协议 | 方法 | SR | 覆盖 | 熵 |
|---|---|---:|---:|---:|
| Standard-120 | MiniLMv2-6000 | 65.0% | 18/24 | 0.805 |
| Standard-120 | Velocity-250 | **70.8%** | **20/24** | **0.810** |
| Standard-480 | MiniLMv2-6000 | 277/480，57.7% | 22/24 | **0.845** |
| Standard-480 | Velocity-250 | **303/480，63.1%** | 22/24 | 0.836 |

Standard-480下Velocity提高5.4个SR百分点、保持22种模式，熵仅下降0.009，说明它主要提高已有模式的闭环执行准确率，没有出现明显额外collapse。但它仍低于原始数据完整训练的`FM-3x48-16-Full`：92.7%/24/H=0.965；当前结果证明Velocity Repair有效，而非Teacher-only结构迁移已经达到student能力上限。

### 15.3 四种方法说明

四种方法都从MiniLMv2-6000继续Repair，不访问原始示范：

1. **Velocity-250**：关系损失与teacher速度场MSE共同更新，`lambda_v=1.0`固定；
2. **Velocity ramp**：前100轮将`lambda_v`由接近0线性升到1.0，试图先稳住关系结构；
3. **Mode-balanced**：保持`lambda_v=1.0`，按buffer轨迹mode频率的倒数进行加权采样；
4. **Ramp＋Mode-balanced**：同时使用上述权重调度与均衡采样。

### 15.4 Velocity-250还能否继续提高

当前结论是“有明确改进空间，但不能简单认定训练不足，也不应直接大幅提高Velocity权重”。

证据与原因：

- 250轮Standard-480相对MiniLMv2-6000提高5.4个SR百分点，说明Velocity监督有效；
- 100轮、250轮和后期loss-best的闭环结果非单调，说明继续增加epoch不保证改善；
- 当前每epoch仅4个batch，250轮约1000次更新，优化预算并不大，但teacher buffer只有240条rollout、覆盖23/24，teacher-state监督与闭环student访问状态之间仍有分布差；
- 训练记录中Velocity MSE约`0.17–0.27`，关系loss约`0.01`；在`lambda_v=1`时Velocity已主导标量loss。直接把权重提高到2或更大，较可能继续牺牲关系与mode，而不是稳定提高SR；
- 旧Student-induced实验与有害的Endpoint同时加入，尚未单独验证`Relation＋Velocity＋Student-induced`，因此不能据旧结果否定它。

下一轮应按信息增益排序：

1. **先消除协议混淆**：统一500轮cosine计划，保存并闭环评估150/200/250/300/350/500；同时评估真实epoch-500，而不是loss-best；
2. **做局部权重扫描**：在相同scheduler和固定checkpoint下比较`lambda_v=0.75/1.0/1.25/1.5`，不优先尝试大于2；
3. **隔离covariate-shift修复**：比较`Velocity`与`Velocity＋Student-induced`，且不加入Endpoint，扫描较弱`lambda_i=0.05/0.1/0.25`；
4. **扩大teacher-only状态覆盖**：增加teacher rollout buffer或对当前student访问状态请求teacher velocity标签，保持无原始示范；
5. **实现异步闭环选点**：每50轮在固定验证集上计算`SR/Coverage/H`，按Pareto约束选checkpoint，再用独立Standard-480确认；
6. 若固定权重仍存在明显SR—多样性冲突，再考虑GradNorm/uncertainty weighting等自适应梯度平衡，而不是继续手工放大Velocity标量权重。

### 15.5 统一协议的四卡密集扫描

为消除scheduler与checkpoint语义混淆，四组全部使用500轮cosine计划、相同MiniLMv2-6000起点、teacher buffer、seed 42、学习率`3e-5`和每轮4 batches，并固定保存/评估`150/200/250/300/350/500`：

| GPU | 配置 | 研究问题 |
|---:|---|---|
| 0 | `lambda_v=1.0, lambda_i=0` | 重建统一协议基线与训练曲线 |
| 1 | `lambda_v=1.25, lambda_i=0` | 局部提高Velocity是否继续提升SR |
| 2 | `lambda_v=1.0, lambda_i=0.1` | 弱Student-induced能否修复闭环偏移 |
| 3 | `lambda_v=1.0, lambda_i=0.25` | 较强Student-induced的收益与多样性代价 |

每个checkpoint使用相同Standard-120；按`SR/Coverage/H`选择Pareto点。Student-induced分支不包含Endpoint，首次解耦评估其独立作用。

#### 完整结果

| 配置 | Epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Velocity `lambda_v=1.0` | 150 | 60.0% | 16 | 0.755 |
|  | 200 | 64.2% | 18 | 0.764 |
|  | **250** | **70.8%** | **20** | **0.810** |
|  | 300 | 65.0% | 18 | 0.771 |
|  | 350 | 68.3% | 17 | 0.761 |
|  | 500 | 67.5% | 18 | 0.780 |
| Velocity `lambda_v=1.25` | 150 | 60.8% | 16 | 0.757 |
|  | 200 | 65.0% | 19 | 0.771 |
|  | 250 | 68.3% | 20 | 0.803 |
|  | 300 | 65.8% | 17 | 0.752 |
|  | 350 | 66.7% | 18 | 0.767 |
|  | 500 | 69.2% | 18 | 0.779 |
| Velocity＋Induced `lambda_i=0.1` | 150 | 62.5% | 17 | 0.736 |
|  | 200 | 63.3% | 17 | 0.731 |
|  | 250 | 67.5% | 19 | 0.765 |
|  | 300 | 64.2% | 18 | 0.757 |
|  | 350 | 63.3% | 17 | 0.737 |
|  | 500 | 65.8% | 18 | 0.752 |
| Velocity＋Induced `lambda_i=0.25` | 150 | 70.0% | 18 | 0.748 |
|  | 200 | 66.7% | 19 | 0.754 |
|  | 250 | 66.7% | 17 | 0.717 |
|  | 300 | 67.5% | 19 | 0.739 |
|  | 350 | 65.0% | 19 | 0.740 |
|  | 500 | 70.0% | 17 | 0.718 |

统一scheduler后，`lambda_v=1.0, epoch=250`仍是唯一同时通过`SR≥70%、Coverage≥20、H≥0.80`的点。`lambda_v=1.25`没有带来收益；Student-induced可在部分epoch提高SR，但稳定降低熵或覆盖，说明当前student-induced状态采样更偏向少数闭环安全区域。当前不应继续简单增大Velocity或Induced权重；Repair局部扫描可阶段性收束，下一步应以Velocity-250进入mode-preserving CTM，同时只为最终候选补必要的多seed确认。

### 15.6 Velocity长训练确认

为确认500轮以后是否存在第二个性能上升区间，使用两张GPU从相同MiniLMv2-6000起点分别训练seed 42/43至1000轮，统一采用1000轮cosine计划，保存并评估750与1000轮。由于旧500轮实验没有optimizer/scheduler状态，本实验从头训练，不能与旧500轮曲线视作无缝续接；seed 42用于主比较，seed 43用于判断趋势稳定性。

Standard-120结果：

| 训练seed | Epoch | SR | 覆盖 | 熵 |
|---:|---:|---:|---:|---:|
| 42 | 750 | 68.3% | 18 | 0.796 |
| 42 | 1000 | 62.5% | 17 | 0.773 |
| 43 | 750 | 64.2% | 19 | 0.803 |
| 43 | 1000 | 62.5% | 17 | 0.773 |

750/1000均未超过500轮计划中的Velocity-250：`70.8%/20/H=0.810`。两个seed在1000轮都退化到`62.5%/17/H=0.773`，说明继续延长当前Repair不会产生稳定收益。按照预设决策边界，停止追加单纯epoch，固定Velocity-250进入mode-preserving CTM。

## 16. 强Teacher结构迁移与闭环恢复机制审计（2026-08-01）

### 16.1 实验边界与共同设置

本节补齐此前尚未写入文档的P6--P8实验。全部实验研究
`FM-4x72-16-Full -> FM-3x48-16`结构蒸馏，Student始终使用16步求解器；尚未进入
`16->1`步数蒸馏。除完整训练容量参照外，训练数据只来自冻结Teacher rollout或
Teacher在Student状态上的在线查询，不读取原始demonstration，也不使用专家动作。

关键参照为：

| 模型/数据 | 协议 | SR | 覆盖 | 熵 |
|---|---|---:|---:|---:|
| FM-4x72-16-Full Teacher | Standard-480 | 460/480，95.8% | 24/24 | 0.945 |
| FM-3x48-16-Full容量上界 | Standard-480 | 445/480，92.7% | 24/24 | 0.965 |
| MiniLMv2-6000 teacher-derived起点 | Standard-480 | 277/480，57.7% | 22/24 | 0.845 |
| Velocity-250既有最佳Repair | Standard-480 | 303/480，63.1% | 22/24 | 0.836 |

`FM-3x48-16-Full`证明3x48容量本身足以达到92.7%/24；当前约60%的结果不能简单归因于
Student参数不足。覆盖和熵均只对成功轨迹统计，因此22/24表示“22种路径至少偶尔成功”，
不表示每种路径都可靠。

### 16.2 P6.1：强Teacher闭环buffer

四卡生成2400条FM-4x72-16 rollout，共151,214个状态样本。严格审计结果为
`SR=96.92%`、`24/24`、`H=0.9454`，最少真实mode也有30条成功轨迹；元数据确认
`uses_original_demonstrations=false`、`uses_expert_actions=false`。层次式无标签发现的最佳
配置为basic trajectory features、无PCA whitening、full-covariance GMM、K=24，跨seed
NMI为0.9786、silhouette为0.444、最小latent 33。该buffer足以作为全局Teacher状态来源，
但其状态分布仍主要来自Teacher自身闭环。

产物：`logs/avoiding/strong_teacher_expanded_2400/`。

### 16.3 P6.2：强Teacher离线结构Repair

四路均从MiniLMv2-6000开始，使用成功Teacher rollout、500 epochs、每epoch 4 batches，
保存100/250/500。下表列出各方法最佳Standard-120点：

| 方法 | 最佳epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| Natural velocity | 250 | 62.5% | 18/24 | 0.808 |
| Velocity＋endpoint 0.03 | 250 | **62.5%** | **19/24** | **0.850** |
| Velocity＋MiniLMv2 relation | 250 | 61.7% | 18/24 | 0.821 |
| Velocity＋relation＋endpoint | 500 | 60.0% | 19/24 | 0.801 |

四组均未通过预注册的`SR>=75%、Coverage>=20`门槛。弱endpoint提高了筛选点的覆盖和熵，
但没有提高SR；关系目标也没有产生额外闭环收益。后续Student-induced实验统一选择
Velocity＋endpoint epoch-250作为起点，是因为它在SR并列时具有更高覆盖和熵，而不是因为
它已经达到结构Repair目标。

### 16.4 P6.3 Round 1：普通Student-induced点查询

使用上述起点生成480条全Student闭环轨迹，在每个Student状态以相同噪声查询Teacher。
Student buffer自身为`SR=58.96%/19 modes/H=0.834`。固定endpoint权重0.03后比较
25%、50%、75% induced采样及50%＋BMD均衡：

| 配置 | 选取epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| 25% induced | 50 | **64.2%** | 17/24 | 0.754 |
| 50% induced | 250 | 62.5% | **19/24** | 0.821 |
| 75% induced | 100 | 60.8% | 16/24 | 0.769 |
| 50% induced＋BMD均衡 | 50 | 62.5% | 17/24 | 0.782 |

75%组在epoch-50曾覆盖20种，但SR仅55%，不能视为通过。静态增加Student状态比例形成明显的
全局能力--局部纠错权衡；BMD均衡没有突破该Pareto前沿。单点Teacher标签不足以描述纠正后的
状态演化，因而主线转向Intervention DAgger连续接管。

### 16.5 P6.3 Round 2：Intervention DAgger

触发条件为Teacher--Student endpoint归一化RMSE超过预注册`q80=0.4181`。Teacher分别连续
接管H=4或H=8步，同时记录Student反事实输出：

| 辅助执行 | Assisted SR | 覆盖 | 熵 | 平均触发次数 | Teacher控制步占比 |
|---|---:|---:|---:|---:|---:|
| H=4 | 82.08% | 24/24 | 0.909 | 5.377 | 33.27% |
| H=8 | 82.29% | 24/24 | 0.905 | 3.748 | 45.47% |

这些是Teacher参与控制的assisted rollout，不是Student评估结果。四路Repair的最佳
Standard-120为：

| Repair方式 | 最佳epoch | SR | 覆盖 | 熵 |
|---|---:|---:|---:|---:|
| H4-all | 50 | 65.0% | 17/24 | 0.773 |
| H4-recovery-only | 250 | 65.8% | **19/24** | **0.837** |
| H8-all | 100 | **68.3%** | 18/24 | 0.789 |
| H8-recovery-only | 50 | 65.0% | 18/24 | 0.812 |

虽然均未过筛选门槛，仍对H4-recovery和H8-all补做Standard-480，防止120条漏掉低频mode：

| 方法 | Standard-480 SR | 覆盖 | 熵 |
|---|---:|---:|---:|
| H4-recovery | 284/480，59.17% | **22/24** | **0.862** |
| H8-all | **291/480，60.63%** | **22/24** | 0.851 |

Standard-120分别高估SR约6.6和7.7个百分点，并低估覆盖3--4种。两者保留22种偶发成功
模式，但SR仍约60%，说明连续短接管没有转化为可靠Student能力。使用两个更新后Student各自
重新采集480条全Student轨迹时，SR都只有55.83%、覆盖21；Teacher--Student disagreement
q80从0.4181仅降至约0.405，局部差距改善很小。

### 16.6 P7：三项机制审计

为区分Teacher恢复能力、初始化和旧训练预算，执行以下受控实验。

#### Teacher接管至终局恢复上限

Student首次超过同一disagreement阈值后，Teacher持续控制至episode结束。Standard-480
assisted结果为`446/480=92.92%`、`24/24`、`H=0.950`；477/480发生干预，Teacher控制
86.48%的记录步。相比H4/H8的约82%，持续接管提高约10.7个百分点。

这证明Teacher在Student偏离状态上并非完全失效，主要失败之一发生在有限接管后的控制权
交还；但92.9%仍是assisted上限，不能写成Student性能，也不能证明所有Student状态都可恢复。

#### 完整遍历Teacher buffer与初始化消融

旧Repair每epoch只有4 batches。新实验让3x48完整遍历成功Teacher buffer：每轮约590
batches，10轮约5900 updates；三路数据、optimizer、seed和评估完全相同，只改变初始化。

| 初始化 | e1 SR/覆盖/H | e3 | e5 | e10 |
|---|---|---|---|---|
| Random | 9.2%/3/.239 | 16.7%/5/.389 | 17.5%/9/.601 | 30.0%/9/.338 |
| Teacher-derived PCA | 7.5%/3/.334 | 47.5%/6/.224 | 55.0%/7/.182 | 57.5%/10/.306 |
| MiniLMv2-6000 | **60.8%/17/.781** | 59.2%/16/.744 | 59.2%/16/.752 | 58.3%/17/.744 |

结论是初始化显著决定可学性与模式保留：MiniLMv2远优于PCA和随机初始化。但单纯完整遍历
Teacher buffer不能解决结构迁移；MiniLMv2继续训练反而轻微退化。因此当前约60%并非仅由
“每epoch 4 batches训练不足”造成，普通Teacher-forced velocity目标与闭环可靠性存在错位。

### 16.7 P8：成功恢复轨迹与Recovery Curriculum

从P7接管至终局buffer筛出446条成功辅助轨迹、24,787个Teacher控制状态。所有组从同一
Velocity＋endpoint Student开始，只使用成功恢复段，保持16步且demo-free。固定比例组训练
250 epochs；课程组依次训练`75% -> 50% -> 25%`，每阶段100 epochs。

固定比例完整Standard-120结果：

| 恢复采样比例 | Epoch 50 | Epoch 100 | Epoch 250 |
|---|---|---|---|
| 25% | **70.8%/18/.791** | 57.5%/18/.772 | 54.2%/19/.799 |
| 50% | 55.8%/16/.766 | 60.0%/17/.782 | 57.5%/17/.777 |
| 75% | 64.2%/16/.785 | 52.5%/16/.754 | 63.3%/18/.804 |

课程式结果：

| 阶段 | 恢复比例 | SR | 覆盖 | 熵 |
|---:|---:|---:|---:|---:|
| 1 | 75% | 53.3% | 15/24 | 0.768 |
| 2 | 50% | 56.7% | 18/24 | 0.786 |
| 3 | 25% | 57.5% | 16/24 | 0.780 |

无候选通过`75%/20`门槛，因此未补Standard-480。25%早期点可恢复到既有Standard-120
SR水平，但仍少2种模式；继续训练明显降低SR。简单把成功恢复状态混入点对点velocity和
endpoint监督，并没有教会Student持续完成恢复。该负结果否定的是当前静态混合目标，不是否定
闭环恢复数据本身。

### 16.8 当前可靠结论与下一步

1. **当前主瓶颈是结构蒸馏，不是步数蒸馏。** 本节全部模型都是16步；3x48 Full的92.7%
   又排除了容量不足这一单一解释。
2. **MiniLMv2 teacher-derived初始化有必要但不充分。** 它显著优于Random/PCA，但仍未把
   Teacher知识转化为高可靠闭环策略。
3. **模式支持与闭环准确率脱钩。** 多个Student在Standard-480覆盖22种，但SR只有约60%；
   这表示稀有模式偶发成功，而不是22种稳定技能。
4. **Teacher具备较高恢复上限，但Student无法稳定接棒。** 接管到终局达到92.9%，有限H只到
   约82%，短恢复片段的点对点训练又未改善Student。
5. **继续扫epoch、混合比例或局部loss优先级低。** 下一步应使用真正的多步闭环恢复目标：
   训练连续恢复片段的状态演化、固定episode级行为身份，并显式评估Teacher交还后Student的
   survival/success曲线。

产物根目录：

- `logs/avoiding/p6_strong_teacher_structure_repair/`
- `logs/avoiding/p6_student_induced_repair_round1/`
- `logs/avoiding/p6_intervention_dagger_round2/`
- `logs/avoiding/p6_intervention_dagger_repair/`
- `logs/avoiding/p6_intervention_pareto_confirm/`
- `logs/avoiding/p7_structure_mechanism_audit/`
- `logs/avoiding/p8_recovery_curriculum/`

## 17. TinySR启发的可恢复深度结构搜索计划（2026-08-01）

TinySR的核心启发是按“有限恢复训练后的任务性能”选择剪枝mask，而不是按静态层重要性或
初始化loss选择结构。当前FM Teacher只有4层，因此不使用Gumbel mask学习，而是穷举全部
四种保留3层的有序子集，获得更准确、可审计的结论。

### 17.1 第一阶段：深度与宽度解耦

四卡并行比较`keep012/keep013/keep023/keep123`，统一构造`FM-3x72-16`。共享模块和保留
Transformer block从4x72 Teacher逐元素精确复制，初始化最大差异必须为0。训练只使用2400条
强Teacher成功rollout；目标固定为velocity MSE加弱endpoint anchor 0.03。四组共用seed 42、
500轮cosine计划、每轮4 batches，保存50/100/250/500。

每个checkpoint使用四worker Standard-120。预注册门槛为`SR>=80%、Coverage>=22`；通过者
按SR、覆盖、熵排序并补Standard-480。Standard-120只负责筛选，且覆盖/熵只在成功轨迹上计算。

### 17.2 第二阶段：宽度压缩

只有最佳3x72通过Standard-480确认后，才固定其深度mask并执行`FM-3x72-16 -> FM-3x48-16`。
该阶段比较teacher-derived width projection与Random对照，但不再改变层组合，从而单独测量
宽度压缩造成的SR和mode损失。若3x72第一阶段都未过门槛，则停止宽度压缩，结论是当前
teacher-rollout恢复目标不足以支持可恢复层选择，而不是任意挑一个失败mask继续。

### 17.3 第三阶段与WAM迁移

D3IL四层Teacher用于验证recoverability criterion；迁移到深层FastWAM/DreamZero后，才采用
TinySR的blockwise mask probability、Dynamic Inter-block Activation和Expansion-Corrosion，
避免深层组合爆炸。机器人版本的结构选择必须同时约束闭环SR、覆盖和逐mode可靠性，不能只用
图像论文中的LPIPS/L1，也不能把高SR严重collapse的mask选为最佳。

当前产物：`logs/avoiding/tinysr_depth_recoverability/`；执行脚本：
`scripts/run_tinysr_depth_recoverability_4gpu.sh`。

### 17.4 完成结果与阶段决策

四种深度mask的recoverability差异非常显著。Standard-120最佳checkpoint分别为：

| Mask | 最佳SR | 对应覆盖 | 熵 |
|---|---:|---:|---:|
| keep012 | 82.5% | 19/24 | 0.825 |
| **keep013** | **91.7%** | **23/24** | **0.921** |
| keep023 | 65.0% | 18/24 | 0.790 |
| keep123 | 27.5% | 8/24 | 0.604 |

自动选择keep013 epoch-500后的Standard-480为`437/480=91.0%`、`24/24`、`H=0.929`。
相对4x72 Teacher只下降4.8个SR百分点和0.016熵，同时参数减少约23%。这推翻了“强Teacher
rollout本身不足以支持高质量结构迁移”的宽泛解释：它足以支持纯深度迁移；旧3x48约60%的
主要问题更可能来自深度与宽度同时变化及不合适的结构映射。

后续固定keep013，即删除Teacher零索引第2层。先补独立suite与多seed，再以该3x72模型为
Teacher单独研究`3x72 -> 3x48`宽度压缩；不再优先在旧3x48 Velocity起点上叠加恢复loss。
完整协议、16组checkpoint、结论边界和产物见`07_可恢复性引导的深度压缩实验.md`。

## 18. 宽度压缩机制实验更新（2026-08-02）

最近三轮实验进一步定位了`3x72 -> 3x48`失败机制。Head-only `3x72, 4→3 heads`经250轮
达到Standard-120 `90.8%/23/H=.911`；相反，保留4 heads但执行逐head 72→48映射仅达到
`49.2%/10/H=.507`（coordinate）或`47.5%/4/H=.171`（PCA）。因此减少head数量不是主因，
破坏共享72维残差坐标更关键。

渐进PCA和PPCL适配器分别达到`85.0%/4/H=.086`与最高`67.5%/5/H=.165`，仍未恢复模式。
保留72维残差、只将FFN从288压到36时，activation方案到epoch-500为
`46.7%/15/H=.763`，表现为覆盖恢复但控制成功不足。这说明模式路由和可靠控制分布在残差、
attention和FFN多个算子，不能通过压缩单一子系统解决。

当前正在评估四种“保留残差72、跨Q/K/V/proj/FFN分布式低秩压缩”方案。完整设置、全部
Standard-120数据、结论边界及实时状态见
`实验日报/2026-08-02_最近12小时宽度压缩实验总结.md`。

### P1：Repair严格确认

- [x] 对Pareto候选补Standard-480；
- [x] 与冻结MiniLMv2-6000在完全相同episode/seed下比较；
- [ ] 补至少两个额外训练seed；
- [ ] 统计每种mode的频次与条件成功率，判断Velocity是否只强化高频模式；
- [x] 保存成功/失败异色轨迹图。

已完成项对应Velocity-250 Standard-480：`303/480，63.1%/22/H=0.836`；冻结MiniLMv2-6000为`277/480，57.7%/22/H=0.845`。轨迹图位于`logs/avoiding/teacher_generated_minilmv2_velocity_followup/epoch250_eval480/eval480/trajectory_comparison.png`。

### P2：Mode-preserving CTM

- [ ] 固定最佳Repair起点，比较Flow-CTM＋DSM基线；
- [ ] 比较`CTM＋弱teacher endpoint anchor`；
- [ ] 比较`CTM＋同状态K=4多噪声输出Gram约束`；
- [ ] 比较endpoint与Gram组合；
- [ ] 每组保存100/250/500 epoch并先做Standard-120。

### P3：论文级验证

- [ ] CTM候选补Standard-480；
- [ ] 补Paired-1000与JS divergence；
- [ ] 补多seed均值与离散程度；
- [ ] 绘制Repair前后、CTM前后的SR、覆盖、熵和模式频次变化；
- [ ] 更新总览、Repair与CTM文档，明确直接/间接原始数据依赖。

## 10. 产物

- `logs/avoiding/teacher_generated_structure_wave2/`
- `logs/avoiding/teacher_generated_structure_wave2_1000/`
- `logs/avoiding/teacher_generated_structure_wave2_teacher_assistant/`
- `logs/avoiding/teacher_generated_structure_wave2_teacher_assistant_stage_probe/`

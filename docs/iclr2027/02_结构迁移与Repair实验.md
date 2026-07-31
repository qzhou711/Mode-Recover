# 结构迁移与Repair实验

## 1. 文档边界

Repair指结构初始化之后、步数蒸馏之前的功能恢复。当前统一保持16步推理，目标是使压缩student恢复teacher的速度场、内部关系和`noise→mode`映射。

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

### P1：Repair严格确认

- [ ] 对Pareto候选补Standard-480；
- [ ] 与冻结MiniLMv2-6000在完全相同episode/seed下比较；
- [ ] 补至少两个额外训练seed；
- [ ] 统计每种mode的频次与条件成功率，判断Velocity是否只强化高频模式；
- [ ] 保存成功/失败异色轨迹图。

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

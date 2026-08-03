# BMD闭环保模态实验计划

长期目标是在不访问原始示范的条件下，为FastWAM、DreamZero等机器人WAM提供
可迁移的跨结构、跨步数保模态蒸馏。当前短期目标是将`FM-3x48-16`压缩为
`FM-3x48-1`时，把成功mode覆盖由约15提升至至少18，并保持`SR>=65%`。

## 顺序TODO与当前状态

- [x] P1：四卡新增1920条Velocity-250 Teacher rollout，与现有480条合并；
- [x] P1：重新运行hierarchical K=24；验收要求为24类全占用、每类至少20条成功
  轨迹且seed 42/43 NMI不低于0.8，实际没有候选通过最小簇20门槛；
- [x] P2：按episode划分数据，训练GRU、Transformer及不同分组候选的
  `q_T(z|trajectory prefix)`；
- [x] P2：要求完整轨迹macro-F1不低于0.85、50% prefix不低于0.70、每类召回
  不低于0.50；
- [x] P3：用同seed生成Teacher与一步Student配对闭环轨迹，并在Student-induced
  states查询Teacher；
- [x] P4a：四卡比较Student-induced Repair、冻结Mode loss、Student-weight和联合约束；
- [x] P4a：Standard-120筛选；无候选同时达到`SR>=65%`和覆盖至少18，因此不补480；
- [ ] P4b：审计Mode loss为何把SR提升到约98%却压缩为1--4 modes；
- [ ] P4b：在Student-induced states上做Teacher同状态多噪声诊断，确认条件多模态
  是否仍存在并量化可恢复上限；
- [ ] P4c：比较冻结Teacher关系保持、per-state set-OT/Sinkhorn和短轨迹set matching，
  不再使用类别原型式交叉熵；
- [ ] P4c：四卡首轮建议为`关系保持`、`endpoint set-OT`、`短轨迹set-OT`、
  `短轨迹set-OT＋弱endpoint anchor`；所有方法共用同一buffer、seed与训练预算；
- [ ] P4c：Standard-120只用于淘汰；只有`SR>=65%`且覆盖至少18才补Standard-480；
- [ ] P5：候选补三训练seed、paired exact-mode retention和per-mode SR；
- [ ] P6：有效约束前移到`4x72->3x48`结构蒸馏，再验证完整结构＋步数链路；
- [ ] P7：将轨迹encoder替换为action/world embedding，迁移FastWAM或DreamZero。

## 资源规则

每个P阶段启动前必须检查Slurm剩余时间、GPU数量和现有进程。时间不足时只保存
可恢复状态，不启动下一阶段。所有长任务使用tmux，阶段验收失败则停止自动后续。

Rollout和评估默认采用每GPU 4个单线程worker；episode分片不得重叠。启动前同时
检查CPU核数，避免worker数超过整机CPU承载能力。

## 方法边界

当前hierarchical clustering＋inference model属于BMD-inspired方法，不冒充论文中
含Steering Policy和PPO的完整BMD。若冻结分类器的离线代理不能阻止闭环collapse，
再升级到BMD intrinsic reward＋PPO。

## P1--P4a验收摘要

- 扩展buffer：2400 episodes，Teacher `SR=59.4%`，成功覆盖23/24，熵0.839；
- 严格最小簇20门槛没有候选通过；shape候选最小簇18，仅作为敏感性边界继续；
- shape-18 Transformer：完整prefix macro-F1 0.927，50% prefix 0.795，最低类召回
  0.667，是唯一通过在线模式推断门槛的模型；
- Student-induced buffer：480配对episodes；Teacher/Student SR分别为59.8%/76.0%；
- 基础Repair最佳SR点为70.0%/13 modes，最高覆盖点为60.8%/15 modes；
- Mode loss达到约98% SR，但仅保留1--4 modes，是明确失败而非正结果；
- Student-weight单独未改善Pareto，联合Mode loss时也不能阻止坍缩。

因此最近的证据把优先级从“继续扫分类loss权重”转向：先验证Student状态上的
Teacher条件多模态是否存在，再使用同状态多噪声的集合/关系匹配。只有该受控目标
能突破18-mode门槛，才值得进入多seed和跨结构迁移。

## P4b--P5更新（2026-07-31）

审计否决了同状态endpoint set-OT：Teacher在同一Student状态更换16组噪声时，
endpoint平均两两距离仅0.000889，模式主要来自闭环历史分叉。旧Mode loss使用独立
状态拼接的分布外伪轨迹，且成功/失败mask会改写成功Student并锚定双失败轨迹。

改用成功Teacher闭环Replay后，Standard-480为：Natural `68.3%/20 modes/H0.741`，
Balanced `65.0%/19/H0.752`，Balanced＋Anchor `64.2%/20/H0.693`，
PCGrad `65.8%/18/H0.645`。Natural是当前最简单且最优的Pareto点。

P5固定协议：

- [x] seed 42：Natural epoch 100、Balanced epoch 50；
- [x] seed 43/44：固定相同epoch，不重新基于Standard-120选点；
- [x] 四卡分别运行Natural-43、Natural-44、Balanced-43、Balanced-44；
- [x] 所有训练seed使用同一Standard-480评估suite；
- [ ] 三seed稳定后补paired retention/per-mode SR，再决定是否进入P6。

## P6：强Teacher结构蒸馏与完整链路计划

### 总体目标

将当前已经稳定的Natural Replay步数蒸馏嵌入完整链路：

```text
FM-4x72-16 Full Teacher
  -> FM-3x48-16 结构蒸馏/Repair
  -> FM-3x48-1 Natural Replay步数蒸馏
```

保持无原始示范、跨结构可迁移和mode-preserving。当前中期门槛：结构阶段
`SR>=80%、Coverage>=22`；步数阶段相对结构Teacher的SR下降不超过5个百分点、
覆盖下降不超过2 modes。

### P6.0：定格现有结论

- [x] Natural Replay三训练seed Standard-480；
- [x] Balanced Replay三训练seed Standard-480；
- [x] solver-only 1/2/4/8步诊断；
- [x] 240条强Teacher buffer直接一步Replay初探；
- [ ] 生成Natural三seed逐mode SR、paired exact-mode retention和轨迹图；
- [ ] 将Natural确定为步数蒸馏主基线，Balanced保留为消融。

### P6.1：扩大强Teacher闭环buffer

- [ ] 启动前确认checkpoint确为`FM-4x72-16 Full`、Standard-480约95.8%、24 modes；
- [ ] 四卡、每GPU四worker生成至少2400条强Teacher闭环rollout；
- [ ] 保存原生`state, noise, 16-step endpoint, episode, timestep, success, path`配对；
- [ ] 审计`uses_original_demonstrations=false`、`uses_expert_actions=false`；
- [ ] 验收`Teacher SR>=90%`、真实覆盖24/24、每个真实mode至少20条成功轨迹；
- [ ] 运行hierarchical无标签mode discovery，仅用二值成功反馈和轨迹几何；
- [ ] 要求跨seed NMI不低于0.8、24类全占用；不以真实mode指标选择聚类。

若强Teacher rollout本身未通过SR/覆盖门槛，则停止，不进入结构Repair。

### P6.2：结构迁移基线与因果拆分

- [ ] 固定同一个teacher-derived 3x48初始化，禁止混用Full-data warm start；
- [ ] 保存初始化后16步Standard-120诊断，但不把它当最终方法；
- [ ] 四路首轮结构Repair：
  1. Natural multi-time velocity distillation；
  2. velocity＋弱successful endpoint anchor；
  3. velocity＋MiniLMv2 relation保持；
  4. velocity＋relation＋弱endpoint anchor；
- [ ] 所有目标只使用强Teacher rollout/query，不接触原始示范；
- [ ] 训练期间保存多个checkpoint，Standard-120按预注册规则筛选；
- [ ] 只有`SR>=75%、Coverage>=20`的候选补Standard-480；最终进入下一阶段要求
  `SR>=80%、Coverage>=22`。

该阶段只训练和评估`FM-3x48-16`，不得同时改成一步，以隔离结构迁移损失。

### P6.3：Student-induced闭环迭代Repair

若离线结构Repair未达到80%/22：

- [ ] 使用当前最佳3x48-16真实闭环rollout产生Student-induced states；
- [ ] 在这些状态上查询冻结4x72-16 Teacher；
- [ ] 成功Teacher自然Replay作为全局模式来源；Student失败/不确定状态作为修正来源；
- [ ] 不再使用“Teacher失败就锚定Student”的旧mask；
- [ ] 比较离线Repair、一次DAgger式聚合、两次聚合；每轮重新rollout而非重复旧buffer；
- [ ] 监控SR、覆盖、逐mode SR和闭环状态分布漂移。

这一步是BMD思想在当前任务中的主要落点：约束真实闭环occupancy和行为覆盖，而非
在Teacher-forced局部endpoint上施加分类器loss。

#### Round 1执行协议（2026-08-01）

- [x] 审计并否决旧的一步Student-induced实现：旧代码硬编码`3x48-16 -> 3x48-1`，
  不适用于当前纯结构蒸馏；
- [x] 固定Teacher为`FM-4x72-16 Full`，Student起点为P6.2中SR并列但熵最高的
  `velocity+endpoint, epoch 250`；两者均保持16步；
- [x] 单轨迹smoke验证原生`student state/noise -> teacher 16-step correction`配对，
  shape、架构和无示范元数据通过；
- [ ] 四卡、每卡四worker收集480条Student真实闭环轨迹，每15条落盘；
- [ ] 合并后验证480个连续episode、无重复state/timestep、tensor有限且不访问示范；
- [ ] 固定弱endpoint权重0.03，四卡比较induced采样比例25%、50%、75%，以及
  `50% induced + hierarchical latent-balanced global Teacher anchor`；
- [ ] 每组训练250 epochs，保存50/100/250，并按同一Standard-120规则筛选；
- [ ] 只有`SR>=75%、Coverage>=20`的候选补Standard-480；最终结构阶段门槛仍为
  `SR>=80%、Coverage>=22`；
- [ ] 若Round 1未过最终门槛，用最佳模型重新闭环采样后做Round 2，不重复使用旧buffer
  伪装成迭代，也不直接增加训练轮数。

因果控制：四路共用相同初始化、Teacher、base buffer、Student-induced buffer、seed、
训练预算与16步求解器；只改变induced/base占比及BMD全局锚采样。强Teacher成功buffer
负责保留全局24-mode支持，Student-induced状态仅负责修复闭环covariate shift。

#### Round 1结果与机制转向

四路均未通过`75%/20`门槛。最佳SR为25% induced、epoch 50的
`64.2%/17/H=0.754`；兼顾覆盖的最佳点为50% induced、epoch 250的
`62.5%/19/H=0.821`。75% induced虽在epoch 50短暂达到20 modes，但SR仅55%。
因此静态混合更多Student状态不能稳定解决问题，且会形成全局能力与局部纠错的权衡。

#### P6.3 Round 2：Intervention DAgger

目标从“给Student失败状态添加单点Teacher标签”切换为“让Teacher演示从危险状态连续
返回可成功流形的恢复段”。

- [x] 用Round 1原生配对标定Teacher--Student endpoint disagreement；归一化RMSE的
  `q80=0.4181`，预注册为首轮触发阈值，不使用评估结果调阈值；
- [ ] 固定P6.2 `velocity+endpoint epoch 250`为共同起点，保持`FM-3x48-16`；
- [ ] Student控制时若disagreement超过0.4181，冻结当前episode/step噪声规则并让
  `FM-4x72-16`连续接管H步；接管期间仍记录Student反事实输出；
- [ ] 分别生成H=4和H=8的480条混合闭环轨迹，保存trigger、controller、连续恢复段、
  intervention rate和assisted success；
- [ ] 四卡比较`H4-all`、`H4-recovery-only`、`H8-all`、`H8-recovery-only`；训练端
  固定25% induced占比、弱endpoint 0.03、seed和预算，避免再次扫loss；
- [ ] 与Round 1 point-query 25%基线直接比较。Standard-120筛选后，只有
  `SR>=75%、Coverage>=20`者补Standard-480；
- [ ] 若连续接管提高SR但压缩模式，再引入episode-fixed latent router/MoE；若接管本身
  无效，再转向FA-OPD式短轨迹occupancy/reward蒸馏，而非继续扫H或权重。

机制假设：point query只有局部动作监督，无法说明纠正后状态如何演化；Teacher takeover
提供连续、可恢复的状态--动作序列，并通过Student/Teacher混合执行把训练分布限制在
Student真实访问但仍可由Teacher恢复的区域。这是DAgger/SafeDAgger式数据生成机制，
不是新增辅助loss。

### P6.4：Natural Replay步数蒸馏

结构阶段通过后：

- [ ] 用最佳`FM-3x48-16`生成至少2400条自身成功闭环rollout；
- [ ] 对`FM-3x48-1`执行Natural Replay，不先加均衡、anchor或PCGrad；
- [ ] seed 42筛checkpoint，固定epoch后补seed 43/44 Standard-480；
- [ ] 报告结构Teacher到一步Student的SR retention、mode retention、JS和逐mode SR；
- [ ] 若一步SR损失超过5个百分点，再训练真正的4步和2步Student，比较
  `16->4->2->1`，不能用solver-only替代。

### P6.5：完整链路与WAM迁移门槛

- [ ] 用相同480个episode/noise对齐评估4x72-16、3x48-16、3x48-1；
- [ ] 分别归因结构损失和步数损失；报告参数量、solver步数和实际推理延迟；
- [ ] 完整链路至少三个训练seed；
- [ ] D3IL链路稳定后，将手工轨迹特征替换为action/world embedding；
- [ ] 在FastWAM或DreamZero先做小规模smoke，再进行完整任务评估。

### Go/No-Go原则

- 不因GPU空闲而跳过审计；
- 不用Standard-120的1--3个mode差异形成结论；
- 不把solver-only当作多步蒸馏；
- 不把无标签聚类的latent当作真实mode；
- 任一阶段未过门槛，先定位数据、结构或闭环分布瓶颈，不自动进入下一阶段。

## P6--P8执行状态更新（2026-08-01）

- [x] P6.1：扩展强Teacher buffer至2400条；`96.92%/24/H=0.945`，无示范审计通过；
- [x] P6.2：四路强Teacher结构Repair；最佳仅`62.5%/19/H=0.850`，未过75/20；
- [x] P6.3 Round 1：普通Student-induced点查询与25/50/75%混合；未过门槛；
- [x] P6.3 Round 2：H4/H8 Intervention DAgger训练与评估；
- [x] H4-recovery、H8-all补Standard-480，分别为`59.2%/22/H=.862`和
  `60.6%/22/H=.851`；
- [x] P7恢复上限：Teacher首次触发后接管至终局为assisted
  `92.9%/24/H=.950`；
- [x] P7完整buffer训练：Random、PCA、MiniLMv2三种初始化公平对照；证明初始化重要，
  但旧4-batch预算不是主要瓶颈；
- [x] P8成功恢复buffer与25/50/75%、75->50->25课程；最佳Standard-120为
  `70.8%/18/H=.791`，无候选过75/20，不补480；
- [x] P8.5：TinySR启发的四种3x72深度mask可恢复性穷举；keep013 Standard-480达到
  `91.0%/24/H=.929`，结构深度阶段通过；
- [ ] P9a：keep013补独立Standard-480 suite与训练seed 43/44；
- [ ] P9b：固定keep013执行`3x72 -> 3x48`宽度压缩四路公平对照；
- [ ] P9c：最佳3x48-16达到至少`80%/22`后，恢复P6.4 Natural Replay步数蒸馏；
- [ ] 连续恢复片段/交还survival保留为宽度压缩失败后的机制支线，不再是当前第一优先级。

当前Go/No-Go决定：停止继续扫描静态recovery比例、普通训练epoch和局部点对点loss。keep013
证明强Teacher rollout可以完成高质量纯深度迁移；当前第一优先级改为单独解决宽度压缩，避免
再次把深度、宽度和闭环恢复混为一个变量。覆盖和熵只在成功轨迹上计算，keep013仍需独立suite、
多seed和逐mode可靠性确认。

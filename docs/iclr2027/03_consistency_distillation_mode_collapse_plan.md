# Consistency Distillation 后的多模态保持实验

## 1. 研究目标

本实验研究 DDPM-Transformer 在 consistency distillation 压缩采样步数后，是否出现 mode collapse，即原始策略的多模态绕障路径被压缩到少数模式。

主 teacher 使用当前闭环成功率最高的 checkpoint：

`logs/avoiding/trained/ddpm_transformer_10000_cosine_seed42/eval_best_ddpm.pth`

其 8-step DDPM 基线为 468/480 成功（97.5%）、24/24 成功模式覆盖、归一化模式熵 0.913。15,000-epoch checkpoint 的模式熵更高（0.936），可作为后续高多样性对照，但不作为第一阶段主 teacher。

## 2. 核心假设

- 如果蒸馏只降低推理步数，而保留噪声到动作的映射，student 应基本保持 teacher 的 24-mode 分布。
- 如果 student 对不同 teacher mode 做均值回归，成功模式覆盖、模式熵和有效模式数会下降。
- 确定性 DDIM 采样本身也可能改变模式分布，因此必须建立 DDIM 基线，避免把 sampler 变化误判为蒸馏导致的 mode collapse。

## 3. 实验原则

1. teacher 与 student 共享初始噪声 `z`，使 `z` 持续充当 mode identifier。
2. 不对同一状态下多个 teacher 动作直接求均值。
3. 采用渐进式 `8 → 4 → 2 → 1` 蒸馏，每个阶段独立评估。
4. teacher 冻结并使用 EMA checkpoint；student 从上一阶段 teacher 权重初始化。
5. 闭环性能与多模态指标同时报告，不能只比较动作 MSE。

## 4. TODO 与状态

### 阶段 A：蒸馏前基线

- [x] 在现有 diffusion model 中加入 deterministic DDIM（`eta=0`）采样，同时保持 DDPM 为默认行为。
- [x] 完成语法、dtype、checkpoint 兼容性和单轨迹 smoke test；修复 DDIM 中间张量被提升为 `float64` 的问题。
- [x] 使用 10,000-epoch teacher 运行 8-step DDIM、480 条 rollout、seed 42。
- [x] 对比 8-step DDPM 与 8-step DDIM，确认 sampler 变化已经降低模式熵，但没有减少模式覆盖。

### 阶段 B：Consistency / Progressive Distillation

- [x] 新增独立蒸馏脚本，不改动原始 DDPM 训练路径。
- [x] 实现共享初始噪声下的 teacher 8-step DDIM target 与 student 4-step 预测。
- [x] 使用 consistency loss，并保留小权重原始 diffusion loss 作为正则项。
- [x] 完成 `8 → 4` 蒸馏，保存 student 与 EMA checkpoint。
- [x] 完成无 diffusion/DSM loss 的 `8 → 4` 消融及 480-rollout 评估。
- [x] 完成 `4 → 2` 蒸馏：DSM 权重 0.1 与 0 两组均完成 500 epochs 和 480-rollout 评估。
- [ ] `2 → 1` 蒸馏正在按相同 DSM 对照配置运行。

### 阶段 C：闭环与多模态评估

- [ ] 对 teacher-8/DDPM、teacher-8/DDIM、student-4、student-2、student-1 使用相同协议评估（student-2 已完成）。
- [ ] Student-4 已完成 seed 42、480 条 rollout；正式结果仍需扩展到至少 6 个 evaluation seeds。
- [x] 统计成功率、24-mode coverage、归一化 mode entropy 和各 mode 频率。
- [x] 计算 effective modes、与 teacher 分布的 JS divergence、每个 mode 的频率变化。
- [ ] 绘制不同采样步数的 mode-count 图及指标折线图。

## 5. Mode Collapse 判定

单次评估少出现一个低频 mode 不直接视为 mode collapse。正式判断应基于多 seed 置信区间，并同时满足以下证据中的多项：

- student 的成功模式覆盖持续低于 teacher；
- normalized mode entropy 或 effective modes 显著下降；
- student 与 teacher 的 mode distribution JS divergence 显著增大；
- 轨迹集中到少数通道组合，且跨 evaluation seeds 可复现；
- 上述变化不能仅由成功率下降解释。

## 6. 当前实现与训练信号

当前实现位于 `distill_ddim_avoiding.py`，属于共享噪声的 8-step teacher 到 4-step student 轨迹蒸馏。它借鉴 consistency/progressive distillation 的核心思想，但不是原始 Consistency Models 中相邻时间点的严格 local consistency objective：当前 consistency 分支直接比较 teacher 和 student 从同一初始噪声生成的最终动作。

### 6.1 输入输出与两个训练分支

两个分支使用同一批 Avoiding 专家状态：

```text
state.shape  = [batch, 5, 4]
action.shape = [batch, 5, 2]
```

Consistency 分支：

```text
teacher_action = teacher_8step_ddim(state, z)
student_action = student_4step_ddim(state, z)
consistency_loss = MSE(student_action, teacher_action)
```

teacher 与 student 共享初始噪声 `z`，使其保持为潜在 mode identifier。Teacher 冻结；student 从 10,000-epoch teacher 权重初始化。Student 使用离散时间表 `[7, 4, 2, 0]`。

Diffusion/DSM 分支：

```text
t       = random integer in [0, 7]
epsilon = Gaussian noise
x_t     = add_noise(expert_action, epsilon, t)
epsilon_pred = student(x_t, t, state)
diffusion_loss = MSE(epsilon_pred, epsilon)
```

两个分支共享同一个 student 和同一批 state，但动作侧噪声独立采样。每个 batch 只做一次联合反向传播：

```text
total_loss = consistency_loss + 0.1 * diffusion_loss
```

Consistency loss 负责压缩 teacher 的生成过程；diffusion loss 让 student 继续接触原始 24-mode 专家动作分布。后者是降低遗忘风险的正则项，但不能保证不会发生 mode collapse。

### 6.2 是否必须使用专家动作

- 当前方案需要原始专家动作，因为 diffusion/DSM 分支必须从干净专家动作构造 `x_t`。
- 去掉 diffusion loss 后，纯 teacher trajectory matching 不需要专家动作，但仍需要有代表性的 state/observation。
- State 可来自原数据、缓存或环境 rollout；如果只使用 teacher rollout，条件分布可能偏向 teacher 的高频路径。
- One-Step Diffusion Policy 一类 distribution/score distillation 可以不直接使用专家动作，但仍通常从示范数据取得 observation。

### 6.3 与从零训练 4-step diffusion 的区别

从零训练 4-step diffusion 只以专家数据上的 diffusion loss 为目标，并重新学习四步噪声过程。当前 student 的主目标是复现 8-step teacher 的生成映射，diffusion loss 仅以 0.1 权重作为辅助正则；此外它沿用原 8-step 时间参数化，只在推理时选择四个离散时间点。

因此必须加入“从零训练的原生 4-step diffusion”作为强基线，才能判断蒸馏是否优于直接训练。

### 6.4 相关工作

- [Consistency Models](https://arxiv.org/abs/2303.01469)：提出 consistency distillation 与 consistency training。
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI)：将确定性 DDIM sampler 逐级从 N 步压缩到 N/2 步。
- [Consistency Policy](https://arxiv.org/abs/2405.07503)：机器人策略中使用 `CTM loss + DSM loss`；论文也报告确定性 consistency trajectory 可能损失部分多模态性。
- [One-Step Diffusion Policy](https://arxiv.org/abs/2410.21257)：通过 teacher score 做 distribution matching，蒸馏损失不直接要求专家动作。

## 7. `8 → 4` 蒸馏设置与结果

### 7.1 训练设置

| 项目 | 设置 |
|---|---:|
| Teacher | 10,000-epoch 最佳 EMA checkpoint |
| Teacher sampler | 8-step deterministic DDIM，`eta=0` |
| Student sampler | 4-step deterministic DDIM，`[7,4,2,0]` |
| 蒸馏 epochs | 500 |
| Batch size | 256 |
| Optimizer | Adam |
| 初始学习率 | `1e-4` |
| LR scheduler | Cosine，最低 `1e-6` |
| Diffusion loss 权重 | 0.1 |
| EMA decay | 0.995 |

训练耗时 17 分 34 秒，最佳 checkpoint 出现在 epoch 475，最佳总损失为 0.01296。训练完成后单轨迹 smoke test 成功，随后完成 seed 42 的 480 条闭环评估。

### 7.2 四组实验对比

| 模型 | 成功率 | 成功模式 | 模式熵 | 有效模式数 |
|---|---:|---:|---:|---:|
| 8-step DDPM Teacher | 97.50%（468/480） | 24/24 | 0.913 | 18.19 |
| 8-step DDIM Teacher | **97.71%（469/480）** | 24/24 | 0.884 | 16.63 |
| 4-step Student，DSM=0.1 | 96.25%（462/480） | 22/24 | 0.880 | 16.37 |
| 4-step Student，DSM=0 | 96.04%（461/480） | 23/24 | **0.898** | **17.34** |

相邻对照的 mode-distribution 差异如下：

| 对照 | 缺失模式 | JS divergence（nats） | 归一化 JS |
|---|---|---:|---:|
| DDPM Teacher → DDIM Teacher | 无 | 0.02194 | 0.03165 |
| DDIM Teacher → Student，DSM=0.1 | `1-2-2`、`1-2-3` | **0.01034** | **0.01491** |
| DDIM Teacher → Student，DSM=0 | `1-2-3` | 0.01785 | 0.02575 |

所有 Student 都没有产生 teacher 之外的新模式。

8-step DDIM Teacher 轨迹图：

![8-step DDIM teacher 的 480 条轨迹](../../logs/avoiding/ddim_teacher_8_seed42_480_rollouts/trajectory_comparison.png)

含 DSM Student 轨迹图：

![4-step distilled student 的 480 条轨迹](../../logs/avoiding/distilled/ddim_student_4step_480_rollouts/trajectory_comparison.png)

无 DSM Student 轨迹图：

![无 DSM 的 4-step student 的 480 条轨迹](../../logs/avoiding/distilled/ddim_student_4step_nodsm_480_rollouts/trajectory_comparison.png)

### 7.3 当前分析

8-step DDPM 改为 8-step DDIM 后，成功率从 97.50% 微升至 97.71%，仍覆盖 24/24 模式；但模式熵由 0.913 降至 0.884，有效模式数由 18.19 降至 16.63。这说明多模态分布压缩首先发生在 stochastic DDPM → deterministic DDIM 的 sampler 变化阶段，而不是全部由蒸馏造成。

以正确的 8-step DDIM Teacher 为基线，含 DSM Student 的成功率下降 1.46 个百分点，缺失两个低频模式，熵仅下降 0.005，有效模式数下降 0.26；其 JS divergence 只有 0.0103 nats。由此看，`8 → 4` 蒸馏带来的额外分布变化较小，当前更接近轻微低频 mode loss，而非严重 mode collapse。

无 DSM Student 的成功率为 96.04%，比含 DSM 版本只低 0.21 个百分点；它保留 23/24 模式，熵和有效模式数反而高于含 DSM Student，唯一缺失的是 Teacher 中仅出现一次的 `1-2-3`。因此当前单 seed 实验没有显示 `0.1 * diffusion loss` 能改善多模态保持；但含 DSM Student 与 DDIM Teacher 的 JS divergence 更小，说明 DSM 对整体频率分布的贴合可能略有帮助。两种指标指向不完全一致，需要多 seed 判断。

当前可以将变化分解为：

```text
DDPM -> DDIM：覆盖不变，但熵明显下降
DDIM -> 4-step：成功率小幅下降，1–2 个低频 mode 消失
```

当前只有一个训练 seed 和一个评估 seed；DDIM Teacher 中 `1-2-2` 与 `1-2-3` 本来就分别只有 2 次和 1 次成功，因此有限采样波动足以造成缺失，尚不能据此宣称统计显著的 mode collapse。

## 8. `4 → 2` 蒸馏设置与结果

### 8.1 训练与评估设置

两组实验分别从各自对应的 4-step checkpoint 继续渐进蒸馏，避免在 DSM 消融链路之间交叉初始化。Teacher 与 student 共享初始噪声，teacher 使用 4-step deterministic DDIM，student 使用 2-step deterministic DDIM；训练后先执行单轨迹 smoke test，再以 seed 42 完成 480 条闭环轨迹评估。

| 项目 | DSM=0.1 | DSM=0 |
|---|---:|---:|
| Teacher | 4-step DSM=0.1 student | 4-step DSM=0 student |
| 蒸馏方向 | 4-step → 2-step | 4-step → 2-step |
| 蒸馏 epochs | 500 | 500 |
| Batch size | 256 | 256 |
| Diffusion/DSM loss 权重 | 0.1 | 0 |
| 最佳 epoch | 482 | 491 |
| 最佳总 loss | 0.02350 | 0.00830 |
| Smoke test | 1/1 成功 | 1/1 成功 |

### 8.2 480-rollout 结果

| 模型 | 成功率 | 成功模式 | 模式熵 | 有效模式数 |
|---|---:|---:|---:|---:|
| 4-step Student，DSM=0.1 | 96.25%（462/480） | 22/24 | 0.880 | 16.37 |
| 2-step Student，DSM=0.1 | **95.42%（458/480）** | 21/24 | 0.818 | 13.45 |
| 4-step Student，DSM=0 | 96.04%（461/480） | 23/24 | 0.898 | 17.34 |
| 2-step Student，DSM=0 | **94.79%（455/480）** | 23/24 | 0.844 | 14.60 |

有效模式数按 `exp(H_norm * ln(24))` 计算。

DSM=0.1 的 2-step 轨迹图：

![DSM=0.1 的 2-step student 480 条轨迹](../../logs/avoiding/distilled/ddim_student_2step_dsm01_480_rollouts/trajectory_comparison.png)

DSM=0 的 2-step 轨迹图：

![DSM=0 的 2-step student 480 条轨迹](../../logs/avoiding/distilled/ddim_student_2step_nodsm_480_rollouts/trajectory_comparison.png)

### 8.3 结论

从 4-step 继续压缩到 2-step 后，两组成功率都只下降约 1.2–1.3 个百分点，说明任务完成能力总体仍较稳定；但模式熵下降更明显：DSM=0.1 从 0.880 降至 0.818，DSM=0 从 0.898 降至 0.844，有效模式数分别减少约 2.92 和 2.74。这表明进一步减少采样步数主要损伤的是模式频率均衡性，而不是立即造成大幅成功率崩溃。

在两种 2-step student 之间，DSM=0.1 的成功率高 0.63 个百分点，但只覆盖 21/24 模式；DSM=0 覆盖 23/24 模式，模式熵也高 0.026。当前单 seed 结果仍未证明 `0.1 * DSM loss` 有助于多模态保持，反而显示纯 consistency 分支在 coverage 与 entropy 上更好；不过差异可能受低频模式采样波动影响，仍需多 seed 验证。

综合 `8 → 4 → 2` 的结果，压缩趋势可以概括为：

```text
采样步数减少：成功率缓慢下降
确定性 DDIM 与进一步蒸馏：模式熵持续下降
DSM=0.1：略偏向成功率，但当前未改善模式覆盖或熵
```

因此 2-step 阶段已经出现可测量的多模态压缩，但尚不是彻底的 mode collapse：成功率仍约 95%，且至少保留 21/24 模式。

### 8.4 两条渐进蒸馏路径的指标曲线

下图使用共同的 8-step deterministic DDIM teacher 作为起点，分别连接 DSM=0.1 与 DSM=0 的 `8 → 4 → 2 → 1` 结果。1-step 的最终结果为：DSM=0.1 成功率 6.46%、模式熵 0.619；DSM=0 成功率与模式熵均为 0。

![两条蒸馏路径的成功率变化](assets/distillation_success_rate_vs_steps.png)

![两条蒸馏路径的模式熵变化](assets/distillation_mode_entropy_vs_steps.png)

两条曲线都显示 8→4→2 阶段成功率缓慢下降，而 2→1 出现断崖式退化。DSM=0.1 在单步阶段仍保留少量成功轨迹和模式多样性，DSM=0 则完全失效；因此当前瓶颈首先是 one-step 任务能力崩溃，其次才是单独的 mode collapse。

## 9. 下一步对照实验

1. 从原始数据直接训练原生 4-step diffusion，使用相同网络与训练预算。
2. 对每组扩展多个 evaluation seeds，报告置信区间和聚合 mode distribution。
3. 对 DSM 权重 `0`、`0.01`、`0.1` 做多 seed 消融，判断其对覆盖和 JS divergence 的真实影响。
4. 完成正在运行的 `2 → 1` 蒸馏，检验单步策略是否使模式熵和覆盖进一步快速下降。

## 10. 实验产物

- 蒸馏脚本：`distill_ddim_avoiding.py`
- 训练日志：`logs/avoiding/distill_8to4_pipeline.log`
- 最佳 student：`logs/avoiding/distilled/ddim_student_4step_seed42/eval_best_ddpm.pth`
- 蒸馏指标：`logs/avoiding/distilled/ddim_student_4step_seed42/distillation_metrics.json`
- 480-rollout 指标：`logs/avoiding/distilled/ddim_student_4step_480_rollouts/metrics.json`
- 原始轨迹：`logs/avoiding/distilled/ddim_student_4step_480_rollouts/ddpm_transformer_trajectories.npz`
- 轨迹图：`logs/avoiding/distilled/ddim_student_4step_480_rollouts/trajectory_comparison.png`
- 8-step DDIM 指标：`logs/avoiding/ddim_teacher_8_seed42_480_rollouts/metrics.json`
- 8-step DDIM 轨迹图：`logs/avoiding/ddim_teacher_8_seed42_480_rollouts/trajectory_comparison.png`
- 无 DSM 最佳 student：`logs/avoiding/distilled/ddim_student_4step_nodsm_seed42/eval_best_ddpm.pth`
- 无 DSM 蒸馏指标：`logs/avoiding/distilled/ddim_student_4step_nodsm_seed42/distillation_metrics.json`
- 无 DSM 480-rollout 指标：`logs/avoiding/distilled/ddim_student_4step_nodsm_480_rollouts/metrics.json`
- 无 DSM 轨迹图：`logs/avoiding/distilled/ddim_student_4step_nodsm_480_rollouts/trajectory_comparison.png`
- 2-step DSM=0.1 最佳 student：`logs/avoiding/distilled/ddim_student_2step_dsm01_seed42/eval_best_ddpm.pth`
- 2-step DSM=0.1 蒸馏日志：`logs/avoiding/distill_4to2_dsm01_pipeline.log`
- 2-step DSM=0.1 指标与轨迹图：`logs/avoiding/distilled/ddim_student_2step_dsm01_480_rollouts/`
- 2-step DSM=0 最佳 student：`logs/avoiding/distilled/ddim_student_2step_nodsm_seed42/eval_best_ddpm.pth`
- 2-step DSM=0 蒸馏日志：`logs/avoiding/distill_4to2_nodsm_pipeline.log`
- 2-step DSM=0 指标与轨迹图：`logs/avoiding/distilled/ddim_student_2step_nodsm_480_rollouts/`

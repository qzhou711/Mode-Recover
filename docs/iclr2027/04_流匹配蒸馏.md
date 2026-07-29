# Flow Matching Policy 压缩与 Mode Collapse 实验

## 1. 研究目标

在 D3IL Avoiding 任务中建立原生 Flow Matching policy，并同时压缩 ODE 求解步数与网络规模。核心问题不是单纯获得高成功率，而是验证：蒸馏后 student 仍能完成任务时，多模态路径分布是否发生压缩；随后用显式多样性约束缓解 mode collapse。

统一报告闭环成功率、24-mode 覆盖、归一化模式熵、参数量和推理步数。所有正式比较使用相同 seed 与 rollout 协议；30/120 条只作筛选，最终结论使用 480 条。

## 2. 原生 Flow Matching 实现

新增实现：

- `agents/models/flow_matching/flow_matching.py`：线性 conditional flow matching（CFM）训练目标，以及 Euler/Heun ODE 求解。
- `agents/flow_matching_agent.py`：复用现有数据缩放、EMA、优化与闭环预测接口。
- `configs/agents/flow_matching_transformer_agent.yaml`：与 DDPM-Transformer 可比的 Transformer backbone 配置。
- `distill_flow_matching_avoiding.py`：teacher shortcut/chord target、原始 CFM 正则、网络缩小与几何保持约束。
- `visualize_avoiding.py`：支持 Flow Matching、任意求解步数及压缩后的 Transformer 结构。

标准 CFM 从同形高斯噪声 `z` 与专家动作 `a` 构造线性路径：

```text
x_t = (1 - t) z + t a
v_target = a - z
L_flow = MSE(v_theta(x_t, t, state), v_target)
```

推理从 `z` 出发，对学习到的速度场从时间 0 积分到 1。步数减少是数值求解压缩；网络层数和 embedding 宽度减少是模型规模压缩，两者分开记录。

## 3. 蒸馏与抗塌缩约束

对 student 网格上的起点 `t`，冻结 teacher 从相同噪声积分得到 `x_t` 和终点 `x_1^T`，student 学习跨越剩余区间的 chord velocity：

```text
v_shortcut_target = (x_1^T - x_t) / (1 - t)
L_shortcut = MSE(v_student(x_t, t, state), v_shortcut_target)
L_total = L_shortcut + lambda_flow L_flow + lambda_geo L_geometry
```

`L_flow` 让 student 继续接触原始专家数据分布。`L_geometry` 比较 batch 内 student/teacher 终点的归一化两两距离，保留样本间相对几何结构；完全塌缩到单一点会受到惩罚，同时该项对整体尺度变化不敏感。它是本项目的抗 mode-collapse 实验约束，不把它表述为已有算法的完整复现。

最优 checkpoint 使用固定验证 batch、固定噪声和固定时间网格选择，而不是随机训练 batch 的瞬时 loss。当前配置的 train/test 数据目录是否严格独立仍需结合主配置解释，因此最终模型选择与论文结论以闭环 rollout 为准。

## 4. 实现验收

| 验收项 | 结果 |
|---|---|
| 线性路径与目标速度 | 通过 |
| Euler/Heun 常速度场全程与局部积分 | 通过 |
| 速度网络梯度有限且可回传 | 通过 |
| shortcut target 在常速度 teacher 上解析正确 | 通过 |
| 几何损失：同形为 0、尺度不敏感、塌缩受罚 | 通过 |
| 小模型 checkpoint 保存/重载输出一致 | 通过 |
| 小模型蒸馏→保存→可视化加载→闭环执行 | 通过 |
| 固定验证 loss 选择 best checkpoint | 通过 |
| Python 编译与 `git diff --check` | 通过 |

默认 teacher 为 4 层、72 维 Transformer，共 274,826 参数；验收用小 student 为 2 层、36 维、3 heads，共 37,982 参数，即参数量压缩 7.24 倍。

两轮端到端 smoke 中，小 student 总训练损失由 0.749 降至 0.658；该 smoke 的 teacher 只训练 2 epochs，因此轨迹成败不作为性能结论。

## 5. 正式实验状态与结果

正式 teacher：`flow_matching_transformer_5000_seed42`，5000 epochs，16-step Heun，seed 42。训练完成后先以 120 条 rollout 验收；暂定进入蒸馏的门槛为成功率至少 80%，且成功模式至少 18/24。之后补充 480 条基线。

<!-- FORMAL_RESULTS -->

## 6. 计划中的对照矩阵

1. 未蒸馏 teacher 在 16/8/4/2/1 步下的 solver-only 基线。
2. 相同网络规模的 shortcut 蒸馏，区分步数压缩与模型压缩。
3. 2 层、36 维 student，测量约 7.24 倍参数压缩的额外影响。
4. `lambda_geo=0` 与正值的配对实验；其余 seed、噪声、训练预算和评估协议保持一致。
5. 对筛选后的关键 checkpoint 做 480-rollout，并保存成功/失败分色轨迹图与 mode 频次。

只有出现“成功率仍可用，但 mode 覆盖/熵显著下降”，才支持蒸馏导致 mode collapse 的目标现象；抗塌缩方案必须在成功率不显著恶化的前提下提高覆盖或熵。

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

完整MiniLMv2相对简化目标在300轮时不损失SR并将覆盖翻倍。训练到1000轮后多样性显著提高、SR下降；2000轮相对1000轮又提高8.3个SR百分点、1种模式和0.082熵，但仍未恢复300轮的SR。关系目标形成非单调Pareto轨迹，不能用单一loss选择checkpoint。

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

## 10. 产物

- `logs/avoiding/teacher_generated_structure_wave2/`
- `logs/avoiding/teacher_generated_structure_wave2_1000/`
- `logs/avoiding/teacher_generated_structure_wave2_teacher_assistant/`
- `logs/avoiding/teacher_generated_structure_wave2_teacher_assistant_stage_probe/`

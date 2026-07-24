# Flow Matching 单步蒸馏：同规模对照与 Student 容量扫描

## 1. 实验问题

本轮实验检验两个相互关联但不同的问题：

1. 改进后的`3×48`模型做同规模`16→1`蒸馏时，是否仍会mode collapse；
2. 固定Large-FM teacher与蒸馏配置，仅增加student容量，能否恢复成功率和多模态。

## 2. 统一配置

- 任务：Avoiding；
- teacher推理步数：16；
- student推理步数：1；
- 蒸馏训练：500 epochs；
- batch size：256；
- 每epoch最多4 batches；
- CFM loss权重：0.1；
- geometry loss权重：0；
- seed：42；
- checkpoint选择：固定验证集loss最优；
- 闭环评估：480条轨迹；
- 指标：成功率、成功模式覆盖、归一化模式熵。

容量扫描的teacher均为Large-FM `4×72`。`3×48→3×48`是独立的同规模因果对照，不混入容量曲线。

## 3. 最终结果

### 3.1 同规模`3×48→3×48`

| Teacher | Student | Student参数 | 成功率 | 模式数 | 模式熵 |
|---|---|---:|---:|---:|---:|
| `3×48` Full-FM-16 | `3×48` distilled-1 | 95,042 | 369/480（76.9%） | 11/24 | 0.122 |

对应基线：

| 模型 | 步数 | 成功率 | 模式数 | 模式熵 |
|---|---:|---:|---:|---:|
| `3×48` Full-FM | 16 | 445/480（92.7%） | 24/24 | 0.965 |
| 同checkpoint直接solver | 1 | 394/480（82.1%） | 14/24 | 0.569 |
| 同规模蒸馏 | 1 | 369/480（76.9%） | 11/24 | 0.122 |

蒸馏没有恢复单步solver丢失的模式，反而进一步降低成功率、模式数和模式熵。这说明在`3×48`上，即使teacher与student结构完全相同，当前pointwise shortcut目标仍会产生明显模式集中。

### 3.2 Large teacher到不同容量student

| Student | 参数量 | 最优epoch | 最优验证loss | 成功率 | 模式数 | 模式熵 |
|---|---:|---:|---:|---:|---:|---:|
| `2×36` | 37,982 | 既有基线 | — | 387/480（80.6%） | 12/24 | 0.147 |
| `3×48` | 95,042 | 460 | 0.2871 | 367/480（76.5%） | 11/24 | 0.193 |
| `4×64` | 217,666 | 467 | 0.2196 | 357/480（74.4%） | 14/24 | 0.212 |

容量增加没有单调提高闭环成功率：成功率从80.6%下降到74.4%。模式熵从0.147缓慢增加到0.212，最大模型覆盖14个模式，但仍远低于Large-FM-16的24模式和约0.945熵。

因此，增加student容量只带来有限的多样性改善，无法单独解决mode collapse。当前还观察到成功率—多样性的轻微交换：较大student分布稍均衡，但闭环成功率更低。

## 4. 更新后的归因

早期`2×36`实验显示：

- Small→Small蒸馏：23模式、熵0.885；
- Large→Small蒸馏：12模式、熵0.147。

这曾支持teacher–student容量差距是主要因素。但本轮新增证据表明：

- `3×48→3×48`同规模蒸馏仍只有11模式、熵0.122；
- Large→`3×48/4×64`即使增加容量，熵也只恢复到0.193/0.212。

因此更准确的结论是：

> teacher–student容量差距会影响蒸馏难度，但不是mode collapse的充分解释。collapse取决于teacher速度场、student架构、一步transport复杂度和pointwise shortcut优化共同作用；单纯扩大student不能恢复teacher的条件多模态分布。

`2×36` Small→Small的良好多样性更可能说明Small teacher的映射更平滑或更容易压缩，而不能推广为“所有同规模蒸馏均不collapse”。

## 5. 并行评估验收

`Large→3×48`的480条评估被拆成两个互不重叠分片：

- GPU 0：episodes 0–239，184/240成功；
- GPU 1：episodes 240–479，183/240成功。

每条轨迹仍使用`seed + episode_id`。合并后为367/480成功，并基于全部成功轨迹重新计算模式数和模式熵。分片指标没有被直接平均。

后续评估将按统一规范在启动前检查可用GPU数量，默认每张GPU使用4个独立环境worker。

## 6. 图表与轨迹

容量扫描：

- 成功率：`logs/avoiding/flow_capacity_scan/capacity_vs_success_rate.png`
- 模式熵：`logs/avoiding/flow_capacity_scan/capacity_vs_mode_entropy.png`
- 模式覆盖：`logs/avoiding/flow_capacity_scan/capacity_vs_mode_count.png`
- 汇总数据：`logs/avoiding/flow_capacity_scan/capacity_scan_results.json`

轨迹图：

- `3×48→3×48`：`logs/avoiding/flow_capacity_scan/same_3x48/eval480/trajectory_comparison.png`
- Large→`3×48`：`logs/avoiding/flow_capacity_scan/large_to_3x48/eval480/trajectory_comparison.png`
- Large→`4×64`：`logs/avoiding/flow_capacity_scan/large_to_4x64/eval480/trajectory_comparison.png`

## 7. 下一步

容量扫描已经表明继续单纯扩大student不是最高优先级。下一步应优先：

1. 比较直接`16→1`与渐进式`16→4→2→1`；
2. 固定状态和初始噪声，建立teacher/student噪声到模式的转移矩阵；
3. 检验distribution matching、IMLE或其他显式覆盖约束；
4. 至少补充多个训练seed，确认当前非单调容量趋势不是单seed波动。

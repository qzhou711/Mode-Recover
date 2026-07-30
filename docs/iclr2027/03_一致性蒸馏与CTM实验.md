# 一致性蒸馏与CTM实验

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

同结构Teacher-init蒸馏不必然导致mode collapse；跨结构与初始化误差才是关键前置因素。

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

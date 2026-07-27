# Flow Matching蒸馏初始化纠正与最新结论

## 1. 纠正摘要

此前同规模蒸馏对照存在初始化方式不一致：

- `FM-4x72-1-Distill-4x72`和`FM-2x36-1-Distill-2x36`通过`deepcopy(teacher)`初始化；
- 原`FM-3x48-1-Distill-3x48`因显式传入student结构而重新实例化网络，实际为Random-init。

因此，原`3×48`的76.9%成功率、11/24模式和0.122熵是有效的Random-init消融数据，但不能用于证明`3×48`规模或同规模蒸馏本身容易collapse。相关旧因果结论作废。

实现现已增加显式`--student-init teacher|random|auto`。Teacher-init要求teacher/student结构相同，并在训练开始时逐参数验证最大绝对差为0。

## 2. 最新完整结果

| 实验名称 | 参数量 | 训练类型 | Teacher/来源 | 初始化 | 协议 | 成功率 | 模式覆盖 | 模式熵 | JS |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| `FM-4x72-16-Full` | 274,826 | 完整训练 | 原始数据 | — | Standard-480 | 460/480，95.8% | 24/24 | 0.945 | — |
| `FM-4x72-1-Solver` | 274,826 | solver-only | `FM-4x72-16-Full` | 继承checkpoint | Standard-480 | 406/480，84.6% | 15/24 | 0.667 | — |
| `FM-4x72-1-Distill-4x72` | 274,826 | 同规模蒸馏 | `FM-4x72-16-Full` | Teacher | Standard-480 | 467/480，97.3% | 24/24 | 0.895 | — |
| `FM-4x72-1-Distill-4x72` | 274,826 | 同规模蒸馏 | `FM-4x72-16-Full` | Teacher | Paired-1000 | 970/1000，97.0% | 24/24 | 0.894 | 0.019 |
| `FM-3x48-16-Full` | 95,042 | 完整训练 | 原始数据 | — | Standard-480 | 445/480，92.7% | 24/24 | 0.965 | — |
| `FM-3x48-1-Solver` | 95,042 | solver-only | `FM-3x48-16-Full` | 继承checkpoint | Standard-480 | 394/480，82.1% | 14/24 | 0.569 | — |
| `FM-3x48-1-Distill-3x48` | 95,042 | 同规模蒸馏 | `FM-3x48-16-Full` | Teacher | Standard-480 | 442/480，92.1% | 23/24 | 0.868 | — |
| `FM-3x48-1-Distill-3x48` | 95,042 | 初始化消融 | `FM-3x48-16-Full` | Random，seed 42 | Standard-480 | 369/480，76.9% | 11/24 | 0.122 | — |
| `FM-3x48-1-Distill-3x48` | 95,042 | 初始化消融复现 | `FM-3x48-16-Full` | Random，seed 43 | Standard-480 | 365/480，76.0% | 10/24 | 0.129 | — |
| `FM-3x48-1-Distill-3x48` | 95,042 | 初始化消融 | `FM-3x48-16-Full` | Random，seed 42 | Paired-1000 | 776/1000，77.6% | 14/24 | 0.149 | 0.466 |
| `FM-2x36-16-Full` | 37,982 | 完整训练 | 原始数据 | — | Standard-480 | 319/480，66.5% | 24/24 | 0.942 | — |
| `FM-2x36-1-Solver` | 37,982 | solver-only | `FM-2x36-16-Full` | 继承checkpoint | Standard-480 | 268/480，55.8% | 16/24 | 0.700 | — |
| `FM-2x36-1-Distill-2x36` | 37,982 | 同规模蒸馏 | `FM-2x36-16-Full` | Teacher | Standard-480 | 287/480，59.8% | 23/24 | 0.885 | — |
| `FM-2x36-1-Distill-2x36` | 37,982 | 同规模蒸馏 | `FM-2x36-16-Full` | Teacher | Paired-1000 | 590/1000，59.0% | 24/24 | 0.887 | 0.044 |
| `FM-2x36-1-Distill-4x72` | 37,982 | 跨规模蒸馏 | `FM-4x72-16-Full` | Random | Standard-480 | 387/480，80.6% | 12/24 | 0.147 | — |
| `FM-2x36-1-Distill-4x72` | 37,982 | 跨规模蒸馏 | `FM-4x72-16-Full` | Random | Paired-1000 | 771/1000，77.1% | 14/24 | 0.141 | 0.515 |
| `FM-3x48-1-Distill-4x72` | 95,042 | 跨规模蒸馏 | `FM-4x72-16-Full` | Random | Standard-480 | 367/480，76.5% | 11/24 | 0.193 | — |
| `FM-4x64-1-Distill-4x72` | 217,666 | 跨规模蒸馏 | `FM-4x72-16-Full` | Random | Standard-480 | 357/480，74.4% | 14/24 | 0.212 | — |
| `FM-4x72-1-Distill-3x48` | 274,826 | 跨结构蒸馏 | `FM-3x48-16-Full` | Random | Standard-480 | 416/480，86.7% | 9/24 | 0.079 | — |
| `FM-2x36-16-PerturbFull` | 37,982 | 扰动完整训练 | 扰动状态＋原标签 | — | Standard-480 | 194/480，40.4% | 7/24 | 0.446 | — |
| `FM-2x36-1-PerturbFull` | 37,982 | solver-only | 同一PerturbFull checkpoint | 继承checkpoint | Standard-480 | 471/480，98.1% | 2/24 | 0.057 | — |

## 3. 纠正后的分析

### 3.1 模型规模与完整训练

三个Full模型都覆盖24种模式且熵为0.942–0.965。模型规模主要影响闭环成功率，不能据此认为小模型本身缺少多模态表达能力。

### 3.2 一步solver误差

三个结构从16步直接改为1步后都损失成功率和模式，说明低步数transport是独立瓶颈。

### 3.3 Teacher-init同规模蒸馏

Teacher-init后：

- `4×72`：97.3%，24模式，熵0.895；
- `3×48`：92.1%，23模式，熵0.868；
- `2×36`：59.8%，23模式，熵0.885。

因此，Flow Matching从16步蒸馏到1步本身不必然导致mode collapse；同结构teacher初始化可以恢复solver-only损失的大部分能力和多样性。

### 3.4 初始化是关键控制变量

`3×48`的Teacher-init与Random-init形成直接消融：成功率92.1%对76.9%，模式23对11，熵0.868对0.122。Random-init的退化在seed 43复现，但它证明的是初始化敏感性，而不是`3×48`规模效应。

### 3.5 跨规模结果的结论边界

当前跨规模/跨结构student均为Random-init。它们普遍保留一定成功率但模式熵很低；甚至`3×48 teacher→4×72 student`也只有9模式、熵0.079。因此严重collapse不能单独归因于student变小或参数压缩。

现阶段最准确的结论是：跨结构、Random-init和pointwise fidelity目标的组合与collapse强相关。若要研究模型规模的独立作用，需要为不同规模student构造公平warm start，例如从对应规模的Full模型初始化后再接受同一`4×72` teacher蒸馏。

### 3.6 成功率不能替代多样性

`FM-2x36-1-PerturbFull`成功率98.1%，但仅2模式、熵0.057。所有后续实验必须联合报告成功率、模式覆盖、模式熵；Paired实验再报告JS。

## 4. 作废或降级的旧结论

- 作废：“`3×48`同规模蒸馏天然比`2×36`和`4×72`更容易collapse。”
- 作废：“`3×48`结果排除了初始化因素。”
- 降级：“容量扫描证明模型压缩导致collapse。”当前扫描同时混入跨结构和Random-init。
- 保留：solver-only一步会损失模式。
- 保留：Random-init fidelity蒸馏可产生高成功率、低多样性的策略。
- 保留：成功率与多样性可以显著解耦。

## 5. 下一步必要对照

1. 为纠正后的`FM-3x48-1-Distill-3x48`补Paired-1000和JS；
2. 固定`FM-4x72-16-Full` teacher，用对应规模Full checkpoint作为跨规模student warm start；
3. 在相同初始化、更新预算和评估seed下扫描student规模；
4. 至少补两个训练seed，报告均值与波动。

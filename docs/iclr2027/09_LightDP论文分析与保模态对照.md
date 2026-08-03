# LightDP论文分析与保模态对照

更新时间：2026-08-02

## 1. 论文与代码状态

论文：Yiming Wu等，*On-Device Diffusion Transformer Policy for Efficient Robot Manipulation*
（LightDP，ICCV 2025，arXiv:2508.00697v1）。截至本次审计，论文项目页虽显示“Code”，但链接未指向
可用仓库；作者公开GitHub和机器人/VLA代码索引也未发现官方实现。因此本项目实现的是明确标注的
`LightDP-style`受控对照，而不是声称复现其官方代码。

## 2. LightDP实际做了什么

LightDP同时处理两种互补压缩：

1. **架构压缩**：为Transformer block学习Bernoulli二值门控，门控用Gumbel-Softmax训练；用各层
   Q/K/V与FFN权重的截断SVD重构误差初始化重要性，联合优化门控与网络后保留Top-N层，再去掉门控
   微调紧凑模型。
2. **步数蒸馏**：使用DDIM轨迹和一致性蒸馏，把约100次去噪降为4次。Student和EMA Target由
   Teacher初始化，训练目标对齐跨时间间隔的`x0`预测。

其DP-T基线为8层、8.97M参数、100 NFE、约90.6 ms；2层＋4 NFE版本约0.97 ms，论文报告约
93倍加速，同时CALVIN成功率从约0.772降至0.730。论文评估任务成功率和延迟，但没有显式测量
同一任务内部的mode coverage、mode entropy、per-mode SR或noise-to-mode basin保存情况。

## 3. 与本项目相似和不同之处

### 相似

- 都把每步模型计算量和求解步数分开压缩；规范术语分别是**架构压缩/跨架构蒸馏**与
  **步数蒸馏**。
- 都采用先压架构、再减步数的分阶段路线；两个阶段可独立验收，也可以组合计算加速。
- 都认为Transformer block存在冗余，并通过Teacher知识恢复压缩模型功能。

### 根本区别

| 维度 | LightDP | 本项目 |
|---|---|---|
| 核心问题 | 端侧速度和平均任务成功率 | 跨架构、跨步数压缩中的多模态破坏与恢复 |
| 数据假设 | pruning/CD期间访问原始demonstration | 主线要求不访问原始示范，仅Teacher rollout/查询 |
| 架构选择 | 静态SVD先验＋可学习门控 | 闭环可恢复性选择；当前最佳为keep013 |
| 步数方法 | DDPM上的DDIM consistency distillation | Flow Matching上的Endpoint/Flow-CTM等 |
| 主要指标 | SR、平均任务长度、latency | SR、coverage、per-mode SR，熵仅作辅助，外加latency |
| 迁移目标 | 具体Diffusion Transformer/CALVIN | FastWAM、DreamZero等跨架构WAM |

因此，LightDP证明“架构压缩＋步数蒸馏”组合是合理且有效的效率路线，但没有回答我们的核心问题：
**平均成功率尚可时，压缩是否已经合并了行为模式。**

## 4. 值得直接借鉴的内容

1. **SVD初始化而非随机门控**：减少门控搜索初期的不稳定，适合作为静态权重重要性基线。
2. **联合学习后再离散Top-N**：避免一次性删除层造成不可逆破坏。
3. **离散后继续repair**：必须评估真正部署的紧凑网络，而非软门控超网。
4. **架构压缩和步数蒸馏分别报告**：同时给出层数、参数量、NFE、latency和任务质量，适合纳入
   我们的最终Pareto表。
5. **端侧latency实测**：FLOPs和参数量不能代替真实延迟；本项目继续使用同硬件、batch=1实测。

不能直接照搬的部分：原论文依赖demonstration，其成功率指标也不能识别mode collapse；DDIM-CD目标
不能原样用于Flow Matching。后续必须保持数据访问和生成过程的差异可见。

## 5. 公平对照设计与审计

本地实现：`train_lightdp_style_depth_pruning.py`；入口：
`scripts/run_lightdp_style_depth_comparison_2gpu.sh`。

固定项：

- Teacher：`FM-4x72-16 Full`；
- Student：`FM-3x72-16`，最终只保留3/4层；
- 数据：相同成功Teacher rollout buffer，不含原始示范和专家动作；
- repair：相同velocity matching＋0.03 endpoint anchor；
- seed与Standard-120评估协议与keep013一致；
- 通过门槛为`SR>=88%`且`coverage>=22`，只有竞争性方案补Standard-480。

变量只有层选择：

- LightDP-style：SVD重构误差初始化，Gumbel-Sigmoid门控联合训练250 epochs，Top-3离散后相同
  repair 500 epochs；
- 当前强基线：四个3层子集分别repair，以闭环SR和mode coverage选择keep013。

这不是原论文的严格复现：将DDPM换成Flow、将demonstration换成Teacher rollout，是为了隔离“层选择
机制”并满足长期的demonstration-free约束。

## 6. 初始机制信号

smoke test的截断SVD分数为：

| Teacher层 | SVD重构误差 | 初始保留概率 |
|---:|---:|---:|
| 0 | 13.569 | 0.621 |
| 1 | 25.726 | 0.911 |
| 2 | 23.212 | 0.876 |
| 3 | 12.781 | 0.592 |

静态Top-3因此为`keep012`，而闭环可恢复性已选择`keep013`并取得Standard-480
`91.0% / 24 modes / H=0.929`。这个冲突支持一个可证伪机制假设：**权重低秩重构误差衡量的是
静态算子信息量，不一定衡量某层对noise-to-mode路由和闭环纠错的因果作用。**正式门控训练将检验
任务监督能否把排序从静态SVD先验纠正过来。

## 7. 验收与下一步

- 若门控最终选择keep013且达到同等闭环指标：LightDP式门控可作为昂贵穷举的可扩展近似，适合
  更深WAM；创新重点转向在门控目标中显式加入mode/闭环信号。
- 若仍选择keep012且coverage/SR显著较差：形成“静态重要性和平均fidelity不足以保护机器人策略
  多模态”的直接对照证据。
- 若选择其它mask但表现相当：需要多seed确认门控排序稳定性，不能据单seed归因。
- 架构对照通过后，给LightDP-style模型接与keep013完全相同的Endpoint步数蒸馏，并比较最终
  SR、coverage、per-mode SR与latency，避免把架构选择和步数目标混为一个变量。


# PEARL Phase 3：精确 PA 的跨投影残差与分割性能

## 范围、实现和协议

本阶段固定原始 CLIP ViT-B/16 checkpoint、Reduced 最后一层、Phase 2 的文本类别与 prompt，以及三个完整验证集：VOC21 和 VOC20 各 1,449 张，ADE20K 2,000 张。ADE20K 沿用 224 crop、112 stride 的 sliding-window 推理。所有新增 PA 关系均使用精确 SVD 正交 Procrustes；模型权重不更新。TLP off/on 分别报告，完整分割指标由逐图 confusion matrix 汇总，不取逐图 mIoU 平均。

令中心化矩阵为 $Q_c,K_c$，$R^*=UV^\top$，其中 $U\Sigma V^\top=K_c^\top Q_c$；$E=K_cR^*-Q_c$。下表和图表使用的四条路径严格区分：

| 简写 | 构造 | 参数含义 |
|---|---|---|
| `g` | $\operatorname{softmax}[(QQ^\top+\gamma QE^\top)/\sqrt d]$ | QQ→PA 的 logit 残差强度；$\gamma=1$ 为 Pure PA |
| `kb` | $\operatorname{softmax}[(KK^\top+\beta(Q-KR^*)(KR^*)^\top)/\sqrt d]$ | KK→PA 的 logit 插值 |
| `cb` | $(1-\beta)A_{\rm CSA}+\beta A_{\rm PA}$ | CSA mean→PA 的 Softmax 后 Attention 插值 |
| `cw` | $\lambda A_{QQ}+(1-\lambda)A_{KK}$ | 不使用 PA 的可调 CSA 对照 |

这里 $A_{\rm CSA}=(A_{QQ}+A_{KK})/2$，它与历史 SCLIP-style **sum** 有不同的行和。`kb` 的残差 $Q-KR^*$ 含未中心化均值差，不能等同于 $E$。$\gamma>1$ 是沿残差方向外插，不表示 Procrustes 对齐误差进一步降低。上述 Pure PA 路径均没有额外 $K_cK_c^\top$ 项；原始 PEARL 单独列为 NS-5 PA+KK。

固定 Q/K/V 数值测试覆盖三条路径的两个端点、SVD 正交性、旋转前后 KK Gram 等价、Softmax 行归一化、Value 聚合，以及可选 residual+KK 与既有 PA+KK 的等价性；实际 ADE sliding inference 的 SVD 缓存前后预测也逐像素相同。见 [测试代码](../experiments/pa_mechanism/phase3/)。默认推理策略仍保持原样。

### 参数选择与解释边界

[预先记录的规则](../results/pa_mechanism_phase3_selection_protocol.json)使用 seed 24003 对 VOC21/VOC20 同一批图像索引划分 435 张开发集与 1,014 张留出集，ADE20K 用 seed 24004 划分 600/1,400 张。统一 gamma 选择在三个开发集上相对固定 CSA mean 的 mIoU 差值之算术平均最大者；统一 CSA 权重以三个开发集的 mIoU 增益平均选择。候选值全部固定在规则文件中。

实际逐图记录核对确认 VOC21/VOC20 的 1,449 个索引对应完全相同且不重复的 image ID，故这两个划分确实保持了图像一致性。

开发集选出 **$\gamma=1.25$、$\lambda=0.75$**。因此这是**不更新权重、但使用有标签开发集选超参数**的设置。完整验证集曲线属于探索性证据，不能将全量峰值解释为独立泛化确认。初始 VOC21 全量探索曲线在划分冻结前已被查看；VOC21 留出集因此也不是对整个研究过程完全盲的独立测试。VOC20 与 VOC21 共享原始图像，只是标签协议不同，不能视作两个独立图像样本。

## 实验 A：TLP off，完整验证集

以下全部为完整验证集 pooled mIoU (%)，每行均为对应数据集的同一评估协议。`g100=kb100=cb100=Pure PA SVD`，其端点只展示一次。QQ、KK、CSA mean、$\gamma\le1.25$、Pure PA SVD 与原始 PEARL 复用配置一致的既有全量记录；其余行来自 Phase 3。完整的 mAcc、pAcc、开发集/留出集 mIoU 和来源文件见 [汇总 CSV](../results/pa_mechanism_phase3_summary.csv)。

| QQ→PA 残差 $\gamma$ | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| 0 (QQ) | 57.819 | 84.636 | 17.318 |
| .25 | 58.244 | 84.741 | 17.488 |
| .50 | 58.587 | **84.762** | 17.623 |
| .75 | 58.861 | 84.719 | 17.724 |
| 1 (Pure PA) | 59.079 | 84.624 | 17.796 |
| 1.25 (开发集统一选择) | 59.238 | 84.451 | 17.859 |
| 1.50 | 59.326 | 84.197 | **17.906** |
| 2.00 | **59.338** | 83.526 | 17.900 |
| 2.50 | 59.134 | 82.656 | 17.755 |
| 3.00 | 58.474 | 81.340 | 17.482 |

| KK→PA logit $\beta$ | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| 0 (KK) | 58.908 | 80.296 | 17.413 |
| .25 | 59.452 | 81.619 | 17.633 |
| .50 | **59.700** | 82.796 | 17.784 |
| .75 | 59.642 | 83.798 | **17.852** |
| 1 (Pure PA) | 59.079 | **84.624** | 17.796 |

| CSA mean→PA Attention $\beta$ | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| 0 (CSA mean) | 59.299 | 83.479 | 17.634 |
| .25 | **59.347** | 83.874 | 17.713 |
| .50 | 59.322 | 84.198 | 17.770 |
| .75 | 59.227 | 84.448 | **17.801** |
| 1 (Pure PA) | 59.079 | **84.624** | 17.796 |

| 可调 CSA 的 QQ 权重 $\lambda$ | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| 0 (KK) | 58.908 | 80.296 | 17.413 |
| .25 | **59.365** | 82.165 | 17.592 |
| .50 (CSA mean) | 59.299 | 83.479 | **17.634** |
| .75 (开发集统一选择) | 58.763 | 84.322 | 17.541 |
| 1 (QQ) | 57.819 | **84.636** | 17.318 |

Pure PA 的跨投影关系相对 QQ 在 VOC21、ADE20K 随 gamma 增强到一定程度后改善，但 VOC20 在 .5 后持续退化。开发集选定的统一 $\gamma=1.25$ 相对 QQ 在 VOC21/ADE20K 提升，在 VOC20 低 0.185 点；它没有在三个数据集统一超过固定 CSA mean：VOC21 低 0.061，VOC20 高 0.972，ADE20K 高 0.225 mIoU 点。**探索性的统一 $\gamma=1.5$ 在三个完整验证集的点估计都高于固定 CSA mean**，差值为 VOC21 +0.026、VOC20 +0.718、ADE20K +0.272 点；其中 VOC21 的成对区间 [−0.171,+0.249] 跨零，不能称为稳健的三数据集改善。$\gamma=2$ 的点估计也在三套数据集均略高于 CSA，但 VOC20 仅 +0.047 点，且属于同一探索性扫描。

KK→PA 的中间 logit 插值在 VOC21 达到 59.700，高于 KK 0.792、Pure PA 0.622、固定 CSA mean 0.401 点；ADE20K 的 $\beta=.75$ 也略高于 Pure PA 0.056 点。VOC20 则随 $\beta$ 增大而提高，终点 Pure PA 最好。CSA→PA 的 Attention 混合在 VOC20 不优于 Pure PA，在 VOC21 与 ADE20K 超过两端的幅度仅 0.048 与 0.005 点；因此没有把它当作稳健互补性证据扩展到 TLP 全量矩阵。可调 CSA 在 VOC20 用较高 QQ 权重即可超过固定 CSA；统一 $\lambda=.75$ 在 VOC21、ADE20K 反而退化。

## 实验 B：TLP on，完整验证集

下表全部是 TLP on 的完整验证集 pooled mIoU (%)。`g125` 是 TLP-off 开发集选定的统一 gamma；`g150` 是额外统一检验点，其解释仍是探索性的。TLP 原实现、文本标签和滑窗设置未改。历史行复用 Phase 2 完整记录。

| Attention + TLP | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| QQ | 61.701 | 86.403 | 18.455 |
| KK | 64.029 | 83.878 | 18.995 |
| CSA mean | 63.988 | 85.970 | 18.994 |
| weighted CSA $\lambda=.75$ | 63.112 | 86.422 | 18.770 |
| Pure PA SVD $\gamma=1$ | 63.660 | 86.634 | 18.973 |
| Pure PA SVD $\gamma=1.25$ | 63.986 | 86.551 | 19.053 |
| Pure PA SVD $\gamma=1.5$，探索 | **64.232** | 86.418 | 19.092 |
| SVD PA+KK，原始 alpha | 64.162 | 86.729 | 19.202 |
| 原始 PEARL，NS-5 PA+KK | 64.066 | **86.905** | **19.363** |

Pure PA $\gamma=1$ + TLP 的缺失基线现已补齐：63.660/86.634/18.973。相对 TLP-off 的 59.079/84.624/17.796，TLP 自身贡献 +4.581/+2.010/+1.178 点，不能把完整推理增益都归给 PA 残差。把 gamma 从 1 改成开发集选定的 1.25 后，TLP-on 差值为 +0.326/−0.083/+0.080 点；VOC20 仍是明确的反例。

统一 $\gamma=1.25$ 在 TLP-on 的 VOC20 超过固定 CSA 0.581 点、在 ADE20K 高 0.059 点、VOC21 几乎持平（−0.002 点）。它没有在三个数据集上都严格超过固定 CSA，也没有超过原始 PEARL。额外探索的统一 $\gamma=1.5$ 在 TLP-on 的三套完整数据集都数值高于固定 CSA，分别 +0.244/+0.447/+0.098 点；成对区间在 VOC21 [−0.002,+0.543] 与 ADE20K [−0.033,+0.205] 跨零，因此不是稳健的三数据集确认。它在 VOC21 达到 64.232，超过原始 PEARL 0.166 点；但 VOC20 低 0.487 点、ADE20K 低 0.271 点，故并非跨数据集统一胜出的 Pure PA 配置。

公平 CSA 对照同样改变了结论：$\lambda=.75$ 相对固定 1:1 CSA 在 VOC20 +0.452 点，但 VOC21 −0.876、ADE20K −0.224 点。Pure PA $\gamma=1.25$ 对这一统一权重 CSA 在三个数据集分别 +0.874/+0.129/+0.284 点；其中 VOC20 的小差值需要结合区间看待。可调权重不是 PA 的廉价等价替代，也没有三数据集一致的最优权重。

### 成对 image bootstrap 与模型选择

[成对区间表](../results/pa_mechanism_phase3_bootstrap.csv)对相同图像逐图重采样 1,000 次，每次先汇总 TP/GT/pred，再计算数据集 mIoU 差值；分别覆盖完整集和未参与参数选择的留出集。区间是这批验证图像条件下的 percentile 95% 区间，不是独立数据集重复，也未对探索中多重比较作校正。ADE20K 每次仅纳入 pooled GT-or-pred union 大于零的类别，稀有类抽样会影响区间。

| TLP-on 比较，后者减前者 | VOC21 全量 Δ [95% CI] | VOC20 全量 Δ [95% CI] | ADE20K 全量 Δ [95% CI] |
|---|---:|---:|---:|
| $\gamma=1$ → 选定 $\gamma=1.25$ | +0.326 [+0.254,+0.397] | −0.083 [−0.155,−0.015] | +0.080 [+0.046,+0.110] |
| CSA mean → 选定 $\gamma=1.25$ | −0.002 [−0.242,+0.276] | +0.581 [+0.376,+0.793] | +0.059 [−0.064,+0.157] |
| weighted CSA $\lambda=.75$ → 选定 $\gamma=1.25$ | +0.874 [+0.645,+1.123] | +0.129 [−0.063,+0.331] | +0.284 [+0.159,+0.387] |
| 原始 PEARL → 选定 $\gamma=1.25$ | −0.081 [−0.215,+0.058] | −0.353 [−0.513,−0.207] | −0.309 [−0.510,−0.150] |
| 原始 PEARL → 探索 $\gamma=1.5$ | +0.166 [−0.009,+0.345] | −0.487 [−0.681,−0.295] | −0.271 [−0.478,−0.111] |

留出集上，TLP-on $\gamma=1.5$ 相对原始 PEARL 的差值是 VOC21 +0.247 [+0.015,+0.451]、VOC20 −0.420 [−0.686,−0.172]、ADE20K −0.256 [−0.522,−0.101]。VOC21 的初始全量曲线在划分前已被看过，且 $\gamma=1.5$ 的 TLP 测试属于后续探索，故不能把这个单一留出集的正区间表述为严格的独立确认。$\gamma=1.25$ 相对固定 CSA 的 TLP-on 留出集差值分别为 +0.084 [−0.230,+0.430]、+0.546 [+0.278,+0.839]、+0.026 [−0.124,+0.145]；只有 VOC20 的区间明确为正。

## 实验 C：条件触发的 Residual-Enhanced PA+KK

Pure PA 残差调节在 VOC21/ADE20K 有收益，但 TLP-on 的统一开发集选择仍低于 VOC20/ADE20K 的原始 PEARL，因此按[条件扩展规则](../results/pa_mechanism_phase3_kk_extension_protocol.json)固定 $\gamma=1.25$，只检查少量 $\alpha K_cK_c^\top$。所有 $\alpha>0$ 的行均称为 **Residual-Enhanced PA+KK**，绝不是 Pure PA。这个扩展在核心全量结果已被查看后才规划，属于探索性消融。

| $\gamma=1.25$，TLP off | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| $\alpha=0$，Pure PA | **59.238** | **84.451** | 17.859 |
| $\alpha=.03$ | 59.213 | 84.415 | **17.876** |
| $\alpha=.08$ | 59.039 | 84.243 | 17.870 |
| $\alpha=.15$ | 58.631 | 83.818 | 17.788 |
| $\alpha=.30$，ADE 原始系数 | — | — | 17.497 |

VOC21/VOC20 的所有新增 alpha 均低于对应 Pure PA；ADE20K 的 alpha=.03/.08 相对 Pure PA 只高 0.017/0.011 点。ADE20K 开发集上 alpha=.08 的差值 +0.037 点，留出集仅 +0.010 点。依照条件规则及[选择记录](../results/pa_mechanism_phase3_kk_extension_selection.json)，只有 ADE20K 的 alpha=.08 进入 TLP 完整验证；VOC 的 alpha=.03 虽在 VOC21 开发集出现 +0.0005 点，但留出集为 −0.032 点，不支持把它作为有效收益继续追逐。

| ADE20K 完整验证，$\gamma=1.25$ | TLP off | TLP on | TLP 增益 |
|---|---:|---:|---:|
| Pure PA，$\alpha=0$ | 17.859 | 19.053 | +1.194 |
| Residual-Enhanced PA+KK，$\alpha=.08$ | 17.870 | **19.251** | +1.382 |
| 差值：附加 KK 减 Pure PA | +0.011 | **+0.198** | +0.187 |

[扩展实验成对区间](../results/pa_mechanism_phase3_kk_extension_bootstrap.csv)表明 ADE20K 的 alpha=.08 在 TLP-off 相对 Pure PA 为 +0.011 [−0.041,+0.091] 点，而 TLP-on 为 +0.198 [+0.131,+0.299]；对应留出集为 +0.010 [−0.049,+0.104] 与 +0.191 [+0.112,+0.322]。这支持 **KK 与 TLP 的配置交互**，不能说小 alpha 已在无传播的 Attention 层面产生稳健性能改善。与原始 PEARL + TLP 比，该扩展仍低 0.111 [−0.239,−0.008] 点；留出集差值 −0.097 [−0.255,+0.003]。比较同时涉及 NS-5/SVD solver 与不同 gamma，无法单独归因这些差距。

## 类别变化与研究结论

[逐类别 IoU 表](../results/pa_mechanism_phase3_class_deltas.csv)保留每个类别的 GT 像素支持和两种方法的 pooled IoU。与原始 PEARL + TLP 相比，探索点 $\gamma=1.5$ 在 VOC21 的 21 类中 14 类改善、7 类下降；VOC20 的 20 类中仅 4 类改善、16 类下降；ADE20K 的 150 类中 54 类改善、90 类下降、6 类持平。单类变化是描述性证据，稀有类较大的差值不应被解读为跨类别普遍收益。

**Q1：残差是否超越 QQ？** 精确 SVD 下，VOC21/ADE20K 随 gamma 从 0 增至 1.5 左右确有收益，VOC21 到 2.0 仍微升；VOC20 在 gamma=.5 后持续回落，外插至 3 时明显受损。正交旋转 $R^*$ 始终相同，改变的是 Attention 对 $QE^\top$ 的权重。

**Q2：Pure PA 是否优于 KK、CSA？** KK→PA 的中间点在 VOC21 大幅优于两端，说明跨空间关系与 K 自相关的组合可以有效，但中间点仍含 KK logit，不能归入 Pure PA。CSA→PA 的微小中间峰在 ADE20K 仅 +0.005 点，不支持强互补性结论。开发集选定的统一 Pure PA $\gamma=1.25$ 的 TLP-off 留出集相对固定 CSA 在 VOC21 −0.054、VOC20 +1.003、ADE20K +0.172 点；探索性的 $\gamma=1.5$ 则在三套完整集上都有正的点估计，但 VOC21 的提升仅 +0.026 点且区间跨零。可调 CSA 权重在 VOC20 弥补部分差距，但同一权重损害另两套数据。

**Q3：残差增强能否改善最终分割？** $\gamma=1.25$ + TLP 相比 $\gamma=1$ 在 VOC21/ADE20K 有正差值，在 VOC20 为负。额外 $\gamma=1.5$ 在 VOC21 全量超过原始 PEARL 0.166 点，但全量成对区间跨零；VOC20、ADE20K 则低于原始 PEARL 0.487/0.271 点。TLP 带来的独立增益已在上节量化，不应混入 PA 残差解释。

**Q4：额外 KK 是否必要？** 原始 PEARL 仍是 VOC20/ADE20K 的较强完整推理配置，但它同时使用 NS-5、KK 与 TLP，不能把相对优势单独归因于 KK。固定精确 SVD 与 $\gamma=1.25$ 后，新增 KK 在两个 VOC 数据集未改善 TLP-off；ADE20K 的 alpha=.08 在 TLP-off 不具有明确增益，却在 TLP-on 提升约 0.20 点，并仍略低于原始 PEARL。这说明额外 KK 在 ADE20K 的这组完整管线上有价值，但没有证据表明 Pure PA 残差方法在各数据集都必须加入它。该正结果属于 **Residual-Enhanced PA+KK**，不能记在 Pure PA 名下。

## 图表、原始记录和复现

- [Figure 1：三条不同插值路径](../results/figures/pa_mechanism_phase3/three_paths.png)；[SVG](../results/figures/pa_mechanism_phase3/three_paths.svg)。图中的 gamma、两种 beta 是不同机制，不应沿同一横轴比较。
- [Figure 2：残差与 TLP](../results/figures/pa_mechanism_phase3/residual_tlp.png)；[SVG](../results/figures/pa_mechanism_phase3/residual_tlp.svg)。同时呈现选定 gamma、额外探索 gamma、CSA、weighted CSA 和原始 PEARL，纵轴是相对固定 CSA 的 mIoU 点数。
- [完整汇总](../results/pa_mechanism_phase3_summary.csv)、[差值表](../results/pa_mechanism_phase3_comparisons.csv)、[模型选择](../results/pa_mechanism_phase3_selection.json)、[运行 manifest](../results/pa_mechanism_phase3_manifest.json) 和[可复现命令](../experiments/pa_mechanism/phase3/README.md)。

逐图 confusion JSONL 和配置 sidecar 位于 `/data/Multimodal-final-project/outputs/pa_mechanism/phase3/runs/`，大文件不进入 Git。评估脚本按图像和 variant 断点续跑，聚合脚本拒绝缺少完整图像的配置；manifest 记录真实图像数、checkpoint 哈希和基础代码 commit。新增策略仅按显式参数启用，旧阶段结果不被覆盖。

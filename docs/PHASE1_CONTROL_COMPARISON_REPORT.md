# Phase1 Morlet 与 matched-control 对照实验分析

归档日期：2026-09-03。当前版本按 Phase1 frozen release 处理；本次仅提取已有结果并进行描述性统计。

## 实验目的

比较固定 Morlet 光学前端与结构匹配控制前端在 CAM16 上的最佳验证集 slide AUROC，判断固定 Morlet 前端在本次 Phase1 对照中是否产生性能优势。

## 实验变量设计

比较变量为固定前端的滤波器构造：

| frontend_variant | 前端 | 已有结果目录 |
| --- | --- | --- |
| morlet | 固定 Morlet 前端 | `artifacts/formal_runs/phase1-cam16-baseline-b32-v2` |
| matched_control | 冻结的包络匹配随机相位控制前端 | `artifacts/formal_runs/phase1-cam16-matched-control-b32-v1` |

matched-control 结果记录的生成器为 `frozen-envelope-matched-random-phase-v1`，随机数生成器为 `PCG64DXSM`，控制前端 seed 为 `20260901`；该 seed 与训练 seed 区分。结果来源 commit 由任务指定为 `fed03b18a1cd9a0d84a06986a78259133df56b3e`，本次未另行执行哈希审计。

两组均使用冻结前端。该比较针对上述两种前端构造，不将结构匹配解释为完整的频谱因素隔离。

## 固定条件

根据两组 seed-3407 的 `training_summary.json` 内嵌 `effective_config`，`data`、`determinism`、`evaluation`、`training` 内容一致；`model` 除前端合同标识及 matched-control 显式增加的 `frontend_variant` 外，其余记录字段一致。

| 条件 | 两组共同设置 |
| --- | --- |
| 数据与划分 | CAM16；已有 train/val 划分；四个最佳 epoch 结果记录的 train/val 划分标识一致 |
| 验证集规模 | 42 张 slide（22 阳性、20 阴性），18,171 个 patch |
| 训练 seed | 1729、3407 |
| 输入与染色 | 256 × 256 RGB；相同固定 H/E stain basis |
| 前端结构设置 | 4 个尺度、8 个方向、105 支持尺寸；FFT 执行 |
| 下游设置 | 金字塔层级 `[1, 2, 4]`，分类器 `linear-logit-v1` |
| 精度与确定性 | `protected-float32-v1`；确定性算法开启；AMP、TF32、float16、bfloat16 关闭 |
| 训练设置 | batch size 32；AdamW；学习率 0.001；weight decay 0.0001；BCE-with-logits；无 scheduler |
| 训练上限与早停 | 最多 20 epochs；patience 5；min_delta 0 |
| checkpoint 选择 | 最佳 validation slide AUROC；相同值选择最早 epoch |
| 评估 | `cam16-eval-v1`；slide 聚合为 `manifest-bounded-max-logit` |

上述核对仅使用已有结果 JSON 与其中保存的配置和标识，未打开数据集或配置文件，未重新计算数据或 checkpoint 哈希。四个完成记录均为 `status=complete`、`test_split_accessed=false`；患者级隔离仍为 `not_evaluated`，`patient_level_claim_allowed=false`。

## 结果表

每个值直接来自对应 `seed-<seed>/completion.json`，并与该 seed 最佳 epoch 的结果 JSON 对照一致。`best_epoch` 保留结果文件的从 0 开始的编号。

| frontend_variant | seed | best_epoch | best_validation_slide_auroc |
| --- | --- | --- | --- |
| morlet | 1729 | 9 | 0.9113636363636364 |
| morlet | 3407 | 1 | 0.865909090909091 |
| matched_control | 1729 | 7 | 0.8295454545454546 |
| matched_control | 3407 | 3 | 0.8159090909090909 |

matched-control 的 seed-1729 `training_summary.json` 是该次调用的历史快照，其中 seed-3407 仍标为 pending。本报告使用各 seed 的最终 `completion.json`，不以该历史快照判定另一 seed 的最终状态。

## 统计结果

统计单位为训练 seed，每组 n = 2。对每个 seed 的最佳验证集 slide AUROC 等权计算均值与样本标准差：

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}.
\]

Difference 均按 Morlet 减 matched-control 计算；Sample SD 行的差值仅描述跨 seed 离散程度差异。

| Metric | Morlet | matched-control | Difference |
| --- | --- | --- | --- |
| Mean AUROC | 0.8886363636363637 | 0.8227272727272728 | +0.0659090909090909 |
| Sample SD | 0.0321412173266612 | 0.0096423651979984 | +0.0224988521286628 |

**ΔAUROC = Morlet mean AUROC − matched-control mean AUROC = +0.0659090909090909**，即约 **6.59 个百分点**。

统计脚本使用 Python 标准库 `statistics.mean` 和 `statistics.stdev`，在原始精度上计算，汇总表显示 16 位小数。Morlet 汇总与已有 `validation_summary.json` 的均值和样本标准差在该显示精度下一致。

在仓库根目录复现：

```powershell
python scripts/phase1_control_comparison.py --self-test
python scripts/phase1_control_comparison.py
```

自检验证均值、n−1 样本标准差和相等值情形；两条命令均已通过。分析脚本只读取四个完成记录并输出 Markdown，不加载训练模块，不读写 checkpoint，不写入训练产物。

按本任务仅分析既有结果的范围，未运行项目 smoke test 或训练相关测试；训练、模型与配置文件均未改动，锁定规范未受影响。分析以用户已确认的实验完成与冻结状态及已有结果记录为依据，不重新审计训练执行过程；无阻碍本次统计与归档的未解决问题。

## Phase1 结论

在本次冻结的 Phase1 CAM16 验证集对照中，固定 Morlet 前端在 seed-1729 和 seed-3407 上均获得更高的最佳 slide AUROC，均值相较结构匹配控制前端提高 0.0659090909。因此，本次两个 seed 的结果支持固定 Morlet 前端具有观察到的验证集性能优势。

该结论限于这两个 seed 的最佳验证集指标；每组 n = 2 的均值与样本标准差属于描述性统计，不构成统计显著性或独立 test 性能优势的证明。

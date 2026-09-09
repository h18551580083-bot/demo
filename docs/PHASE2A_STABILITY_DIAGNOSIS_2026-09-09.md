# Phase2-A Morlet stability diagnosis — 2026-09-09

判定：NOT_READY_FOR_5SEED_RERUN。候选已实现；两 seed 实测状态见文末。

本报告只使用指定 baseline 的 5 个 completion、5 个 training_summary、45 个 epoch JSON，以及直接相关代码和 train/val 清单。未读取 test 清单或图片，未读取 checkpoint，未重算全量 manifest/hash，未做全仓库审计。

前提修正：存在 5 个标记 complete 的结果；本次恢复核验发现最高/最低 seed 的 epoch 与 summary 已齐全，旧报告的缺失结论已失效。当前代码及 baseline TOML 是五 seed，而 TRAINING_PROTOCOL 与此前 Decisions 尚未记录相应五 seed 扩展。本报告保留并描述所有结果，不替这项文档缺口补签正式批准。

## 完成记录与统计

epoch 为代码的零基索引；例如 best_epoch=1 是第 2 轮。停止轮数来自 epochs_completed。

| seed | best AUROC | best epoch（零基） | 最佳轮次 | 已完成轮数 | 最后 epoch（零基） | epoch 日志 |
|---|---:|---:|---:|---:|---:|---:|
| 1729 | 0.9113636364 | 9 | 10 | 15 | 14 | 15 |
| 3407 | 0.8659090909 | 1 | 2 | 7 | 6 | 7 |
| 5113 | 0.8977272727 | 3 | 4 | 9 | 8 | 9 |
| 7717 | 0.8863636364 | 1 | 2 | 7 | 6 | 7 |
| 9109 | 0.8795454545 | 1 | 2 | 7 | 6 | 7 |

mean=0.8881818182；sample SD（n−1）=0.0173383919；min=0.8659090909；max=0.9113636364；range=0.0454545455；median=0.8863636364。

汇总以每个 seed 的 completion 为单位，未重复计算 summary 内累计 runs。5113/7717 的 summary 是当时的累计快照，不能将其中 pending 当作当前未完成。

## 完整可用曲线

validation loss、逐 batch loss、梯度范数、参数更新范数和逐 epoch LR 均未记录。LR=0.001 是五个 effective_config 与无 scheduler 代码支持的常量，不是独立的 LR 实测日志。训练 loss 是 batch 均值的算术平均，最后不满 batch 与满 batch 等权。下面保留可用的全部 epoch；各阈值指标、校准指标和 CI 的完整数值仍在原始 epoch JSON。

### seed 1729

| epoch（零基） | train loss | patch AUROC | slide AUROC | patch BA | slide BA |
|---|---:|---:|---:|---:|---:|
| 0 | 0.514306 | 0.917020 | 0.809091 | 0.839149 | 0.806818 |
| 1 | 0.486441 | 0.884354 | 0.702273 | 0.814346 | 0.690909 |
| 2 | 0.478448 | 0.925312 | 0.804545 | 0.851429 | 0.761364 |
| 3 | 0.468017 | 0.913757 | 0.740909 | 0.848652 | 0.727273 |
| 4 | 0.462894 | 0.934368 | 0.850000 | 0.861990 | 0.838636 |
| 5 | 0.451261 | 0.937277 | 0.856818 | 0.864688 | 0.840909 |
| 6 | 0.455474 | 0.925762 | 0.800000 | 0.858616 | 0.786364 |
| 7 | 0.451408 | 0.915869 | 0.831818 | 0.834070 | 0.813636 |
| 8 | 0.443730 | 0.938666 | 0.852273 | 0.862695 | 0.811364 |
| 9 | 0.439269 | 0.939040 | 0.911364 | 0.866164 | 0.884091 |
| 10 | 0.435090 | 0.939118 | 0.881818 | 0.872786 | 0.861364 |
| 11 | 0.437058 | 0.928410 | 0.784091 | 0.865487 | 0.804545 |
| 12 | 0.436508 | 0.939907 | 0.831818 | 0.871161 | 0.829545 |
| 13 | 0.432446 | 0.944093 | 0.895455 | 0.874745 | 0.884091 |
| 14 | 0.423034 | 0.933256 | 0.863636 | 0.854785 | 0.827273 |

最佳点前/当轮/后：0.852273 / 0.911364 / 0.881818；全曲线 range=0.209091；最大相邻绝对变化=0.109091。最佳点后五轮均未超越最佳值，停止逻辑符合 patience=5。

### seed 3407

| epoch（零基） | train loss | patch AUROC | slide AUROC | patch BA | slide BA |
|---|---:|---:|---:|---:|---:|
| 0 | 0.523576 | 0.911188 | 0.713636 | 0.844887 | 0.718182 |
| 1 | 0.485285 | 0.926854 | 0.865909 | 0.856476 | 0.811364 |
| 2 | 0.475024 | 0.918591 | 0.763636 | 0.853464 | 0.777273 |
| 3 | 0.467868 | 0.926523 | 0.795455 | 0.851653 | 0.763636 |
| 4 | 0.469646 | 0.927300 | 0.811364 | 0.861246 | 0.770455 |
| 5 | 0.460641 | 0.924640 | 0.779545 | 0.859316 | 0.759091 |
| 6 | 0.454778 | 0.909908 | 0.727273 | 0.841066 | 0.713636 |

最佳点前/当轮/后：0.713636 / 0.865909 / 0.763636；全曲线 range=0.152273；最大相邻绝对变化=0.152273。最佳点后五轮均未超越最佳值，停止逻辑符合 patience=5。

### seed 5113

| epoch（零基） | train loss | patch AUROC | slide AUROC | patch BA | slide BA | slide AUROC 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.513939 | 0.924441 | 0.879545 | 0.855845 | 0.818182 | [0.761364, 0.970455] |
| 1 | 0.487477 | 0.929255 | 0.836364 | 0.854837 | 0.811364 | [0.693182, 0.950000] |
| 2 | 0.474216 | 0.929914 | 0.795455 | 0.863639 | 0.781818 | [0.647727, 0.918182] |
| 3 | 0.471030 | 0.934822 | 0.897727 | 0.862529 | 0.838636 | [0.786364, 0.979545] |
| 4 | 0.460518 | 0.920582 | 0.777273 | 0.854859 | 0.784091 | [0.627273, 0.909091] |
| 5 | 0.449735 | 0.932319 | 0.838636 | 0.858257 | 0.788636 | [0.709091, 0.945455] |
| 6 | 0.450178 | 0.931436 | 0.820455 | 0.867751 | 0.788636 | [0.681818, 0.940909] |
| 7 | 0.454060 | 0.935754 | 0.809091 | 0.869927 | 0.831818 | [0.665909, 0.938636] |
| 8 | 0.443204 | 0.934050 | 0.765909 | 0.869250 | 0.806818 | [0.606818, 0.909091] |

最佳点前/当轮/后：0.795455 / 0.897727 / 0.777273；全曲线 range=0.131818；最大相邻绝对变化=0.120455。
可用轨迹复核：最佳 epoch 后连续 5 轮无严格改善，在 epoch 8 停止，与 patience=5 一致。

### seed 7717

| epoch（零基） | train loss | patch AUROC | slide AUROC | patch BA | slide BA | slide AUROC 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.514888 | 0.921020 | 0.836364 | 0.852698 | 0.784091 | [0.697727, 0.947727] |
| 1 | 0.490000 | 0.924996 | 0.886364 | 0.854870 | 0.886364 | [0.756818, 0.984091] |
| 2 | 0.469362 | 0.931427 | 0.845455 | 0.860920 | 0.831818 | [0.711364, 0.952273] |
| 3 | 0.466490 | 0.934214 | 0.856818 | 0.867449 | 0.856818 | [0.715909, 0.968182] |
| 4 | 0.460043 | 0.919479 | 0.802273 | 0.853694 | 0.784091 | [0.652273, 0.927273] |
| 5 | 0.462927 | 0.921704 | 0.788636 | 0.856950 | 0.786364 | [0.640909, 0.915909] |
| 6 | 0.444111 | 0.930457 | 0.736364 | 0.859255 | 0.763636 | [0.565909, 0.884091] |

最佳点前/当轮/后：0.836364 / 0.886364 / 0.845455；全曲线 range=0.150000；最大相邻绝对变化=0.054545。
可用轨迹复核：最佳 epoch 后连续 5 轮无严格改善，在 epoch 6 停止，与 patience=5 一致。

### seed 9109

| epoch（零基） | train loss | patch AUROC | slide AUROC | patch BA | slide BA | slide AUROC 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.507023 | 0.913004 | 0.736364 | 0.849035 | 0.711364 | [0.575000, 0.881818] |
| 1 | 0.500634 | 0.927267 | 0.879545 | 0.853042 | 0.881818 | [0.745455, 0.990909] |
| 2 | 0.473523 | 0.927941 | 0.863636 | 0.863416 | 0.809091 | [0.738636, 0.965909] |
| 3 | 0.469435 | 0.936576 | 0.822727 | 0.868633 | 0.831818 | [0.672727, 0.947727] |
| 4 | 0.462693 | 0.932756 | 0.815909 | 0.861689 | 0.784091 | [0.679545, 0.938636] |
| 5 | 0.460924 | 0.929677 | 0.809091 | 0.863534 | 0.829545 | [0.654545, 0.945455] |
| 6 | 0.460984 | 0.928394 | 0.765909 | 0.864332 | 0.802273 | [0.595455, 0.911364] |

最佳点前/当轮/后：0.736364 / 0.879545 / 0.863636；全曲线 range=0.143182；最大相邻绝对变化=0.143182。
可用轨迹复核：最佳 epoch 后连续 5 轮无严格改善，在 epoch 6 停止，与 patience=5 一致。

## 原因判断（证据与假设分开）

1. B：seed 改变的主要训练因素是 batch 顺序。当前模型所有 9473 个可训练标量均初始化为零，五 seed 构造检查已通过；因此 A“不同随机初始化”不适用于当前实现。nn.Linear 构造时消耗随机数，但参数随后显式清零。没有随机增广或 dropout。现有日志不能独立证明历史环境与当前代码完全一致。
2. E / slide aggregation：验证仅 42 slides（22 阳性、20 阴性），440 对正负配对。无 ties 时 AUROC 步长为 1/440=0.0022727273，有 ties 时可出现半步。五 seed 极差相当于 20 个净 winning pairs，但没有 high/low 逐 slide logits，不能识别具体哪张 slide 改变。最大值聚合只取一张 slide 中最大的 patch logit；patch AUROC 改善并不保证这些最大值排名改善。5113 在 epoch 3→4 的 patch AUROC 0.934822→0.920582，而 slide AUROC 0.897727→0.777273。其后 patch 恢复到约 0.934，slide 仍下降。该现象支持聚合敏感性，不单独证明 LR 过大。
3. C/G/D：存在优化路径与 slide 指标波动，但不是已证实的梯度爆炸或 early stopping 过早。五个有日志 seed 的 loss 总体下降，slide 后期退化；最高 seed 到第 10 轮才达峰，低 seed 第 7 轮停止；低 seed 最后三轮 AUROC 为 0.811364 → 0.779545 → 0.727273，没有停止时持续改善的直接证据。这使 early stopping 成为可检验假设，却没有低 seed 后续反弹的证据。暂不增大 patience，不加 clipping，不调整 weight decay。

同一固定验证集本身不会让完全相同的预测产生不同 AUROC；小样本使预测排名变化更显著，并使泛化估计不确定。不能将 bootstrap CI、seed SD 和验证集抽样方差混为一谈。已有 CI 较宽不是“seed 差异无效”的统计检验。

## 随机性与优化实现

- training.py::configure_determinism 覆盖 Python、NumPy、torch CPU 和所有 CUDA；严格 deterministic_algorithms、CUBLAS_WORKSPACE_CONFIG=:4096:8、cuDNN deterministic 均已有，不新增性能开关。
- data.py::_WorkerSeeder 对每个 worker 设置 Python/NumPy/torch seed；DataLoader 使用独立 CPU generator。SHA-256(seed, epoch, patch_id) 决定全局排列，drop_last=False；没有额外 shuffle、weighted/class-balanced/slide-balanced sampler。
- seed 也驱动验证 bootstrap CI，但不通过 CI 决定 checkpoint。验证行排序与 raw-logit 最大值聚合固定；未发现当前相关路径中明显未受控的随机算子。未做同 seed CUDA 完整重复训练，不能据静态检查保证跨环境 bitwise 一致。
- AdamW，LR 0.001，betas 0.9/0.999，epsilon 1e-8，weight decay 1e-4，batch 32，float32 BCEWithLogits；无 scheduler/warmup/clipping，最大 20 轮。训练逐步检查 loss 和梯度有限性，但未记录其范数，无法判断有限值范围内的尖峰。

## 数据与采样

只读取两份 split-specific CSV。train：79,570 patches（41,121 阴性、38,449 阳性），171 slides（82 阴性、89 阳性）；val：18,171 patches（9,942 阴性、8,229 阳性），42 slides。按清单 slide_id 的 train/val 交集为零；不作 patient 隔离声明。
每 slide patch 数 train min/median/max=2/503/1024；val=6/505/1024。train 单 slide 最大占比 1.287%，val 最大占比 5.635%。不均衡确实存在，但所有 seed 每轮均访问全部行一次，因此整轮类别/slide/hard-easy 样本总量不随 seed 改变；仅 batch 分组及提前停止导致的总轮数不同。hard/easy 没有可靠定义，未按元数据臆断。

使用生产 hash_epoch_order 重放 epoch 0 的全部 train ID，不读取图片：

| seed | batch 阳性比例 sample SD | batch 内同 slide 最大 patch 数 | batch 内最多同 slide patch 数的均值 | 纯单类别 batch |
|---|---:|---:|---:|---:|
|1729|0.091656|5|2.239646|0|
|3407|0.088430|5|2.254524|0|
|5113|0.088786|5|2.253317|0|
|7717|0.091170|5|2.249698|0|
|9109|0.086746|5|2.250905|0|

这不支持“某个 seed 被 sampler 大量重复/丢失样本”。只重放首轮，不宣称覆盖后续全部 batch。暂不引入 slide-balanced sampler；它会改变训练权重，需额外独立实验。

## stability-v1 候选

唯一科学变量：learning_rate="0.001" → "0.0005"。理由是改变 batch 顺序下优化更新的尺度，检查是否减少 slide 排名震荡；现有日志支持测试该假设，不足以宣称 LR 已被定位为根因。其余训练、前端、split、聚合和 checkpoint 选择不变。未同时延长 patience，避免混杂。
config.py 修复了 Phase2 exploratory 不可能同时满足“五 seed 强制校验”和“仅单 seed”校验的问题；五 seed 约束仍只用于正式模式。只有 Phase2 exploratory 允许候选 LR 0.0005 与原 LR 0.001，Phase1/正式 LR 约束不变。
新增 configs/phase2a_morlet_stability_v1.toml；使用既有 exploratory 路径，保留 formal_experiment=false / experiment_mode=exploratory_train 和非覆盖输出。

## 两 seed 验证与验收

| 指标 | 原 baseline | stability-v1 |
|---|---:|---|
| high seed 1729 | 0.9113636364 | 待实测完成 |
| low seed 3407 | 0.8659090909 | 待实测完成 |
| gap | 0.0454545455 | 不可计算 |
| two-seed mean | 0.8886363636 | 不可计算 |
| best epoch（零基） | high 9 / low 1 | 不可计算 |
| convergence | high/low 曲线已恢复，均有震荡 | 云端未执行，不可比较 |

历史极值配对是定向 sanity check，不能估计新的五 seed SD；还须警惕极值选择与向均值回归。用户给出的“实质改善/明显缩小/不低太多”未提供数值阈值，不在观察结果后倒设验收门槛。结果完成前不得判 READY。

## 执行、测试与边界

2026-09-09 本次恢复：STOP: CLOUD_ENV_UNAVAILABLE。
SSH 使用原主机别名 codex-gpufree-30610，BatchMode=yes、ConnectTimeout=10；返回 `banner exchange: Connection to UNKNOWN port -1: Connection refused`。未进入 ~/demo，未同步代码、核验远端 HEAD、访问远端数据或启动 preflight/训练。无法仅凭此错误确定云实例、网络或 SSH 转发中哪一层故障。

原报告记录过本地 low-seed 启动尝试；这是历史执行记录，不符合本次原云端要求，不计入 sanity check。当前未发现 Python 训练进程；原本地 seed-3407 目录无 JSON 结果。保留该目录，不删除、不重试。本次没有本地 GPU 训练、CUDA benchmark、test 访问或新正式五 seed。

测试：python -m pytest tests/test_pipeline_config.py tests/test_pipeline_entrypoints.py -q：49 passed（包含项目 smoke）。新增实际 config→optimizer LR 传递检查在修复前失败，修复后通过；拒绝非法候选 LR 与正式 LR 放宽的检查通过。五 seed 零初始化检查通过。python -m ruff check src/cg_pipeline/config.py tests/test_pipeline_config.py：通过。未运行全套测试、正式 preflight 或正式训练。

修改文件：configs/phase2a_morlet_stability_v1.toml、src/cg_pipeline/config.py、tests/test_pipeline_config.py、docs/DECISIONS.md、本报告。数据整理另新增两份本地 train/val 清单，禁止纳入 commit。
固定光学前端与既有正式 baseline/seed 集合未改；新增的是本任务批准的 exploratory LR 例外。锁定 DEVELOPMENT_SPEC 未改，未填 TBD。原正式五 seed 文档配套缺口尚未解决。
基于当前工作区实现解释历史训练是明确假设；high/low epoch 日志已恢复；历史完整环境信息仍不能补造。
commit：本报告随本任务五个文件提交，实际 SHA 见最终回复。

本次仅更新报告/Decision 的证据与执行状态，训练代码和测试未再修改，复用既有 49 passed 与 Ruff 通过记录，不重复运行。

原因置信度：B batch order / sampler — high（seed 控制排列的机制；对观察方差的贡献大小未定）；E validation statistical variance / aggregation — medium（固定 42 slides、max 聚合与曲线支持敏感性）；G LR / regularization — low（仅待检验的 LR 假设）。不将 D early stopping 排为已证实主要原因。

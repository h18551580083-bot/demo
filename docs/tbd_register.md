# TBD 决策台账（TBD Decision Register）

> 只读盘点产物，盘点日期 2026-08-04。
> 本文件是一次"只读盘点"的交付物：**不修改**任何规范、代码、配置或测试，
> **不构成**决策记录（决策记录统一在 `docs/DECISIONS.md`），**不填充**任何 TBD。
> 台账状态变化一律遵循：提出候选 → 比较科学性/复现性/实现风险 → 推荐 →
> 人工批准 → 写入 `DECISIONS` → 同步规范 → 编写测试 → 验证后冻结。

## 1. 盘点范围与口径

- 基线：`e:\cg` 工作树。Phase 0 已于 2026-08-03 关闭
  （[DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md) §2；
  [PHASE0_ACCEPTANCE_MATRIX.md](PHASE0_ACCEPTANCE_MATRIX.md) 摘要 11 complete / 0 blocked）。
- 搜索模式（全仓、大小写不敏感）：
  `TBD | 待定 | 未冻结 | provisional | blocked | 待批准 | 未决`；
  补充探针：`unresolved | not decided | still unresolved | not yet frozen | TODO | FIXME | placeholder | NotImplementedError`。
- 覆盖：`docs/**`（含 ADR）、`src/**`、`tests/**`、`configs/**`、根文档（AGENTS.md、CONTEXT.md、README.md）。
- 关闭判据（五条全部满足才算"真正关闭"）：
  ① 有明确值；② 决策有记录；③ 代码已实现；④ 测试可验证；⑤ 无冲突表述。

## 2. 结论摘要

- **存活 TBD：0**（配置、源码、测试、ADR 中不存在未决的 `TBD` 值；`config.py` 的校验器
  在 `phase0-experiment-config-v1` 下显式拒绝字符串 `TBD`）。
- **五大阻断组：5/5 已冻结**，全部在 2026-08-03 获得人工授权处置并同步到
  规范 §6、两份协议、唯一正式配置与测试。
- **观察项 3 条**：
  - `OBS-01`：验收脚本 `acceptance.py` 引用的规范章节标题已陈旧，导致"残留 TBD"
    检查空转（代码-文档冲突表述，未阻断，但属门禁空洞）；
  - `OBS-02`：非规范评估契约草稿保留在磁盘（明确无规范效力，与冻结契约无冲突）；
  - `OBS-03`：CUDA 条件跳过测试（文档化行为，正式门禁绑定 Decision 30 RTX 4090 报告）。
- 患者级隔离状态在全部文档中一致：`patient_level_isolation = not_evaluated`、
  `patient_level_claim_allowed = false`、患者级门禁 `NOT APPLICABLE`，未发现任何
  越级声明或文档间冲突。

## 3. 台账主表

| 编号 | 问题（一句话） | 候选方案 | 影响文件（主要） | 阻断阶段 | 验证方法 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| TBD-01 | 分类器结构、参数预算、H/E 交换性质未定 | MLP / 带归一化分类头 / `linear-logit-v1` 零初始化（采纳） | 规范 §3.11；`configs/*.toml [model]`；`model.py`；`test_model_contract.py` | Phase 0 门禁（阻断组 1） | 模型契约测试断言 9408/9409/9473 | 已冻结 / 关闭 |
| TBD-02 | 损失、损失内精度、优化器与状态精度未定 | 加权/平衡损失、focal、SAM、AMP/TF32/float16（拒绝）；float32 BCE-with-logits + AdamW（采纳） | 规范 §3.12/§6；`configs/*.toml [training]`；`training.py`、`config.py`；协议测试 | Phase 0 门禁（阻断组 2） | 配置校验器 + 协议/配置测试 + preflight determinism 门 | 已冻结 / 关闭 |
| TBD-03 | `cam16-eval-v1` 清单/聚合/AUROC/Youden/阈值/校准/不确定度/例外未冻结 | 库默认 AUROC、中点/分位数阈值、patient-cluster bootstrap（拒绝）；manifest-bounded max-logit + Mann-Whitney + 2000 分层 bootstrap（采纳） | 规范 §3.15；`EVALUATION_PROTOCOL.md`；`configs [evaluation]`；`evaluation.py`；`test_evaluation_protocol.py` | Phase 0 门禁（阻断组 3） | 评估协议测试；final-once 测试门保持 false | 已冻结 / 关闭（附注 OBS-02） |
| TBD-04 | 种子/批次/纪元预算/早停/检查点/resume/失败运行/CI/最终一次性测试门未定 | 失败自动重试、非严格 resume（拒绝）；三种子 20 纪元不可变检查点（采纳） | `TRAINING_PROTOCOL.md`；`configs`；`training.py`、`pipeline.py`；训练/入口测试 | Phase 0 门禁（阻断组 4） | 训练/评估测试；dry-run 两轮重复 + resume | 已冻结 / 关闭 |
| TBD-05 | 迁移数据集、物理尺度自适应、迁移协议未定 | 现引入迁移数据集（拒绝）；不属于 Phase 1 起点、后续独立预注册（采纳） | `ADR 0010`；规范 §6 组 5 / §7 | 非 CAM16 Phase 1 阻断（组 5） | 源码/配置审计（无迁移代码与配置） | 显式延后决策（非 TBD） |
| TBD-06 | 无可靠 patient-to-slide 映射，患者级门禁曾误判为阻断 | (a) 以映射缺失阻断发布（否决，范围升级错误）；(b) 声明限定 `group_id/slide_id`、门禁 `NOT APPLICABLE`（采纳） | `phase0_release.json`；规范 §3.15/§5/§8；`claims.py`、`data.py`、`pipeline.py`；相关测试 | 原 Phase 0 门禁（已裁定 NOT APPLICABLE） | 声明审计测试 + claims 正则审计 | 关闭（NOT APPLICABLE） |
| OBS-01 | `acceptance.py` 按已不存在的章节标题切分规范，残留 TBD 检查空转 | (a) 更新代码锚点为现标题；(b) 规范 §6 改回旧标题（否决）；(c) 结构式 TBD 扫描（推荐） | `acceptance.py:176-182`；`DEVELOPMENT_SPEC.md:1283` | 不阻断（Phase 0 已关）；影响未来门禁完整性 | 修复后注入 TBD 副本触发 `run_phase0_acceptance` | 开放（待人工批准） |
| OBS-02 | 非规范评估契约草稿保留在磁盘，含 "not yet frozen" 措辞 | (a) 保留为历史审计草稿（现状）；(b) 归档；(c) 删除 | `CAM16_EVAL_V1_CALCULATION_CONTRACT_DRAFT.md`；`DECISIONS.md:1492-1494` | 无 | 文档自声明 + 决策记录 | 非冲突（处置待人工决定） |
| OBS-03 | 校准测试在无 CUDA 环境下条件跳过 | 无（正式验收绑定 Decision 30 RTX 4090 报告） | `test_calibration.py:12`、`test_calibration_autograd.py:16`；`acceptance.py:46-76` | 无 | `audit_decision30_report` 固定 SHA-256 审计 | 非 TBD（文档化行为） |

## 4. 逐项追踪

### TBD-01 — 分类器架构与精确参数预算（阻断组 1）

- **问题**：可训练后端架构与参数预算、H/E 交换性质未定
  （`DECISIONS.md` 2026-07-28 "Not decided"）。
- **候选方案**：多层 MLP / 含激活与归一化的分类头 / `linear-logit-v1` 零初始化。
  拒绝项见 `DECISIONS.md:1478-1483`（hidden framework defaults、data-adaptive
  parameter selection、MLP classifiers、learned/normalized optical features）。
- **决策**：2026-08-03 "Approve `linear-logit-v1` and the exact backend budget"
  （`DECISIONS.md:1326-1363`）；规范 `DEVELOPMENT_SPEC.md:642-670`（§3.11）。
  冻结值：`w ∈ [1, 9408]`、`b_head ∈ [1]`、显式零初始化、9409 + 64 门控标量 =
  **9473**（等式约束，非上限）；无 sigmoid/隐藏层/归一化/丢弃/注意力/残差/可训练温度。
- **影响文件**：`DEVELOPMENT_SPEC.md` §3.11；`DECISIONS.md`；
  `configs/phase1_baseline.toml` `[model]`（`classifier = "linear-logit-v1"`）；
  `src/cg_pipeline/model.py`；`tests/test_model_contract.py`。
- **阻断阶段**：Phase 0 验收门禁（阻断组 1）。
- **验证方法**：`pytest tests/test_model_contract.py -q`；测试断言
  `sum(numel()) == 9473`（`test_model_contract.py:105`）、
  `classifier.weight.shape == (1, 9408)`（`:107`）；preflight `optimizer_ownership` 门。
- **状态**：已冻结 / 关闭。
- **关闭判据核查**：① 有明确值（9409/9473 等式）；② 决策有记录（DECISIONS 8/3）；
  ③ 代码已实现（`model.py`）；④ 测试可验证（模型契约测试）；⑤ 无冲突表述
  （§3.11 与 §3.12、验收矩阵交付物 2/6 一致）。

### TBD-02 — 损失 / 优化器 / 优化器状态精度（阻断组 2）

- **问题**：损失定义、损失内精度、优化器算法与优化器状态精度未定
  （`DECISIONS.md:1353-1354` "Still unresolved"）。
- **候选方案**：加权/平衡采样损失、标签平滑、focal、SAM/Adam/AdamW、
  AMP/TF32/float16 状态。拒绝项：weighted or balanced sampling、augmentation、
  TF32/AMP/float16/bfloat16/梯度缩放（规范 §3.12；`DECISIONS.md:1478-1483`）。
- **决策**：2026-08-03 冻结（`DECISIONS.md:1418-1429`）：mean float32
  BCE-with-logits（无类权、无平滑、无 focal、无概率预转换）；AdamW 全 float32 状态；
  lr `0.001`、betas `(0.9, 0.999)`、eps `0.00000001`、wd `0.0001`、无 scheduler。
  精度基线另由 2026-07-30 决策冻结（`DECISIONS.md:903-961` → 规范 §3.12）。
- **影响文件**：`configs/phase1_baseline.toml` `[training]`；
  `docs/TRAINING_PROTOCOL.md`；`src/cg_pipeline/training.py`、`config.py`；
  `tests/test_training_protocol.py`、`test_pipeline_config.py`。
- **阻断阶段**：Phase 0 验收门禁（阻断组 2）。
- **验证方法**：`pytest tests/test_training_protocol.py tests/test_pipeline_config.py -q`；
  配置校验器拒绝未知/缺失/非法/浮点/`TBD` 值（`config.py:232-242, 392`，
  测试 `test_pipeline_config.py:135, 147`）；preflight `precision_and_determinism` 门。
- **状态**：已冻结 / 关闭。
- **关闭判据核查**：① 有明确值（TOML `[training]` + `[determinism]`）；
  ② 有记录（DECISIONS 7/30 + 8/3）；③ 代码实现（`training.py`、`config.py`）；
  ④ 测试可验证；⑤ 无冲突表述（`TRAINING_PROTOCOL.md` 表格与 TOML 逐项一致）。

### TBD-03 — `cam16-eval-v1` 计算契约（阻断组 3）

- **问题**：现有补丁清单规范化、采样、聚合、AUROC、Youden 阈值、校准、不确定度、
  例外情况全部未冻结；DECISIONS 8/3 原则决策明确 "**not yet frozen**"
  （`DECISIONS.md:1391-1401`）。
- **候选方案**：库默认 AUROC 实现、中点/分位数阈值、patient-cluster bootstrap、
  拟合校准变换（均拒绝）；采纳 manifest-bounded max-logit 聚合 + 精确 Mann-Whitney
  + 验证集 distinct-logits Youden（最大 J、最大 logit 破平）+ 2000 复现分层 slide
  bootstrap 百分位区间（下标 49 / 1949）。
- **决策**：2026-08-03 计算契约冻结（`DECISIONS.md:1447-1467`）；
  规范 `DEVELOPMENT_SPEC.md:999-1036`（§3.15）；`docs/EVALUATION_PROTOCOL.md` 全篇。
- **影响文件**：`EVALUATION_PROTOCOL.md`；`DEVELOPMENT_SPEC.md` §3.15；
  `configs/*.toml` `[evaluation]`；`src/cg_pipeline/evaluation.py`；
  `tests/test_evaluation_protocol.py`。
- **阻断阶段**：Phase 0 验收门禁（阻断组 3）。
- **验证方法**：`pytest tests/test_evaluation_protocol.py -q`；
  最终一次性测试门保持 false（`phase0_release.json:8` `test_access_authorized = false`）。
- **状态**：已冻结 / 关闭（草稿问题见 OBS-02）。
- **关闭判据核查**：① 有明确值（`[evaluation]` 全部字段）；② 有记录；
  ③ 代码实现（`evaluation.py`）；④ 测试可验证；⑤ 无冲突表述（冻结契约与
  §3.15、验收矩阵交付物 11 一致；草稿明确无规范效力，不构成冲突）。

### TBD-04 — 训练预算 / 种子 / 检查点 / CI / 最终一次性测试门（阻断组 4）

- **问题**：seeds、batch/epoch 预算、早停、checkpoint/resume、失败运行、
  多种子聚合、2000 复现置信区间、最终一次性测试规则未定
  （`DECISIONS.md:1357-1358` "Still unresolved"）。
- **候选方案**：失败自动重试/替换种子、宽松 resume（拒绝）；采纳：种子
  `1729/3407/7919`、batch 4、max 20 epochs、早停 val slide AUROC patience 5 /
  min delta 0、每个完整 epoch 存不可变检查点、最大 val AUROC（平局取最早 epoch）、
  resume 仅接受连续零基不可变配对且身份完全一致、失败种子记录并排除。
- **决策**：2026-08-03 训练基线冻结（`DECISIONS.md:1430-1445`）；
  `docs/TRAINING_PROTOCOL.md` 全篇。
- **影响文件**：`TRAINING_PROTOCOL.md`；`configs/phase1_baseline.toml` `[training]`；
  `src/cg_pipeline/training.py`、`pipeline.py`；
  `tests/test_training_protocol.py`、`test_pipeline_entrypoints.py`。
- **阻断阶段**：Phase 0 验收门禁（阻断组 4）。
- **验证方法**：训练/入口测试；dry-run 全链两轮重复与 resume 负向路径
  （`PHASE0_ACCEPTANCE_MATRIX.md` 交付物 9）。
- **状态**：已冻结 / 关闭。
- **关闭判据核查**：全部满足（值、记录、代码、测试、无冲突均验证）。

### TBD-05 — 迁移数据集 / 物理尺度 / 迁移协议（阻断组 5）

- **问题**：迁移数据集选择、物理尺度自适应、迁移协议未定
  （`DECISIONS.md:1359-1360` "Still unresolved"）。
- **候选方案**：在 Phase 1 起点引入某个迁移数据集（拒绝）；不纳入起点、
  由后续独立预注册决定（采纳）。
- **决策**：2026-08-03 transfer disposition（`DECISIONS.md:1469-1477`；
  `ADR 0010`；规范 §6 组 5 与 §7 禁止使用迁移数据做开发/调参/选择）。
- **影响文件**：`docs/adr/0010-phase1-preregistered-baseline.md`；
  `DEVELOPMENT_SPEC.md` §6/§7；（源码/配置中无迁移实现，审计确认）。
- **阻断阶段**：对 CAM16 Phase 1 入口**非阻断**；属未来独立预注册范围。
- **验证方法**：源码/配置审计——`configs/`、`src/` 无任何迁移数据集/尺度自适应条目。
- **状态**：显式延后决策（**非 TBD**，有明确处置，不得再以"未决"表述）。
- **关闭判据核查**：① 有明确值（"不属于 Phase 1 起点"）；② 有记录；③ 代码
  未引入（符合范围）；④ 验证 = 审计通过；⑤ 无冲突表述（与规范 §7、README 一致）。

### TBD-06 — 患者级隔离声明门（原患者映射阻断）

- **问题**：当前 CAM16 包无可靠 patient-to-slide 映射；早期曾将患者级隔离
  误升级为 Phase 0 外部阻断器（`DECISIONS.md:1496-1505` "Still externally unresolved"）。
- **候选方案**：(a) 以映射缺失阻断发布（**否决**——被 8/3 修正裁定为范围升级错误）；
  (b) 声明严格限定在 `group_id/slide_id` 层面、患者级门禁标记 `NOT APPLICABLE`（采纳）。
- **决策**：2026-08-03 "Correct the patient-level gate scope and close Phase 0"
  （`DECISIONS.md:1527-1574`），并取代同日更早的阻断表述。
- **影响文件**：`configs/phase0_release.json`（`patient_level_isolation = "not_evaluated"`、
  `patient_level_claim_allowed = false`）；`DEVELOPMENT_SPEC.md` §3.15/§5/§8；
  `AGENTS.md`；`CONTEXT.md`；`src/cg_pipeline/claims.py`、`data.py`、`pipeline.py`；
  `tests/test_data_contract.py:90-91`、`test_pipeline_entrypoints.py`、`test_phase0_acceptance.py`。
- **阻断阶段**：原 Phase 0 门禁（已裁定 NOT APPLICABLE，非 FAIL；对 Phase 1
  train/validation preflight 非阻断）。
- **验证方法**：`pytest tests/test_phase0_acceptance.py tests/test_data_contract.py -q`；
  claims.py 正则审计（`claims.py:12-49`）拒绝任何肯定式患者级声明与非规范字段值。
- **状态**：关闭（NOT APPLICABLE）。患者级隔离保持 `not_evaluated`，患者级声明被禁止，
  不得从文件名/标识符推断患者身份。
- **关闭判据核查**：① 有明确值（`not_evaluated` / `false`）；② 有记录
  （DECISIONS 8/3 修正）；③ 代码实现（claims 审计 + release 校验）；④ 测试可验证；
  ⑤ 无冲突表述（验收矩阵、GAP_REGISTER P0-05、README、协议文档全仓一致）。

### OBS-01 — 验收脚本章节锚点陈旧（代码-文档冲突表述）

- **问题**：`acceptance.py:176` 以 `"## 6. Blocking unresolved decisions"` 切分
  `DEVELOPMENT_SPEC.md`，但规范 §6 现行标题为
  `"## 6. Blocking decision groups and current disposition"`
  （`DEVELOPMENT_SPEC.md:1283`）。锚点永不命中 → `blocking_section` 退化为
  "§7 之前的全文"，而 `acceptance.py:180-181` 探测的两条短语
  （`"The following values remain \`TBD\`"`、`"must not be inferred, defaulted, or selected"`）
  在规范中均不存在 → `active_tbd` 恒为 False，该门成为**空转检查**。
- **候选方案**：
  - (a) 将代码锚点改为现行标题并同步更新探测短语（最小改动）；
  - (b) 将规范 §6 标题改回旧名（**否决**——破坏现行文档体系、ADR 0010 与协议的引用一致性）；
  - (c) 结构式检查：对 §6 隔离区间正则匹配 `\bTBD\b`，并对五大阻断组逐一断言存在
    显式处置语句（**推荐**，能真正拦截未来回归）。
- **影响文件**：`src/cg_pipeline/acceptance.py:176-182`；
  `docs/DEVELOPMENT_SPEC.md:1283`。
- **阻断阶段**：不阻断已关闭的 Phase 0；影响**未来阶段**门禁完整性——若 §6 未来
  再现 TBD，当前实现无法拦截。
- **验证方法**：修复后运行 `pytest tests/test_phase0_acceptance.py -q`，并以人为注入
  TBD 的规范副本运行 `run_phase0_acceptance(...)`，确认 `active_blocking_tbd = True`。
- **状态**：**开放**——待人工批准方案后，按流程写入 DECISIONS → 修改 `acceptance.py`
  → 补负向测试 → 验证后冻结。本次只读盘点不实施修改。

### OBS-02 — 非规范评估契约草稿保留

- **问题**：`docs/CAM16_EVAL_V1_CALCULATION_CONTRACT_DRAFT.md` 声明
  `Normative effect: none`（草稿 §0，:7），全文使用 propose/would 措辞，含
  `"clauses are not yet frozen"`（:499）。该文件存在于磁盘但未被 git 跟踪
  （`git ls-files` 无匹配），与已冻结的 `EVALUATION_PROTOCOL.md` 无冲突。
- **候选方案**：(a) 保留为历史审计草稿（现状，DECISIONS 8/3 声明
  "neither overwritten nor required by this decision"，`DECISIONS.md:1492-1494`）；
  (b) 移入归档目录；(c) 删除。
- **影响文件**：`docs/CAM16_EVAL_V1_CALCULATION_CONTRACT_DRAFT.md`；
  `DECISIONS.md:1492-1494`。
- **阻断阶段**：无。
- **验证方法**：文档自声明 + DECISIONS 记录；冻结契约以 `EVALUATION_PROTOCOL.md`
  与 §3.15 为准。
- **状态**：**非冲突**（明确无规范效力）；归档或删除由人工另行决定。

### OBS-03 — CUDA 条件跳过测试

- **问题**：`tests/test_calibration.py:12` 与 `tests/test_calibration_autograd.py:16`
  在无 CUDA 环境条件跳过。
- **候选方案**：无。Decision 30 正式跨设备数值等价验收绑定单一非 CPU 设备
  RTX 4090（`DECISIONS.md:1256-1293`），本地 skip 属文档化行为。
- **影响文件**：`tests/test_calibration.py`、`tests/test_calibration_autograd.py`；
  `acceptance.py:46-76`（`audit_decision30_report` 以固定 SHA-256
  `fe0c0a3d...1836117` 校验正式报告）。
- **阻断阶段**：无（正式门禁不受本地 skip 影响）。
- **验证方法**：`audit_decision30_report` 校验报告哈希、模式与必需字段。
- **状态**：非 TBD，文档化行为。

## 5. 交叉核对记录

| 文档 / 工件 | 声称状态 | 证据 | 一致性 |
| --- | --- | --- | --- |
| `DEVELOPMENT_SPEC.md` §2 | Phase 0 于 2026-08-03 关闭；formal train/validation 已授权；test/transfer 仍关闭 | `DEVELOPMENT_SPEC.md:21-25` | ✓ |
| `configs/phase0_release.json` | `phase0_closed = true`、`formal_training_authorized = true`、`test_access_authorized = false`、患者字段 not_evaluated/false | `phase0_release.json:3-8` | ✓（与 §2 一致） |
| `DECISIONS.md` | 历史 "Not decided / Still unresolved / remains TBD / not yet frozen" 均为后续同日决策解决（superseded），无存活项 | `DECISIONS.md:30-40, 236, 578-899, 1352-1360, 1391-1401, 1496-1505` → 8/3 冻结与修正决策 | ✓（历史为审计轨迹，非规范） |
| `PHASE0_ACCEPTANCE_MATRIX.md` | 11 complete / 0 blocked | `PHASE0_ACCEPTANCE_MATRIX.md:24` | ✓ |
| `PHASE0_GAP_REGISTER.md` | P0-01…15 全部 resolved / not applicable | `PHASE0_GAP_REGISTER.md:9-23` | ✓ |
| `TRAINING_PROTOCOL.md` ↔ `phase1_baseline.toml [training]` ↔ `config.py` | 训练契约逐项一致；校验器拒绝 TBD/浮点 | 协议表格；`config.py:232-242, 392` | ✓ |
| `EVALUATION_PROTOCOL.md` ↔ 规范 §3.15 ↔ `[evaluation]` | 评估契约一致；final-once 测试门关闭 | `EVALUATION_PROTOCOL.md:79-90` | ✓ |
| `ADR 0010` ↔ 规范 §6 组 5 | 迁移不属于 Phase 1 起点 | `ADR 0010:6-18` | ✓ |
| 测试断言（9473/9408/not_evaluated/false）↔ 规范 §3.11/§3.15 | 参数等式与患者字段被测试锁定 | `test_model_contract.py:105-115`；`test_data_contract.py:90-91` | ✓ |
| `acceptance.py:176` 章节锚点 ↔ 规范 §6 标题 | 锚点 `"## 6. Blocking unresolved decisions"` ≠ 现行标题 `"## 6. Blocking decision groups and current disposition"` | `acceptance.py:176`；`DEVELOPMENT_SPEC.md:1283` | ✗（OBS-01，未阻断） |

## 6. 待人工决策项（下一步）

1. **OBS-01**：批准候选 (c)（推荐）或 (a)，随后按流程：写入 `DECISIONS.md` →
   同步 `acceptance.py` → 补负向测试 → 运行
   `pytest tests/test_phase0_acceptance.py -q` → 验证后冻结。
2. **OBS-02**：批准草稿保留 (a) / 归档 (b) / 删除 (c) 之一。
3. **未来新科学选择**（迁移数据集、消融基线、`j = 4` 尺度、新统计量等）必须先经
   独立决策流程；规范 §1 与 §7 规定 TBD 不得未经人工批准被填充——本台账只记录
   追踪状态，不代为决策。

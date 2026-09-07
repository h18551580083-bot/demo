# 仓库瘦身审计 — 2026-09-07

## 范围与事实

- 工作树 `E:/cg`，origin 为 `https://github.com/h18551580083-bot/demo.git`。
- 瘦身前 HEAD：`7e8d8cffaa14d99de596c9e11bdf3db611d1e062`。
- 已创建本地 tag `archive/pre-slim-2026-09-07`，验证指向该 HEAD。
  没有 push、commit、改写历史或修改分支指向。初始仅 `picture/` 未跟踪。
- 远端 main 的只读查询结果为 `034dc8935471e89aaf2db6e67418df5e936a90ff`；
  本地 tag/HEAD 不等于云端备份。归档云端化尚未完成。
- 扫描主仓库 Git 路径、主要目录、源代码 import/CLI、配置及文档引用，
  记录 5043 个文件元数据（修改迁移测试后、删除前快照）。真实数据、Git
  内部对象、独立 gitlink 与图片内容不递归读取。主要目录均列于表中；
  这不是对数据集、独立仓库或二进制论文素材的内容审计。
- 代码链：`__main__ → pipeline → preflight/runtime → training_runs → training`；
  模型链：`model → frontend → morlet/control_bank` 与 `interaction/pooling`；
  评估链：`evaluation → evaluation_metrics`。AST 反向 import 与文本引用交叉核对。
- Phase2A 配置/代码与结果已存在；DECISIONS 的 2026-09-05 记录批准了有限参数化
  与 validity gate 实现。不能因目标示例只写 Phase1 就删除它们；本次未推进 phase。

## 审计表

代码/配置/测试逐文件列示；同类临时产物和正式 run 按目录合并。
ARCHIVE 表示已在本地 tag 保留并从工作树移除，不能据此宣称已完成云端归档。
DELETE 行的 Action 区分实际移除与因审批限制未执行。

| Path | Category | Current dependency | Reason | Action |
| --- | --- | --- | --- | --- |
| `.gitignore` | KEEP | 包安装/CLI 配置/项目入口 | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `AGENTS.md` | KEEP | 包安装/CLI 配置/项目入口 | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `CONTEXT.md` | KEEP | 包安装/CLI 配置/项目入口 | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `README.md` | KEEP | 包安装/CLI 配置/项目入口 | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `configs/exploratory_train.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/formal_training_authorization.json` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase0_release.json` | ARCHIVE | 仅历史验收入口/闭环文档；当前 pipeline/preflight/runtime 无调用 | 删除前与归档 tag 逐字节归一换行校验；患者声明测试已迁移 | 已从索引和工作树移除 |
| `configs/phase1_baseline.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase1_matched_control.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase2a_morlet_baseline.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase2a_morlet_gamma_0p625.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase2a_morlet_sigma0_0p7.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `configs/phase2a_morlet_xi0_2pi3.toml` | KEEP | config.load_experiment_config；训练 CLI 显式 --config / --authorization | Phase1 baseline/control 论文对照；Phase2A 为已存在批准实现，不能误删 | 原样保留 |
| `demo-rehearsal` | DEPENDENCY-CHECK | 独立 Git gitlink 35d2b946；不是当前 Python 包 | 独立仓库，当前用途与可清理范围未确认 | 保留，不递归审计内部内容 |
| `docs/DECISIONS.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/DEVELOPMENT_SPEC.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/EVALUATION_PROTOCOL.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/PHASE0_ACCEPTANCE_MATRIX.md` | ARCHIVE | 仅历史验收入口/闭环文档；当前 pipeline/preflight/runtime 无调用 | 删除前与归档 tag 逐字节归一换行校验；患者声明测试已迁移 | 已从索引和工作树移除 |
| `docs/PHASE0_GAP_REGISTER.md` | ARCHIVE | 仅历史验收入口/闭环文档；当前 pipeline/preflight/runtime 无调用 | 删除前与归档 tag 逐字节归一换行校验；患者声明测试已迁移 | 已从索引和工作树移除 |
| `docs/PHASE1_CONTROL_COMPARISON_REPORT.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/PHASE1_TRAINING_RUNBOOK.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/PHASE2_A_MORLET_ABLATION_DESIGN.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/TRAINING_PROTOCOL.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0001-fixed-he-stain-basis.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0002-complex-morlet-primary-wavelets.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0003-first-order-wavelet-modulus-frontend.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0004-explicit-discrete-morlet-generation.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0005-reflection-padding-and-valid-support-mask.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0006-domain-separated-canonical-hashes.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0007-electronic-he-interaction-boundary.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0008-linear-logit-primary-classifier.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0009-existing-patch-data-entry.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0010-phase1-preregistered-baseline.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/adr/0011-exploratory-and-formal-training-modes.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/agents/domain.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | AGENTS 仍显式引用的三份小型约定；不是新增 Agent framework | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/agents/issue-tracker.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | AGENTS 仍显式引用的三份小型约定；不是新增 Agent framework | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/agents/triage-labels.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | AGENTS 仍显式引用的三份小型约定；不是新增 Agent framework | 保留历史叙述；已移除路径从归档 tag 读取 |
| `docs/tbd_register.md` | KEEP | AGENTS/CONTEXT/README 文档入口或研究决策引用 | 有效科研规则、协议、ADR、Phase1 论文比较或已有 Phase2A 设计；TBD 保留原状 | 保留历史叙述；已移除路径从归档 tag 读取 |
| `pyproject.toml` | KEEP | 包安装/CLI 配置/项目入口 | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `scripts/phase1_control_comparison.py` | KEEP | PHASE1_CONTROL_COMPARISON_REPORT 与两组 completion.json | 当前可执行项目与论文复现入口 | 保留；README/.gitignore 更新 |
| `src/cg_acceptance/__init__.py` | KEEP | 包导入/模块 CLI | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/__main__.py` | KEEP | 包导入/模块 CLI | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/calibration.py` | KEEP | `src/cg_acceptance/__init__.py`, `src/cg_acceptance/__main__.py` | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/calibration_environment.py` | KEEP | 间接包导出/同包调用 | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/comparator.py` | KEEP | `src/cg_acceptance/__init__.py`, `src/cg_acceptance/calibration.py` | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/cpu_reference.py` | KEEP | 间接包导出/同包调用 | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/device_operator.py` | KEEP | 间接包导出/同包调用 | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_acceptance/fixture.py` | KEEP | `src/cg_acceptance/__init__.py`, `src/cg_acceptance/__main__.py`, `src/cg_acceptance/calibration.py`, `src/cg_acceptance/calibration_environment.py` | 最高规格 §3.14 仍要求独立 CPU reference、比较器、校准及负例；非启动热路径，不能因 Phase 0 名称删除 | 保留 |
| `src/cg_pipeline/__init__.py` | KEEP | 包导入/模块 CLI | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/__main__.py` | KEEP | `tests/test_pipeline_entrypoints.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/acceptance.py` | ARCHIVE | 仅历史验收入口/闭环文档；当前 pipeline/preflight/runtime 无调用 | 删除前与归档 tag 逐字节归一换行校验；患者声明测试已迁移 | 已从索引和工作树移除 |
| `src/cg_pipeline/artifacts.py` | KEEP | `src/cg_pipeline/pipeline.py`, `src/cg_pipeline/preflight.py`, `src/cg_pipeline/training_runs.py`, `tests/test_claims.py`, `tests/test_phase2a_morlet.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/claims.py` | KEEP | `src/cg_pipeline/artifacts.py`, `src/cg_pipeline/data.py`, `src/cg_pipeline/pipeline.py`, `src/cg_pipeline/preflight.py`, `src/cg_pipeline/training_runs.py`, `tests/test_claims.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/config.py` | KEEP | `src/cg_pipeline/__init__.py`, `src/cg_pipeline/pipeline.py`, `src/cg_pipeline/preflight.py`, `src/cg_pipeline/runtime.py`, `src/cg_pipeline/training_runs.py`, `tests/test_frontend_variants.py`, `tests/test_phase2a_morlet.py`, `tests/test_pipeline_config.py`, `tests/test_pipeline_entrypoints.py`, `tests/test_pipeline_safety.py`, `tests/test_training_protocol.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/control_bank.py` | KEEP | `src/cg_pipeline/frontend.py`, `tests/test_frontend_variants.py`, `tests/test_pipeline_safety.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/data.py` | KEEP | `src/cg_pipeline/preflight.py`, `src/cg_pipeline/runtime.py`, `src/cg_pipeline/training_runs.py`, `tests/test_data_contract.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/evaluation.py` | KEEP | `src/cg_pipeline/runtime.py`, `src/cg_pipeline/training_runs.py`, `tests/test_evaluation_protocol.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/evaluation_metrics.py` | KEEP | `src/cg_pipeline/evaluation.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/frontend.py` | KEEP | `src/cg_pipeline/model.py`, `tests/test_frontend_variants.py`, `tests/test_morlet_frontend.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/identity.py` | KEEP | `src/cg_pipeline/control_bank.py`, `src/cg_pipeline/data.py`, `src/cg_pipeline/evaluation.py`, `src/cg_pipeline/frontend.py`, `src/cg_pipeline/morlet.py`, `src/cg_pipeline/training_runs.py`, `tests/test_pipeline_config.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/interaction.py` | KEEP | `src/cg_pipeline/model.py`, `tests/test_frontend_variants.py`, `tests/test_model_contract.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/model.py` | KEEP | `src/cg_pipeline/preflight.py`, `src/cg_pipeline/runtime.py`, `src/cg_pipeline/training.py`, `src/cg_pipeline/training_runs.py`, `tests/test_frontend_variants.py`, `tests/test_model_contract.py`, `tests/test_phase2a_morlet.py`, `tests/test_training_protocol.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/morlet.py` | KEEP | `src/cg_pipeline/config.py`, `src/cg_pipeline/control_bank.py`, `src/cg_pipeline/frontend.py`, `src/cg_pipeline/morlet_validity.py`, `src/cg_pipeline/preflight.py`, `tests/test_frontend_variants.py`, `tests/test_morlet_frontend.py`, `tests/test_morlet_spectral.py`, `tests/test_phase2a_morlet.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/morlet_validity.py` | KEEP | `src/cg_pipeline/preflight.py`, `tests/test_phase2a_morlet.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/pipeline.py` | KEEP | `src/cg_pipeline/__main__.py`, `tests/test_pipeline_entrypoints.py`, `tests/test_pipeline_safety.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/pooling.py` | KEEP | `src/cg_pipeline/model.py`, `tests/test_frontend_variants.py`, `tests/test_model_contract.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/preflight.py` | KEEP | `src/cg_pipeline/pipeline.py`, `tests/test_pipeline_safety.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/runtime.py` | KEEP | `src/cg_pipeline/pipeline.py`, `src/cg_pipeline/preflight.py`, `src/cg_pipeline/training_runs.py`, `tests/test_pipeline_safety.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/training.py` | KEEP | `src/cg_pipeline/data.py`, `src/cg_pipeline/pipeline.py`, `src/cg_pipeline/preflight.py`, `src/cg_pipeline/runtime.py`, `src/cg_pipeline/training_runs.py`, `tests/test_frontend_variants.py`, `tests/test_training_protocol.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `src/cg_pipeline/training_runs.py` | KEEP | `src/cg_pipeline/pipeline.py`, `tests/test_training_protocol.py` | 当前 CLI → pipeline → runtime/preflight/training_runs 的训练、模型、数据或评估依赖 | 保留 |
| `tests/test_calibration.py` | KEEP | `cg_acceptance` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_calibration_autograd.py` | KEEP | `cg_acceptance` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_comparator.py` | KEEP | `cg_acceptance` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_data_contract.py` | KEEP | `cg_pipeline.data` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_evaluation_protocol.py` | KEEP | `cg_pipeline.evaluation` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_frontend_variants.py` | KEEP | `cg_pipeline.morlet`, `cg_pipeline.training`, `cg_pipeline.control_bank`, `cg_pipeline.frontend`, `cg_pipeline.interaction`, `cg_pipeline.pooling`, `cg_pipeline.config`, `cg_pipeline.model` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_model_contract.py` | KEEP | `cg_pipeline.interaction`, `cg_pipeline.pooling`, `cg_pipeline.model` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_morlet_frontend.py` | KEEP | `cg_pipeline.morlet`, `cg_pipeline.frontend` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_morlet_spectral.py` | KEEP | `cg_pipeline.morlet` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_phase0_acceptance.py` | ARCHIVE | 仅历史验收入口/闭环文档；当前 pipeline/preflight/runtime 无调用 | 删除前与归档 tag 逐字节归一换行校验；患者声明测试已迁移 | 已从索引和工作树移除 |
| `tests/test_phase2a_morlet.py` | KEEP | `cg_pipeline.morlet`, `cg_pipeline.artifacts`, `cg_pipeline.config`, `cg_pipeline.model`, `cg_pipeline.morlet_validity`, `cg_pipeline` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_pipeline_config.py` | KEEP | `cg_pipeline.identity`, `cg_pipeline.config` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_pipeline_entrypoints.py` | KEEP | `cg_pipeline.config`, `cg_pipeline.pipeline`, `cg_pipeline.__main__` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_pipeline_safety.py` | KEEP | `cg_pipeline.control_bank`, `cg_pipeline.config`, `cg_pipeline.preflight`, `cg_pipeline.runtime`, `cg_pipeline.pipeline`, `cg_pipeline` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `tests/test_training_protocol.py` | KEEP | `cg_pipeline.training`, `cg_pipeline.config`, `cg_pipeline.training_runs`, `cg_pipeline.model`, `cg_pipeline` | 当前科研合同；calibration/comparator 仍覆盖 §3.14 独有约束 | 保留（运行范围见下文） |
| `.pytest-formal-run-id-20260819-{c,d,e,f,g,h,i,j}/` | DELETE | 已核对 57 个 tracked TOML 为测试变异配置；无当前引用 | .gitignore 的 .pytest-*/ 已覆盖 | 57 个文件已 git rm；历史在本地 tag |
| `artifacts/__pycache__/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 2 files, 24122 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/codex-pytest-full-morlet-waiver-20260813/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 360 files, 20762940 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/codex-pytest-full-morlet-waiver-20260813-final/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 360 files, 20762940 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/codex-pytest-release-waiver-20260813-a/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/codex-pytest-release-waiver-20260813-b/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 267 files, 228573 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/decision30_calibration_cuda0.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/decision30_cuda0_20260731T061120Z.tar.gz` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/decision30_doc_baseline_smoke_cuda0_20260803.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/decision30_formal_acceptance_rtx4090_20260802.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/decision30_local_smoke_autograd_cuda0_20260731.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/decision30_local_smoke_cuda0_20260731T061120Z.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/edit_plan_docx.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/environment_cuda0_20260731T061120Z.txt` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/exploratory_runs/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/formal_runs/phase1-cam16-baseline-b32-v2/` | KEEP | 当前 Phase1 baseline/control 或已有 Phase2A 结果 | 保留全部结果、checkpoint 和 summary；本次不加载/评估 | 原样保留，未确认论文需求也不删 |
| `artifacts/formal_runs/phase1-cam16-matched-control-b32-v1/` | KEEP | 当前 Phase1 baseline/control 或已有 Phase2A 结果 | 保留全部结果、checkpoint 和 summary；本次不加载/评估 | 原样保留，未确认论文需求也不删 |
| `artifacts/formal_runs/phase2a-cam16-morlet-baseline-b32-v1/` | KEEP | 当前 Phase1 baseline/control 或已有 Phase2A 结果 | 保留全部结果、checkpoint 和 summary；本次不加载/评估 | 原样保留，未确认论文需求也不删 |
| `artifacts/formal_runs/phase2a-cam16-morlet-sigma0-0p7-b32-v1/` | KEEP | 当前 Phase1 baseline/control 或已有 Phase2A 结果 | 保留全部结果、checkpoint 和 summary；本次不加载/评估 | 原样保留，未确认论文需求也不删 |
| `artifacts/generate_plan_figures.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/inspect_plan_docx.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/learning_probe_summary_800.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/make_contact_sheets.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/patient_gate_na_20260803/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/patient_gate_na_20260803_post_commit/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_dry_run_v1/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_dry_run_v1.stale-pre-commit/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_dry_run_v1.stale-pre-final/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight.stale-pre-commit.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight.stale-pre-final.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight_patient_gate_na_20260803.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight_patient_gate_na_20260803_post_commit.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight_phase0-closed-v1_0cdf5af.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight_phase0-closed-v1_0cdf5af_evidence.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_preflight_recheck_20260804.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_total_acceptance.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_total_acceptance.stale-pre-commit.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_total_acceptance_patient_gate_na_20260803.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase0_total_acceptance_patient_gate_na_20260803_post_commit.json` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/phase1-b32-controlled-smoke/` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/preflight/` | DEPENDENCY-CHECK | formal CLI 消费显式 --preflight-report | 混合版本；路径/历史 PASS 不能证明适用于当前代码配置 | 全保留，未执行真实数据 preflight |
| `artifacts/pytest/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 1073 files, 103318614 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-affected-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 131 files, 35274896 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-baseline-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 82 files, 11935578 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-epoch-log-full-20260819/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 178 files, 20649331 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-epoch-log-green-20260819/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-epoch-log-json-20260819/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 2 files, 573 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-epoch-log-red-20260819/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-epoch-log-related-20260819/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 88 files, 20443555 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-final-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 348 files, 20748078 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-focused/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-focused-2/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-focused-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 277 files, 278602 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-full-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 343 files, 20744940 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-lasttest-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 7 files, 9171 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-postreview-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 277 files, 278626 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-exploratory-reviewfix-host/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 307 files, 294239 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-final-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 138 files, 35365577 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-full-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 138 files, 35365568 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-green-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 97 files, 14911482 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-postreview-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 138 files, 35365577 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-red-phase1-b32/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 37 files, 47455 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-split-manifest-red-20260820-a/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest-workers-config/` | DELETE | 无 src/tests/config/docs 当前引用；子目录为 test_* / decision300 合成测试夹具 | 0 files, 0 bytes；不属正式训练输出 | 未删除：自动审批阻止递归清理 |
| `artifacts/pytest_cuda0_20260731T061120Z.txt` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/render_docx.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/update_word_fields.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `artifacts/validate_plan_docx.py` | DEPENDENCY-CHECK | 旧报告/文档工具/探索产物；用途或 Git 备份未证实 | 历史验收结果可能仍支撑规范或论文；不从名称推断可删除 | 保留 |
| `cam16_patch/` | KEEP | 数据边界/版本历史 | 真实数据不读取；图片与独立材料不猜测用途 | 保留，排除内容审计 |
| `picture/` | DEPENDENCY-CHECK | 初始未跟踪的图片和汇报材料 | 真实数据不读取；图片与独立材料不猜测用途 | 保留 |
| `tools/` | DEPENDENCY-CHECK | 数据边界/版本历史 | 真实数据不读取；图片与独立材料不猜测用途 | 保留 |
| `.git/` | KEEP | 数据边界/版本历史 | 真实数据不读取；图片与独立材料不猜测用途 | 保留，排除内容审计 |
| `.pytest_cache/; .ruff_cache/; src/**/__pycache__/; tests/__pycache__/` | DELETE | 工具缓存，可重建 | 递归删除审批受阻；验证也会重新生成缓存 | 保留于忽略范围，不入索引 |
| `docs/医院前期合作研究方案_泛癌病理光学表征.docx` | DEPENDENCY-CHECK | 未跟踪文档/工作材料 | 用途未确认，不读取二进制内容 | 保留 |

## 删除/迁移与覆盖证据

- 已移除 Phase0 总验收 `acceptance.py`、release JSON、两份闭环报告。
  当前 CLI、preflight、training_runs 不导入这些文件。
- 原 `test_phase0_acceptance.py` 的 11 个声明反例和两个文档/报告保护测试
  迁至 `test_claims.py`；文档审计原算法迁入既有 `claims.py`，仅从必需文档列表
  移除两份历史报告。新增当前文档完整/缺失的回归检查。
- 退役两个绑定历史 Decision30 报告和旧 tracked-file allowlist 的测试：
  它们不验证当前训练 gate，旧白名单只支持 baseline seed-1729，已不能描述
  当前多 seed/多 run 仓库。其历史行为在 tag 保留，不声称被当前测试等价覆盖。
  当前数值约束仍由保留的 calibration/comparator/model 测试保护；Git 禁止新增
  checkpoint 等政策仍由 AGENTS/最高规范约束，旧白名单不再冒充当前审计。
- 57 个 `.pytest-formal-run-id-*` TOML 已从索引和工作树移除，原内容仍在 tag。
- 其余 pytest/cache 递归清理被自动审批两次拒绝（仅返回 blocked by policy），
  未继续绕过。表中 DELETE 但 Action 为未删除的目录仍在磁盘。
- 没有新建 archive 目录；没有删除训练产物或触碰独立 demo-rehearsal。

## 最小科研安全测试集

以下为日常核心集；其余保留测试在对应模块/科学定义发生授权变更时运行。
“最小”不意味着删除仍覆盖独有锁定约束的测试。

| 保留测试 | 保护的科研约束 |
| --- | --- |
| tests/test_data_contract.py | group/slide 隔离、显式 split、路径/标签/manifest 合同 |
| tests/test_pipeline_config.py | 严格配置、锁定值、禁止 TBD/test 开关、run/output 一致 |
| tests/test_pipeline_entrypoints.py | 非训练 CLI/config/pipeline 控制流端到端 smoke；train/val 构造 |
| tests/test_pipeline_safety.py | 授权/preflight、test 禁用、跨 split 拒绝、seed 与不可覆盖输出 |
| tests/test_model_contract.py | 固定前端、9473 参数、pooling/precision、优化器步不改光学状态 |
| tests/test_training_protocol.py | 确定性、checkpoint/resume、早停、日志合同；合成或 mock |
| tests/test_claims.py | 患者声明 not_evaluated/false、文档和报告拒绝越界声明 |
| tests/test_frontend_variants.py | Morlet/control 对齐、固定状态及合成前向/反向/checkpoint smoke |

额外保留：`test_morlet_frontend.py`、`test_morlet_spectral.py`、
`test_phase2a_morlet.py`、`test_evaluation_protocol.py`，分别保护实际前端、频谱
门禁、已批准的参数化路径和论文指标计算。`test_calibration.py`、
`test_calibration_autograd.py`、`test_comparator.py` 保护最高规范 §3.14，
本次未修改这些模块，因此未重跑其专门 CUDA 校准测试。

## 验证

在 PowerShell 设置 `PYTHONPATH=E:/cg/src`、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。
以下 pytest 全部使用临时合成输入；训练 epoch 被 mock 的 gate 测试不是正式训练。
模型不变性测试执行合成张量的单个优化器步，不读取 CAM16。
为严格避免执行 test 评估路径，以下两个合成测试亦 deselect，文件仍保留。

```powershell
python -m pytest tests/test_claims.py tests/test_data_contract.py tests/test_pipeline_config.py tests/test_pipeline_entrypoints.py tests/test_pipeline_safety.py tests/test_model_contract.py tests/test_training_protocol.py tests/test_frontend_variants.py tests/test_morlet_frontend.py tests/test_morlet_spectral.py tests/test_phase2a_morlet.py tests/test_evaluation_protocol.py -k 'not test_prediction_ledger_requires_every_authorized_row_and_test_gate_is_identity_bound and not test_final_test_authorization_rechecks_independent_approval_artifact' -q -o addopts=''
python -m compileall -q src tests
python -m ruff check .
git diff --check
git diff --cached --check
```

- pytest：最终 **182 collected / 182 passed**，包含项目 smoke。
- 初次运行两个旧 mock 配置缺少 sigma0/xi0/gamma 而失败；夹具改读现有 baseline
  model 配置后全量所选测试通过。未给训练代码添加隐藏默认值。
- compileall：通过。
- `ruff check .`：失败，35 个既有问题；34 个在未跟踪 picture 汇报脚本，
  1 个在未修改的 scripts/phase1_control_comparison.py（I001）。未用排除规则遮掩。
- `python -m ruff check src tests`：通过。
- unstaged/staged diff whitespace 检查：通过。
- 已检查 `.gitignore` 匹配，`git ls-files .pytest-*` 为空。
- 受保护的 artifacts、当前 configs、frontend、preflight、training_runs、
  DEVELOPMENT_SPEC、TRAINING_PROTOCOL、EVALUATION_PROTOCOL、CONTEXT
  与 HEAD 比较无改动。AGENTS 的陈旧 formal gate 说明已按有效 Decision 同步；
  未加载 checkpoint、未运行 preflight/真实训练或 test 评估。

## 未决事项与假设

1. 22 个 baseline checkpoint 已在初始 Git 索引，与“不得提交 checkpoint”规则
   不一致；为保持论文证据，本次不删除、不新增或重新提交它们，也不改历史。
2. preflight 目录混合版本，未检查真实数据，无法认定哪个报告当前有效；全保留。
3. 旧忽略产物没有经过可恢复备份验证；Phase0 历史产物也不擅自删除。
4. Phase2A 与 matched-control 可能仍是论文证据，保守保留；不推断新实验许可。
5. 本地 tag 可恢复已跟踪历史文件，但不包含 ignored/untracked 产物，也不是
   云端归档。若之后明确要推送，可执行 `git push origin archive/pre-slim-2026-09-07`；
   本任务没有执行推送。无需再次创建同名 tag。
6. Skills 仅有评估规范，没有评估任何实际 Skill 或编造增益；四类 Automation
   只有 trigger/input/checks/output/forbidden actions 设计，未安装调度。
7. 递归临时目录清理受策略阻止，因此磁盘清理目标部分未完成；不确定文件均保留。

## 全部 changed files

以下为本任务改动（不包含进入任务前已有的 picture/ 未跟踪文件）。
删除文件已 staged；其余修改/新增保留待审，未提交。

| Path | Change |
| --- | --- |
| `AGENTS.md` | M |
| `.gitignore` | M |
| `.pytest-formal-run-id-20260819-c/test_formal_config_accepts_new0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-d/test_formal_config_accepts_new0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-e/test_formal_config_rejects_run0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-f/test_formal_config_rejects_run0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv1/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv2/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv3/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv4/formal.toml` | D |
| `.pytest-formal-run-id-20260819-g/test_formal_config_rejects_inv5/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv1/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv2/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv3/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv4/formal.toml` | D |
| `.pytest-formal-run-id-20260819-h/test_formal_config_rejects_inv5/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv1/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv2/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv3/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv4/formal.toml` | D |
| `.pytest-formal-run-id-20260819-i/test_formal_config_rejects_inv5/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_is_strict_normaliz0/config.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi0/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi1/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi2/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi3/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi4/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_config_rejects_unknown_mi5/bad.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_engineering_o0/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai0/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai1/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai2/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai3/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai4/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai5/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_exploratory_overrides_fai6/exploratory.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_accepts_new0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin1/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin2/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin3/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin4/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin5/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin6/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin7/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_keeps_engin8/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv1/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv2/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv3/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv4/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_inv5/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_formal_config_rejects_run0/formal.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_locked_scientific_fields_0/mutated.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_locked_scientific_fields_1/mutated.toml` | D |
| `.pytest-formal-run-id-20260819-j/test_locked_scientific_fields_2/mutated.toml` | D |
| `README.md` | M |
| `configs/phase0_release.json` | D |
| `docs/DECISIONS.md` | M |
| `docs/CONTRACT_RECONCILIATION_2026-09-07.md` | A |
| `docs/PHASE0_ACCEPTANCE_MATRIX.md` | D |
| `docs/PHASE0_GAP_REGISTER.md` | D |
| `src/cg_pipeline/acceptance.py` | D |
| `src/cg_pipeline/claims.py` | M |
| `tests/test_phase0_acceptance.py` | D |
| `tests/test_training_protocol.py` | M |
| `tests/test_claims.py` | A (untracked) |
| `docs/MAINTENANCE.md` | A (untracked) |
| `docs/REPOSITORY_AUDIT_2026-09-07.md` | A (untracked) |

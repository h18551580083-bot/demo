# 最小维护规范

本文件只定义维护评估，不改变 DEVELOPMENT_SPEC、研究 phase、训练或评估合同。
不安装 Skill，不创建调度任务，不引入 Agent 或 routing framework。

## Skills 生命周期

建议独立放在 `C:/Users/sunnet/skill-evals/`，本次不创建该目录：

```text
skill-evals/
  registry.json
  cases/<skill-name>.json
  results/<model-version>/<skill-name>.json
```

每项登记 `name`、`purpose`、`trigger`、`eval_cases`、`baseline_score`、
`skill_score`、`last_model_tested`、`dependencies`、`status`。
分数初值为 null，未执行评估不得填入估计成绩。status 只允许
KEEP / MERGE / RETIRE / NEEDS_REWRITE；首次 NEEDS_REWRITE 须注明尚未评估，
这只是待审状态，不代表已经证明需要重写。

每项用 3–5 个稳定案例，包含正常任务、边界输入和误触发案例。
baseline 与 skill 使用相同模型版本、输入、工具权限、预算与验收标准，
在独立干净上下文执行；唯一变量是是否加载该 Skill。项目科研规则两组
始终生效，绝不为了 baseline 去掉 AGENTS、数据 protocol 或安全约束。
每案例至少重复 3 次，保存逐例结果、错误、耗时与 token 消耗；不能仅存总分。

评分先看科研/安全硬约束是否全部通过，再看任务成功率与人工返工量，
耗时与成本作为次要指标。明确增益的阈值须在运行前按案例登记，不能看完
结果再改阈值。小样本只能作为维护判断，不宣称统计显著性。

| 状态 | 判定 | 动作 |
| --- | --- | --- |
| KEEP | 配对重复评估有稳定增益，且无硬约束退步 | 保留；记录已测模型版本 |
| MERGE | trigger、工具步骤或案例高度重叠 | 提出合并稿，合并后重跑共同案例 |
| RETIRE | 连续两个模型版本未达预登记增益阈值 | 人工确认后停用，保留版本与结果 |
| NEEDS_REWRITE | 专用任务仍重要，但提示词主要补偿旧模型能力，或评估不足 | 删通用提示、保留专用步骤，再评估 |

优先保留项目科研规则、专用工具流程、高风险 guardrail 与数据/实验 protocol；
优先评估淘汰通用写代码技巧、通用 reasoning 和“逐步思考”提示词。
与安全规则重叠的 Skill 可以退役，但规则必须继续留在项目权威文档中。
没有实际 baseline/skill 配对结果时，本项目不对任何已安装 Skill 下退役结论。

## 四类 Automation 设计

以下是逻辑触发设计，不声称 Codex 已原生订阅提交、模型升级或结果事件。
日后接入时，可由人工启动，或用低频调度只读比较上次记录的身份；
发现相关变化才检查。未发生有意义变化时保持安静。当前任务不创建调度。
执行前读 AGENTS、CONTEXT、最高规格与本次授权；冲突时报告并停止相关动作。
默认输出当前任务中的简短报告；保存报告须使用新文件名，不覆盖研究证据。
不自动发外部消息，不创建 PR 或推送，除非另有明确授权。

| Automation | Trigger | Input | Checks | Output | Forbidden actions |
| --- | --- | --- | --- | --- | --- |
| research-gate | 重要代码变更后；先比对提交/改动文件清单 | diff、当前配置、权威规范、保留测试 | frontend 参数/优化器所有权；合成 split 泄漏反例；test 访问禁用；授权与 preflight 拒绝路径；CLI smoke；规范与代码冲突 | PASS/FAIL/BLOCKED、失败位置、测试命令及跳过项；不将单测通过写成正式训练许可 | 真实训练、真实数据读取、test 评估、改 gate/阈值/TBD、自动推进 phase |
| repo-hygiene | 每周一次 | Git 跟踪/忽略状态、路径/大小/时间元数据、import/CLI/配置/文档引用 | cache、pytest 目录、大文件、孤立产物、无引用脚本、陈旧配置、重复文档；先验证依赖与历史可恢复性 | KEEP/ARCHIVE/DELETE/DEPENDENCY-CHECK 表；必要时经授权做 PR | 删除不确定文件、递归读取数据集、删除 checkpoint、改历史、自动推送 |
| skill-regression | 模型升级后，或每季度；身份未变且无新案例可跳过 | 独立 registry、固定案例、模型/Skill 版本、相同预算与工具权限 | baseline vs skill 重复配对；成功率、硬约束、返工、成本；依赖可用性与重叠 | 逐例成绩及 KEEP/MERGE/RETIRE/NEEDS_REWRITE 建议，注明证据不足 | 自动删除/改写 Skill、降低科研规则、真实数据/正式训练、为评测安装新框架 |
| experiment-audit | 新 completion/epoch/summary 身份出现后 | 指定 run 目录的 JSON、配置、seed、provenance 与 checkpoint 文件元数据 | run id/output root；完整 seed/epoch 序列；配置与 frontend 身份；checkpoint/report 配对及存在性；validation summary 使用仓库聚合；formal/exploratory 标记；test_split_accessed=false | 完整/缺失/矛盾清单；区分观察证据与未验证字段；陈旧 summary 用最终 completion 交叉核对 | 加载 checkpoint、读 test split、重跑训练、补造 provenance、覆盖/提升探索产物、自动推进 phase |

experiment-audit 若需核对 checkpoint SHA，应另行明确读取范围；仅检查存在性
时不得声称内容完整性已验证。发现 formal/exploratory 字段缺失时报告缺失，
不能从目录名称推断其正式性。初期只做这四类报告，无总控 Agent。

## Legacy tracked artifact exception

`artifacts/formal_runs/phase1-cam16-baseline-b32-v2/` 下历史已跟踪的 22 个
`epoch-*.pt` 是当前正式 baseline 的训练证据，予以保留。这是一次历史遗留例外，
不表示允许提交任何新 checkpoint；`.gitignore` 继续阻止新增 checkpoint。
本仓库不改写 Git 历史。若以后迁移到 GitHub Release、对象存储或归档 tag，须先
由人工批准并验证外部副本、引用与恢复路径，之后另行决定是否从当前分支移除。

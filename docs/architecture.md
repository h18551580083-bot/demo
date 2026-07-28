# 架构说明

数据流为：`DatasetRegistry → InputAdapter → OpticalBackbone → SquareLawDetector →
LightweightBackend → Evaluation/Audit`。

`OpticalBackbone` 的核是 buffer 而非可训练参数。`digital_ideal` 使用空间域卷积，
`fourier_4f` 使用频域相乘，两者共享 KernelSpec、通道顺序和探测器。迁移阶段必须
加载 P4 锁定前端清单并比较 `kernel_bank_hash`；目标数据不得反向参与核族、尺度、
方向数、后处理或后端预算选择。

## 阶段边界

- P0：锁定研究与配置契约；当前仓库提供模板，不替代研究者对 TBD 项的决策。
- P1：验证 CAM16 数字理想固定特征和等容量基线。
- P2：加入采样、孔径、相位-only、噪声与量化等 4f 非理想因素。
- P3/P4：消融、多 seed、全量训练与 frozen test 最终评估。
- P5：使用完全相同前端做线性探针、轻量后端和少样本迁移。

## 产物规范

每次正式运行应创建唯一的 `artifacts/<run_id>`，保存解析后的锁定配置、代码版本、
清单哈希、核哈希、逐样本预测、指标、模型、门禁结论和 SHA-256 清单。失败运行同样
保留状态与原因。


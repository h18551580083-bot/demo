# CAM16 固定小波散射式光电分类系统

面向 CAM16 patch 二分类与跨病理迁移研究的可复现工程骨架。当前版本落实 P0/P1
基础设施：数据注册与患者/WSI 级隔离审计、确定性固定小波核、数字理想与 4f 等效
计算、平方律探测、轻量电子后端、配置锁定和哈希审计。

> 本项目只实现数字仿真与 4f 等效仿真，不代表实际光路搭建、器件制造或临床验证。

## 快速开始

```powershell
python -m pip install -e ".[dev]"
cam16-wavelet validate-dataset --config configs/datasets/cam16.yaml
cam16-wavelet smoke-test --config configs/experiments/cam16_p1_smoke.yaml
pytest
```

默认数据路径为 `E:/cg/cam16_patch`，也可在 YAML 中修改。训练与正式实验不得使用
test split 做结构或超参数选择。
`fixed_wavelet_v1.yaml` 中的数值仅用于验证工程链路，状态为 provisional；正式 P1
前必须完成 P0 决策并生成独立的锁定配置。CLI 可用 `--output artifacts/<run_id>`
写入配置哈希、代码版本和运行摘要。

## 目录

- `src/cam16_wavelet/data`：数据契约、注册、清单审计与输入适配
- `src/cam16_wavelet/optics`：核库、数字/4f 前端与探测器
- `src/cam16_wavelet/models`：受控容量的电子后端
- `src/cam16_wavelet/audit`：配置、数据与前端哈希
- `configs`：数据、前端和实验配置
- `tests`：契约、数值等价性与泄漏测试
- `artifacts`：运行产物（不纳入版本控制）
- `docs`：开发路线与决策记录

详见 [docs/architecture.md](docs/architecture.md) 与
[docs/p0_decisions.md](docs/p0_decisions.md)。

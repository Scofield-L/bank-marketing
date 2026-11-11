# Bank Marketing 数据挖掘大作业

## 项目简介

基于 **UCI Bank Marketing Data Set** 的银行营销预测任务：根据客户特征预测客户是否会订阅银行定期存款（二分类）。

- **数据集**：Bank Marketing Data Set（已核验 bank-full.csv：45,211 条，16 个特征 + 1 个目标变量 `y`）
- **数据来源**：UCI Machine Learning Repository
- **课程**：数据挖掘（大作业）
- **提交截止**：2026-11-30（延河课堂）

## 小组成员

| 角色 | 姓名 | 学号 | 主责 |
|------|------|------|------|
| 成员 A（数据/工程） | （待填） | （待填） | 数据预处理 + 代码整合 |
| 成员 B（算法/建模） | （待填） | （待填） | EDA + 建模 + 调参 |
| 组长 C（统筹/评估/汇报） | （待填） | （待填） | 评估 + 结果可视化 + 报告 + PPT/视频 |

## 目录结构

```
bank-marketing/
├── data/          # 数据集（原始 + 预处理后；大文件不入库，见 data/README.md）
├── code/          # 代码：01_preprocessing / 02_eda / 03_modeling / 04_evaluation
├── figures/       # 图表（EDA 图、ROC/PR、混淆矩阵、模型对比等）
├── docs/          # 文档：报告草稿、分工计划、数据集清单、预处理说明
├── output/        # 最终提交物（报告/PPT/视频；大二进制另存云盘，不入库）
├── .gitignore     # 忽略大数据集、视频、Python 缓存等
└── README.md      # 本文件
```

## 协作约定

1. **每人只改自己主责的文件**，改前在群里说一声。
2. **报告/PPT 在线协同编辑**（腾讯文档/石墨文档），定稿后导出放入 `output/`（或另存云盘）。
3. **大数据集、视频不入库**（已在 `.gitignore` 中忽略），统一放共享云盘。
4. **代码提交**：每人建独立分支，组长负责合并到 `master`。
5. 提交信息用清晰的中文说明，例如 `feat: 完成数据预处理脚本`。

## 快速开始

使用 Python 3.12；已在 Windows / Python 3.12.7 验证 A 的流程。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python code/00_download_data.py
.\.venv\Scripts\python code/01_preprocessing.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

若 ensurepip 失败，可用 `python -m pip --python .venv/Scripts/python.exe install -r requirements.txt` 向该环境安装。镜像不可用时可添加 `--index-url https://pypi.org/simple`。

已有原始文件时跳过下载，下载器拒绝覆盖。原始数据和产物不入 Git。默认输出 data/processed/，包括保留原始行号的 train/test.csv、矩阵、预处理器及含校验值和特征名的 manifest.json。可通过 --input、--output、--seed 指定配置，默认路径与工作目录无关。

默认排除 duration。通话后对照运行 `python code/01_preprocessing.py --include-duration --output data/processed_with_duration`。

B 调参应使用原始 train.csv 和 code/bank_preprocessing.py 的 Pipeline 接口，保证预处理和采样在训练折内完成。详见 [A 交接与复现记录](docs/A/交接与复现记录.md)、[数据说明](docs/A/数据说明.md) 和 [预处理说明](docs/A/数据预处理说明.md)。

## B 的实验与 A 的交接核验

已整合 B 的 EDA、建模与调参代码。主实验为已安排电话的拨号前评分，假设 day/month、contact、campaign 已知，仅排除 duration。推荐 xgb_none_tuned，保留原始分布和 pdays=-1；duration_control 为独立通话后对照。

```powershell
.\.venv\Scripts\python code/02_eda.py
.\.venv\Scripts\python code/03_modeling.py --jobs 4
.\.venv\Scripts\python code/05_verify_handoff.py --refit
```

建模命令会重跑完整五折比较和网格调参，并写入 data/modeling/；仅核对已交付结果时直接运行 05_verify_handoff.py 即可，不加 --refit 不重训。核验报告在 data/integration/handoff_check.json，模型二进制按仓库规则不入库。B 的结果表已入库，原始 CSV 和 processed/ 仍需本地准备。

完整接口见 [B 的交接文档](docs/B/建模与评估交接.md)。C 的 04_evaluation.py 尚待交付，完整项目端到端复现尚未完成。

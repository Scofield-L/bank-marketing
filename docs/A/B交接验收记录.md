# A 对 B 的交接验收（2026-11-11）

核验基于主分支 642f909，在 A 分支进行。Windows / Python 3.12.7，使用此前 A 的独立 .venv 并补装共享 requirements；不是重新创建的全新环境。没有重新执行全部模型比较及网格搜索。

## 场景和配置

主实验为已安排电话的拨号前评分，假设 day/month、contact、campaign 已知，只排除 duration。保留 pdays=-1，不采样；bank-full、分层 80/20、seed=42、yes=1 不变。原始最后联系字段不能证明历史时点可用性，必须披露该假设。排期前圈选与通话后评分均不属于主实验。

## 实际核验结果

| 检查 | 结果 |
|---|---|
| 原始数据 SHA256 与 B manifest | 一致 |
| A/B train.csv、test.csv 的 SHA256 | 均一致 |
| 从原始文件重新划分并比对输入及标签 | 一致 |
| 测试预测表 | 19 份，每份 9043 行，ID、标签、行序均一致 |
| 预测概率与默认标签 | 无空值、概率有限且在 [0,1]，标签对应阈值 0.5 |
| recommended_oof.csv | 36168 行，与训练 ID、标签、行序一致 |
| 推荐方案选型 | 不含 duration 的实验中五折平均 AP 最高，xgb_none_tuned |
| 推荐模型重训后的测试概率 | 与 B 表一致（rtol=1e-6、atol=1e-7） |
| 推荐模型重训五折 OOF 概率 | 与 B 表一致（相同容差） |
| 推荐模型独立五折平均 AP | 0.44265713147628427，与 B 汇总一致 |
| 测试 AP 独立重算 | 0.460443577352844 |
| 测试 ROC AUC 独立重算 | 0.8038358192878188 |
| 重训 Pipeline 保存后重新加载 | 测试概率逐项一致 |
| EDA 重跑 | 6 份统计表与 B 交付一致，图表生成成功 |
| 原有预处理及对照实验接口测试 | 5 项通过 |
| pip check | 无依赖冲突 |

核心依赖：NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.5.2、imbalanced-learn 0.12.4、XGBoost 2.1.4、Matplotlib 3.10.0。C 预留的 seaborn/lightgbm 尚未锁版本，不把其安装成功当作 C 的兼容性验收。

## 复现命令

```powershell
python -m pip --python .venv/Scripts/python.exe install -r requirements.txt
.\.venv\Scripts\python code/05_verify_handoff.py --refit
.\.venv\Scripts\python code/02_eda.py --figures data/integration/figures --tables data/integration/eda
.\.venv\Scripts\python -m unittest discover -s tests -v
python -m pip --python .venv/Scripts/python.exe check
```

核验报告、推荐模型与 EDA 临时输出在 data/integration/，均不入库、不覆盖 B 的产物。全量实验命令见 B 交接文档；本次只重训推荐方案，不根据测试集再次选型或调整阈值。

## 交给 C 的材料与剩余工作

从 cv_summary.csv 定位预测；主实验比较排除 duration_control。阈值只用 recommended_oof.csv 选择，再固定应用到测试集。AP 和梯形 PR-AUC 必须区分。等待 C 提供评估脚本及运行说明后，A 再完成整个项目的新环境复现和最终代码包。本次没有向其他成员发送消息或交付包，文档已就绪待共享。

## AI 与人工核验边界

Codex 辅助同步代码、审阅训练折处理方式、编写核验入口、执行推荐模型及 OOF 重训、复算指标并整理记录。上表为实际工具运行结果，不代表 A 已完成人工逐项审核。A/B 按本次交接统一输入、标签和预测时点；C 的最终验收、阈值与业务权衡仍由 C 完成。

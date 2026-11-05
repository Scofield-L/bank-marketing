# data/ 目录说明

本目录用于存放数据集，但**数据文件本身不纳入 Git 版本管理**（见根目录 `.gitignore`），请从以下地址下载后放入本目录。

## 下载地址

**UCI Bank Marketing Data Set**

- 数据集主页：https://archive.ics.uci.edu/ml/datasets/bank+marketing
- 完整版数据（含额外特征，约 41,188 条）：
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip`
- 精简版数据（约 45,211 条，20 个特征）：
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip`

> 说明：`bank.zip`（bank-full.csv）为精简版，行数更多（45,211 条），满足 ≥5000 条要求；`bank-additional.zip`（bank-additional-full.csv）特征更丰富。二选一即可，建议先用 `bank-full.csv`。

## 下载后放置方式

```
data/
├── README.md          # 本文件（已入库）
└── bank-full.csv      # 下载后放这里（不入库，已被 .gitignore 忽略）
```

## 字段说明（主要）

| 字段 | 含义 |
|------|------|
| age | 年龄 |
| job | 职业 |
| marital | 婚姻状况 |
| education | 教育程度 |
| default | 是否有信用违约 |
| balance | 年均余额 |
| housing | 是否有住房贷款 |
| loan | 是否有个人贷款 |
| contact | 联系方式 |
| day / month | 最后联系日期 |
| duration | 最后联系通话时长（秒） |
| campaign | 本次营销联系次数 |
| pdays | 距上次联系的天数（999 表示从未联系） |
| previous | 本次营销前的联系次数 |
| poutcome | 上次营销结果 |
| y | **目标变量**：是否订阅定期存款（yes/no） |

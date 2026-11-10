# data/ 目录说明

默认使用 UCI bank-full.csv：45,211 行、16 个输入字段和目标 y；分号分隔。原始文件及产物被 Git 忽略。

来源：https://archive.ics.uci.edu/dataset/222/bank+marketing

下载：https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

运行 `python code/00_download_data.py` 下载 bank-full.csv、bank-names.txt，记录来源与校验值；拒绝覆盖已有文件。也可手动下载并将这两个文件放在本目录。

bank.csv 为 4,521 行抽样，不满足课程规模要求；bank-additional-full.csv 为 41,188 行、20 个输入字段的不同版本。本项目拒绝混用结构。

bank-full 的 pdays=-1 表示此前未联系；additional 使用 999，不能套用。

运行 `python code/01_preprocessing.py` 生成 processed/train.csv、test.csv、matrices.npz、preprocessor.joblib 和 manifest.json。默认排除 duration，分层 80/20、seed=42，原始文件保持不变。

详见 [数据说明](../docs/A/数据说明.md)、[交接记录](../docs/A/交接与复现记录.md)。

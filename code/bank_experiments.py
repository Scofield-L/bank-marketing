"""B 的数据接口与可随模型保存的转换器。"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from bank_preprocessing import CATEGORICAL, NUMERIC, load_data


def read_splits(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['include_duration'] or manifest['label_mapping'] != {'no': 0, 'yes': 1}:
        raise ValueError('请提供 A 的通话前数据目录（不含 duration，no=0、yes=1）')
    expected = set(NUMERIC + CATEGORICAL + ['y']) - {'duration'}
    splits = []
    for name in ('train', 'test'):
        frame = pd.read_csv(directory / f'{name}.csv', index_col='sample_id')
        if (set(frame.columns) != expected or not frame.index.is_unique
                or not pd.api.types.is_integer_dtype(frame.index) or frame.index.hasnans
                or set(frame.y.unique()) != {0, 1}
                or len(frame) != manifest[f'{name}_rows']
                or frame.y.sum() != manifest[f'{name}_positive']):
            raise ValueError(f'{name}.csv 的字段、样本 ID 或标签与 manifest 不符')
        splits.append(frame)
    train, test = splits
    if set(train.index).intersection(test.index):
        raise ValueError('训练集与测试集的 sample_id 存在重叠')
    if set(train.index).union(test.index) != set(range(manifest['rows'])):
        raise ValueError('训练集与测试集的 sample_id 未覆盖原始行号')
    return train, test, manifest


def original_data(path, manifest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('原始数据的 SHA256 与 A 的 manifest 不一致')
    return load_data(path)


def add_duration(frame, raw):
    original = raw.loc[frame.index]
    if not original.y.map({'no': 0, 'yes': 1}).eq(frame.y).all():
        raise ValueError('原始数据与预处理表的标签未按 sample_id 对齐')
    return frame.assign(duration=original.duration)


def pdays_as_missing(frame):
    """pdays=-1 表示未联系；保留 previous，pdays 缺失由训练折中位数填补。"""
    return frame.assign(pdays=frame.pdays.mask(frame.pdays.eq(-1)))


class BalancedXGBClassifier(XGBClassifier):
    """在每次 fit 时只用当前训练标签计算平衡权重。"""

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, sample_weight=compute_sample_weight('balanced', y), **kwargs)

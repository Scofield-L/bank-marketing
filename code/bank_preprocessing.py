"""Bank-full 数据接口；所有有状态处理必须放入交叉验证 Pipeline。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
SOURCE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'


def load_data(path=ROOT / 'data/bank-full.csv'):
    """仅接受 bank-full schema；原始行号作为稳定 sample_id。"""
    frame = pd.read_csv(path, sep=';')
    if set(frame.columns) != set(NUMERIC + CATEGORICAL + ['y']):
        raise ValueError('需要 bank-full.csv 的 16 个输入字段和 y；不支持混用 additional 版本')
    if len(frame) < 5000:
        raise ValueError('样本不足 5000，请使用完整文件')
    if frame.y.isna().any() or not set(frame.y.unique()) <= {'yes', 'no'} or frame.y.nunique() != 2:
        raise ValueError('目标必须同时包含 yes/no，且无缺失')
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if np.isinf(frame[col]).any():
            raise ValueError(f'{col} 包含无穷值')
    bounds = {'age': (1, None), 'day': (1, 31), 'duration': (0, None),
              'campaign': (1, None), 'pdays': (-1, None), 'previous': (0, None)}
    for col, (low, high) in bounds.items():
        if (frame[col] < low).any() or (high is not None and (frame[col] > high).any()):
            raise ValueError(f'{col} 超出允许范围')
    frame.index.name = 'sample_id'
    return frame


def split_data(frame, seed=42, test_size=0.2, include_duration=False):
    X = frame.drop(columns=['y'] + ([] if include_duration else ['duration']))
    y = frame.y.map({'no': 0, 'yes': 1}).rename('y')
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _feature_columns(include_duration=False, drop_features=None):
    if isinstance(drop_features, str):
        raise ValueError('drop_features 应为字段名列表，不能是字符串')
    dropped = set(drop_features or [])
    unknown = dropped - set(NUMERIC + CATEGORICAL)
    if unknown:
        raise ValueError(f'未知的 drop_features 字段：{sorted(unknown)}')
    if not include_duration:
        dropped.add('duration')
    numeric = [c for c in NUMERIC if c not in dropped]
    categorical = [c for c in CATEGORICAL if c not in dropped]
    if not numeric and not categorical:
        raise ValueError('不能排除所有输入字段')
    return numeric, categorical


def make_preprocessor(include_duration=False, scale=True, drop_features=None):
    """排除指定数值/类别字段；drop_features 优先于 include_duration。"""
    numeric, categorical = _feature_columns(include_duration, drop_features)
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=True)


def make_model_pipeline(estimator, strategy='none', include_duration=False, scale=True, seed=42,
                        drop_features=None):
    """strategy: none / class_weight / undersample / smote (= SMOTENC)。

    传原始 X 给 GridSearchCV；禁止使用导出的矩阵做交叉验证。
    SMOTENC 在填补、数值缩放之后、One-Hot 之前运行，保留离散类别。
    drop_features 同时排除数值与类别字段；输入表可保留或提前删除这些列。
    """
    from sklearn.base import clone
    from imblearn.pipeline import Pipeline as SamplingPipeline
    from imblearn.over_sampling import SMOTENC
    from imblearn.under_sampling import RandomUnderSampler
    estimator = clone(estimator)
    preprocessor = make_preprocessor(include_duration, scale, drop_features)
    if strategy == 'class_weight':
        if 'class_weight' not in estimator.get_params():
            raise ValueError('该模型不支持 class_weight，请由 B 配置专用权重参数')
        estimator.set_params(class_weight='balanced')
    if strategy in ('none', 'class_weight'):
        return SamplingPipeline([('preprocess', preprocessor), ('model', estimator)])
    if strategy == 'undersample':
        return SamplingPipeline([('preprocess', preprocessor),
                                 ('sample', RandomUnderSampler(random_state=seed)), ('model', estimator)])
    if strategy != 'smote':
        raise ValueError(f'未知采样策略：{strategy}')
    numeric, categorical = _feature_columns(include_duration, drop_features)
    if not numeric or not categorical:
        raise ValueError('smote 使用 SMOTENC，需要同时保留数值字段和类别字段')
    # 数值管线复用；类别暂不编码，以便 SMOTENC 识别类别。
    before = ColumnTransformer([
        ('numeric', preprocessor.transformers[0][1], numeric),
        ('categorical', SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True), categorical),
    ])
    cat_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    after = ColumnTransformer([
        ('numeric', 'passthrough', list(range(len(numeric)))),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_indices),
    ])
    return SamplingPipeline([('prepare', before),
                             ('sample', SMOTENC(categorical_features=cat_indices, random_state=seed)),
                             ('encode', after), ('model', estimator)])


def quality_report(frame, path):
    return {
        'source_url': SOURCE, 'file': Path(path).name,
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'rows': len(frame), 'columns': len(frame.columns), 'separator': ';',
        'target': 'y', 'label_mapping': {'no': 0, 'yes': 1},
        'duplicates': int(frame.duplicated().sum()),
        'null_counts': frame.isna().sum().to_dict(),
        'unknown_counts': {c: int(frame[c].eq('unknown').sum()) for c in CATEGORICAL},
        'category_values': {c: sorted(frame[c].dropna().unique().tolist()) for c in CATEGORICAL},
        'numeric_summary': frame[NUMERIC].describe().to_dict(),
        'target_counts': frame.y.value_counts().to_dict(),
        'positive_rate': float(frame.y.eq('yes').mean()),
        'pdays_never_contacted': int(frame.pdays.eq(-1).sum()),
    }


def run(path, output, seed=42, include_duration=False):
    import joblib
    from datetime import datetime, timezone
    frame = load_data(path)
    X_train, X_test, y_train, y_test = split_data(frame, seed, include_duration=include_duration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        X.join(y).to_csv(output / f'{name}.csv', index=True)
    processor = make_preprocessor(include_duration)
    train_matrix = processor.fit_transform(X_train)
    test_matrix = processor.transform(X_test)
    np.savez_compressed(output / 'matrices.npz', X_train=train_matrix, X_test=test_matrix,
                        y_train=y_train.to_numpy(), y_test=y_test.to_numpy(),
                        train_ids=X_train.index.to_numpy(), test_ids=X_test.index.to_numpy())
    joblib.dump(processor, output / 'preprocessor.joblib')
    report = quality_report(frame, path)
    report.update({'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
                   'include_duration': include_duration, 'test_size': 0.2,
                   'train_rows': len(X_train), 'test_rows': len(X_test),
                   'train_positive': int(y_train.sum()), 'test_positive': int(y_test.sum()),
                   'feature_names': processor.get_feature_names_out().tolist()})
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(frame)} rows -> train {train_matrix.shape}, test {test_matrix.shape}; {output}')
    return report

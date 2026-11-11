"""A 对 B 交付的整合核验；不使用测试集选模型或阈值。"""
import argparse
import hashlib
import importlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from bank_preprocessing import ROOT, split_data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_table(path, expected, threshold):
    table = pd.read_csv(path)
    require(list(table.columns) == ['sample_id', 'y_true', 'y_pred', 'y_probability'], f'{path}: 列不匹配')
    require(not table.isna().any().any(), f'{path}: 包含空值')
    require(table.sample_id.is_unique, f'{path}: ID 重复')
    require(np.array_equal(table.sample_id, expected.index), f'{path}: ID 或行序不匹配')
    require(np.array_equal(table.y_true, expected.y), f'{path}: 标签不匹配')
    require(np.isfinite(table.y_probability).all() and table.y_probability.between(0, 1).all(),
            f'{path}: 非法概率')
    require(np.array_equal(table.y_pred, (table.y_probability >= threshold).astype(int)),
            f'{path}: 标签与阈值不一致')
    return table


def run(processed, modeling, output, refit=False):
    from bank_experiments import read_splits, original_data
    train, test, manifest = read_splits(processed)
    raw = original_data(ROOT / 'data/bank-full.csv', manifest)
    X_train, X_test, y_train, y_test = split_data(raw, manifest['seed'], manifest['test_size'])
    pd.testing.assert_frame_equal(train, X_train.join(y_train))
    pd.testing.assert_frame_equal(test, X_test.join(y_test))
    b = json.loads((modeling / 'manifest.json').read_text(encoding='utf-8'))
    require(b['data_sha256'] == manifest['sha256'], 'A/B 原始数据校验值不一致')
    require(b['seed'] == 42 and b['positive_label'] == 1 and b['cv_folds'] == 5
            and b['selection_metric'] == 'average_precision', '实验配置不符合交接约定')
    for name in ('train', 'test'):
        require(hashlib.sha256((processed / f'{name}.csv').read_bytes()).hexdigest() == b['split_sha256'][name],
                f'{name}: A/B 划分文件校验值不一致')
    summary = pd.read_csv(modeling / 'cv_summary.csv')
    require(summary.experiment.is_unique, '实验名重复')
    tables = {}
    for row in summary.itertuples():
        tables[row.experiment] = verify_table(modeling / row.predictions, test, b['threshold'])
    verify_table(modeling / b['recommended_oof'], train, b['threshold'])
    main = summary.loc[~summary.include_duration].sort_values(
        ['cv_mean_ap', 'cv_mean_roc_auc'], ascending=False, kind='stable')
    require(main.iloc[0].experiment == b['recommended'], '推荐方案不是通话前五折最优方案')
    table = tables[b['recommended']]
    metrics = {'ap': average_precision_score(table.y_true, table.y_probability),
               'roc_auc': roc_auc_score(table.y_true, table.y_probability)}
    report = {
        'python': platform.python_version(), 'data_sha256': manifest['sha256'],
        'versions': {name: importlib.import_module(name).__version__
                     for name in ('numpy', 'pandas', 'sklearn', 'imblearn', 'xgboost')},
        'split_sha256': b['split_sha256'], 'prediction_tables_verified': len(tables),
        'test_rows_per_table': len(test), 'oof_rows_verified': len(train),
        'recommended': b['recommended'], 'cv_ap': float(main.iloc[0].cv_mean_ap),
        'test_metrics_recomputed': metrics, 'refit_verified': False,
    }
    if refit:
        import joblib
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
        module = importlib.import_module('03_modeling')
        row = main.iloc[0]
        experiment = module.Experiment(row.experiment, row.model, row.strategy,
                                       bool(row.mask_pdays), bool(row.include_duration))
        pipeline = module.build_pipeline(experiment, b['seed']).set_params(**json.loads(row.params))
        X, y = train.drop(columns='y'), train.y
        pipeline.fit(X, y)
        probabilities = pipeline.predict_proba(test.drop(columns='y'))[:, list(pipeline.classes_).index(1)]
        np.testing.assert_allclose(probabilities, table.y_probability, rtol=1e-6, atol=1e-7)
        oof = cross_val_predict(pipeline, X, y, method='predict_proba',
                                cv=StratifiedKFold(5, shuffle=True, random_state=b['seed']))[:, 1]
        expected_oof = pd.read_csv(modeling / b['recommended_oof'])
        np.testing.assert_allclose(oof, expected_oof.y_probability, rtol=1e-6, atol=1e-7)
        cv_scores = cross_val_score(pipeline, X, y, scoring='average_precision',
                                    cv=StratifiedKFold(5, shuffle=True, random_state=b['seed']))
        np.testing.assert_allclose(cv_scores.mean(), row.cv_mean_ap, rtol=1e-6, atol=1e-7)
        report['refit_cv_ap_mean'] = float(cv_scores.mean())
        output.parent.mkdir(parents=True, exist_ok=True)
        model_path = output.parent / 'recommended_refit.joblib'
        joblib.dump(pipeline, model_path)
        reloaded = joblib.load(model_path)
        np.testing.assert_array_equal(probabilities, reloaded.predict_proba(test.drop(columns='y'))[:, 1])
        report['refit_verified'] = True
        report['oof_refit_verified'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--modeling', type=Path, default=ROOT / 'data/modeling')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/integration/handoff_check.json')
    parser.add_argument('--refit', action='store_true', help='重训推荐方案及五折 OOF，并验证序列化')
    args = parser.parse_args()
    run(args.processed, args.modeling, args.output, args.refit)

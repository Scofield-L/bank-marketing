"""同一划分上比较模型和采样，按训练集五折 AP 选型和调参。"""
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from bank_experiments import (
    BalancedXGBClassifier, add_duration, original_data, pdays_as_missing, read_splits,
)
from bank_preprocessing import CATEGORICAL, NUMERIC, ROOT, make_model_pipeline


STRATEGIES = ('none', 'class_weight', 'undersample', 'smote')
SCORING = {
    'ap': 'average_precision',
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': make_scorer(f1_score, zero_division=0),
}
GRIDS = {
    'lr': {'model__C': [1.0, 0.1, 10.0]},
    'dt': {'model__max_depth': [None, 6, 12], 'model__min_samples_leaf': [1, 20]},
    'rf': {'model__max_depth': [None, 12, 24], 'model__min_samples_leaf': [1, 5]},
    'xgb': {'model__max_depth': [6, 3, 4], 'model__n_estimators': [100, 200]},
}
THRESHOLD = 0.5


@dataclass
class Experiment:
    name: str
    model: str
    strategy: str
    mask_pdays: bool = False
    include_duration: bool = False


def build_pipeline(experiment, seed):
    estimators = {
        'lr': LogisticRegression(max_iter=1500, random_state=seed),
        'dt': DecisionTreeClassifier(random_state=seed),
        'rf': RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed),
        'xgb': xgboost.XGBClassifier(
            n_estimators=100, max_depth=6, tree_method='hist', eval_metric='logloss',
            n_jobs=1, random_state=seed,
        ),
    }
    estimator = estimators[experiment.model]
    strategy = experiment.strategy
    if experiment.model == 'xgb' and strategy == 'class_weight':
        estimator = BalancedXGBClassifier(**estimator.get_params())
        strategy = 'none'
    pipeline = make_model_pipeline(
        estimator, strategy=strategy, include_duration=experiment.include_duration,
        scale=experiment.model == 'lr' or experiment.strategy == 'smote', seed=seed,
    )
    if experiment.mask_pdays:
        pipeline.steps.insert(0, ('pdays', FunctionTransformer(pdays_as_missing)))
    return pipeline


def select(experiment, train, grid, cv, seed, jobs):
    print(f'CV {experiment.name}', flush=True)
    search = GridSearchCV(
        build_pipeline(experiment, seed), grid, scoring=SCORING,
        refit=False, cv=cv, n_jobs=jobs, error_score='raise',
    )
    search.fit(train.drop(columns='y'), train.y)
    candidates = pd.DataFrame(search.cv_results_)
    candidates = candidates.drop(columns=candidates.filter(regex=r'^param_').columns)
    candidates.insert(0, 'experiment', experiment.name)
    best = candidates.sort_values(
        ['mean_test_ap', 'mean_test_roc_auc'], ascending=False, kind='stable',
    ).iloc[0]
    print(f'  AP={best.mean_test_ap:.4f} ROC AUC={best.mean_test_roc_auc:.4f}', flush=True)
    return best, candidates


def save_predictions(frame, probabilities, path):
    pd.DataFrame({
        'sample_id': frame.index,
        'y_true': frame.y.to_numpy(),
        'y_pred': (probabilities >= THRESHOLD).astype(int),
        'y_probability': probabilities,
    }).to_csv(path, index=False)


def save_importance(pipeline, experiment, path):
    if experiment.strategy == 'smote':
        columns = [name for name in NUMERIC if experiment.include_duration or name != 'duration']
        features = pipeline.named_steps['encode'].get_feature_names_out(columns + CATEGORICAL)
    else:
        features = pipeline.named_steps['preprocess'].get_feature_names_out()
    model = pipeline.named_steps['model']
    if experiment.model == 'lr':
        scores = model.coef_[0]
        kind = 'signed_coefficient'
    else:
        scores = model.feature_importances_
        kind = 'gain' if experiment.model == 'xgb' else 'impurity_decrease'
    pd.DataFrame({
        'feature': features, 'importance': scores, 'importance_type': kind,
    }).sort_values('importance', key=lambda values: values.abs(), ascending=False).to_csv(path, index=False)


def run(processed, raw_path, output, jobs):
    train, test, manifest = read_splits(processed)
    raw = original_data(raw_path, manifest)
    seed = manifest['seed']
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    experiments = []
    all_candidates = []

    def evaluate(experiment, grid, frame=train):
        best, candidates = select(experiment, frame, grid, cv, seed, jobs)
        experiments.append((experiment, best))
        all_candidates.append(candidates)
        return experiments[-1]

    for model in GRIDS:
        for strategy in STRATEGIES:
            evaluate(Experiment(f'{model}_{strategy}', model, strategy), {})

    def rank(result):
        return result[1].mean_test_ap, result[1].mean_test_roc_auc

    initial, initial_scores = max(experiments, key=rank)
    feature_check = evaluate(replace(initial, name=f'{initial.name}_pdays_missing', mask_pdays=True), {})
    # AP、ROC AUC 都相等时保留原始特征。
    selected, _ = max([(initial, initial_scores), feature_check], key=rank)
    tuned, _ = evaluate(replace(selected, name=f'{selected.name}_tuned'), GRIDS[selected.model])
    recommended, recommended_scores = max(experiments, key=rank)

    duration_train = add_duration(train, raw)
    duration_test = add_duration(test, raw)
    control = replace(recommended, name='duration_control', include_duration=True)
    evaluate(control, {key: [value] for key, value in recommended_scores.params.items()}, duration_train)

    output.mkdir(parents=True, exist_ok=True)
    for directory in ('predictions', 'importance', 'models'):
        (output / directory).mkdir(exist_ok=True)
    rows = []
    for experiment, result in experiments:
        fit_data, test_data = (duration_train, duration_test) if experiment.include_duration else (train, test)
        pipeline = build_pipeline(experiment, seed).set_params(**result.params)
        pipeline.fit(fit_data.drop(columns='y'), fit_data.y)
        positive = list(pipeline.classes_).index(1)
        probabilities = pipeline.predict_proba(test_data.drop(columns='y'))[:, positive]
        paths = {
            'predictions': f'predictions/{experiment.name}.csv',
            'importance': f'importance/{experiment.name}.csv',
            'model_file': f'models/{experiment.name}.joblib',
        }
        save_predictions(test_data, probabilities, output / paths['predictions'])
        save_importance(pipeline, experiment, output / paths['importance'])
        joblib.dump(pipeline, output / paths['model_file'], compress=3)
        rows.append({
            'experiment': experiment.name, 'model': experiment.model, 'strategy': experiment.strategy,
            'mask_pdays': experiment.mask_pdays, 'include_duration': experiment.include_duration,
            **paths, 'params': json.dumps(result.params),
            **{f'cv_{stat}_{metric}': result[f'{stat}_test_{metric}']
               for metric in SCORING for stat in ('mean', 'std')},
        })
        print(f'Exported {experiment.name}', flush=True)

    pipeline = build_pipeline(recommended, seed).set_params(**recommended_scores.params)
    # 每行概率由不包含该行的训练折模型给出，供 C 在训练数据上选择阈值。
    oof = cross_val_predict(
        pipeline, train.drop(columns='y'), train.y, cv=cv, n_jobs=jobs, method='predict_proba',
    )[:, 1]
    save_predictions(train, oof, output / 'recommended_oof.csv')
    pd.DataFrame(rows).to_csv(output / 'cv_summary.csv', index=False)
    pd.concat(all_candidates, ignore_index=True).to_csv(output / 'cv_candidates.csv', index=False)
    run_manifest = {
        'data_sha256': manifest['sha256'],
        'split_sha256': {name: hashlib.sha256((processed / f'{name}.csv').read_bytes()).hexdigest()
                         for name in ('train', 'test')},
        'train_rows': len(train), 'test_rows': len(test), 'test_size': manifest['test_size'],
        'seed': seed, 'cv_folds': cv.n_splits, 'selection_metric': 'average_precision',
        'selection_tie_breaker': 'roc_auc', 'positive_label': 1, 'threshold': THRESHOLD,
        'recommended': recommended.name, 'initial_winner': initial.name, 'tuned': tuned.name,
        'duration_control': control.name, 'recommended_oof': 'recommended_oof.csv',
        'versions': {module.__name__: module.__version__
                     for module in (np, pd, sklearn, imblearn, xgboost, joblib)},
    }
    (output / 'manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')
    print(f'Selected {recommended.name}: AP={recommended_scores.mean_test_ap:.4f}; {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/bank-full.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/modeling')
    parser.add_argument('--jobs', type=int, default=1, help='交叉验证并行进程数')
    args = parser.parse_args()
    run(args.processed, args.raw, args.output, args.jobs)

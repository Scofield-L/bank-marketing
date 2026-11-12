"""04_evaluation.py — 评估 B 的建模结果（组长 C 主责）。

从 data/modeling/cv_summary.csv 定位各实验的测试集预测，计算留出测试集指标；
对推荐方案在训练集折外预测（recommended_oof.csv）上做阈值调优，再把固定阈值
应用到测试集；输出 ROC/PR 曲线、混淆矩阵、阈值曲线和模型对比图。

指标口径与 B 一致：AP 用 sklearn 的 average_precision_score，主指标为
AP / ROC AUC，辅以精确率、召回率、F1。概率对应正类 1（订阅）。

用法：
    python code/04_evaluation.py                    # 完整评估 + 出图
    python code/04_evaluation.py --no-plot          # 只算指标不出图
    python code/04_evaluation.py --criterion f1     # 阈值准则 f1 或 target_recall
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / 'data' / 'modeling'
PRED_DIR = MODELING / 'predictions'
FIG_DIR = ROOT / 'figures' / 'evaluation'
OUT_DIR = ROOT / 'data' / 'evaluation'

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 中文字体支持（Windows）；否则图中中文会显示为方框
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_manifest():
    with open(MODELING / 'manifest.json', encoding='utf-8') as f:
        return json.load(f)


def load_cv_summary():
    return pd.read_csv(MODELING / 'cv_summary.csv')


def compute_metrics(y_true, y_prob, threshold=0.5):
    """在给定阈值下计算各指标；AP/AUC 与阈值无关。"""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'average_precision': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold(oof_df, criterion='f1', target_recall=0.5):
    """在训练集折外概率上扫描阈值。

    criterion='f1'：最大化 F1；
    criterion='target_recall'：召回率不低于 target_recall 时最大化精确率。
    """
    y_true = oof_df.y_true.to_numpy()
    y_prob = oof_df.y_probability.to_numpy()
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        if criterion == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif criterion == 'target_recall':
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            score = prec if rec >= target_recall else -1.0
        else:
            raise ValueError(f'未知准则：{criterion}')
        if score > best_score:
            best_score, best_thr = score, thr
    return float(best_thr), float(best_score)


def evaluate_all_experiments(summary):
    """对每个实验的测试集预测计算指标，返回带 test 前缀的汇总表。"""
    rows = []
    for _, exp in summary.iterrows():
        pred_path = PRED_DIR / Path(exp['predictions']).name
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        m = compute_metrics(pred.y_true, pred.y_probability, threshold=0.5)
        rows.append({
            'experiment': exp['experiment'],
            'model': exp['model'],
            'strategy': exp['strategy'],
            'include_duration': exp['include_duration'],
            'test_ap': m['average_precision'],
            'test_roc_auc': m['roc_auc'],
            'test_accuracy': m['accuracy'],
            'test_precision': m['precision'],
            'test_recall': m['recall'],
            'test_f1': m['f1'],
        })
    return pd.DataFrame(rows)


def evaluate_recommended(summary, manifest, criterion):
    """对推荐方案做阈值调优，返回详细结果。"""
    rec_name = manifest['recommended']
    rec = summary[summary.experiment == rec_name].iloc[0]
    pred = pd.read_csv(PRED_DIR / Path(rec['predictions']).name)
    oof = pd.read_csv(MODELING / manifest['recommended_oof'])

    # 校验：预测表 y_true 与 OOF 的 sample_id 范围应分别为测试集/训练集
    assert set(pred.columns) == {'sample_id', 'y_true', 'y_pred', 'y_probability'}
    assert set(oof.columns) == {'sample_id', 'y_true', 'y_pred', 'y_probability'}

    best_thr, _ = find_best_threshold(oof, criterion=criterion)
    m_default = compute_metrics(pred.y_true, pred.y_probability, threshold=0.5)
    m_tuned = compute_metrics(pred.y_true, pred.y_probability, threshold=best_thr)
    cm = confusion_matrix(pred.y_true, (pred.y_probability >= best_thr).astype(int))

    result = {
        'recommended': rec_name,
        'params': rec['params'],
        'cv_mean_ap': float(rec['cv_mean_ap']),
        'cv_mean_roc_auc': float(rec['cv_mean_roc_auc']),
        'threshold_criterion': criterion,
        'tuned_threshold': best_thr,
        'test_default(0.5)': m_default,
        'test_tuned': m_tuned,
        'confusion_matrix_tuned': cm.tolist(),
    }
    return result, pred, oof


def plot_curves(pred, oof, result, fig_dir):
    """ROC、PR、阈值扫描、混淆矩阵、模型对比。"""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC / PR（测试集，含 duration 对照）
    dur = pd.read_csv(PRED_DIR / 'duration_control.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, df in [('推荐 xgb_none_tuned', pred), ('通话后对照 duration', dur)]:
        fpr, tpr, _ = roc_curve(df.y_true, df.y_probability)
        prec, rec, _ = precision_recall_curve(df.y_true, df.y_probability)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(df.y_true, df.y_probability):.3f})')
        axes[1].plot(rec, prec, label=f'{name} (AP={average_precision_score(df.y_true, df.y_probability):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'roc_pr_curves.png', dpi=150)
    plt.close(fig)

    # 2. 阈值扫描（训练集 OOF）
    y_true, y_prob = oof.y_true.to_numpy(), oof.y_probability.to_numpy()
    thrs = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thrs:
        p = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, p, zero_division=0))
        recs.append(recall_score(y_true, p, zero_division=0))
        f1s.append(f1_score(y_true, p, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, precs, label='Precision')
    ax.plot(thrs, recs, label='Recall')
    ax.plot(thrs, f1s, label='F1')
    ax.axvline(result['tuned_threshold'], color='r', ls='--',
               label=f"选定阈值 {result['tuned_threshold']:.2f}")
    ax.set(xlabel='Threshold', ylabel='Score', title='训练集折外阈值扫描')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'threshold_sweep.png', dpi=150)
    plt.close(fig)

    # 3. 混淆矩阵（测试集，选定阈值）
    cm = np.array(result['confusion_matrix_tuned'])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='预测', ylabel='实际',
           title=f"混淆矩阵（阈值 {result['tuned_threshold']:.2f}）")
    ax.set_xticklabels(['否(0)', '是(1)'])
    ax.set_yticklabels(['否(0)', '是(1)'])
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def plot_model_comparison(summary, test_df, fig_dir):
    """主实验各模型的测试集 AP 对比柱状图。"""
    main = test_df[~test_df.include_duration].sort_values('test_ap')
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#d62728' if e == 'xgb_none_tuned' else '#4c72b0' for e in main.experiment]
    ax.barh(main.experiment, main.test_ap, color=colors)
    ax.set(xlabel='测试集 Average Precision', title='主实验（通话前）各实验测试集 AP')
    ax.axvline(main.test_ap.max(), color='#d62728', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / 'model_comparison.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plot', action='store_true', help='只算指标不出图')
    parser.add_argument('--criterion', default='f1', choices=['f1', 'target_recall'],
                        help='阈值选择准则（默认 f1）')
    args = parser.parse_args()

    summary = load_cv_summary()
    manifest = load_manifest()

    # 1. 全部实验的测试集指标
    test_df = evaluate_all_experiments(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(OUT_DIR / 'test_metrics.csv', index=False)

    # 2. 推荐方案阈值调优
    result, pred, oof = evaluate_recommended(summary, manifest, args.criterion)
    (OUT_DIR / 'recommended_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    # 3. 打印摘要
    print('=' * 60)
    print(f"推荐方案：{result['recommended']}  {result['params']}")
    print(f"CV 五折：AP={result['cv_mean_ap']:.4f}  ROC AUC={result['cv_mean_roc_auc']:.4f}")
    print(f"测试集（阈值 0.5）：AP={result['test_default(0.5)']['average_precision']:.4f}  "
          f"AUC={result['test_default(0.5)']['roc_auc']:.4f}  "
          f"F1={result['test_default(0.5)']['f1']:.4f}")
    print(f"测试集（阈值 {result['tuned_threshold']:.2f}）："
          f"精确率={result['test_tuned']['precision']:.4f}  "
          f"召回率={result['test_tuned']['recall']:.4f}  "
          f"F1={result['test_tuned']['f1']:.4f}")
    print('混淆矩阵（行=实际，列=预测）：', result['confusion_matrix_tuned'])
    print('=' * 60)
    print('\n主实验测试集 AP 排名（前 5）：')
    print(test_df[~test_df.include_duration].nlargest(5, 'test_ap')
          [['experiment', 'test_ap', 'test_roc_auc']].to_string(index=False))
    print(f'\n结果已写入 {OUT_DIR}')

    # 4. 出图
    if HAS_PLOT and not args.no_plot:
        plot_curves(pred, oof, result, FIG_DIR)
        plot_model_comparison(summary, test_df, FIG_DIR)
        print(f'图已写入 {FIG_DIR}')
    elif not HAS_PLOT:
        print('（未安装 matplotlib，跳过出图）')


if __name__ == '__main__':
    main()

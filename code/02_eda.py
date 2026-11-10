"""对 A 的训练集生成 EDA 图和分组统计；duration 仅作通话后描述。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from bank_experiments import add_duration, original_data, read_splits
from bank_preprocessing import CATEGORICAL, NUMERIC, ROOT


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def group_rates(group, y):
    return y.groupby(group, observed=True, dropna=False).agg(
        samples='size', positives='sum', positive_rate='mean'
    ).rename_axis('group').reset_index()


def rate_plot(ax, table, title, baseline):
    ax.barh(table.group.astype(str), table.positive_rate, color='#226b83')
    ax.invert_yaxis()
    ax.axvline(baseline, color='#b94836', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    for position, row in enumerate(table.itertuples()):
        ax.text(min(row.positive_rate + .015, .82), position, f'n={row.samples:,}',
                va='center', fontsize=7)


def run(processed, raw_path, figures, tables):
    train, _, manifest = read_splits(processed)
    train = add_duration(train, original_data(raw_path, manifest))
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    y = train.y
    baseline = y.mean()
    counts = y.value_counts().reindex([0, 1]).rename_axis('y').rename('samples').to_frame()
    counts['share'] = counts.samples / len(y)
    counts.to_csv(tables / 'target_distribution.csv')
    fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
    bars = ax.bar(['No (0)', 'Yes (1)'], counts.samples, color=['#436886', '#c16e38'])
    ax.bar_label(bars, labels=[f'{n:,} ({p:.1%})' for n, p in zip(counts.samples, counts.share)])
    ax.set(ylim=(0, counts.samples.max() * 1.15), title='Training target distribution', ylabel='Samples')
    save_figure(fig, figures / 'target_distribution.png')

    numeric = train[NUMERIC].copy()
    numeric['pdays'] = numeric.pdays.mask(numeric.pdays.eq(-1))
    pd.DataFrame({
        'missing': train.isna().sum(),
        'unknown': train.eq('unknown').sum(),
    }).rename_axis('feature').to_csv(tables / 'data_quality.csv')
    numeric.describe().T.to_csv(tables / 'numeric_summary.csv')
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained')
    for name, ax in zip(NUMERIC, axes.flat):
        ax.hist(numeric[name].dropna(), bins=35, color='#226b83')
        ax.set_title({'pdays': 'pdays (previously contacted)',
                      'duration': 'duration (post-call)'}.get(name, name))
    axes.flat[-1].axis('off')
    fig.suptitle('Numeric features in the training split')
    save_figure(fig, figures / 'numeric_histograms.png')

    corr = numeric.assign(y=y).corr()
    corr.to_csv(tables / 'numeric_correlations.csv')
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    chart = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    for i, values in enumerate(corr.to_numpy()):
        for j, value in enumerate(values):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color='white' if abs(value) > .65 else 'black')
    fig.colorbar(chart, ax=ax)
    ax.set_title('Pearson correlation (duration is post-call)')
    save_figure(fig, figures / 'numeric_correlations.png')

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), layout='constrained')
    category_tables = []
    for name, ax in zip(CATEGORICAL, axes.flat):
        table = group_rates(train[name].fillna('(missing)'), y)
        rate_plot(ax, table, name, baseline)
        category_tables.append(table.assign(feature=name))
    pd.concat(category_tables, ignore_index=True).to_csv(tables / 'category_rates.csv', index=False)
    fig.suptitle('Subscription rate by category (dashed line: overall training rate)')
    save_figure(fig, figures / 'category_rates.png')

    bins = {
        'duration': [-1, 60, 120, 300, 600, float('inf')],
        'campaign': [0, 1, 2, 4, 8, float('inf')],
        'previous': [-1, 0, 1, 3, float('inf')],
        'pdays': [-2, -1, 30, 90, 180, float('inf')],
    }
    labels = {
        'duration': ['0-60', '61-120', '121-300', '301-600', '>600'],
        'campaign': ['1', '2', '3-4', '5-8', '>8'],
        'previous': ['0', '1', '2-3', '>3'],
        'pdays': ['Never (-1)', '0-30', '31-90', '91-180', '>180'],
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    key_tables = []
    for name, ax in zip(bins, axes.flat):
        groups = pd.cut(train[name], bins[name], labels=labels[name])
        table = group_rates(groups, y)
        rate_plot(ax, table, 'duration (post-call)' if name == 'duration' else name, baseline)
        key_tables.append(table.assign(feature=name))
    pd.concat(key_tables, ignore_index=True).to_csv(tables / 'key_feature_rates.csv', index=False)
    fig.suptitle('Subscription rate by feature range (dashed line: overall training rate)')
    save_figure(fig, figures / 'key_feature_rates.png')
    print(f'EDA: {len(train):,} training rows, {baseline:.2%} positive -> {figures} / {tables}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/bank-full.csv')
    parser.add_argument('--figures', type=Path, default=ROOT / 'figures/eda')
    parser.add_argument('--tables', type=Path, default=ROOT / 'data/eda')
    args = parser.parse_args()
    run(args.processed, args.raw, args.figures, args.tables)

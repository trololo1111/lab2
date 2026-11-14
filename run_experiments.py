from dataclasses import asdict
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pandas_tasks import analyze_titanic
from correlation_tasks import FEATURES, analyze_brain_correlations
from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput


def main():
    root = Path(__file__).resolve().parent
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    results = {
        'titanic': asdict(analyze_titanic(TitanicInput(str(root / 'titanic.csv')))),
        'brain': asdict(analyze_brain_correlations(BrainDataInput(str(root / 'brainsize.txt')))),
    }
    frame = pd.read_csv(root / 'brainsize.txt', sep='\t', na_values=['NA'])
    results['brain_missing_by_column'] = {key: int(value) for key, value in frame.isna().sum().items()}
    results['pair_counts'] = {
        gender: {feature: len(group[[feature, 'MRI_Count']].dropna()) for feature in FEATURES}
        for gender, group in frame.groupby('Gender')
    }
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
    for row, gender in enumerate(('Female', 'Male')):
        group = frame.loc[frame['Gender'].eq(gender)]
        for col, feature in enumerate(FEATURES):
            pair = group[[feature, 'MRI_Count']].dropna()
            ax = axes[row, col]
            sns.scatterplot(data=pair, x=feature, y='MRI_Count', ax=ax,
                            color='#247ba0' if row == 0 else '#cf5c36', s=45)
            r = pair[feature].corr(pair['MRI_Count'])
            ax.set_title(f'{gender}: r = {r:.3f}, n = {len(pair)}')
    fig.suptitle('Brain size: MRI_Count versus each feature, by gender', fontsize=17)
    fig.savefig(figures / 'brain_scatter.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, gender in zip(axes, ('Female', 'Male')):
        corr = frame.loc[frame['Gender'].eq(gender), [*FEATURES, 'MRI_Count']].corr(method='pearson')
        sns.heatmap(corr, ax=ax, annot=True, fmt='.3f', cmap='vlag',
                    vmin=-1, vmax=1, center=0, square=True)
        ax.set_title(f'{gender}: Pearson correlations')
    fig.savefig(figures / 'brain_heatmaps.png', dpi=160)
    plt.close(fig)
    (root / 'experiment_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

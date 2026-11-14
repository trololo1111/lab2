from __future__ import annotations

import math
import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ('FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height')


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    frame = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    men = frame.loc[frame['Gender'].eq('Male')]
    women = frame.loc[frame['Gender'].eq('Female')]

    def correlations(group: pd.DataFrame) -> dict[str, float]:
        return {
            feature: float(group[feature].corr(group['MRI_Count'], method='pearson'))
            for feature in FEATURES
        }

    men_correlations = correlations(men)
    women_correlations = correlations(women)
    candidates = [
        (feature, value)
        for group in (women_correlations, men_correlations)
        for feature, value in group.items()
        if math.isfinite(value)
    ]
    if not candidates:
        raise ValueError('Нет определённых корреляций для выбора strongest_mri_feature')
    strongest = max(candidates, key=lambda item: abs(item[1]))[0]
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest,
    )

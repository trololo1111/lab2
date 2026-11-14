from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    frame = pd.read_csv(data.csv_path)
    grouped = frame.groupby('Pclass')
    return TitanicSummary(
        row_count=len(frame),
        missing_by_column={str(key): int(value) for key, value in frame.isna().sum().items()},
        adults_over_30_count=int(frame['Age'].gt(30).sum()),
        mean_age_by_pclass={int(key): float(value) for key, value in grouped['Age'].mean().items()},
        survival_rate_by_pclass={int(key): float(value) for key, value in grouped['Survived'].mean().items()},
        highest_fares=frame['Fare'].nlargest(5).astype(float).tolist(),
    )

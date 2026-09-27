"""Mode (most-frequent) imputation matching Phase 1 categorical behavior."""

import pandas as pd

from predictor.ml.strategies.base import (
    ImputationStrategy,
    ensure_dataframe,
)


class ModeStrategy(ImputationStrategy):
    name = "Mode Imputation"
    applicable_types = ("CATEGORICAL", "BOOLEAN")

    def __init__(self) -> None:
        super().__init__()
        self._fill_values = None

    def fit(self, X, y=None):
        df = ensure_dataframe(X)
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not cat_cols:
            self._fail("ModeStrategy requires at least one categorical column.")

        fill_values = {}
        for col in cat_cols:
            mode_series = df[col].dropna().mode()
            if len(mode_series) > 0:
                fill_values[col] = mode_series.iloc[0]

        if not fill_values:
            self._fail("Cannot compute mode: no observed categorical values.")

        self._fill_values = fill_values
        self._mark_fitted(fill_values={k: str(v) for k, v in fill_values.items()})
        return self

    def transform(self, X):
        self._require_fitted()
        df = ensure_dataframe(X).copy()
        for col, value in self._fill_values.items():
            if col in df.columns:
                # Phase 1 parity: masked assignment (avoids pandas' object-dtype
                # fillna downcasting FutureWarning, e.g. bool columns with NaN).
                null_mask = df[col].isnull()
                if null_mask.any():
                    df.loc[null_mask, col] = value
        return df

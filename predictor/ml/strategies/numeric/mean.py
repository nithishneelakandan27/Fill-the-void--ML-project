"""Mean imputation for numeric columns."""

import pandas as pd

from predictor.ml.strategies.base import (
    ImputationStrategy,
    ensure_dataframe,
)


class MeanStrategy(ImputationStrategy):
    name = "Mean Imputation"
    applicable_types = ("NUMERIC",)

    def __init__(self) -> None:
        super().__init__()
        self._fill_values = None

    def fit(self, X, y=None):
        df = ensure_dataframe(X)
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            self._fail("MeanStrategy requires at least one numeric column.")

        self._fill_values = numeric.mean()
        if self._fill_values.isna().all():
            self._fail("Cannot compute mean: no observed numeric values.")

        self._mark_fitted(fill_values=self._fill_values.dropna().to_dict())
        return self

    def transform(self, X):
        self._require_fitted()
        df = ensure_dataframe(X).copy()
        for col, value in self._fill_values.items():
            if col in df.columns and pd.notna(value):
                df[col] = df[col].fillna(value)
        return df

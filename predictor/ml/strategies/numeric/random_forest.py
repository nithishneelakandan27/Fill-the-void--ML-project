"""
Random Forest regression imputation matching Phase 1:

- Target column is predicted from other numeric columns
- Feature NaNs filled with column median (then 0)
- Requires at least 5 observed target rows
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from predictor.ml.strategies.base import (
    ImputationStrategy,
    ensure_dataframe,
)


class RandomForestStrategy(ImputationStrategy):
    name = "Random Forest Regression"
    applicable_types = ("NUMERIC",)

    def __init__(
        self,
        target_column=None,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.target_column = target_column
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None
        self._feature_cols = None
        self._feature_medians = None

    def fit(self, X, y=None):
        df = ensure_dataframe(X)
        if y is not None:
            y = pd.Series(y, index=df.index)
            if self.target_column is None:
                self.target_column = y.name if y.name is not None else "target"
            df = df.copy()
            df[self.target_column] = y

        if not self.target_column or self.target_column not in df.columns:
            self._fail("RandomForestStrategy requires a target_column present in X.")

        numeric = df.select_dtypes(include="number")
        if self.target_column not in numeric.columns:
            self._fail("RandomForestStrategy target must be numeric.")

        feature_cols = [c for c in numeric.columns if c != self.target_column]
        if not feature_cols:
            self._fail("RandomForestStrategy requires at least one numeric predictor column.")

        observed = df[self.target_column].notna()
        if observed.sum() < 5:
            self._fail("RandomForestStrategy requires at least 5 observed target values.")

        X_all = numeric[feature_cols].copy()
        feature_medians = X_all.median()
        X_all = X_all.fillna(feature_medians).fillna(0.0)

        try:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            model.fit(X_all.loc[observed], df.loc[observed, self.target_column])
        except Exception as exc:
            self._fail(f"Random Forest fit failed: {exc}")

        self._model = model
        self._feature_cols = feature_cols
        self._feature_medians = feature_medians
        self._mark_fitted(
            n_train=int(observed.sum()),
            n_features=len(feature_cols),
            n_estimators=self.n_estimators,
        )
        return self

    def transform(self, X):
        self._require_fitted()
        df = ensure_dataframe(X).copy()
        if self.target_column not in df.columns:
            self._fail(f"Target column {self.target_column!r} is missing from transform input.")

        null_mask = df[self.target_column].isnull()
        if not null_mask.any():
            return df

        X_all = df[self._feature_cols].copy()
        X_all = X_all.fillna(self._feature_medians).fillna(0.0)

        try:
            preds = self._model.predict(X_all.loc[null_mask])
        except Exception as exc:
            self._fail(f"Random Forest transform failed: {exc}")

        df.loc[null_mask, self.target_column] = preds
        return df

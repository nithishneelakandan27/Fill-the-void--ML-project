"""
KNN imputation matching Phase 1 behavior:

- StandardScaler on numeric columns
- KNNImputer with k = min(5, n_samples)
- 100% missing numeric columns are excluded (as in Phase 1) and left untouched
- Only originally missing cells are replaced
"""

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from predictor.ml.strategies.base import (
    ImputationStrategy,
    ensure_dataframe,
)


class KNNStrategy(ImputationStrategy):
    name = "KNN Imputation"
    applicable_types = ("NUMERIC",)

    def __init__(self, n_neighbors: int = 5) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self._columns = None
        self._k = None
        self._scaler = None
        self._imputer = None

    def fit(self, X, y=None):
        df = ensure_dataframe(X)
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            num_cols = {}
            for col in df.columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().any():
                    num_cols[col] = converted
            if num_cols:
                numeric = pd.DataFrame(num_cols, index=df.index)

        if numeric.empty:
            self._fail("KNNStrategy requires at least one numeric column.")
        # Phase 1 parity: 100% missing columns are not usable as KNN features
        # (KNNImputer drops them, which would break inverse_transform).
        numeric = numeric.loc[:, numeric.notna().any()]
        if numeric.empty:
            self._fail("Cannot fit KNN: no observed numeric values.")

        self._columns = list(numeric.columns)
        self._k = min(self.n_neighbors, max(1, len(numeric)))
        self._scaler = StandardScaler()
        self._imputer = KNNImputer(n_neighbors=self._k)

        try:
            numeric_float = numeric[self._columns].apply(pd.to_numeric, errors="coerce").astype(float)
            scaled = self._scaler.fit_transform(numeric_float)
            self._imputer.fit(scaled)
        except Exception as exc:
            self._fail(f"KNN fit failed: {exc}")

        self._mark_fitted(k=self._k, n_columns=len(self._columns))
        return self

    def transform(self, X):
        self._require_fitted()
        df = ensure_dataframe(X).copy()
        original_missing = df[self._columns].isnull()

        try:
            numeric_float = df[self._columns].apply(pd.to_numeric, errors="coerce").astype(float)
            scaled = self._scaler.transform(numeric_float)
            imputed_scaled = self._imputer.transform(scaled)
            imputed = self._scaler.inverse_transform(imputed_scaled)
        except Exception as exc:
            self._fail(f"KNN transform failed: {exc}")

        imputed_df = pd.DataFrame(imputed, index=df.index, columns=self._columns)
        for col in self._columns:
            mask = original_missing[col]
            if mask.any():
                df.loc[mask, col] = imputed_df.loc[mask, col]
        return df

"""
Iterative (MICE-style) imputation for numeric columns using sklearn's
IterativeImputer.

This is an OPTIONAL candidate strategy. It models each column with missing
values as a function of the other numeric columns, round-robin, for a fixed
number of iterations. It is not assumed to be better than KNN / median / etc.;
selection between strategies is a later step.

Behavior:
- Numeric columns only; non-numeric columns pass through untouched.
- 100% missing numeric columns are excluded from the model and left as NaN
  (IterativeImputer would otherwise drop them and corrupt the output shape).
- Requires at least 2 usable numeric columns (one column has nothing to
  regress on) and at least 2 rows.
- Deterministic via random_state.
- Only originally missing cells are replaced; observed values are preserved.
- Never mutates the caller's DataFrame.
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from predictor.ml.strategies.base import (
    ImputationStrategy,
    ensure_dataframe,
)


class IterativeStrategy(ImputationStrategy):
    name = "Iterative Imputation"
    applicable_types = ("NUMERIC",)

    def __init__(self, max_iter: int = 10, random_state: int = 42) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self._columns = None
        self._imputer = None

    def fit(self, X, y=None):
        df = ensure_dataframe(X)
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            self._fail("IterativeStrategy requires at least one numeric column.")

        # Exclude 100% missing columns: they carry no information and
        # IterativeImputer silently drops them (shape mismatch on transform).
        numeric = numeric.loc[:, numeric.notna().any()]
        if numeric.empty:
            self._fail("Cannot fit IterativeImputer: no observed numeric values.")
        if numeric.shape[1] < 2:
            self._fail(
                "IterativeStrategy requires at least 2 numeric columns with "
                "observed values (nothing to regress on)."
            )
        if len(numeric) < 2:
            self._fail("IterativeStrategy requires at least 2 rows.")

        self._columns = list(numeric.columns)
        self._imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            keep_empty_features=True,
        )

        try:
            self._imputer.fit(numeric)
        except Exception as exc:
            self._fail(f"IterativeImputer fit failed: {exc}")

        self._mark_fitted(
            max_iter=self.max_iter,
            n_columns=len(self._columns),
            n_iter_run=int(getattr(self._imputer, "n_iter_", 0)),
        )
        return self

    def transform(self, X):
        self._require_fitted()
        df = ensure_dataframe(X).copy()

        missing_cols = [c for c in self._columns if c not in df.columns]
        if missing_cols:
            self._fail(
                f"Columns {missing_cols!r} seen at fit are missing from transform input."
            )

        original_missing = df[self._columns].isnull()
        if not original_missing.any().any():
            return df

        try:
            imputed = self._imputer.transform(df[self._columns])
        except Exception as exc:
            self._fail(f"IterativeImputer transform failed: {exc}")

        if imputed.shape != (len(df), len(self._columns)):
            self._fail(
                f"IterativeImputer returned shape {imputed.shape}, expected "
                f"{(len(df), len(self._columns))}."
            )

        imputed_df = pd.DataFrame(imputed, index=df.index, columns=self._columns)
        for col in self._columns:
            mask = original_missing[col]
            if mask.any():
                df.loc[mask, col] = imputed_df.loc[mask, col]
        return df

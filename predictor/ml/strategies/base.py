"""
Base interface for Fill the Void imputation strategies.

Sklearn-compatible: fit / transform / impute.
Strategies must not mutate the caller's input.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import pandas as pd


class ImputationStrategyError(Exception):
    """Raised when a strategy cannot be fitted or applied."""


def ensure_dataframe(X) -> pd.DataFrame:
    """Convert array-like input to a DataFrame without assuming a copy."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    return pd.DataFrame(X)


class ImputationStrategy(ABC):
    """Common interface for numeric and categorical imputation strategies."""

    name: str = "base"
    applicable_types: Sequence[str] = ()

    def __init__(self) -> None:
        self._is_fitted = False
        self._metrics: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

    @abstractmethod
    def fit(self, X, y=None):
        """Learn fill values or models from observed data. Returns self."""

    @abstractmethod
    def transform(self, X):
        """Return a copy of X with missing values filled."""

    def impute(self, X):
        """Alias for transform()."""
        return self.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_metrics(self) -> Dict[str, Any]:
        """Optional evaluation / fit metadata. Empty until the strategy is fit."""
        return dict(self._metrics)

    def get_explanation(self) -> str:
        """Short description of this strategy. Full explainability is a later step."""
        return f"{self.name}."

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ImputationStrategyError(
                f"{self.name} must be fit before transform/impute."
            )

    def _mark_fitted(self, **metrics: Any) -> None:
        self._is_fitted = True
        self._last_error = None
        self._metrics = {"method": self.name, **metrics}

    def _fail(self, message: str) -> None:
        self._last_error = message
        self._is_fitted = False
        raise ImputationStrategyError(message)

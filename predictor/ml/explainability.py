from typing import Any, Dict
from predictor.ml.strategies.base import ImputationStrategy
from predictor.ml.quality_metrics import calculate_imputation_metrics
import pandas as pd

def generate_imputation_explanation(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    col_name: str,
    strategy: ImputationStrategy,
    evaluation_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generates a structured, factual explanation of the imputation process.
    """
    metrics = calculate_imputation_metrics(
        original_df, imputed_df, col_name, evaluation_result
    )
    
    explanation = {
        "target_column": col_name,
        "strategy": strategy.name,
        "n_missing_before": metrics["n_missing_before"],
        "n_imputed": metrics["n_imputed"],
        "n_remaining_missing": metrics["n_remaining_missing"],
        "is_fully_imputed": metrics["is_fully_imputed"],
    }
    
    if "evaluation" in metrics:
        explanation["evaluation_metrics"] = metrics["evaluation"]
    elif "evaluation_error" in metrics:
        explanation["evaluation_error"] = metrics["evaluation_error"]
        
    return explanation

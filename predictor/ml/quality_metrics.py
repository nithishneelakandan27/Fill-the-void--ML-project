from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def calculate_imputation_metrics(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    col_name: str,
    evaluation_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate transparent quality metrics for an imputation result.
    """
    if col_name not in original_df.columns or col_name not in imputed_df.columns:
        raise ValueError(f"Column {col_name} must exist in both DataFrames.")

    orig_series = original_df[col_name]
    imputed_series = imputed_df[col_name]
    
    null_mask = orig_series.isnull()
    total_missing_before = int(null_mask.sum())
    
    # Values that were missing and are now filled
    filled_values_mask = null_mask & imputed_series.notnull()
    n_imputed = int(filled_values_mask.sum())
    
    # Check for remaining missing values
    remaining_missing = int(imputed_series.isnull().sum())
    
    metrics = {
        "column": col_name,
        "n_missing_before": total_missing_before,
        "n_imputed": n_imputed,
        "n_remaining_missing": remaining_missing,
        "is_fully_imputed": bool(remaining_missing == 0),
    }
    
    # Integrate Step 5 evaluation if available
    if evaluation_result and "error" not in evaluation_result:
        metrics["evaluation"] = evaluation_result
    elif evaluation_result and "error" in evaluation_result:
        metrics["evaluation_error"] = evaluation_result["error"]
        
    return metrics


import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

from predictor.ml.strategies.base import ImputationStrategy, ensure_dataframe

def evaluate_imputation_strategy(
    df: pd.DataFrame, 
    col_name: str, 
    strategy: ImputationStrategy, 
    mask_percentage: int = 20, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate imputation strategy using artificial masking.
    
    1. Select subset of observed values.
    2. Mask them.
    3. Run imputation.
    4. Calculate metrics.
    """
    df = ensure_dataframe(df).copy()
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in DataFrame.")
    
    # Identify observed values
    observed = df[df[col_name].notna()]
    if len(observed) < 5:  # Need sufficient samples
        return {"error": "Insufficient observed values for evaluation."}
        
    n_to_hide = max(1, int(len(observed) * (mask_percentage / 100)))
    rng = np.random.RandomState(random_state)
    
    # Mask values
    mask_indices = rng.choice(observed.index, size=n_to_hide, replace=False)
    
    true_values = df.loc[mask_indices, col_name].copy()
    df.loc[mask_indices, col_name] = np.nan
    
    # Impute
    try:
        imputed_df = strategy.fit_transform(df)
        predicted_values = imputed_df.loc[mask_indices, col_name]
    except Exception as e:
        return {"error": f"Strategy failed during evaluation: {e}"}
        
    # Calculate Metrics
    col_type = strategy.applicable_types[0] # Assuming first type is primary
    metrics = {"method": strategy.name, "n_evaluated": n_to_hide}
    
    if col_type == "NUMERIC":
        metrics["mae"] = mean_absolute_error(true_values, predicted_values)
        metrics["rmse"] = np.sqrt(mean_squared_error(true_values, predicted_values))
        metrics["r2"] = r2_score(true_values, predicted_values)
    else:
        # Categorical: normalize true and predicted values to string representation for evaluation
        true_str = true_values.astype(str)
        pred_str = predicted_values.astype(str)
        metrics["accuracy"] = accuracy_score(true_str, pred_str)
        
    return metrics

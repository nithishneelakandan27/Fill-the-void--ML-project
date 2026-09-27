import pandas as pd
from typing import Dict, List, Any, Optional

from predictor.ml.column_classifier import classify_columns_enhanced, ColumnInfo
from predictor.ml.evaluator import evaluate_imputation_strategy
from predictor.ml.quality_metrics import calculate_imputation_metrics
from predictor.ml.explainability import generate_imputation_explanation
from predictor.ml.strategies.numeric.mean import MeanStrategy
from predictor.ml.strategies.numeric.median import MedianStrategy
from predictor.ml.strategies.numeric.knn import KNNStrategy
from predictor.ml.strategies.numeric.iterative import IterativeStrategy
from predictor.ml.strategies.categorical.mode import ModeStrategy

# Map types to candidate strategies
STRATEGY_MAP = {
    "NUMERIC": [MeanStrategy(), MedianStrategy(), KNNStrategy(), IterativeStrategy()],
    "CATEGORICAL": [ModeStrategy()],
}

def orchestrate_imputation(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Orchestrates candidate selection, evaluation, and execution for a column.
    """
    column_info = classify_columns_enhanced(df)[target_col]
    
    if not column_info.is_imputable:
        return {"error": f"Column {target_col} is not imputable (type: {column_info.type})."}
        
    candidates = STRATEGY_MAP.get(column_info.type, [])
    if not candidates:
        return {"error": f"No imputation strategies available for type: {column_info.type}"}
        
    # Evaluate candidates
    best_strategy = None
    best_metric_val = float('inf') if column_info.type == "NUMERIC" else -1.0
    evaluation_results = {}
    
    for strategy in candidates:
        eval_res = evaluate_imputation_strategy(df, target_col, strategy)
        evaluation_results[strategy.name] = eval_res


        
        if "error" not in eval_res:
            # Simple heuristic for selection
            if column_info.type == "NUMERIC":
                # Lower is better (MAE)
                val = eval_res.get("mae", float('inf'))
                if val < best_metric_val:
                    best_metric_val = val
                    best_strategy = strategy
            else:
                # Higher is better (accuracy)
                val = eval_res.get("accuracy", -1.0)
                if val > best_metric_val:
                    best_metric_val = val
                    best_strategy = strategy
                    
    # Fallback to first strategy if no successful evaluations
    if not best_strategy:
        best_strategy = candidates[0]
        
    # Execution
    imputed_df = best_strategy.fit_transform(df)
    metrics = calculate_imputation_metrics(df, imputed_df, target_col, evaluation_results.get(best_strategy.name))
    explanation = generate_imputation_explanation(df, imputed_df, target_col, best_strategy, evaluation_results.get(best_strategy.name))
    
    return {
        "target_column": target_col,
        "type": column_info.type,
        "selected_strategy": best_strategy.name,
        "imputed_df": imputed_df,
        "metrics": metrics,
        "explanation": explanation,
        "evaluation_history": evaluation_results
    }

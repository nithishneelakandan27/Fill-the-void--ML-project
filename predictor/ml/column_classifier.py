"""
Enhanced Column Classification for Fill the Void

Detects column types and provides detailed metadata for imputation strategy selection.
Conservative classification prioritizing accuracy over comprehensiveness.

Types:
  NUMERIC       - float/int numerical data
  CATEGORICAL   - text/category data with reasonable cardinality
  BOOLEAN       - binary true/false-like values
  DATETIME      - date/time data
  IDENTIFIER    - unique IDs (user_id, email, UUID, etc.) - excluded from ML predictors
  FREE_TEXT     - long-form unstructured text
  UNSUPPORTED   - mixed/ambiguous/unhandled types

Behavior:
  - Does NOT transform the original DataFrame
  - Returns comprehensive metadata per column
  - Conservative: prefers safer classifications when uncertain
  - Identifies which columns are usable as ML predictors
  - Identifies which columns can be imputed
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Metadata for a single column after classification."""
    
    column_name: str
    dtype: str
    type: str  # NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, IDENTIFIER, FREE_TEXT, UNSUPPORTED
    cardinality: int
    unique_ratio: float
    missing_count: int
    missing_pct: float
    is_usable_predictor: bool  # Can be used in ML model
    is_imputable: bool  # Can missing values be imputed
    classification_reason: str
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ColumnInfo(name={self.column_name!r}, type={self.type}, "
            f"missing={self.missing_count}/{self.cardinality + self.missing_count} "
            f"({self.missing_pct:.1f}%), predictor={self.is_usable_predictor}, "
            f"imputable={self.is_imputable})"
        )


# ─────────────────────────────────────────────────────────────────────
# DETECTION FUNCTIONS - Conservative, ordered by priority
# ─────────────────────────────────────────────────────────────────────

def _is_empty_column(series: pd.Series) -> Tuple[bool, str]:
    """
    Check if column is completely empty (100% missing).
    
    Returns: (is_empty, reason)
    """
    if series.isnull().all():
        return True, f"100% missing values ({len(series)} total)"
    return False, ""


def _is_native_numeric(series: pd.Series) -> Tuple[bool, str]:
    """
    Check if column is natively numeric (int64, float64, etc.).
    
    Returns: (is_numeric, reason)
    """
    if pd.api.types.is_numeric_dtype(series):
        return True, f"Native numeric dtype: {series.dtype}"
    return False, ""


def _is_native_boolean(series: pd.Series) -> Tuple[bool, str]:
    """
    Check if column is natively boolean dtype.
    
    Returns: (is_boolean, reason)
    """
    if pd.api.types.is_bool_dtype(series):
        return True, "Native bool dtype"
    return False, ""


def _is_native_datetime(series: pd.Series) -> Tuple[bool, str]:
    """Check if column is already a datetime dtype."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, f"Native datetime dtype: {series.dtype}"
    return False, ""


def _is_datetime_column(series: pd.Series) -> Tuple[bool, str]:
    """
    Detect datetime columns using conservative parse-success rate.
    
    Requires:
    - At least 3 non-null values to evaluate
    - At least 80% of non-null values successfully parse as datetime
    - No single successful parse is accepted (too lenient)
    
    Returns: (is_datetime, reason)
    """
    non_null = series.dropna()
    
    if len(non_null) < 3:
        return False, "Insufficient non-null values to detect datetime"
    
    # Sample up to 100 values for evaluation
    sample_size = min(100, len(non_null))
    sample_indices = np.random.choice(len(non_null), size=sample_size, replace=False)
    sample = non_null.iloc[sample_indices]
    
    successful_parses = 0
    
    for value in sample:
        try:
            # Try to parse as datetime
            pd.to_datetime(value, errors='raise')
            successful_parses += 1
        except (ValueError, TypeError, AttributeError):
            pass
    
    parse_rate = successful_parses / len(sample)
    
    if parse_rate >= 0.80:
        return True, f"Datetime detection: {parse_rate*100:.0f}% parse success rate"
    
    return False, f"Low datetime parse rate: {parse_rate*100:.0f}%"


def _is_boolean_like(series: pd.Series) -> Tuple[bool, str]:
    """
    Detect boolean-like columns with common representations.
    
    Supports: Yes/No, True/False, Y/N, On/Off, 1/0 (as strings)
    
    Requirements:
    - Exactly 2-3 unique non-null values
    - All values match known boolean patterns (case-insensitive)
    - Cannot be native boolean (already caught)
    
    Returns: (is_boolean_like, reason)
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False, "No non-null values"
    
    unique_values = non_null.unique()
    
    # Must have 2-3 unique values
    if len(unique_values) < 2 or len(unique_values) > 3:
        return False, f"Wrong number of unique values: {len(unique_values)}"
    
    # Known boolean representations
    boolean_patterns = {
        frozenset({'yes', 'no'}),
        frozenset({'true', 'false'}),
        frozenset({'y', 'n'}),
        frozenset({'on', 'off'}),
        frozenset({'1', '0'}),
        frozenset({'t', 'f'}),
        frozenset({'y', 'no'}),  # Mixed
    }
    
    # Normalize and check
    unique_lower = frozenset(str(v).lower().strip() for v in unique_values)
    
    boolean_reason_labels = {
        frozenset({'yes', 'no'}): "Yes/No",
        frozenset({'true', 'false'}): "True/False",
        frozenset({'y', 'n'}): "Y/N",
        frozenset({'on', 'off'}): "On/Off",
        frozenset({'1', '0'}): "1/0",
        frozenset({'t', 'f'}): "T/F",
        frozenset({'y', 'no'}): "Y/No",
    }

    for pattern in boolean_patterns:
        if unique_lower == pattern:
            label = boolean_reason_labels.get(pattern, "/".join(sorted(pattern)))
            return True, f"Boolean-like values detected: {label}"
    
    return False, f"Not a recognized boolean pattern: {unique_lower}"


def _is_identifier_column(series: pd.Series) -> Tuple[bool, str]:
    """
    Detect identifier columns (user_id, email, UUID, etc.).
    
    Uses heuristics:
    1. Column name contains ID-like keywords
    2. Very high uniqueness (>95% of non-null values are unique)
    3. Special patterns: UUID, email, phone-like strings
    4. Sequential numeric pattern (incremental IDs)
    
    But NOT just because unique_ratio is high for a numeric measurement.
    
    Returns: (is_identifier, reason)
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False, "All values missing"

    if pd.api.types.is_datetime64_any_dtype(series):
        return False, "Datetime dtype is not an identifier"
    
    col_name = series.name.lower() if series.name else ""
    
    # Strong signal: Column name contains ID keywords
    id_keywords = ['id', '_id', 'code', 'key', 'pid', 'uid', 'guid', 'email', 'phone']
    name_has_id = any(kw in col_name for kw in id_keywords)
    
    # Uniqueness metrics
    n_unique = len(non_null.unique())
    unique_ratio = n_unique / len(non_null)
    
    # Special pattern detection
    has_uuid_pattern = False
    has_email_pattern = False
    has_phone_pattern = False
    
    # Sample up to 20 values for pattern matching
    sample_size = min(20, len(non_null))
    sample_indices = np.random.choice(len(non_null), size=sample_size, replace=False)
    sample = non_null.iloc[sample_indices].astype(str)
    
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$', re.I)
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    phone_pattern = re.compile(r'^[\d\-\+\(\)\s]{10,}$')
    
    uuid_matches = sum(1 for v in sample if uuid_pattern.match(v))
    email_matches = sum(1 for v in sample if email_pattern.match(v))
    phone_matches = sum(1 for v in sample if phone_pattern.match(v))
    
    has_uuid_pattern = uuid_matches >= sample_size * 0.7
    has_email_pattern = email_matches >= sample_size * 0.7
    has_phone_pattern = phone_matches >= sample_size * 0.7
    
    # Sequential numeric ID detection
    has_sequential_numeric = False
    if pd.api.types.is_numeric_dtype(series):
        try:
            sorted_vals = sorted(non_null.dropna().unique())
            if len(sorted_vals) >= 5:
                # Check if mostly sequential (with some gaps OK)
                diffs = np.diff(sorted_vals)
                median_diff = np.median(diffs)
                sequential_count = sum(1 for d in diffs if abs(d - median_diff) <= median_diff * 0.5)
                if sequential_count >= len(diffs) * 0.8:
                    has_sequential_numeric = True
        except:
            pass
    
    # Decision logic
    reasons = []
    
    if name_has_id:
        reasons.append("Name contains ID keyword")
    
    if has_uuid_pattern:
        reasons.append("UUID pattern detected")
    elif has_email_pattern:
        reasons.append("Email pattern detected")
    elif has_phone_pattern:
        reasons.append("Phone pattern detected")
    
    if has_sequential_numeric:
        reasons.append("Sequential numeric pattern")
    
    # Strong evidence only: ID-like name or special ID patterns, plus high uniqueness.
    # Sequential numeric pattern is supporting evidence, not enough on its own
    # (age, salary, quantity can look sequential).
    has_strong_id_evidence = name_has_id or has_uuid_pattern or has_email_pattern or has_phone_pattern
    if has_strong_id_evidence and unique_ratio > 0.95:
        return True, f"Identifier: {'; '.join(reasons)} (unique_ratio={unique_ratio:.1%})"
    
    # Don't classify purely based on high uniqueness alone
    # (normal measurements like exact prices or exact measurements can have high uniqueness)
    return False, f"Not an identifier (unique_ratio={unique_ratio:.1%})"


def _is_free_text_column(series: pd.Series) -> Tuple[bool, str]:
    """
    Detect free-text columns (descriptions, comments, etc.).
    
    Requirements:
    - String/object dtype
    - Average string length > 50 characters
    - High uniqueness (>50% of non-null values unique)
    
    Returns: (is_free_text, reason)
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False, "All values missing"
    
    # Must be object dtype
    if not pd.api.types.is_object_dtype(series):
        return False, f"Not object dtype: {series.dtype}"
    
    try:
        # Calculate mean string length
        str_lengths = non_null.astype(str).str.len()
        mean_length = str_lengths.mean()
        
        # Uniqueness ratio
        n_unique = len(non_null.unique())
        unique_ratio = n_unique / len(non_null)
        
        if mean_length > 50 and unique_ratio > 0.5:
            return True, f"Free-text: mean_length={mean_length:.0f}, unique_ratio={unique_ratio:.1%}"
        
        return False, f"Mean length too short ({mean_length:.0f}) or low uniqueness ({unique_ratio:.1%})"
    except Exception as e:
        logger.warning(f"Error detecting free-text: {e}")
        return False, f"Error during detection: {e}"


def _is_categorical_column(series: pd.Series) -> Tuple[bool, str]:
    """
    Fallback: classify as categorical if none of the above.
    
    Returns: (is_categorical, reason)
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False, "All values missing (cannot classify)"
    
    n_unique = len(non_null.unique())
    
    # Basic categorical: object/category dtype with reasonable cardinality
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True, f"Object/category dtype with {n_unique} unique values"
    
    return False, f"Cannot classify as categorical (dtype={series.dtype})"


# ─────────────────────────────────────────────────────────────────────
# MAIN CLASSIFICATION FUNCTION
# ─────────────────────────────────────────────────────────────────────

def classify_columns_enhanced(df: pd.DataFrame) -> Dict[str, ColumnInfo]:
    """
    Enhanced column classification for the Fill the Void imputation engine.
    
    Classifies each column into one of 7 types and returns detailed metadata.
    Does NOT modify the input DataFrame.
    
    Args:
        df: Input DataFrame to classify
    
    Returns:
        Dict mapping column_name → ColumnInfo with classification metadata
    
    Classification Priority (in order):
        1. Empty columns (100% missing)
        2. Native boolean (bool dtype)
        3. Native datetime dtype
        4. Identifier (ID keywords, high uniqueness, special patterns)
        5. Native numeric (int64, float64, etc.)
        6. Datetime strings (conservative parse-rate detection)
        6. Boolean-like (Yes/No, True/False, etc.)
        7. Free-text (long strings, high uniqueness)
        8. Categorical (fallback for object/category columns)
        9. Unsupported (everything else)
    """
    column_info = {}
    
    for col_name in df.columns:
        series = df[col_name]
        
        # Basic metadata
        dtype_str = str(series.dtype)
        n_total = len(series)
        n_missing = series.isnull().sum()
        pct_missing = 100 * n_missing / n_total if n_total > 0 else 0.0
        n_unique = len(series.dropna().unique())
        unique_ratio = n_unique / (n_total - n_missing) if (n_total - n_missing) > 0 else 0.0
        
        # Classification logic (priority order)
        col_type = "UNSUPPORTED"
        reason = "Default unsupported classification"
        is_predictor = False
        is_imputable = False
        
        # 1. Empty column
        is_empty, reason = _is_empty_column(series)
        if is_empty:
            col_type = "UNSUPPORTED"
            reason = f"Empty column: {reason}"
            is_predictor = False
            is_imputable = False
        
        # 2. Native boolean (check BEFORE numeric, since bool can be treated as numeric)
        elif (result := _is_native_boolean(series))[0]:
            is_bool, reason = result
            col_type = "BOOLEAN"
            is_predictor = False  # Will need encoding for ML
            is_imputable = True
        
        # 3. Native datetime (before identifier; date strings can look phone-like)
        elif (result := _is_native_datetime(series))[0]:
            is_dt, reason = result
            col_type = "DATETIME"
            is_predictor = False  # Will need feature engineering
            is_imputable = True
        
        # 4. Identifier (before native numeric so user_id is not treated as a measurement)
        elif (result := _is_identifier_column(series))[0]:
            is_id, reason = result
            col_type = "IDENTIFIER"
            is_predictor = False  # Excluded from ML predictors
            is_imputable = False  # Usually no imputation needed
        
        # 5. Native numeric
        elif (result := _is_native_numeric(series))[0]:
            is_numeric, reason = result
            col_type = "NUMERIC"
            is_predictor = True
            is_imputable = True
        
        # 6. Datetime strings
        elif (result := _is_datetime_column(series))[0]:
            is_dt, reason = result
            col_type = "DATETIME"
            is_predictor = False  # Will need feature engineering
            is_imputable = True
        
        # 7. Boolean-like
        elif (result := _is_boolean_like(series))[0]:
            is_bool_like, reason = result
            col_type = "BOOLEAN"
            is_predictor = False  # Will need encoding for ML
            is_imputable = True
        
        # 8. Free-text
        elif (result := _is_free_text_column(series))[0]:
            is_text, reason = result
            col_type = "FREE_TEXT"
            is_predictor = False  # Too high-dimensional for simple ML
            is_imputable = False  # Cannot impute unstructured text reasonably
        
        # 9. Categorical (fallback)
        elif (result := _is_categorical_column(series))[0]:
            is_cat, reason = result
            col_type = "CATEGORICAL"
            is_predictor = False  # Will need encoding for ML
            is_imputable = True
        
        # Store column info
        column_info[col_name] = ColumnInfo(
            column_name=col_name,
            dtype=dtype_str,
            type=col_type,
            cardinality=n_unique,
            unique_ratio=round(unique_ratio, 4),
            missing_count=int(n_missing),
            missing_pct=round(pct_missing, 2),
            is_usable_predictor=is_predictor,
            is_imputable=is_imputable,
            classification_reason=reason,
        )
    
    return column_info


# ─────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def get_classifier_summary(column_info: Dict[str, ColumnInfo]) -> str:
    """
    Generate a human-readable summary of classification results.
    
    Args:
        column_info: Output from classify_columns_enhanced()
    
    Returns:
        Formatted summary string
    """
    lines = ["Column Classification Summary", "=" * 50]
    
    # Group by type
    by_type = {}
    for col_name, info in column_info.items():
        col_type = info.type
        if col_type not in by_type:
            by_type[col_type] = []
        by_type[col_type].append((col_name, info))
    
    # Print by type
    for col_type in ["NUMERIC", "CATEGORICAL", "BOOLEAN", "DATETIME", "IDENTIFIER", "FREE_TEXT", "UNSUPPORTED"]:
        if col_type in by_type:
            lines.append(f"\n{col_type} ({len(by_type[col_type])}):")
            for col_name, info in by_type[col_type]:
                lines.append(
                    f"  {col_name:30} | "
                    f"missing={info.missing_pct:5.1f}% | "
                    f"cardinality={info.cardinality:4} | "
                    f"predictor={str(info.is_usable_predictor):5} | "
                    f"imputable={str(info.is_imputable):5}"
                )
    
    return "\n".join(lines)

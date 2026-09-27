# PHASE 2 ML ARCHITECTURE PROPOSAL
## Fill the Void – ML/Data Science Improvements

**Date:** September 4, 2026  
**Status:** DESIGN PHASE (No code changes yet)

---

## 1. CURRENT ML PIPELINE

### 1.1 Overview
The current `ml_impute()` function in [predictor/views.py](predictor/views.py#L141) implements a straightforward imputation strategy:

```
INPUT: DataFrame
    ↓
Classify: numeric vs categorical (classify_columns)
    ↓
1. 100% missing columns → SKIP (report unable to impute)
2. Categorical columns → Mode imputation
3. Numeric columns:
   - ≤40% missing → KNN Imputation (scaled, k=min(5, n_samples))
   - >40% missing → Random Forest Regression
    ↓
OUTPUT: cleaned_df, metrics, filled_mask
```

### 1.2 Current Components

**`classify_columns(df)`** (Line 48-50)
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
```
- Binary classification: numeric vs categorical
- Simple pandas `select_dtypes()` approach
- No analysis of column content or characteristics

**KNN Imputation** (Lines 189-217)
- StandardScaler normalization on all numeric features
- KNNImputer with k = min(5, n_samples)
- Uses ALL usable numeric columns as distance predictors (including complete columns)
- Applied to all columns with 0% < missingness ≤ 40%
- Fallback to median if KNN fails
- No evaluation metrics

**Random Forest Regression** (Lines 219-276)
- Applied to columns with > 40% missing
- Uses other numeric columns as features
- Missing feature values filled with column median before training
- Requires ≥5 training rows
- RMSE evaluation on train_test_split if ≥10 rows available
- Fallback to median if RF fails or insufficient training data

**Categorical Imputation** (Lines 166-183)
- Mode (most frequent value) imputation
- Applied to object/category/string columns with any missing values
- No cardinality analysis
- No evaluation or confidence scores
- Fallback to "Unable to impute (no observed values)" if no mode exists

---

## 2. CURRENT ML PROBLEMS

### 2.1 Column Classification Issues
**Problem:** Over-simplified classification treats all object columns as ordinary categorical data.

**Impact:**
- Potential ID columns (user_id, transaction_id) treated as predictors → no predictive value
- Free text columns (descriptions, comments) treated as categorical → high cardinality, slow KNN
- Datetime strings treated as categorical → should be parsed
- Boolean strings ("Yes"/"No", "True"/"False") treated as categorical → ordinal nature ignored
- Mixed-type columns (some numeric strings, some text) → parsing errors

**Example:**
```python
df = pd.DataFrame({
    'user_id': [1001, 1002, 1003, 1004, 1005],  # Should be IDENTIFIER, not predictor
    'age': [25, np.nan, 35, 40, 45],            # Numeric
    'hired_date': ['2020-01-15', '2020-02-20', np.nan, '2020-04-10', '2020-05-01'],  # Date
    'is_active': ['Yes', 'No', np.nan, 'Yes', 'No'],  # Boolean
    'description': ['...', '...', np.nan, '...', '...'],  # Free text
})
```
Current output: numeric=['age'], categorical=['user_id', 'hired_date', 'is_active', 'description']  
**Problem:** user_id and description are useless as predictors.

### 2.2 Arbitrary Missingness Threshold
**Problem:** Hard-coded 40% threshold for KNN vs Random Forest decision.

**Impact:**
- Not adaptive to dataset size or number of features
- 40% of 5 rows (2 rows) ≠ 40% of 1000 rows (400 rows)
- KNN with 2 training rows is very different from KNN with 400 rows
- No consideration of feature correlation strength

**Better approach:** Adaptive threshold based on:
- Dataset size (n_samples)
- Number of useful features
- Observed training set size
- Estimated imputation difficulty

### 2.3 No Evaluation of Imputation Quality
**Problem:** Imputation methods are not compared against baselines or evaluated rigorously.

**Current RF evaluation:**
```python
X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
rf.fit(X_tr, y_tr)
rmse = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))
```
**Problem:** This RMSE is on *observed data*, not on data with *artificial missingness*.  
**Impact:** We never know if KNN/RF imputation is actually better than simple mean/median.

**Missing:** Masking-based evaluation
```
1. Take complete dataset
2. Artificially mask k% of observed values
3. Run imputation algorithm
4. Compare imputed ≠ original
5. Calculate MAE, RMSE
6. Compare vs baseline (mean, median)
```

### 2.4 Categorical Imputation Has No Evaluation
**Problem:** Mode imputation applied without any quality assessment.

**Current code:**
```python
mode_series = df_clean.loc[~null_mask, col].mode()
if len(mode_series) > 0:
    df_clean.loc[null_mask, col] = mode_series.iloc[0]
    metrics[col] = {'method': 'Mode Imputation', 'type': 'categorical'}
```

**Missing:**
- Mode frequency (how confident is this imputation?)
- Comparison to baseline (random selection from observed values)
- Cardinality analysis (mode imputation unreliable for high-cardinality columns)
- Evaluation using masking-based accuracy

### 2.5 No Explainability
**Problem:** Users don't know why a method was chosen or how confident the results are.

**Current output:**
```python
metrics[col] = {
    'method': 'KNN Imputation (k=5, scaled)',
    'type': 'numeric',
}
```

**Missing:**
- Why KNN over RF? (missing % calculation)
- Why k=5? (dataset size reasoning)
- Model quality (RMSE, R², accuracy, confidence interval)
- Alternative methods considered
- Estimated uncertainty

### 2.6 Data Leakage in RF Evaluation
**Problem:** RF RMSE calculation uses test split on observed values, not artificial missingness.

**Current flow:**
```
Observed values (e.g., age: [25, 30, 35, 40, 45])
    ↓ train_test_split
[25, 30, 35] → train_model
[40, 45] → evaluate RMSE
    ↓
RMSE = 2.1 (evaluated on real data)
```

**Problem:** This tells us "RF predicts observed age well" but NOT "RF can impute missing age well".

**Better approach:** Masking-based evaluation
```
Observed values (e.g., age: [25, 30, 35, 40, 45])
    ↓ artificially hide 20% (40%)
[25, 30, 35, hidden, 45]
    ↓ train_model
[25, 30, 35, 45] → train_model
[40] → predict
    ↓
RMSE = true_value(40) - imputed_value(41) = 1
```

### 2.7 KNN Has No Quality Metrics
**Problem:** KNN imputation applied with no RMSE, MAE, or accuracy reporting.

**Current output:**
```python
metrics[col] = {'method': f'KNN Imputation (k={k}, scaled)', 'type': 'numeric'}
```

**Missing:**
- How well did KNN perform on artificially masked data?
- Is KNN actually better than mean/median for this column?
- What is the uncertainty in the imputed values?

### 2.8 No Feature Selection
**Problem:** All numeric columns used as KNN distance predictors, even if unrelated.

**Current code:**
```python
X_num = df_clean[usable_numeric_cols].copy()
knn = KNNImputer(n_neighbors=k)
```

**Problem Example:**
```python
df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, 45],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'customer_id': [1001, 1002, 1003, 1004, 1005],
    'random_noise': [0.1, 0.2, 0.3, 0.4, 0.5],
})
```
KNN uses ALL 3 numeric columns to predict 'age', including unrelated 'customer_id' and 'random_noise'.

**Better approach:** Correlation-based feature selection or statistical significance testing.

### 2.9 No Datetime Column Handling
**Problem:** Datetime strings treated as categorical.

**Current behavior:**
```python
df['date'] = ['2020-01-01', '2020-01-02', np.nan, '2020-01-04']
# Treated as categorical → mode = '2020-01-02'? (incorrect)
```

**Missing:**
- Datetime parsing
- Temporal feature engineering (year, month, day)
- Time-series aware imputation (interpolation)

### 2.10 No Boolean Column Handling
**Problem:** Boolean columns treated as ordinary categorical, losing ordinal information.

**Current behavior:**
```python
df['is_active'] = ['Yes', 'No', np.nan, 'Yes', 'No']
# Mode imputation on strings, no encoding awareness
```

**Better approach:**
- Detect boolean-like columns
- Use classification (mode or RF classification)
- Report confidence

---

## 3. PROPOSED COLUMN CLASSIFICATION SYSTEM

### 3.1 Enhanced Column Types
Instead of binary (numeric/categorical), propose 7-tier classification:

```
1. NUMERIC
   - Integer or float
   - pandas dtype: int64, float32, float64
   - Used as: model features, KNN distance, regression target

2. CATEGORICAL
   - String or category type
   - Cardinality: reasonable (< 100 unique values typically)
   - No datetime or boolean interpretation
   - Used as: mode imputation or RF classification

3. BOOLEAN
   - Binary-like strings ("Yes"/"No", "True"/"False", "1"/"0")
   - OR actual bool dtype
   - Cardinality: 2-3 unique values
   - Used as: mode imputation (appropriate) or RF classification

4. DATETIME
   - Parseable as date/time
   - Examples: "2020-01-15", "01/15/2020", "2020-01-15 14:30:00"
   - Used as: temporal interpolation or feature engineering

5. IDENTIFIER
   - Unique-value ratio > 0.9 (cardinality ≈ n_rows)
   - Examples: user_id, transaction_id, email, phone
   - Used as: NOT used as predictors (removed from feature set)

6. FREE_TEXT
   - Long strings (mean length > 50 chars)
   - High unique ratio (> 0.5)
   - Examples: descriptions, comments, addresses
   - Used as: NOT used for numeric imputation (low information density)

7. UNSUPPORTED
   - Mixed types
   - Unrecognized format
   - Action: skip or handle with caution

```

### 3.2 Classification Heuristics

#### 3.2.1 NUMERIC Detection
```python
if pd.api.types.is_numeric_dtype(col):
    return 'NUMERIC'
```

#### 3.2.2 DATETIME Detection
```python
def is_datetime_column(series):
    """Check if column can be parsed as datetime."""
    # Remove NaNs
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    
    # Try parsing sample with pd.to_datetime
    try:
        pd.to_datetime(non_null.sample(min(100, len(non_null))))
        return True
    except:
        return False
```

#### 3.2.3 BOOLEAN Detection
```python
def is_boolean_column(series):
    """Check if column contains boolean-like values."""
    non_null = series.dropna().unique()
    
    # Known boolean values
    bool_values = {
        ('yes', 'no'),
        ('true', 'false'),
        ('y', 'n'),
        ('1', '0'),
        ('on', 'off'),
        (True, False),  # actual booleans
        (1, 0),
    }
    
    non_null_lower = {str(v).lower() for v in non_null}
    return non_null_lower in bool_values and len(non_null_lower) <= 3
```

#### 3.2.4 IDENTIFIER Detection
```python
def is_identifier_column(series):
    """Check if column is likely an ID."""
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return False
    
    # Heuristic 1: Column name contains ID-like words
    name_patterns = ['id', 'code', 'key', 'pid', 'uid', 'guid']
    if any(pattern in series.name.lower() for pattern in name_patterns):
        return True
    
    # Heuristic 2: Unique value ratio > 0.9
    unique_ratio = len(non_null.unique()) / len(non_null)
    if unique_ratio > 0.9:
        # Further check: all values numeric or UUID-like
        try:
            # Try parsing as int
            pd.to_numeric(non_null)
            return True
        except:
            # Check if UUID-like
            if all(isinstance(v, str) and len(v) in [36, 32] for v in non_null.sample(min(10, len(non_null)))):
                return True
    
    return False
```

#### 3.2.5 FREE_TEXT Detection
```python
def is_free_text_column(series):
    """Check if column contains free-form text."""
    non_null = series.dropna().astype(str)
    
    if len(non_null) == 0:
        return False
    
    # Mean string length > 50 chars
    mean_length = non_null.str.len().mean()
    if mean_length < 50:
        return False
    
    # High unique ratio (> 0.5)
    unique_ratio = len(non_null.unique()) / len(non_null)
    if unique_ratio > 0.5:
        return True
    
    return False
```

#### 3.2.6 CATEGORICAL Detection
```python
def is_categorical_column(series):
    """Check if column is ordinary categorical."""
    # If not numeric, date, boolean, identifier, or free text
    # It's categorical
    return (
        not is_numeric_column(series) and
        not is_datetime_column(series) and
        not is_boolean_column(series) and
        not is_identifier_column(series) and
        not is_free_text_column(series)
    )
```

### 3.3 Classification Algorithm
```python
def classify_columns_v2(df):
    """
    Enhanced column classification.
    
    Returns:
        column_info: dict[column_name] -> {
            'type': 'NUMERIC' | 'CATEGORICAL' | 'BOOLEAN' | 'DATETIME' | 'IDENTIFIER' | 'FREE_TEXT' | 'UNSUPPORTED',
            'cardinality': int,
            'unique_ratio': float (0-1),
            'mean_length': float (for text),
            'is_usable_predictor': bool (can be used in KNN/RF),
            'is_imputable': bool (should we try to impute),
        }
    """
    column_info = {}
    
    for col in df.columns:
        non_null = df[col].dropna()
        n_unique = len(non_null.unique())
        unique_ratio = n_unique / len(df) if len(df) > 0 else 0
        
        col_type = determine_column_type(df[col])
        
        # Determine if usable as predictor and imputable
        is_predictor = col_type in ('NUMERIC', 'CATEGORICAL', 'BOOLEAN')
        is_predictor = is_predictor and col_type not in ('IDENTIFIER', 'FREE_TEXT')
        
        is_imputable = col_type in ('NUMERIC', 'CATEGORICAL', 'BOOLEAN', 'DATETIME')
        
        column_info[col] = {
            'type': col_type,
            'cardinality': n_unique,
            'unique_ratio': unique_ratio,
            'missing_count': df[col].isnull().sum(),
            'missing_pct': round(100 * df[col].isnull().sum() / len(df), 2),
            'is_usable_predictor': is_predictor,
            'is_imputable': is_imputable,
        }
    
    return column_info
```

---

## 4. PROPOSED NUMERIC IMPUTATION STRATEGIES

### 4.1 Strategy Hierarchy
For each numeric column with missing values, evaluate in order:

```
1. COMPLETE COLUMN
   - 0% missing → no imputation needed
   - Include as predictor in KNN/RF

2. SINGLE VALUE
   - Only one unique observed value
   - Fill all missing with that value
   - High confidence

3. MEAN IMPUTATION
   - 1-5% missing (< 5 missing values, high confidence baseline)
   - Use if few missing values
   - Compare vs KNN baseline

4. MEDIAN IMPUTATION
   - Simple, robust baseline
   - Always evaluate as baseline
   - Good for non-normal distributions

5. KNN IMPUTATION
   - 5-40% missing AND
   - ≥ 3 useful predictor features AND
   - ≥ 10 training rows
   - Scaled, k = adaptive(n_rows, n_features)
   - Requires baseline evaluation

6. ITERATIVE IMPUTATION (IterativeImputer)
   - 25-60% missing AND
   - ≥ 2 useful predictor features AND
   - ≥ 20 training rows
   - MICE-style approach, more flexible than KNN
   - Requires baseline evaluation

7. RANDOM FOREST REGRESSION
   - > 40% missing AND
   - ≥ 2 useful predictor features AND
   - ≥ 30 training rows
   - Uses categorical features too
   - Requires baseline evaluation

8. INTERPOLATION (Time-series aware)
   - If datetime column exists (temporal ordering)
   - Linear or spline interpolation
   - Applies only to temporal-dependent columns

9. DOMAIN-SPECIFIC
   - Custom rules (e.g., age minimum 0, salary minimum threshold)
   - Applied after main imputation

10. UNABLE_TO_IMPUTE
   - No information available
   - 100% missing column
   - No predictors available
```

### 4.2 Proposed Numeric Imputation Logic
```python
def select_numeric_strategy(
    col_name,
    series,           # Column with missing values
    df_predictors,    # DataFrame of usable predictors
    n_samples,        # Total dataset rows
):
    """
    Select best imputation strategy for numeric column.
    
    Returns: {
        'strategy': str,
        'reasoning': str,
        'expected_quality': 'low' | 'medium' | 'high',
    }
    """
    n_missing = series.isnull().sum()
    n_observed = len(series) - n_missing
    pct_missing = 100 * n_missing / len(series)
    
    # Check for single unique value
    if len(series.dropna().unique()) == 1:
        return {
            'strategy': 'SINGLE_VALUE',
            'reasoning': f'Only one observed value: {series.dropna().iloc[0]}',
            'expected_quality': 'high',
        }
    
    # Evaluate baseline metrics first
    baseline_mean = series.mean()
    baseline_median = series.median()
    
    # Small missingness
    if n_missing < 5 and pct_missing < 5:
        return {
            'strategy': 'MEAN',
            'reasoning': f'{n_missing} values missing (<5%), use mean as baseline',
            'expected_quality': 'high',
        }
    
    # No predictors available
    n_predictors = len(df_predictors.columns)
    if n_predictors == 0:
        return {
            'strategy': 'MEDIAN',
            'reasoning': 'No predictor features available, use median',
            'expected_quality': 'medium',
        }
    
    # KNN is viable
    if n_observed >= 10 and n_predictors >= 3 and pct_missing <= 40:
        k = min(5, max(3, n_observed // 5))  # Adaptive k
        return {
            'strategy': 'KNN',
            'reasoning': f'{pct_missing:.1f}% missing, {n_observed} observed, {n_predictors} predictors → KNN(k={k})',
            'expected_quality': 'high' if n_observed >= 30 else 'medium',
        }
    
    # IterativeImputer is viable
    if n_observed >= 20 and n_predictors >= 2 and 25 <= pct_missing <= 60:
        return {
            'strategy': 'ITERATIVE',
            'reasoning': f'{pct_missing:.1f}% missing, {n_observed} observed, {n_predictors} predictors → IterativeImputer',
            'expected_quality': 'high' if n_observed >= 50 else 'medium',
        }
    
    # Random Forest is viable
    if n_observed >= 30 and n_predictors >= 2 and pct_missing > 40:
        return {
            'strategy': 'RANDOM_FOREST',
            'reasoning': f'{pct_missing:.1f}% missing (heavy), {n_observed} observed → Random Forest',
            'expected_quality': 'medium' if n_observed >= 100 else 'low',
        }
    
    # Fallback
    return {
        'strategy': 'MEDIAN',
        'reasoning': f'Insufficient data for advanced methods ({n_observed} observed, {n_predictors} predictors)',
        'expected_quality': 'medium',
    }
```

---

## 5. PROPOSED CATEGORICAL IMPUTATION STRATEGIES

### 5.1 Strategy Hierarchy
```
1. COMPLETE COLUMN
   - 0% missing → no imputation

2. SINGLE VALUE
   - Only one unique observed value
   - Fill all missing with that value
   - High confidence

3. MODE IMPUTATION (Most Frequent)
   - Baseline for categorical
   - Always evaluate against random baseline
   - Report mode frequency / total observed

4. RANDOM FOREST CLASSIFICATION
   - < 50 unique values (not high-cardinality)
   - ≥ 3 useful predictor features
   - ≥ 20 training rows
   - Use other numeric + categorical features
   - Requires baseline evaluation (accuracy vs mode)

5. STRATIFIED RANDOM
   - Sample from observed values proportionally
   - Conservative baseline
   - Good for balanced classes

6. UNABLE_TO_IMPUTE
   - 100% missing
   - No observed values
```

### 5.2 Proposed Categorical Imputation Logic
```python
def select_categorical_strategy(
    col_name,
    series,           # Column with missing values
    df_predictors,    # DataFrame of usable predictors
    n_samples,
):
    """Select best strategy for categorical column."""
    n_missing = series.isnull().sum()
    n_observed = len(series) - n_missing
    pct_missing = 100 * n_missing / len(series)
    cardinality = len(series.dropna().unique())
    
    # Single unique value
    if cardinality == 1:
        return {
            'strategy': 'SINGLE_VALUE',
            'reasoning': f'Only one observed value in category',
            'expected_quality': 'high',
        }
    
    # No predictors
    if len(df_predictors.columns) == 0:
        mode = series.mode()
        if len(mode) > 0:
            return {
                'strategy': 'MODE',
                'reasoning': f'No predictors, use mode: "{mode[0]}"',
                'expected_quality': 'low' if pct_missing > 30 else 'medium',
            }
    
    # High cardinality (too many classes for RF)
    if cardinality > 50:
        return {
            'strategy': 'MODE',
            'reasoning': f'High cardinality ({cardinality} values), mode is safest',
            'expected_quality': 'low',
        }
    
    # RF Classification viable
    if (n_observed >= 20 and 
        len(df_predictors.columns) >= 2 and 
        cardinality <= 50 and
        pct_missing <= 40):
        return {
            'strategy': 'RF_CLASSIFICATION',
            'reasoning': f'{pct_missing:.1f}% missing, {cardinality} classes, {n_observed} observed → RF',
            'expected_quality': 'high' if n_observed >= 50 else 'medium',
        }
    
    # Default to mode
    mode = series.mode()
    if len(mode) > 0:
        return {
            'strategy': 'MODE',
            'reasoning': f'Fallback to mode: "{mode[0]}"',
            'expected_quality': 'medium' if n_observed >= 10 else 'low',
        }
    
    return {
        'strategy': 'UNABLE_TO_IMPUTE',
        'reasoning': 'No observed values (100% missing)',
        'expected_quality': 'none',
    }
```

---

## 6. STRATEGY SELECTION LOGIC & EXPLAINABILITY

### 6.1 Proposed Output Format
For each column, the user should see:

```yaml
age:
  type: NUMERIC
  missing_count: 5 / 100 (5%)
  imputation:
    strategy: KNN
    reasoning: >-
      5% missing with 8 useful numeric predictors and 95 observed training rows
      enables KNN with k=5 neighbors (standardized features, adaptive k)
    expected_quality: HIGH
    

salary:
  type: NUMERIC
  missing_count: 42 / 100 (42%)
  imputation:
    strategy: RANDOM_FOREST
    reasoning: >-
      42% missing (heavy). Random Forest regression selected over KNN due to
      high missingness. Uses age, experience, department as numeric/encoded
      predictors. 58 observed training rows available.
    expected_quality: MEDIUM
    estimated_rmse: 8500 (estimated from cross-validation)
    

department:
  type: CATEGORICAL
  cardinality: 4 unique values
  missing_count: 2 / 100 (2%)
  imputation:
    strategy: MODE
    reasoning: >-
      Low missingness (2%, only 2 values). Mode imputation selected as baseline.
      Most frequent value: "Engineering" (45% of observed).
    expected_quality: HIGH
    mode_frequency: 45%
    

user_id:
  type: IDENTIFIER
  cardinality: 100 unique values
  missing_count: 0 / 100 (0%)
  imputation:
    strategy: SKIPPED
    reasoning: >-
      Identified as ID column (unique_ratio=1.0, name contains "id").
      Not used as imputation predictor. No missing values.
    expected_quality: N/A


dataset_quality:
  overall_score: 8.2 / 10
  missing_summary: 49 missing cells (7.0% of dataset)
  imputable_missing: 49 / 49 (100%)
  quality_breakdown:
    high_confidence_imputations: 2 columns (age, department)
    medium_confidence_imputations: 1 column (salary)
    unimputable: 0 columns
```

### 6.2 Explanation Template
Each strategy selection generates an explanation covering:

1. **Data Profile**: Missing %, observed count, cardinality
2. **Method Selection Criteria**: Why this method over alternatives
3. **Quality Assessment**: Expected accuracy/confidence
4. **Alternative Considered**: Why not KNN? Why not mode?
5. **Uncertainty**: Confidence interval or quality metric

---

## 7. EVALUATION METHODOLOGY

### 7.1 Current Problems
**Current approach:**
- RF RMSE evaluated on train_test_split of *observed* values
- KNN has no evaluation at all
- Mode has no quality metrics
- Never compare methods against baselines

**Problem:** Observed value prediction ≠ missing value imputation quality.

### 7.2 Proposed Masking-Based Evaluation

#### 7.2.1 Evaluation Pseudocode
```python
def evaluate_imputation_strategy(
    df,                  # Full dataset
    col_name,           # Column to impute
    strategy,           # 'MEAN' | 'MEDIAN' | 'KNN' | 'RF' | 'MODE'
    mask_percentage=20, # Hide 20% of observed values
):
    """
    Evaluate strategy using artificial missingness.
    
    1. Select subset of observed values to hide
    2. Train model on remaining observed values
    3. Predict hidden values
    4. Compare predictions vs true values
    5. Calculate RMSE, MAE, Accuracy, etc.
    """
    
    # Step 1: Randomly hide mask_percentage of observed values
    observed_mask = ~df[col_name].isnull()
    n_to_hide = max(1, int(mask_percentage * observed_mask.sum() / 100))
    
    hide_indices = np.random.choice(
        df[observed_mask.index],
        size=n_to_hide,
        replace=False
    )
    
    df_masked = df.copy()
    df_masked.loc[hide_indices, col_name] = np.nan
    
    # Step 2: Train model on df_masked
    imputed_values = impute_single_column(df_masked, col_name, strategy)
    
    # Step 3: Compare predictions vs true values
    true_values = df.loc[hide_indices, col_name]
    predicted_values = imputed_values.loc[hide_indices]
    
    # Step 4: Calculate metrics
    if df[col_name].dtype in ('int64', 'float64'):
        # Numeric metrics
        mae = np.mean(np.abs(true_values - predicted_values))
        rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
        r2 = 1 - (np.sum((true_values - predicted_values) ** 2) /
                  np.sum((true_values - true_values.mean()) ** 2))
        return {
            'strategy': strategy,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_evaluated': n_to_hide,
        }
    else:
        # Categorical metrics
        accuracy = np.mean(true_values == predicted_values)
        return {
            'strategy': strategy,
            'accuracy': accuracy,
            'n_evaluated': n_to_hide,
        }
```

#### 7.2.2 Evaluation Process for Column
```python
def evaluate_column(df, col_name):
    """
    Evaluate multiple strategies and rank them.
    """
    strategies_to_test = select_applicable_strategies(df, col_name)
    
    # Always include baseline
    if is_numeric(col_name):
        baselines = ['MEAN', 'MEDIAN']
    else:
        baselines = ['MODE']
    
    all_strategies = baselines + strategies_to_test
    
    results = []
    for strategy in all_strategies:
        try:
            result = evaluate_imputation_strategy(df, col_name, strategy, mask_percentage=20)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate {strategy}: {e}")
    
    # Rank strategies
    if is_numeric(col_name):
        results.sort(key=lambda r: r['rmse'])  # Lower RMSE is better
    else:
        results.sort(key=lambda r: r['accuracy'], reverse=True)  # Higher accuracy is better
    
    return {
        'column': col_name,
        'best_strategy': results[0]['strategy'],
        'all_results': results,
        'recommendation': results[0],
    }
```

#### 7.2.3 Evaluation Results Output
```python
# For numeric column 'age' with 5% missing:
{
    'column': 'age',
    'evaluation': [
        {
            'strategy': 'MEAN',
            'mae': 2.3,
            'rmse': 3.1,
            'r2': 0.89,
            'n_evaluated': 20,  # 20% of 100 rows
        },
        {
            'strategy': 'MEDIAN',
            'mae': 2.1,
            'rmse': 3.0,
            'r2': 0.90,
        },
        {
            'strategy': 'KNN',
            'mae': 1.8,
            'rmse': 2.4,
            'r2': 0.93,  # Best!
        },
    ]
}

# For categorical column 'department':
{
    'column': 'department',
    'evaluation': [
        {
            'strategy': 'RANDOM_SELECTION',
            'accuracy': 0.40,
            'n_evaluated': 10,
        },
        {
            'strategy': 'MODE',
            'accuracy': 0.75,  # Best!
            'mode_value': 'Engineering',
            'mode_frequency': 0.45,
        },
    ]
}
```

### 7.3 Computational Cost
**Evaluation overhead:**
- For each column: hide 20%, retrain, evaluate
- 10 columns × 5 strategies × 2 train cycles = ~100 model trains per dataset
- For small datasets (100-1000 rows): <1 second
- For medium datasets (10,000 rows): 5-10 seconds
- For large datasets (100,000 rows): 30-60 seconds

**Recommendation:**
- For Phase 2, run evaluation for all columns during development
- For Phase 3+, make evaluation optional (toggle on/off for speed)

---

## 8. DATA QUALITY METRICS

### 8.1 Per-Column Metrics
```python
column_metrics = {
    'column_name': {
        'type': 'NUMERIC',
        'missing_count': 5,
        'missing_pct': 5.0,
        'missing_mechanism': 'likely_random',  # or 'likely_mcar', 'likely_mar'
        
        # Imputation metadata
        'imputation_strategy': 'KNN',
        'strategy_reasoning': '...',
        
        # Evaluation results
        'evaluation_metrics': {
            'mae': 1.8,
            'rmse': 2.4,
            'r2': 0.93,
            'comparison_to_baseline': 'RMSE 14% better than median',
        },
        
        # Quality indicators
        'quality_score': 8.5,  # 0-10
        'confidence': 'HIGH',  # HIGH | MEDIUM | LOW
        'imputation_notes': 'Good predictor availability, sufficient training data',
    }
}
```

### 8.2 Dataset-Level Metrics
```python
dataset_metrics = {
    'shape': (100, 15),
    'total_cells': 1500,
    'missing_cells': 49,
    'missing_pct': 3.3,
    
    'column_summary': {
        'numeric': 8,
        'categorical': 5,
        'boolean': 1,
        'datetime': 1,
        'identifier': 0,  # Excluded from analysis
    },
    
    'imputation_summary': {
        'imputable_columns': 14,
        'unimputable_columns': 0,  # 100% missing
        'high_confidence': 10,
        'medium_confidence': 3,
        'low_confidence': 1,
    },
    
    'missing_mechanism': {
        'likely_mcar': 12,  # Missing completely at random (good)
        'likely_mar': 2,    # Missing at random (OK)
        'likely_mnar': 0,   # Missing not at random (problematic)
    },
    
    'overall_quality_score': 8.2,  # Weighted average
    'estimated_completion_time': 2.3,  # seconds
    'dataset_readiness': 'PRODUCTION_READY',  # or 'NEEDS_REVIEW'
}
```

### 8.3 Quality Score Calculation
```python
def calculate_quality_score(dataset_metrics):
    """
    Calculate overall dataset quality score (0-10).
    
    Weighted formula:
    - Missing percentage: 30%
    - Imputation confidence: 40%
    - Predictor availability: 20%
    - Data type diversity: 10%
    """
    
    # Score 1: Missing percentage (lower is better)
    missing_score = max(0, 10 - dataset_metrics['missing_pct'])  # 0-10
    
    # Score 2: Confidence distribution
    total_imputable = len(dataset_metrics['imputation_summary']['imputable_columns'])
    high = dataset_metrics['imputation_summary']['high_confidence']
    medium = dataset_metrics['imputation_summary']['medium_confidence']
    confidence_score = 10 * (high + 0.5 * medium) / total_imputable
    
    # Score 3: Predictor availability
    avg_predictors_per_col = ...  # calculate
    predictor_score = min(10, 2 + avg_predictors_per_col)
    
    # Score 4: Type diversity (more types = more challenge)
    type_diversity_score = 10 - len(dataset_metrics['column_summary']) * 0.5
    
    overall = (0.30 * missing_score +
               0.40 * confidence_score +
               0.20 * predictor_score +
               0.10 * type_diversity_score)
    
    return min(10, max(0, overall))
```

---

## 9. EXPLAINABILITY DESIGN

### 9.1 User-Facing Output
The application should display explanations at multiple levels:

#### 9.1.1 Quick Summary (Homepage)
```
📊 Dataset Quality: 8.2/10 GOOD
✅ 49 values imputable (100% of missing data)
⏱️  Estimated processing: 2.3 seconds
```

#### 9.1.2 Column-Level Detail (Analysis Page)
```
SALARY
├─ Type: NUMERIC
├─ Missing: 42 of 100 (42%)
└─ Imputation Plan:
   ├─ Strategy: Random Forest Regression
   ├─ Why RF?: 42% missing is heavy; KNN needs <40%
   ├─ Training Data: 58 observed salary values
   ├─ Predictors: age, experience, department (3 features)
   ├─ Expected Quality: MEDIUM
   └─ Estimated Accuracy: RMSE ~$8,500
```

#### 9.1.3 Results Page (Post-Imputation)
```
SALARY
├─ Filled: 42 values
├─ Method: Random Forest Regression
├─ Quality: MEDIUM ⚠️
├─ Performance Metrics:
│  ├─ Cross-val RMSE: $8,500
│  ├─ vs Median Baseline: 22% better
│  └─ Confidence Interval: ±$3,200 (95%)
└─ Recommendation: Review filled high-income values manually
```

### 9.2 Why This Method? Examples

#### Example 1: Simple Baseline
```
AGE
→ Type: NUMERIC
→ Missing: 5 of 100 (5%)
→ Strategy: MEAN IMPUTATION
→ Why?: Very low missingness. Simple mean imputation (mean=35.2 years) 
         is appropriate baseline with minimal error.
→ Quality: HIGH
```

#### Example 2: KNN Selection
```
INCOME
→ Type: NUMERIC
→ Missing: 8 of 100 (8%)
→ Strategy: KNN IMPUTATION
→ Why?: 8% missing ≤ 40% threshold. 9 usable numeric predictors available.
         92 training rows sufficient for k=5 neighbors. Standardized 
         features used for distance calculation. KNN expected to outperform 
         simple mean baseline for correlated features.
→ Quality: HIGH
```

#### Example 3: RF Selection
```
SALARY
→ Type: NUMERIC
→ Missing: 42 of 100 (42%)
→ Strategy: RANDOM FOREST REGRESSION
→ Why?: 42% missing exceeds 40% threshold. KNN not recommended for heavy 
         missingness (limited training neighbors). Random Forest captures 
         non-linear relationships better. 58 observed values + 3 predictors
         (age, experience, department) sufficient for robust training.
→ Expected Quality: MEDIUM (more difficult than light missingness)
→ Cross-val RMSE: ~$8,500
```

#### Example 4: Mode Selection
```
DEPARTMENT
→ Type: CATEGORICAL (4 unique values)
→ Missing: 2 of 100 (2%)
→ Strategy: MODE IMPUTATION
→ Why?: Low missingness + discrete categorical data. Mode (most frequent 
         value) is standard baseline. "Engineering" appears 45% of the time 
         in observed data. Low cardinality makes this reliable.
→ Quality: HIGH
→ Mode Value: "Engineering"
→ Mode Frequency: 45%
```

#### Example 5: Skipped Column
```
USER_ID
→ Type: IDENTIFIER
→ Missing: 0 of 100 (0%)
→ Strategy: SKIPPED (no imputation)
→ Why?: Detected as ID column (100% unique values, name pattern "id").
         ID columns have low information value as ML predictors and are
         not imputed. No missing values anyway.
→ Usage: Preserved as-is, excluded from feature sets.
```

---

## 10. PERFORMANCE CONSIDERATIONS

### 10.1 Computational Complexity Analysis

| Dataset Size | KNN Time | RF Time | IterativeImp | Eval Overhead | Total | Notes |
|--------------|----------|---------|-------------|---------------|-------|-------|
| 100 rows | 10ms | 50ms | 100ms | 500ms | ~700ms | Evaluation dominates |
| 1K rows | 50ms | 200ms | 500ms | 2s | ~2.7s | Still manageable |
| 10K rows | 200ms | 1s | 3s | 15s | ~19s | Evaluation slow |
| 100K rows | 2s | 10s | 30s | 120s | ~162s | Needs async |

### 10.2 Memory Complexity
- KNN: O(n_rows × n_features) for distance matrix
- RF: O(n_trees × n_rows) for tree splits
- IterativeImputer: O(n_rows × n_features) per iteration

**Recommendations:**
- Datasets < 50K rows: OK for sync processing
- Datasets 50-500K rows: Optional async (Phase 3)
- Datasets > 500K rows: Require async processing + sampling

### 10.3 Safe Limits for Phase 2 (Synchronous)
```python
# Recommend skipping evaluation if:
if n_rows * n_cols > 100_000:  # 100K cells
    skip_evaluation = True
    
# Cap number of evaluations:
max_evaluations = 15  # Evaluate max 15 columns per dataset

# Disable expensive methods:
if n_rows < 10:
    disable_strategies = ['RANDOM_FOREST', 'ITERATIVE']
if n_rows < 50:
    disable_strategies = ['RANDOM_FOREST']
```

### 10.4 Optimization Strategies
1. **Lazy Evaluation**: Only evaluate selected columns, not all
2. **Stratified Sampling**: For evaluation, sample 80% of data
3. **Parallel Processing**: Use scikit-learn's `n_jobs=-1` for RF, KNN
4. **Method Caching**: Cache StandardScaler fitted transforms
5. **Early Stopping**: Stop RF evaluation if baseline is clearly better

---

## 11. PROPOSED MODULE/FILE STRUCTURE

### 11.1 Current Structure
```
predictor/
├─ views.py          # Monolithic: routing, forms, ML, tables
├─ forms.py          # Upload validation
├─ tests.py          # Tests
├─ urls.py           # URL routing
└─ models.py         # (empty, no DB models)
```

### 11.2 Proposed Modular Structure
```
predictor/
├─ views.py          # Keep for routing + session management
├─ forms.py          # Keep as-is
├─ tests.py          # Expand with Phase 2 tests
├─ urls.py           # Keep as-is
│
├─ ml/                # NEW: ML pipeline module
│  ├─ __init__.py
│  ├─ column_classifier.py    # NEW: Enhanced column classification
│  ├─ imputer.py              # NEW: Main imputation orchestrator
│  ├─ strategies/
│  │  ├─ __init__.py
│  │  ├─ base.py               # Base strategy class
│  │  ├─ numeric/
│  │  │  ├─ mean.py
│  │  │  ├─ median.py
│  │  │  ├─ knn.py
│  │  │  ├─ iterative.py       # NEW: IterativeImputer wrapper
│  │  │  └─ random_forest.py
│  │  └─ categorical/
│  │     ├─ mode.py
│  │     └─ random_forest.py   # NEW: RF classification
│  │
│  ├─ evaluator.py             # NEW: Masking-based evaluation
│  ├─ quality_metrics.py        # NEW: Quality scoring
│  └─ explainability.py         # NEW: Explanation generation
│
└─ visualization.py   # Keep existing, extend for new metrics
```

### 11.3 New Module Responsibilities

**`ml/column_classifier.py`**
```python
def classify_columns(df) -> Dict[str, ColumnInfo]
    - Detect NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, IDENTIFIER, FREE_TEXT
    - Return detailed column metadata
```

**`ml/strategies/base.py`**
```python
class ImputationStrategy:
    def fit(self, X, y): pass
    def impute(self, X_missing): pass
    def get_metrics(self) -> Dict: pass
    def get_explanation(self) -> str: pass
```

**`ml/strategies/numeric/knn.py`**
```python
class KNNStrategy(ImputationStrategy):
    - Wraps sklearn KNNImputer
    - Handles scaling, k selection
    - Returns RMSE evaluation
```

**`ml/strategies/numeric/iterative.py`** (NEW)
```python
class IterativeStrategy(ImputationStrategy):
    - Wraps sklearn IterativeImputer
    - MICE-style imputation
    - For 25-60% missing numeric data
```

**`ml/evaluator.py`**
```python
def evaluate_imputation_strategy(
    df, col_name, strategy, mask_percentage=20
) -> EvaluationResult
    - Artificial masking
    - Train/test evaluation
    - Return RMSE, MAE, accuracy, etc.
```

**`ml/quality_metrics.py`**
```python
def calculate_column_quality_score(col_info, evaluation_result) -> float
def calculate_dataset_quality_score(dataset_metrics) -> float
```

**`ml/explainability.py`**
```python
def generate_column_explanation(
    col_name, col_info, selected_strategy, evaluation_results
) -> ExplanationText
    - Generates user-friendly explanations
    - Why this method? Why not alternatives?
```

---

## 12. MIGRATION STRATEGY

### 12.1 Phase 2a: Foundation (Week 1)
1. Create `ml/` module structure
2. Implement `column_classifier.py`
3. Write tests for classification
4. Verify backward compatibility

### 12.2 Phase 2b: Strategies (Week 2)
1. Implement strategy base class and wrappers
2. Migrate existing KNN, RF from `views.py` to strategies
3. Add IterativeImputer strategy
4. Ensure evaluation remains backward compatible

### 12.3 Phase 2c: Evaluation (Week 3)
1. Implement `evaluator.py` with masking
2. Add evaluation to each strategy
3. Integrate quality metrics
4. Add optional evaluation toggle

### 12.4 Phase 2d: Explainability (Week 4)
1. Implement explanation generation
2. Update UI to show explanations
3. Test explanation accuracy
4. User feedback and refinement

### 12.5 Backward Compatibility
- Keep existing `ml_impute()` function working
- New modular code called internally
- Gradual replacement over 4 weeks
- All tests passing throughout

---

## 13. RISKS & MITIGATION

### Risk 1: Performance Degradation
**Risk:** Evaluation + additional strategies could slow down imputation.  
**Mitigation:**
- Make evaluation optional (toggle on/off)
- Use stratified sampling for large datasets
- Parallel processing with n_jobs=-1
- Set max evaluation time budget (e.g., 30 seconds)

### Risk 2: Feature Creep
**Risk:** Over-engineering column classification could introduce complexity.  
**Mitigation:**
- Start with heuristics, not full statistical tests
- Unit test each heuristic independently
- Fall back to simple classification if uncertain
- Conservative defaults (better to underclassify than misclassify)

### Risk 3: Explanation Confusion
**Risk:** Too much explanation could overwhelm users.  
**Mitigation:**
- Layered explanations (summary → details)
- Show only critical info by default
- "Expand for details" option
- A/B test explanation clarity

### Risk 4: Breaking Changes
**Risk:** Refactoring could break existing functionality.  
**Mitigation:**
- Comprehensive test suite before refactoring
- Feature flags for new strategies
- Keep old code path available
- Gradual rollout (10% → 25% → 50% → 100%)

### Risk 5: False Confidence
**Risk:** High quality scores on poor datasets could mislead users.  
**Mitigation:**
- Conservative quality scoring (err on low side)
- Clear confidence indicators (HIGH/MEDIUM/LOW)
- Manual review recommendations
- Show worst-case scenarios

---

## 14. RECOMMENDED IMPLEMENTATION ORDER

### STEP 1: Enhanced Column Classification
**Deliverable:** `ml/column_classifier.py`
- Implement `classify_columns_v2()` function
- Detect: NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, IDENTIFIER, FREE_TEXT
- Write comprehensive unit tests
- Verify against current classification (backward compat check)
- Update [predictor/tests.py](predictor/tests.py) with new test cases

**Why first?** Foundation for all downstream decisions.

---

### STEP 2: Strategy Base Class & Module Structure
**Deliverable:** `ml/strategies/base.py` + directory structure
- Define `ImputationStrategy` abstract base class
- Methods: `fit()`, `impute()`, `get_metrics()`, `get_explanation()`
- Create empty wrapper classes for each strategy
- Write mock tests (will implement actual logic next)

**Why second?** Defines the interface before implementation.

---

### STEP 3: Migrate Existing Strategies
**Deliverable:** `ml/strategies/numeric/knn.py`, `random_forest.py`, `ml/strategies/categorical/mode.py`
- Refactor existing KNN code from `views.py` into `KNNStrategy` class
- Refactor existing RF code into `RandomForestStrategy` class
- Refactor existing Mode code into `ModeStrategy` class
- Update `views.ml_impute()` to use new strategy classes
- All tests must pass (behavior unchanged)

**Why third?** Modularization without changing behavior.

---

### STEP 4: Add IterativeImputer Strategy
**Deliverable:** `ml/strategies/numeric/iterative.py`
- Implement IterativeStrategy wrapper around sklearn.impute.IterativeImputer
- MICE-style approach (iterative regression)
- For 25-60% missing numeric data
- Add unit tests with sample data
- Compare RMSE to baseline

**Why fourth?** New capability (iterative imputation).

---

### STEP 5: Implement Masking-Based Evaluation
**Deliverable:** `ml/evaluator.py`
- `evaluate_imputation_strategy()` function
- Mask 10-20% of observed values
- Train strategy on masked data
- Evaluate predictions vs true values
- Return MAE, RMSE, accuracy
- Test with synthetic data (known ground truth)

**Why fifth?** Provides quality metrics for all strategies.

---

### STEP 6: Quality Metrics & Scoring
**Deliverable:** `ml/quality_metrics.py`
- `calculate_column_quality_score()` function
- `calculate_dataset_quality_score()` function
- Implement weighted scoring formula
- Return confidence levels (HIGH/MEDIUM/LOW)
- Unit tests with sample datasets

**Why sixth?** Enables explainability and user guidance.

---

### STEP 7: Explainability Module
**Deliverable:** `ml/explainability.py`
- `generate_column_explanation()` function
- Template-based text generation
- Include: why method selected, alternatives considered, quality metrics
- Generate dataset-level summary
- Test explanation accuracy and clarity

**Why seventh?** User-facing output that explains decisions.

---

### STEP 8: Orchestrator (Refactor ml_impute)
**Deliverable:** `ml/imputer.py`
- New `ImputationEngine` class that coordinates:
  - Column classification
  - Strategy selection (based on column_info)
  - Evaluation (optional)
  - Quality scoring
  - Explanation generation
- Backward-compatible with existing `ml_impute()` calls
- Feature flag for new behavior vs old

**Why eighth?** Ties all components together.

---

### STEP 9: UI Integration (Views & Templates)
**Deliverable:** Updates to [predictor/views.py](predictor/views.py) and templates
- Update `analysis()` view to show strategy selection reasoning
- Add "Why KNN?" explanations to results page
- Show quality scores for each column
- Display dataset-level quality summary
- Add optional evaluation results (if enabled)

**Why ninth?** User-facing integration.

---

### STEP 10: Comprehensive Testing & Documentation
**Deliverable:** [predictor/tests.py](predictor/tests.py) + documentation
- Tests for all new strategies
- Tests for column classification edge cases
- Tests for evaluation correctness
- Tests for quality scoring
- Integration tests (full workflow)
- Performance benchmarks (timing)
- Add docstrings to all modules
- Create Phase 2 implementation guide

**Why last?** QA and knowledge preservation.

---

## SUMMARY

**Phase 2 Goal:** Transform the Fill the Void imputation engine from simple/single-method to sophisticated/adaptive/explainable.

**Key Improvements:**
1. ✅ Enhanced column classification (7 types vs 2)
2. ✅ Adaptive strategy selection (based on data characteristics)
3. ✅ Rigorous evaluation (masking-based, not just train accuracy)
4. ✅ Full explainability (why KNN? why not RF? expected quality?)
5. ✅ Modular architecture (easy to add new strategies)
6. ✅ Quality metrics (confidence scoring)
7. ✅ Alternative strategies (IterativeImputer, RF classification)

**Non-Goals (for Phase 2):**
- ❌ No async processing (add in Phase 3)
- ❌ No new frontend framework (React/Vue in Phase 3)
- ❌ No database refactor (PostgreSQL in Phase 3)
- ❌ No DRF/API (Phase 4)

**Implementation Timeline:** ~4 weeks (10 steps, ~1 week per 2-3 steps)


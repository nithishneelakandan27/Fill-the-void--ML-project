# Fill the Void – Smart Data Cleaner (ML Missing Value Imputation)

Fill the Void is a Django-based web application for automated missing value imputation using machine learning strategies.

## Key Features & Capabilities

- **Dataset Upload & Analysis**: Upload CSV or Excel datasets (up to 10MB) to visualize missing value distributions, missingness heatmaps, and column statistics.
- **Enhanced Column Classification**: Detects column types including `NUMERIC`, `CATEGORICAL`, `BOOLEAN`, `DATETIME`, `IDENTIFIER`, `FREE_TEXT`, and `UNSUPPORTED`.
- **Orchestrated Strategy Selection**: Dynamically evaluates candidate imputation strategies (Mean, Median, KNN, Mode, Iterative MICE) using artificial masking.
- **Transparent Quality Metrics & Explanations**: Generates MAE, RMSE, R², Accuracy metrics, and step-by-step imputation explanations.
- **Cleaned Data Export**: Download fully imputed datasets with visual cell-by-cell diff highlighting in the UI.

## Quick Start

### 1. Run System Check
```bash
python3 manage.py check
```

### 2. Run Test Suite
```bash
python3 manage.py test predictor
```

### 3. Start Development Server
```bash
python3 manage.py runserver
```
Then navigate to `http://127.0.0.1:8000/`.

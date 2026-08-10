import os
import io
import json
import base64
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.conf import settings

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
#  Helper: dataframe ↔ session serialization
# ─────────────────────────────────────────────

def df_to_session(df):
    """Convert DataFrame to JSON-serializable dict for session storage."""
    return df.to_json(orient='split')


def df_from_session(data):
    """Restore DataFrame from session JSON string."""
    return pd.read_json(io.StringIO(data), orient='split')


# ─────────────────────────────────────────────
#  Helper: detect column types
# ─────────────────────────────────────────────

def classify_columns(df):
    """Return lists of numeric and categorical column names."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, cat_cols


# ─────────────────────────────────────────────
#  Helper: generate missing-value heatmap
# ─────────────────────────────────────────────

def generate_heatmap(df):
    """Return base64 PNG string of missing value heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.8), 4))
    missing = df.isnull()

    # If dataset has many rows sample for display
    if len(df) > 100:
        missing = missing.sample(100, random_state=42)

    sns.heatmap(
        missing,
        cbar=False,
        cmap=['#E8F0FE', '#1967D2'],
        ax=ax,
        linewidths=0.1,
        linecolor='#F1F3F4',
    )
    ax.set_title('Missing Values Heatmap  (blue = missing)', fontsize=11, pad=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
#  Helper: generate bar chart of missing counts
# ─────────────────────────────────────────────

def generate_bar_chart(missing_info):
    """Return base64 PNG string of missing values bar chart."""
    cols = [r['column'] for r in missing_info if r['missing_count'] > 0]
    counts = [r['missing_count'] for r in missing_info if r['missing_count'] > 0]

    if not cols:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.9), 4))
    bars = ax.barh(cols, counts, color='#1967D2', alpha=0.85, height=0.55)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count), va='center', fontsize=9, color='#3C4043'
        )

    ax.set_xlabel('Missing Values Count', fontsize=10)
    ax.set_title('Missing Values per Column', fontsize=11, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
#  ML: impute missing values
# ─────────────────────────────────────────────

def ml_impute(df):
    """
    Impute missing values using:
    - RandomForest-based iterative imputation for numeric columns
    - Most-frequent (mode) for categorical columns

    Returns (cleaned_df, metrics_dict, filled_mask_df)
    """
    df_clean = df.copy()
    numeric_cols, cat_cols = classify_columns(df)
    metrics = {}
    filled_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # ── Categorical columns: simple mode imputation ──────────────────────
    for col in cat_cols:
        null_mask = df_clean[col].isnull()
        if null_mask.any():
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean.loc[null_mask, col] = mode_val[0]
                filled_mask.loc[null_mask, col] = True
            metrics[col] = {'method': 'Mode Imputation', 'type': 'categorical'}

    # ── Numeric columns: KNN then RF for columns with many missing ────────
    numeric_df = df_clean[numeric_cols].copy()
    missing_frac = numeric_df.isnull().mean()

    # Columns with < 40% missing → KNN imputer
    knn_cols = [c for c in numeric_cols if 0 < missing_frac[c] <= 0.40]
    rf_cols  = [c for c in numeric_cols if missing_frac[c] > 0.40]

    # KNN impute
    if knn_cols:
        knn = KNNImputer(n_neighbors=5)
        temp = numeric_df[knn_cols].copy()
        imputed = knn.fit_transform(temp)
        for i, col in enumerate(knn_cols):
            null_mask = df[col].isnull()
            df_clean.loc[null_mask, col] = imputed[:, i][null_mask]
            filled_mask.loc[null_mask, col] = True
            metrics[col] = {'method': 'KNN Imputation (k=5)', 'type': 'numeric'}

    # RF impute for heavily-missing columns
    for col in rf_cols:
        null_mask = df[col].isnull()
        not_null_mask = ~null_mask

        # Feature columns: all other numeric cols (already partially imputed)
        feature_cols = [c for c in numeric_cols if c != col]
        if not feature_cols:
            # fallback to median
            median_val = df_clean[col].median()
            df_clean.loc[null_mask, col] = median_val
            filled_mask.loc[null_mask, col] = True
            metrics[col] = {'method': 'Median Imputation (fallback)', 'type': 'numeric'}
            continue

        X_all = df_clean[feature_cols].copy()
        # Fill any remaining NaN in features with column median
        X_all = X_all.fillna(X_all.median())

        X_train = X_all[not_null_mask]
        y_train = df_clean.loc[not_null_mask, col]
        X_pred  = X_all[null_mask]

        if len(X_train) < 5:
            # Not enough samples → median fallback
            df_clean.loc[null_mask, col] = df_clean[col].median()
            filled_mask.loc[null_mask, col] = True
            metrics[col] = {'method': 'Median (insufficient data)', 'type': 'numeric'}
            continue

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_pred)
        df_clean.loc[null_mask, col] = preds
        filled_mask.loc[null_mask, col] = True

        # Evaluate with cross-val on known data
        if len(X_train) >= 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            rf_eval = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf_eval.fit(X_tr, y_tr)
            preds_eval = rf_eval.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds_eval))
            metrics[col] = {
                'method': 'Random Forest Regression',
                'type': 'numeric',
                'rmse': round(float(rmse), 4),
            }
        else:
            metrics[col] = {'method': 'Random Forest Regression', 'type': 'numeric'}

    return df_clean, metrics, filled_mask


# ─────────────────────────────────────────────
#  VIEW: Home
# ─────────────────────────────────────────────

def home(request):
    return render(request, 'predictor/home.html', {'active': 'home'})


# ─────────────────────────────────────────────
#  VIEW: Upload
# ─────────────────────────────────────────────

def upload_dataset(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('dataset')
        if not uploaded_file:
            messages.error(request, 'Please select a file to upload.')
            return redirect('upload')

        filename = uploaded_file.name
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ['.csv', '.xlsx', '.xls']:
            messages.error(request, 'Only CSV and Excel files are supported.')
            return redirect('upload')

        try:
            if ext == '.csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            messages.error(request, f'Error reading file: {e}')
            return redirect('upload')

        if df.empty:
            messages.error(request, 'The uploaded file is empty.')
            return redirect('upload')

        # Store in session
        request.session['df_original'] = df_to_session(df)
        request.session['filename'] = filename
        request.session['df_shape'] = list(df.shape)

        messages.success(request, f'✅ "{filename}" uploaded successfully!')
        return redirect('analysis')

    return render(request, 'predictor/upload.html', {'active': 'upload'})


# ─────────────────────────────────────────────
#  VIEW: Analysis
# ─────────────────────────────────────────────

def analysis(request):
    df_json = request.session.get('df_original')
    if not df_json:
        messages.warning(request, 'Please upload a dataset first.')
        return redirect('upload')

    df = df_from_session(df_json)
    filename = request.session.get('filename', 'dataset')

    # Missing value stats per column
    total_rows = len(df)
    missing_info = []
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        pct = round((missing_count / total_rows) * 100, 2) if total_rows else 0
        dtype = str(df[col].dtype)
        missing_info.append({
            'column': col,
            'dtype': dtype,
            'missing_count': missing_count,
            'missing_pct': pct,
            'non_missing': total_rows - missing_count,
        })

    total_missing = int(df.isnull().sum().sum())
    total_cells = int(df.size)

    # Generate charts
    heatmap_b64 = generate_heatmap(df) if total_missing > 0 else None
    bar_chart_b64 = generate_bar_chart(missing_info) if total_missing > 0 else None

    # Preview (first 20 rows)
    preview_html = df.head(20).to_html(
        classes='preview-table', border=0, index=True, na_rep='<span class="na-val">NaN</span>'
    )

    context = {
        'active': 'analysis',
        'filename': filename,
        'shape': df.shape,
        'missing_info': missing_info,
        'total_missing': total_missing,
        'total_cells': total_cells,
        'missing_pct_overall': round((total_missing / total_cells) * 100, 2) if total_cells else 0,
        'heatmap': heatmap_b64,
        'bar_chart': bar_chart_b64,
        'preview_html': preview_html,
    }
    return render(request, 'predictor/analysis.html', context)


# ─────────────────────────────────────────────
#  VIEW: Run ML Prediction (AJAX / POST)
# ─────────────────────────────────────────────

def run_prediction(request):
    if request.method != 'POST':
        return redirect('analysis')

    df_json = request.session.get('df_original')
    if not df_json:
        messages.warning(request, 'Session expired. Please upload again.')
        return redirect('upload')

    df = df_from_session(df_json)

    try:
        df_clean, metrics, filled_mask = ml_impute(df)
    except Exception as e:
        messages.error(request, f'Prediction error: {e}')
        return redirect('analysis')

    # Store results in session
    request.session['df_cleaned'] = df_to_session(df_clean)
    request.session['filled_mask'] = df_to_session(filled_mask.astype(int))
    request.session['metrics'] = json.dumps(metrics)

    return redirect('results')


# ─────────────────────────────────────────────
#  VIEW: Results
# ─────────────────────────────────────────────

def results(request):
    df_clean_json = request.session.get('df_cleaned')
    df_orig_json  = request.session.get('df_original')
    mask_json     = request.session.get('filled_mask')
    metrics_json  = request.session.get('metrics')

    if not df_clean_json:
        messages.warning(request, 'No results yet. Please run ML prediction first.')
        return redirect('analysis')

    df_clean = df_from_session(df_clean_json)
    df_orig  = df_from_session(df_orig_json)
    mask_df  = df_from_session(mask_json).astype(bool)
    metrics  = json.loads(metrics_json)

    filename = request.session.get('filename', 'dataset.csv')
    total_filled = int(mask_df.values.sum())

    # Build highlighted HTML table
    rows_html = _build_highlighted_table(df_clean, mask_df)

    # Metrics summary list
    metrics_list = []
    for col, info in metrics.items():
        metrics_list.append({
            'column': col,
            'method': info.get('method', '—'),
            'type': info.get('type', '—'),
            'rmse': info.get('rmse', None),
        })

    context = {
        'active': 'results',
        'filename': filename,
        'total_filled': total_filled,
        'df_shape': df_clean.shape,
        'rows_html': rows_html,
        'columns': list(df_clean.columns),
        'metrics_list': metrics_list,
    }
    return render(request, 'predictor/results.html', context)


def _build_highlighted_table(df, mask_df, max_rows=50):
    """Build an HTML table string with highlighted (filled) cells."""
    cols = list(df.columns)
    header = '<thead><tr><th>#</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr></thead>'
    body_rows = []
    for i, (idx, row) in enumerate(df.head(max_rows).iterrows()):
        cells = [f'<td class="row-num">{idx}</td>']
        for col in cols:
            val = row[col]
            is_filled = bool(mask_df.loc[idx, col]) if idx in mask_df.index else False
            cell_class = 'filled-cell' if is_filled else ''
            display_val = '' if pd.isna(val) else str(round(val, 4) if isinstance(val, float) else val)
            cells.append(f'<td class="{cell_class}">{display_val}</td>')
        body_rows.append('<tr>' + ''.join(cells) + '</tr>')
    body = '<tbody>' + ''.join(body_rows) + '</tbody>'
    return f'<table class="result-table">{header}{body}</table>'


# ─────────────────────────────────────────────
#  VIEW: Download cleaned CSV
# ─────────────────────────────────────────────

def download_cleaned(request):
    df_clean_json = request.session.get('df_cleaned')
    if not df_clean_json:
        messages.warning(request, 'No cleaned data to download.')
        return redirect('results')

    df_clean = df_from_session(df_clean_json)

    filename = request.session.get('filename', 'dataset.csv')
    base_name = os.path.splitext(filename)[0]
    safe_base = base_name.replace(' ', '_')
    download_name = f'{safe_base}_cleaned.csv'

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{download_name}"'

    df_clean.to_csv(response, index=False, encoding='utf-8-sig')

    return response


# ─────────────────────────────────────────────
#  VIEW: Clear session / start over
# ─────────────────────────────────────────────

def clear_session(request):
    request.session.flush()
    messages.info(request, 'Session cleared. You can start fresh!')
    return redirect('home')

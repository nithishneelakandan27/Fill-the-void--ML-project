import os
import io
import json
import base64
import logging
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
from django.utils.html import escape, format_html
from django.utils.safestring import mark_safe

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .forms import DatasetUploadForm
from predictor.ml.orchestrator import orchestrate_imputation


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Helper: dataframe ↔ session serialization
# ─────────────────────────────────────────────

def df_to_session(df):
    """Convert DataFrame to JSON-serializable string for session storage."""
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
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
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

def ml_impute_orchestrated(df):
    """
    Impute missing values using the Orchestrator.
    Returns (cleaned_df, metrics_dict, filled_mask_df)
    """
    df_clean = df.copy()
    filled_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    metrics = {}
    
    # Identify columns with missing values
    cols_to_impute = df.columns[df.isnull().any()].tolist()
    
    for col in cols_to_impute:
        res = orchestrate_imputation(df, col)
        
        if "error" in res:
            logger.warning("Failed to impute %s: %s", col, res["error"])
            metrics[col] = {
                'method': f"Unable to impute ({res['error']})",
                'type': 'UNSUPPORTED',
                'imputed_count': 0,
                'remaining_missing': int(df[col].isnull().sum()),
                'evaluation': None,
                'evaluation_error': res['error'],
                'explanation': {'error': res['error']},
                'rmse': None,
            }
            continue
            
        # Update dataframe and mask
        df_clean[col] = res["imputed_df"][col]
        # Only mark filled values
        filled_mask[col] = df[col].isnull()
        
        # Format metrics for result view
        res_metrics = res.get("metrics", {})
        evaluation = res_metrics.get("evaluation", {})
        
        rmse_val = None
        if isinstance(evaluation, dict) and "rmse" in evaluation:
            rmse_val = round(float(evaluation["rmse"]), 4)
        
        metrics[col] = {
            'method': res["selected_strategy"],
            'type': res["type"],
            'imputed_count': res_metrics.get("n_imputed", int(df[col].isnull().sum())),
            'remaining_missing': res_metrics.get("n_remaining_missing", 0),
            'evaluation': evaluation if isinstance(evaluation, dict) else None,
            'evaluation_error': res_metrics.get("evaluation_error"),
            'explanation': res.get('explanation'),
            'rmse': rmse_val
        }
        
    return df_clean, metrics, filled_mask


def ml_impute(df):
    """Entry point for dataset imputation using Orchestrator."""
    return ml_impute_orchestrated(df)


def ml_impute_legacy(df):
    """
    Phase 1 Legacy Imputation Pipeline (preserved for backward compatibility).
    """
    df_clean = df.copy()
    numeric_cols, cat_cols = classify_columns(df)
    metrics = {}
    filled_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    all_missing_cols = [c for c in df.columns if df[c].isnull().all()]
    for col in all_missing_cols:
        col_type = 'numeric' if col in numeric_cols else 'categorical'
        metrics[col] = {
            'method': 'Unable to impute (100% missing values)',
            'type': col_type,
        }

    usable_cat_cols = [c for c in cat_cols if c not in all_missing_cols]
    for col in usable_cat_cols:
        null_mask = df_clean[col].isnull()
        if null_mask.any():
            mode_series = df_clean.loc[~null_mask, col].mode()
            if len(mode_series) > 0:
                df_clean.loc[null_mask, col] = mode_series.iloc[0]
                filled_mask.loc[null_mask, col] = True
                metrics[col] = {'method': 'Mode Imputation', 'type': 'categorical'}
            else:
                metrics[col] = {'method': 'Unable to impute (no observed values)', 'type': 'categorical'}

    usable_numeric_cols = [c for c in numeric_cols if c not in all_missing_cols]
    if usable_numeric_cols:
        numeric_df = df_clean[usable_numeric_cols].copy()
        missing_frac = numeric_df.isnull().mean()

        knn_cols = [c for c in usable_numeric_cols if 0 < missing_frac[c] <= 0.40]
        rf_cols = [c for c in usable_numeric_cols if missing_frac[c] > 0.40]

        if knn_cols:
            try:
                X_num = df_clean[usable_numeric_cols].copy()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_num)

                k = min(5, max(1, len(df_clean)))
                knn = KNNImputer(n_neighbors=k)
                X_imputed_scaled = knn.fit_transform(X_scaled)
                X_imputed = scaler.inverse_transform(X_imputed_scaled)

                imputed_df = pd.DataFrame(X_imputed, index=df_clean.index, columns=usable_numeric_cols)

                for col in knn_cols:
                    null_mask = df[col].isnull()
                    if null_mask.any():
                        df_clean.loc[null_mask, col] = imputed_df.loc[null_mask, col]
                        filled_mask.loc[null_mask, col] = True
                        metrics[col] = {'method': f'KNN Imputation (k={k}, scaled)', 'type': 'numeric'}
            except Exception as e:
                logger.warning("KNN imputation encountered an error: %s. Falling back to median.", e)
                for col in knn_cols:
                    null_mask = df[col].isnull()
                    median_val = df_clean.loc[~null_mask, col].median()
                    if pd.notna(median_val):
                        df_clean.loc[null_mask, col] = median_val
                        filled_mask.loc[null_mask, col] = True
                        metrics[col] = {'method': 'Median Imputation (KNN fallback)', 'type': 'numeric'}
                    else:
                        metrics[col] = {'method': 'Unable to impute', 'type': 'numeric'}

        for col in rf_cols:
            null_mask = df[col].isnull()
            not_null_mask = ~null_mask

            feature_cols = [c for c in usable_numeric_cols if c != col]
            if not feature_cols:
                median_val = df_clean.loc[not_null_mask, col].median()
                if pd.notna(median_val):
                    df_clean.loc[null_mask, col] = median_val
                    filled_mask.loc[null_mask, col] = True
                    metrics[col] = {'method': 'Median Imputation (fallback)', 'type': 'numeric'}
                else:
                    metrics[col] = {'method': 'Unable to impute', 'type': 'numeric'}
                continue

            X_all = df_clean[feature_cols].copy()
            col_medians = X_all.median()
            X_all = X_all.fillna(col_medians).fillna(0.0)

            X_train = X_all[not_null_mask]
            y_train = df_clean.loc[not_null_mask, col]
            X_pred = X_all[null_mask]

            if len(X_train) < 5:
                median_val = y_train.median()
                if pd.notna(median_val):
                    df_clean.loc[null_mask, col] = median_val
                    filled_mask.loc[null_mask, col] = True
                    metrics[col] = {'method': 'Median (insufficient data)', 'type': 'numeric'}
                else:
                    metrics[col] = {'method': 'Unable to impute', 'type': 'numeric'}
                continue

            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                preds = rf.predict(X_pred)
                df_clean.loc[null_mask, col] = preds
                filled_mask.loc[null_mask, col] = True

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
            except Exception as e:
                logger.warning("Random Forest failed for column %s: %s. Falling back to median.", col, e)
                median_val = y_train.median()
                if pd.notna(median_val):
                    df_clean.loc[null_mask, col] = median_val
                    filled_mask.loc[null_mask, col] = True
                    metrics[col] = {'method': 'Median Imputation (RF fallback)', 'type': 'numeric'}
                else:
                    metrics[col] = {'method': 'Unable to impute', 'type': 'numeric'}

    return df_clean, metrics, filled_mask


# ─────────────────────────────────────────────
#  HTML Table Helpers (Safe Escaping)
# ─────────────────────────────────────────────

def _build_preview_table(df, max_rows=20):
    """Build an HTML preview table string with safely escaped values."""
    cols = list(df.columns)
    escaped_headers = ''.join(format_html('<th>{}</th>', str(c)) for c in cols)
    header = format_html('<thead><tr><th>#</th>{}</tr></thead>', mark_safe(escaped_headers))

    body_rows = []
    for idx, row in df.head(max_rows).iterrows():
        cells = [format_html('<th>{}</th>', str(idx))]
        for col in cols:
            val = row[col]
            if pd.isna(val):
                cells.append(mark_safe('<td><span class="na-val">NaN</span></td>'))
            else:
                display_val = str(round(float(val), 4) if isinstance(val, (float, np.floating)) else val)
                cells.append(format_html('<td>{}</td>', display_val))
        body_rows.append(format_html('<tr>{}</tr>', mark_safe(''.join(cells))))

    body = format_html('<tbody>{}</tbody>', mark_safe(''.join(body_rows)))
    return format_html('<table class="preview-table">{}{}</table>', header, body)


def _build_highlighted_table(df, mask_df, max_rows=50):
    """Build an HTML table string with highlighted (filled) cells, fully HTML-escaped."""
    cols = list(df.columns)
    escaped_headers = ''.join(format_html('<th>{}</th>', str(c)) for c in cols)
    header = format_html('<thead><tr><th>#</th>{}</tr></thead>', mark_safe(escaped_headers))

    body_rows = []
    for idx, row in df.head(max_rows).iterrows():
        cells = [format_html('<td class="row-num">{}</td>', str(idx))]
        for col in cols:
            val = row[col]
            is_filled = False
            if idx in mask_df.index and col in mask_df.columns:
                is_filled = bool(mask_df.loc[idx, col])
            cell_class = 'filled-cell' if is_filled else ''

            if pd.isna(val):
                display_val = ''
            elif isinstance(val, (float, np.floating)):
                display_val = str(round(float(val), 4))
            else:
                display_val = str(val)

            if cell_class:
                cells.append(format_html('<td class="{}">{}</td>', cell_class, display_val))
            else:
                cells.append(format_html('<td>{}</td>', display_val))

        body_rows.append(format_html('<tr>{}</tr>', mark_safe(''.join(cells))))

    body = format_html('<tbody>{}</tbody>', mark_safe(''.join(body_rows)))
    return format_html('<table class="result-table">{}{}</table>', header, body)


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
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['dataset']
            df = form.cleaned_df
            filename = uploaded_file.name

            # Store in session
            try:
                request.session['df_original'] = df_to_session(df)
                request.session['filename'] = filename
                request.session['df_shape'] = list(df.shape)
            except Exception as e:
                logger.error("Session serialization failed: %s", e)
                messages.error(request, "Failed to store dataset in session. Please try again.")
                return redirect('upload')

            messages.success(request, f'✅ "{filename}" uploaded successfully!')
            return redirect('analysis')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
            return render(request, 'predictor/upload.html', {'form': form, 'active': 'upload'})

    form = DatasetUploadForm()
    return render(request, 'predictor/upload.html', {'form': form, 'active': 'upload'})


# ─────────────────────────────────────────────
#  VIEW: Analysis
# ─────────────────────────────────────────────

def analysis(request):
    df_json = request.session.get('df_original')
    if not df_json:
        messages.warning(request, 'Please upload a dataset first.')
        return redirect('upload')

    try:
        df = df_from_session(df_json)
    except Exception as e:
        logger.error("Failed to restore dataframe from session: %s", e)
        messages.error(request, 'Session data was corrupted. Please upload your dataset again.')
        return redirect('upload')

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

    # Preview (first 20 rows) - safely generated and escaped
    preview_html = _build_preview_table(df, max_rows=20)

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

    try:
        df = df_from_session(df_json)
    except Exception as e:
        logger.error("Failed to restore dataframe in run_prediction: %s", e)
        messages.error(request, 'Session corrupted. Please upload again.')
        return redirect('upload')

    # Run Orchestrator via ml_impute
    df_clean, metrics, filled_mask = ml_impute(df)

    # Store results in session
    try:
        request.session['df_cleaned'] = df_to_session(df_clean)
        request.session['filled_mask'] = df_to_session(filled_mask.astype(int))
        request.session['metrics'] = json.dumps(metrics)
    except Exception as e:
        logger.error("Failed to serialize cleaned data to session: %s", e)
        messages.error(request, 'Failed to store cleaned dataset. Please try again.')
        return redirect('analysis')

    return redirect('results')


# ─────────────────────────────────────────────
#  VIEW: Results
# ─────────────────────────────────────────────

def results(request):
    df_clean_json = request.session.get('df_cleaned')
    df_orig_json = request.session.get('df_original')
    mask_json = request.session.get('filled_mask')
    metrics_json = request.session.get('metrics')

    if not df_clean_json:
        messages.warning(request, 'No results yet. Please run ML prediction first.')
        return redirect('analysis')

    try:
        df_clean = df_from_session(df_clean_json)
        mask_df = df_from_session(mask_json).astype(bool)
        metrics = json.loads(metrics_json) if metrics_json else {}
    except Exception as e:
        logger.error("Failed to restore cleaned results from session: %s", e)
        messages.error(request, 'Session results corrupted. Please run prediction again.')
        return redirect('analysis')

    filename = request.session.get('filename', 'dataset.csv')
    total_filled = int(mask_df.values.sum())

    # Build safely-escaped highlighted HTML table
    rows_html = _build_highlighted_table(df_clean, mask_df, max_rows=50)

    # Metrics summary list
    metrics_list = []
    for col, info in metrics.items():
        metrics_list.append({
            'column': col,
            'method': info.get('method', '—'),
            'type': info.get('type', '—'),
            'imputed_count': info.get('imputed_count', 0),
            'remaining_missing': info.get('remaining_missing', 0),
            'evaluation': info.get('evaluation'),
            'evaluation_error': info.get('evaluation_error'),
            'explanation': info.get('explanation'),
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


# ─────────────────────────────────────────────
#  VIEW: Download cleaned CSV
# ─────────────────────────────────────────────

def download_cleaned(request):
    df_clean_json = request.session.get('df_cleaned')
    if not df_clean_json:
        messages.warning(request, 'No cleaned data to download.')
        return redirect('results')

    try:
        df_clean = df_from_session(df_clean_json)
    except Exception as e:
        logger.error("Failed to restore cleaned data for download: %s", e)
        messages.error(request, 'Session data corrupted. Please run prediction again.')
        return redirect('results')

    filename = request.session.get('filename', 'dataset.csv')
    base_name = os.path.splitext(filename)[0]
    safe_base = "".join(c for c in base_name if c.isalnum() or c in ('_', '-')).strip() or 'dataset'
    download_name = f'{safe_base}_cleaned.csv'

    response = HttpResponse(content_type='text/csv; charset=utf-8')
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


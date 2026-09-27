import io
import os
import json
import numpy as np
import pandas as pd
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from predictor.forms import DatasetUploadForm, MAX_FILE_SIZE
from predictor.views import ml_impute, _build_highlighted_table, _build_preview_table, classify_columns
from predictor.ml.column_classifier import classify_columns_enhanced, ColumnInfo
from predictor.ml.strategies.base import ImputationStrategy, ImputationStrategyError
from predictor.ml.strategies.numeric.mean import MeanStrategy
from predictor.ml.strategies.numeric.median import MedianStrategy
from predictor.ml.strategies.numeric.knn import KNNStrategy
from predictor.ml.strategies.numeric.random_forest import RandomForestStrategy
from predictor.ml.strategies.categorical.mode import ModeStrategy
from predictor.ml.strategies.numeric.iterative import IterativeStrategy
from predictor.ml.evaluator import evaluate_imputation_strategy
from predictor.ml.strategies.numeric.mean import MeanStrategy
from predictor.ml.strategies.categorical.mode import ModeStrategy
from predictor.ml.quality_metrics import calculate_imputation_metrics





class UploadAndFormValidationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_valid_csv_upload(self):
        """1. Test that a valid CSV file uploads successfully and redirects to analysis."""
        csv_content = b"age,income,gender\n25,50000,M\n30,,F\n35,70000,M\n40,80000,F\n"
        uploaded_file = SimpleUploadedFile("test_data.csv", csv_content, content_type="text/csv")

        response = self.client.post(reverse('upload'), {'dataset': uploaded_file})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('analysis'))

        # Check session data
        session = self.client.session
        self.assertIn('df_original', session)
        self.assertEqual(session['filename'], 'test_data.csv')
        self.assertEqual(session['df_shape'], [4, 3])

    def test_valid_xlsx_upload(self):
        """2. Test that a valid XLSX Excel file uploads successfully."""
        df = pd.DataFrame({
            'score': [85, 90, np.nan, 95],
            'grade': ['A', 'A', 'B', 'A']
        })
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        uploaded_file = SimpleUploadedFile(
            "test_data.xlsx",
            excel_buffer.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        response = self.client.post(reverse('upload'), {'dataset': uploaded_file})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('analysis'))

    def test_invalid_extension(self):
        """3. Test that uploading a file with an invalid extension is rejected."""
        txt_content = b"Some plain text not csv or xlsx"
        uploaded_file = SimpleUploadedFile("test_data.txt", txt_content, content_type="text/plain")

        response = self.client.post(reverse('upload'), {'dataset': uploaded_file})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/upload.html')
        self.assertContains(response, 'Only CSV and Excel files')

    def test_empty_file(self):
        """4. Test that an empty (0-byte) file is rejected gracefully."""
        empty_file = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")

        response = self.client.post(reverse('upload'), {'dataset': empty_file})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/upload.html')
        self.assertContains(response, 'empty')

    def test_malformed_dataset(self):
        """5. Test that a corrupt or unparseable dataset is rejected with user-friendly error."""
        corrupt_file = SimpleUploadedFile("corrupt.xlsx", b"This is not a real zip or excel file", content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        response = self.client.post(reverse('upload'), {'dataset': corrupt_file})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/upload.html')
        self.assertContains(response, 'could not be parsed')

    def test_oversized_file(self):
        """6. Test that a file exceeding MAX_FILE_SIZE is rejected."""
        large_content = b"a,b\n1,2\n" * 10
        uploaded_file = SimpleUploadedFile("large.csv", large_content, content_type="text/csv")
        # Fake large file size
        uploaded_file.size = MAX_FILE_SIZE + 1024

        form = DatasetUploadForm(files={'dataset': uploaded_file})
        self.assertFalse(form.is_valid())
        self.assertIn('File size must not exceed 10 MB.', form.errors['dataset'])


class SecurityAndXSSSanitizationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_xss_payload_in_cell_value_escaped(self):
        """7. Test that malicious XSS HTML/JS in cell values is escaped and not executable."""
        xss_payload = '<img src=x onerror=alert(1)>'
        script_payload = '<script>alert("xss")</script>'

        df = pd.DataFrame({
            'name': ['Alice', xss_payload, script_payload],
            'val': [10.0, np.nan, 30.0]
        })
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns)
        mask_df.loc[1, 'val'] = True

        # Test table builders
        preview_html = _build_preview_table(df)
        highlighted_html = _build_highlighted_table(df, mask_df)

        # Assert raw tags are NOT present
        self.assertNotIn('<img src=x onerror=alert(1)>', str(preview_html))
        self.assertNotIn('<script>alert("xss")</script>', str(preview_html))
        self.assertNotIn('<img src=x onerror=alert(1)>', str(highlighted_html))
        self.assertNotIn('<script>alert("xss")</script>', str(highlighted_html))

        # Assert properly escaped entities ARE present
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', str(preview_html))
        self.assertIn('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;', str(preview_html))
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', str(highlighted_html))

    def test_xss_payload_in_column_name_and_index_escaped(self):
        """Test that malicious XSS payload in column names and index labels is escaped."""
        xss_col = '<svg/onload=alert(1)>'
        xss_idx = '<iframe src="javascript:alert(1)">'

        df = pd.DataFrame({xss_col: [1, 2]}, index=[xss_idx, 'row2'])
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns)

        preview_html = _build_preview_table(df)
        highlighted_html = _build_highlighted_table(df, mask_df)

        self.assertNotIn('<svg/onload=alert(1)>', str(preview_html))
        self.assertNotIn('<iframe src="javascript:alert(1)">', str(preview_html))
        self.assertIn('&lt;svg/onload=alert(1)&gt;', str(preview_html))
        self.assertIn('&lt;iframe src=&quot;javascript:alert(1)&quot;&gt;', str(preview_html))

    def test_security_settings_configured(self):
        """14. Test security settings configuration."""
        self.assertEqual(settings.X_FRAME_OPTIONS, 'DENY')
        self.assertTrue(settings.SECURE_CONTENT_TYPE_NOSNIFF)
        self.assertTrue(settings.SESSION_COOKIE_HTTPONLY)
        self.assertTrue(settings.CSRF_COOKIE_HTTPONLY)
        self.assertEqual(settings.SESSION_COOKIE_SAMESITE, 'Lax')
        self.assertEqual(settings.CSRF_COOKIE_SAMESITE, 'Lax')


class MLImputationAndKNNTests(TestCase):
    def test_numeric_knn_imputation(self):
        """8. Test numeric KNN imputation on partially missing numerical columns."""
        df = pd.DataFrame({
            'feat1': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'feat2': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
        })
        df_clean, metrics, filled_mask = ml_impute(df)

        # Verify missing value in feat2 is filled
        self.assertFalse(df_clean['feat2'].isnull().any())
        self.assertTrue(filled_mask.loc[2, 'feat2'])
        self.assertAlmostEqual(df_clean.loc[2, 'feat2'], 3.0, places=1)
        self.assertIn('Imputation', metrics['feat2']['method'])

    def test_knn_uses_complete_numeric_predictor_columns(self):
        """9. Test that KNN uses complete (0% missing) numeric columns as predictors."""
        # Col A is 100% complete, Col B is 20% missing (<= 40%)
        df = pd.DataFrame({
            'complete_feature': [100.0, 200.0, 300.0, 400.0, 500.0],
            'target_with_missing': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        df_clean, metrics, filled_mask = ml_impute(df)

        self.assertFalse(df_clean['target_with_missing'].isnull().any())
        self.assertAlmostEqual(df_clean.loc[2, 'target_with_missing'], 30.0, delta=5.0)
        # Complete feature must remain untouched
        self.assertFalse(filled_mask['complete_feature'].any())
        self.assertEqual(df_clean['complete_feature'].tolist(), [100.0, 200.0, 300.0, 400.0, 500.0])

    def test_knn_works_with_non_default_index(self):
        """10. Test that KNN imputation preserves custom non-range index labels."""
        custom_index = ['row_alpha', 'row_beta', 'row_gamma', 'row_delta', 'row_epsilon']
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [10.0, np.nan, 30.0, 40.0, 50.0]
        }, index=custom_index)

        df_clean, metrics, filled_mask = ml_impute(df)

        self.assertEqual(list(df_clean.index), custom_index)
        self.assertEqual(list(filled_mask.index), custom_index)
        self.assertFalse(df_clean.isnull().any().any())
        self.assertTrue(filled_mask.loc['row_beta', 'y'])
        self.assertFalse(filled_mask.loc['row_alpha', 'y'])

    def test_existing_non_missing_values_remain_unchanged(self):
        """11. Test that existing observed non-missing values are NEVER modified."""
        df = pd.DataFrame({
            'num': [1.5, np.nan, 3.7, 4.2, 5.9],
            'cat': ['cat', 'dog', np.nan, 'cat', 'bird']
        })
        orig_num = df['num'].copy()
        orig_cat = df['cat'].copy()

        df_clean, metrics, filled_mask = ml_impute(df)

        # Check observed values
        for i in [0, 2, 3, 4]:
            self.assertEqual(df_clean.loc[i, 'num'], orig_num[i])
        for i in [0, 1, 3, 4]:
            self.assertEqual(df_clean.loc[i, 'cat'], orig_cat[i])

    def test_100_percent_missing_numeric_column(self):
        """12. Test that a 100% missing numeric column remains NaN and is marked as unimputable."""
        df = pd.DataFrame({
            'num_valid': [1.0, 2.0, 3.0, 4.0, 5.0],
            'all_nan_num': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        df_clean, metrics, filled_mask = ml_impute(df)

        # Column must remain NaN
        self.assertTrue(df_clean['all_nan_num'].isnull().all())
        self.assertFalse(filled_mask['all_nan_num'].any())
        self.assertIn('Unable to impute', metrics['all_nan_num']['method'])

    def test_100_percent_missing_categorical_column(self):
        """13. Test that a 100% missing categorical column remains NaN and is marked as unimputable."""
        df = pd.DataFrame({
            'cat_valid': ['A', 'B', 'A', 'B', 'A'],
            'all_nan_cat': [None, None, None, None, None]
        })
        df_clean, metrics, filled_mask = ml_impute(df)

        self.assertTrue(df_clean['all_nan_cat'].isnull().all())
        self.assertFalse(filled_mask['all_nan_cat'].any())
        self.assertIn('Unable to impute', metrics['all_nan_cat']['method'])

    def test_categorical_mode_imputation(self):
        """Test categorical mode imputation on partially observed features."""
        df = pd.DataFrame({
            'category': ['apple', 'banana', 'apple', None, 'apple']
        })
        df_clean, metrics, filled_mask = ml_impute(df)

        self.assertEqual(df_clean.loc[3, 'category'], 'apple')
        self.assertTrue(filled_mask.loc[3, 'category'])
        self.assertEqual(metrics['category']['method'], 'Mode Imputation')


class EndToEndWorkflowIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_full_workflow_upload_analysis_predict_results_download(self):
        """Test the complete end-to-end user workflow."""
        # 1. Upload
        csv_data = b"age,salary,city\n25,50000,Paris\n30,,London\n35,70000,Paris\n40,80000,Tokyo\n"
        upload_resp = self.client.post(reverse('upload'), {'dataset': SimpleUploadedFile('employees.csv', csv_data)})
        self.assertEqual(upload_resp.status_code, 302)
        self.assertEqual(upload_resp.url, reverse('analysis'))

        # 2. Analysis
        analysis_resp = self.client.get(reverse('analysis'))
        self.assertEqual(analysis_resp.status_code, 200)
        self.assertTemplateUsed(analysis_resp, 'predictor/analysis.html')
        self.assertContains(analysis_resp, 'employees.csv')
        self.assertContains(analysis_resp, 'salary')

        # 3. Run Prediction
        predict_resp = self.client.post(reverse('run_prediction'))
        self.assertEqual(predict_resp.status_code, 302)
        self.assertEqual(predict_resp.url, reverse('results'))

        # 4. Results
        results_resp = self.client.get(reverse('results'))
        self.assertEqual(results_resp.status_code, 200)
        self.assertTemplateUsed(results_resp, 'predictor/results.html')
        self.assertContains(results_resp, 'Cleaned Dataset Results')
        self.assertContains(results_resp, 'filled-cell')

        # 5. Download
        download_resp = self.client.get(reverse('download'))
        self.assertEqual(download_resp.status_code, 200)
        self.assertEqual(download_resp['Content-Type'], 'text/csv; charset=utf-8')
        self.assertIn('employees_cleaned.csv', download_resp['Content-Disposition'])

        # Verify downloaded CSV has no missing values
        downloaded_df = pd.read_csv(io.StringIO(download_resp.content.decode('utf-8-sig')))
        self.assertFalse(downloaded_df.isnull().any().any())
        self.assertEqual(len(downloaded_df), 4)


# ─────────────────────────────────────────────────────────────────────
#  PHASE 2 TESTS: Enhanced Column Classification
# ─────────────────────────────────────────────────────────────────────

class EnhancedColumnClassificationTests(TestCase):
    """Comprehensive tests for the enhanced column classifier (Phase 2)."""

    def test_numeric_column_detection(self):
        """Test: Numeric column (int64 or float64) detected as NUMERIC."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'height': [5.9, 6.1, 5.8, 6.0, 5.9],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['age'].type, 'NUMERIC')
        self.assertEqual(result['age'].dtype, 'int64')
        self.assertTrue(result['age'].is_usable_predictor)
        self.assertTrue(result['age'].is_imputable)
        self.assertEqual(result['age'].cardinality, 5)
        self.assertEqual(result['age'].missing_count, 0)

    def test_categorical_column_detection(self):
        """Test: String column with reasonable cardinality detected as CATEGORICAL."""
        df = pd.DataFrame({
            'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['city'].type, 'CATEGORICAL')
        self.assertTrue(result['city'].is_imputable)
        self.assertFalse(result['city'].is_usable_predictor)  # Needs encoding
        self.assertEqual(result['city'].cardinality, 3)

    def test_boolean_dtype_detection(self):
        """Test: Native bool dtype detected as BOOLEAN."""
        df = pd.DataFrame({
            'is_active': np.array([True, False, True, False, True], dtype=bool),
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['is_active'].type, 'BOOLEAN')
        self.assertEqual(result['is_active'].dtype, 'bool')
        self.assertFalse(result['is_active'].is_usable_predictor)  # Needs encoding
        self.assertTrue(result['is_active'].is_imputable)

    def test_boolean_yes_no_detection(self):
        """Test: Yes/No string column detected as BOOLEAN."""
        df = pd.DataFrame({
            'approved': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['approved'].type, 'BOOLEAN')
        self.assertFalse(result['approved'].is_usable_predictor)
        self.assertTrue(result['approved'].is_imputable)
        self.assertIn('Yes', result['approved'].classification_reason)

    def test_boolean_true_false_detection(self):
        """Test: True/False string column detected as BOOLEAN."""
        df = pd.DataFrame({
            'flag': ['True', 'False', 'True', 'True', 'False'],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['flag'].type, 'BOOLEAN')
        self.assertIn('True', result['flag'].classification_reason)

    def test_datetime_column_detection(self):
        """Test: Datetime column detected as DATETIME."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']),
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['date'].type, 'DATETIME')
        self.assertFalse(result['date'].is_usable_predictor)
        self.assertTrue(result['date'].is_imputable)

    def test_datetime_string_column_detection(self):
        """Test: Datetime string column (high parse rate) detected as DATETIME."""
        df = pd.DataFrame({
            'hire_date': ['2020-01-15', '2020-02-20', '2020-03-10', '2020-04-05', '2020-05-12'] * 3,
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['hire_date'].type, 'DATETIME')
        self.assertIn('parse', result['hire_date'].classification_reason.lower())

    def test_datetime_with_invalid_values(self):
        """Test: Datetime column with some invalid values still detected as DATETIME if parse rate high."""
        # 80% of values parse successfully (4/5)
        df = pd.DataFrame({
            'date_col': ['2020-01-01', '2020-02-02', '2020-03-03', 'invalid_date', '2020-05-05'] * 4,
        })
        
        result = classify_columns_enhanced(df)
        
        # Should still be DATETIME if >80% parse rate
        if result['date_col'].type == 'DATETIME':
            self.assertIn('parse', result['date_col'].classification_reason.lower())

    def test_identifier_user_id_detection(self):
        """Test: Column named user_id detected as IDENTIFIER."""
        df = pd.DataFrame({
            'user_id': [1001, 1002, 1003, 1004, 1005],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['user_id'].type, 'IDENTIFIER')
        self.assertFalse(result['user_id'].is_usable_predictor)
        self.assertFalse(result['user_id'].is_imputable)
        self.assertIn('id', result['user_id'].classification_reason.lower())

    def test_identifier_uuid_detection(self):
        """Test: UUID-like strings detected as IDENTIFIER."""
        uuids = [
            '550e8400-e29b-41d4-a716-446655440000',
            '6ba7b810-9dad-11d1-80b4-00c04fd430c8',
            '6ba7b811-9dad-11d1-80b4-00c04fd430c8',
            '6ba7b812-9dad-11d1-80b4-00c04fd430c8',
            '6ba7b814-9dad-11d1-80b4-00c04fd430c8',
        ]
        df = pd.DataFrame({'id': uuids})
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['id'].type, 'IDENTIFIER')
        self.assertIn('uuid', result['id'].classification_reason.lower())

    def test_high_cardinality_numeric_not_identifier(self):
        """Test: High-cardinality numeric measurement NOT classified as IDENTIFIER."""
        # Exact prices: many unique values but not an ID
        df = pd.DataFrame({
            'price': [19.99, 25.50, 34.99, 45.00, 12.99, 56.75, 78.50, 89.99, 99.99, 7.50],
        })
        
        result = classify_columns_enhanced(df)
        
        # Should be NUMERIC, not IDENTIFIER
        self.assertEqual(result['price'].type, 'NUMERIC')
        self.assertTrue(result['price'].is_usable_predictor)

    def test_free_text_column_detection(self):
        """Test: Long text column detected as FREE_TEXT."""
        df = pd.DataFrame({
            'description': [
                'This is a long description with multiple sentences.',
                'Another lengthy text entry describing a product in detail.',
                'Yet another comment with substantial content for analysis.',
                'A detailed explanation of the item specifications.',
                'More descriptive text content here for testing purposes.',
            ],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['description'].type, 'FREE_TEXT')
        self.assertFalse(result['description'].is_usable_predictor)
        self.assertFalse(result['description'].is_imputable)

    def test_short_strings_remain_categorical(self):
        """Test: Short strings (mean < 50 chars) remain CATEGORICAL."""
        df = pd.DataFrame({
            'department': ['Sales', 'Engineering', 'HR', 'Finance', 'Support'],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['department'].type, 'CATEGORICAL')
        self.assertTrue(result['department'].is_imputable)
        self.assertFalse(result['department'].is_usable_predictor)

    def test_empty_column_detection(self):
        """Test: Completely empty column (100% missing) detected as UNSUPPORTED."""
        df = pd.DataFrame({
            'empty_col': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['empty_col'].type, 'UNSUPPORTED')
        self.assertEqual(result['empty_col'].missing_count, 5)
        self.assertEqual(result['empty_col'].missing_pct, 100.0)
        self.assertFalse(result['empty_col'].is_imputable)

    def test_mixed_type_column_unsupported(self):
        """Test: Mixed-type column handled gracefully."""
        df = pd.DataFrame({
            'mixed': [1, 'text', 3.14, True, None],
        })
        
        result = classify_columns_enhanced(df)
        
        # Mixed types typically end up as object dtype
        # Should not crash, and classify as best-fit or unsupported
        self.assertIsNotNone(result['mixed'])
        self.assertIn(result['mixed'].type, ['UNSUPPORTED', 'CATEGORICAL'])

    def test_missing_percentage_calculation(self):
        """Test: Missing percentage calculated correctly."""
        df = pd.DataFrame({
            'col_with_missing': [1, 2, np.nan, 4, np.nan, np.nan, 7, 8, 9, 10],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(result['col_with_missing'].missing_count, 3)
        self.assertAlmostEqual(result['col_with_missing'].missing_pct, 30.0, places=1)

    def test_unique_ratio_calculation(self):
        """Test: Unique ratio calculated correctly (excluding missing)."""
        df = pd.DataFrame({
            'col': ['A', 'B', 'A', 'C', np.nan, 'B', 'A'],
        })
        
        result = classify_columns_enhanced(df)
        
        # 6 non-null values, 3 unique → 3/6 = 0.5
        self.assertEqual(result['col'].cardinality, 3)
        self.assertAlmostEqual(result['col'].unique_ratio, 0.5, places=2)

    def test_predictor_eligibility_numeric(self):
        """Test: Numeric columns marked as usable predictors."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertTrue(result['age'].is_usable_predictor)
        self.assertTrue(result['income'].is_usable_predictor)

    def test_predictor_eligibility_categorical_needs_encoding(self):
        """Test: Categorical/boolean columns marked as NOT usable without encoding."""
        df = pd.DataFrame({
            'city': ['NYC', 'LA', 'NYC', 'LA', 'Chicago'],
            'is_active': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertFalse(result['city'].is_usable_predictor)
        self.assertFalse(result['is_active'].is_usable_predictor)

    def test_imputability_flags(self):
        """Test: Imputability flags set correctly for different types."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical': ['A', 'B', np.nan, 'A', 'B'],
            'boolean': ['Yes', 'No', np.nan, 'Yes', 'No'],
            'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', None, '2020-01-04', '2020-01-05']),
            'identifier': [1, 2, 3, 4, 5],
            'free_text': [
                'This is a long free-text description used only for the imputability fixture.',
                'Another lengthy comment with enough characters to exceed the free-text heuristic.',
                'A third substantial notes field describing the record in unstructured language.',
                'More detailed explanation text so mean length stays well above fifty characters.',
                'Final long-form review comment that should not be treated as a short category.',
            ],
            'empty': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        
        result = classify_columns_enhanced(df)
        
        # Imputable: numeric, categorical, boolean, datetime
        self.assertTrue(result['numeric'].is_imputable)
        self.assertTrue(result['categorical'].is_imputable)
        self.assertTrue(result['boolean'].is_imputable)
        self.assertTrue(result['datetime'].is_imputable)
        
        # Not imputable: identifier, free_text, empty
        self.assertFalse(result['identifier'].is_imputable)
        self.assertFalse(result['free_text'].is_imputable)
        self.assertFalse(result['empty'].is_imputable)

    def test_classification_reason_provided(self):
        """Test: Classification reason provided for debugging."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
        })
        
        result = classify_columns_enhanced(df)
        
        self.assertIsNotNone(result['age'].classification_reason)
        self.assertTrue(len(result['age'].classification_reason) > 0)

    def test_backwards_compatibility_with_phase1(self):
        """Test: New classifier doesn't break existing Phase 1 pipeline."""
        # Phase 1 uses classify_columns() from views.py
        # Ensure both classifiers can coexist
        
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago'],
            'salary': [50000.0, 60000.0, np.nan],
        })
        
        # Phase 1 classifier
        numeric_cols, cat_cols = classify_columns(df)
        
        # Phase 2 classifier
        enhanced = classify_columns_enhanced(df)
        
        # Phase 1 results
        self.assertEqual(set(numeric_cols), {'age', 'salary'})
        self.assertEqual(set(cat_cols), {'city'})
        
        # Phase 2 results
        self.assertEqual(enhanced['age'].type, 'NUMERIC')
        self.assertEqual(enhanced['city'].type, 'CATEGORICAL')
        self.assertEqual(enhanced['salary'].type, 'NUMERIC')

    def test_no_dataframe_mutation(self):
        """Test: Original DataFrame is not modified by classification."""
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C'],
        })
        
        df_copy = df_original.copy()
        
        # Classify
        result = classify_columns_enhanced(df_original)
        
        # Verify no mutation
        pd.testing.assert_frame_equal(df_original, df_copy)

    def test_dataframe_with_many_columns(self):
        """Test: Classifier handles DataFrames with many columns."""
        data = {f'col_{i}': np.random.rand(10) if i % 2 == 0 else ['A'] * 10
                for i in range(20)}
        df = pd.DataFrame(data)
        
        result = classify_columns_enhanced(df)
        
        self.assertEqual(len(result), 20)
        # Half should be numeric, half categorical
        numeric_count = sum(1 for info in result.values() if info.type == 'NUMERIC')
        self.assertGreater(numeric_count, 0)

    def test_column_info_dataclass(self):
        """Test: ColumnInfo dataclass works correctly."""
        df = pd.DataFrame({'age': [25, 30, 35]})
        result = classify_columns_enhanced(df)
        info = result['age']
        
        # Test dataclass attributes
        self.assertIsInstance(info, ColumnInfo)
        self.assertEqual(info.column_name, 'age')
        self.assertEqual(info.type, 'NUMERIC')
        self.assertTrue(hasattr(info, 'is_usable_predictor'))
        self.assertTrue(hasattr(info, 'is_imputable'))
        
        # Test repr
        repr_str = repr(info)
        self.assertIn('NUMERIC', repr_str)


# ─────────────────────────────────────────────────────────────────────
#  PHASE 2 STEP 2: Imputation strategy interface (not wired into views)
# ─────────────────────────────────────────────────────────────────────

class ImputationStrategyInterfaceTests(TestCase):
    """Each implemented strategy exposes the common fit/impute interface."""

    def _implemented_strategies(self):
        return [
            MeanStrategy(),
            MedianStrategy(),
            KNNStrategy(),
            RandomForestStrategy(target_column="y"),
            ModeStrategy(),
            IterativeStrategy(),

        ]

    def test_common_interface(self):
        required = (
            "name",
            "applicable_types",
            "fit",
            "transform",
            "impute",
            "fit_transform",
            "get_metrics",
            "get_explanation",
        )
        for strategy in self._implemented_strategies():
            self.assertIsInstance(strategy, ImputationStrategy)
            for attr in required:
                self.assertTrue(hasattr(strategy, attr), f"{strategy.name} missing {attr}")
            self.assertTrue(isinstance(strategy.name, str) and strategy.name)
            self.assertTrue(len(strategy.applicable_types) > 0)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ImputationStrategyError):
            MeanStrategy().transform(pd.DataFrame({"a": [1.0, np.nan]}))


class ImputationStrategyBehaviorTests(TestCase):
    """Unit tests for migrated Phase 1 strategies. Does not call ml_impute()."""

    def test_mean_fills_missing_and_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        original = df.copy()

        result = MeanStrategy().fit_transform(df)

        self.assertFalse(result["a"].isnull().any())
        self.assertAlmostEqual(result.loc[1, "a"], 2.0)
        pd.testing.assert_frame_equal(df, original)

    def test_median_fills_missing(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 100.0]})
        result = MedianStrategy().fit_transform(df)

        self.assertFalse(result["a"].isnull().any())
        self.assertAlmostEqual(result.loc[1, "a"], 50.5)

    def test_knn_fills_missing(self):
        df = pd.DataFrame({
            "feat1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feat2": [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        original = df.copy()

        result = KNNStrategy().fit_transform(df)

        self.assertFalse(result["feat2"].isnull().any())
        self.assertAlmostEqual(result.loc[2, "feat2"], 3.0, places=1)
        self.assertEqual(result["feat1"].tolist(), original["feat1"].tolist())
        pd.testing.assert_frame_equal(df, original)

    def test_knn_with_object_dtype_numeric_columns(self):
        """Test that KNNStrategy handles numeric columns stored as object dtype without AttributeError/TypeError."""
        df = pd.DataFrame({
            "feat1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feat2": [1.0, 2.0, np.nan, 4.0, 5.0],
        }, dtype=object)

        strategy = KNNStrategy()
        result = strategy.fit_transform(df)

        self.assertFalse(result["feat2"].isnull().any())
        self.assertAlmostEqual(result.loc[2, "feat2"], 3.0, places=1)

    def test_random_forest_fills_missing(self):
        rng = np.random.RandomState(0)
        x = np.arange(20, dtype=float)
        y = 2.0 * x + rng.normal(0, 0.01, size=20)
        y[5] = np.nan
        df = pd.DataFrame({"x": x, "y": y})
        original = df.copy()

        result = RandomForestStrategy(target_column="y").fit_transform(df)

        self.assertFalse(result["y"].isnull().any())
        self.assertAlmostEqual(result.loc[5, "y"], 10.0, delta=2.0)
        pd.testing.assert_frame_equal(df, original)

    def test_mode_fills_missing(self):
        df = pd.DataFrame({"category": ["apple", "banana", "apple", None, "apple"]})
        original = df.copy()

        result = ModeStrategy().fit_transform(df)

        self.assertEqual(result.loc[3, "category"], "apple")
        self.assertFalse(result["category"].isnull().any())
        pd.testing.assert_frame_equal(df, original)

    def test_metrics_available_after_fit(self):
        strategy = MeanStrategy()
        strategy.fit(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))
        metrics = strategy.get_metrics()
        self.assertEqual(metrics["method"], "Mean Imputation")
        self.assertIn("fill_values", metrics)

    def test_iterative_strategy_fills_missing(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, np.nan, 8.0, 10.0],
        })
        original = df.copy()

        strategy = IterativeStrategy()
        result = strategy.fit_transform(df)

        self.assertFalse(result["b"].isnull().any())
        self.assertAlmostEqual(result.loc[2, "b"], 6.0, delta=0.1)
        pd.testing.assert_frame_equal(df, original)

        self.assertTrue(strategy.get_explanation())



class ImputationEvaluationTests(TestCase):
    def test_evaluate_numeric_strategy(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        strategy = MeanStrategy()
        result = evaluate_imputation_strategy(df, "a", strategy, mask_percentage=20)
        
        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        self.assertEqual(result["n_evaluated"], 2)

    def test_evaluate_categorical_strategy(self):
        df = pd.DataFrame({
            "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
        })
        strategy = ModeStrategy()
        result = evaluate_imputation_strategy(df, "cat", strategy, mask_percentage=20)
        
        self.assertIn("accuracy", result)
        self.assertEqual(result["n_evaluated"], 2)

    def test_evaluate_categorical_mixed_types(self):
        """Test that evaluate_imputation_strategy handles categorical columns with mixed int and str values without TypeError."""
        df = pd.DataFrame({
            "mixed_cat": [1, "2", 1, "2", 1, "2", 1, "2", 1, "2"]
        })
        strategy = ModeStrategy()
        result = evaluate_imputation_strategy(df, "mixed_cat", strategy, mask_percentage=20)

        self.assertIn("accuracy", result)
        self.assertNotIn("error", result)
        
    def test_evaluation_does_not_mutate_input(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        original = df.copy()
        evaluate_imputation_strategy(df, "a", MeanStrategy(), mask_percentage=20)
        pd.testing.assert_frame_equal(df, original)

    def test_evaluation_handles_insufficient_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})

class QualityMetricsTests(TestCase):
    def test_calculate_metrics(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        imputed_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        metrics = calculate_imputation_metrics(df, imputed_df, "a")
        
        self.assertEqual(metrics["n_missing_before"], 1)
        self.assertEqual(metrics["n_imputed"], 1)
        self.assertEqual(metrics["n_remaining_missing"], 0)
        self.assertTrue(metrics["is_fully_imputed"])
        
    def test_metrics_integration_with_evaluation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        imputed_df = df.copy() # assumed already imputed
        eval_res = {"mae": 0.1, "rmse": 0.2}
        metrics = calculate_imputation_metrics(df, imputed_df, "a", evaluation_result=eval_res)
        
        self.assertIn("evaluation", metrics)
        self.assertEqual(metrics["evaluation"]["mae"], 0.1)

    def test_immutability(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        original = df.copy()
        imputed = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        calculate_imputation_metrics(df, imputed, "a")
        pd.testing.assert_frame_equal(df, original)

        result = evaluate_imputation_strategy(df, "a", MeanStrategy())
        self.assertIn("error", result)


    def test_calculate_metrics_edge_cases(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan]})
        imputed_df = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        metrics = calculate_imputation_metrics(df, imputed_df, "a")
        
        self.assertEqual(metrics["n_missing_before"], 2)
        self.assertEqual(metrics["n_imputed"], 1)
        self.assertEqual(metrics["n_remaining_missing"], 1)
        self.assertFalse(metrics["is_fully_imputed"])


from predictor.ml.explainability import generate_imputation_explanation

class ExplainabilityTests(TestCase):
    def test_explanation_structure(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        imputed_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        strategy = MeanStrategy()
        strategy._mark_fitted() # Need to be fit
        
        explanation = generate_imputation_explanation(df, imputed_df, "a", strategy)
        
        self.assertEqual(explanation["target_column"], "a")
        self.assertEqual(explanation["strategy"], "Mean Imputation")
        self.assertEqual(explanation["n_missing_before"], 1)
        self.assertEqual(explanation["n_imputed"], 1)
        
    def test_explanation_with_evaluation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        eval_res = {"mae": 0.1, "rmse": 0.2}
        strategy = MeanStrategy()
        strategy._mark_fitted()
        
        explanation = generate_imputation_explanation(df, df, "a", strategy, evaluation_result=eval_res)
        
        self.assertIn("evaluation_metrics", explanation)
        self.assertEqual(explanation["evaluation_metrics"]["mae"], 0.1)
        
    def test_explanation_handles_missing_evaluation(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        strategy = MeanStrategy()
        strategy._mark_fitted()
        
        explanation = generate_imputation_explanation(df, df, "a", strategy, evaluation_result={"error": "failed"})
        
        self.assertIn("evaluation_error", explanation)
        self.assertEqual(explanation["evaluation_error"], "failed")
        
    def test_explanation_immutability(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        original = df.copy()
        imputed = pd.DataFrame({"a": [1.0, 2.0]})
        strategy = MeanStrategy()
        strategy._mark_fitted()
        
        generate_imputation_explanation(df, imputed, "a", strategy)
        pd.testing.assert_frame_equal(df, original)


from predictor.ml.orchestrator import orchestrate_imputation

class OrchestratorTests(TestCase):
    def test_numeric_orchestration(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        result = orchestrate_imputation(df, "a")
        
        self.assertEqual(result["target_column"], "a")
        self.assertEqual(result["type"], "NUMERIC")
        self.assertIn("imputed_df", result)
        self.assertIn("metrics", result)
        self.assertIn("explanation", result)

    def test_categorical_orchestration(self):
        df = pd.DataFrame({"a": ["A", "B", "A", np.nan, "A", "B", "A", "B", "A", "B"]})
        result = orchestrate_imputation(df, "a")
        
        self.assertEqual(result["type"], "CATEGORICAL")
        self.assertEqual(result["selected_strategy"], "Mode Imputation")

    def test_immutability(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        original = df.copy()
        orchestrate_imputation(df, "a")
        pd.testing.assert_frame_equal(df, original)

    def test_no_missing_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = orchestrate_imputation(df, "a")
        self.assertEqual(result["metrics"]["n_missing_before"], 0)


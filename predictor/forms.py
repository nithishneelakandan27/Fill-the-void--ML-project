import os
import io
import logging
import pandas as pd
from django import forms

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(
        label='Upload Dataset',
        help_text='Supported formats: CSV, Excel (.xlsx, .xls). Max size: 10 MB.',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv,.xlsx,.xls',
            'id': 'file-input',
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleaned_df = None

    def clean_dataset(self):
        f = self.cleaned_data.get('dataset')
        if not f:
            raise forms.ValidationError('Please select a dataset file to upload.')

        if f.size > MAX_FILE_SIZE:
            raise forms.ValidationError('File size must not exceed 10 MB.')

        ext = os.path.splitext(f.name)[1].lower()
        if ext not in ['.csv', '.xlsx', '.xls']:
            raise forms.ValidationError('Only CSV and Excel files (.csv, .xlsx, .xls) are supported.')

        # Validate file content by attempting to parse into a DataFrame
        try:
            f.seek(0)
            if ext == '.csv':
                try:
                    df = pd.read_csv(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, encoding='latin1')
            else:
                df = pd.read_excel(f)
            f.seek(0)
        except Exception as e:
            logger.warning("Dataset parsing failed during upload: %s", e)
            raise forms.ValidationError(
                'The uploaded file could not be parsed as a valid dataset. Please ensure the file is not corrupted.'
            )

        if df is None or df.empty or len(df.columns) == 0:
            raise forms.ValidationError('The uploaded dataset is empty.')

        self.cleaned_df = df
        return f


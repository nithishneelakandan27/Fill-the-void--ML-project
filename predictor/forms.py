from django import forms

class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(
        label='Upload Dataset',
        help_text='Supported formats: CSV, Excel (.xlsx, .xls). Max size: 10 MB.',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv,.xlsx,.xls',
            'id': 'file-input',
        })
    )

    def clean_dataset(self):
        f = self.cleaned_data.get('dataset')
        if f:
            ext = f.name.rsplit('.', 1)[-1].lower()
            if ext not in ['csv', 'xlsx', 'xls']:
                raise forms.ValidationError('Only CSV and Excel files are supported.')
            if f.size > 10 * 1024 * 1024:
                raise forms.ValidationError('File size must not exceed 10 MB.')
        return f

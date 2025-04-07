from django import forms

class UploadZipForm(forms.Form):
    zip_file = forms.FileField(label='Zip Dosyası') 
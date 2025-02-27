import os
import zipfile
import tempfile
import subprocess
import json
import pandas as pd
import joblib
from django.shortcuts import render
from .forms import UploadFileForm

def handle_uploaded_file(f):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, 'uploaded.zip')
        with open(zip_path, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Run Bandit and save the report as JSON
        report_path = os.path.join(tmpdirname, 'bandit_report.json')
        subprocess.run(['bandit', '-r', tmpdirname, '-f', 'json', '-o', report_path], capture_output=True, text=True)
        
        # Read the Bandit report
        with open(report_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract the "results" section
        results = data.get("results", [])
        
        # Extract issue_text and line_number
        issues = [{"issue_text": result.get("issue_text"), "line_number": result.get("line_number")} for result in results]
        
        # Process the metrics into a DataFrame
        metrics_data = data.get("metrics", {})
        df_list = []
        for file_path, values in metrics_data.items():
            row = {"file_path": file_path}
            row.update(values)
            df_list.append(row)
        
        # Create a DataFrame
        df = pd.DataFrame(df_list)
        
        # Create a summary row
        project_summary = df.drop(columns=["file_path"]).sum().to_frame().T
        project_summary.insert(0, "folder_name", "uploaded_project")  # Add a project name
        
        return project_summary, issues

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            report_df, issues = handle_uploaded_file(request.FILES['file'])
            
            # Load the model
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_combined.pkl')
            loaded_models = joblib.load(model_path)

            # Check if the model is a dictionary
            if isinstance(loaded_models, dict):
                quality_model = loaded_models["quality_model"]
                security_model = loaded_models["security_model"]
            else:
                raise ValueError("Unexpected model format. The model should be saved as a dictionary.")

            # Prepare the data for prediction
            X_test = report_df.drop(columns=["folder_name"], errors="ignore")

            # Make predictions
            predicted_quality = quality_model.predict(X_test)
            predicted_security = security_model.predict(X_test)

            # Add predictions to the DataFrame
            report_df["predicted_software_quality"] = predicted_quality
            report_df["predicted_software_security"] = predicted_security

            # Convert DataFrame to HTML for rendering
            report_html = report_df.to_html(index=False)
            
            # Format issues for display
            issues_html = "<br>".join([f"Line {issue['line_number']}: {issue['issue_text']}" for issue in issues])
            
            return render(request, 'security/report.html', {'report': report_html, 'issues': issues_html})
    else:
        form = UploadFileForm()
    return render(request, 'security/upload.html', {'form': form}) 
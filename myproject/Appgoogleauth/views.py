from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from .forms import UploadZipForm
from .models import FirestoreModel
from django.http import JsonResponse

def home(request):
    if request.user.is_authenticated:
        return redirect('view_files')
    return render(request, 'home.html')

def logout_view(request):
    logout(request)
    return redirect('/')

@login_required
def upload_file(request):
    if request.method == 'POST':
        form = UploadZipForm(request.POST, request.FILES)
        if form.is_valid():
            zip_file = form.cleaned_data['zip_file']
            # Dosyayı Firestore'a kaydet
            FirestoreModel.create_upload(
                user=request.user.username,
                zip_name=zip_file.name,
                bandit_results={'test': 'sample bandit result'},  # Test için örnek veri
                ai_results={'test': 'sample AI result'}  # Test için örnek veri
            )
            return redirect('view_files')
    else:
        form = UploadZipForm()
    return render(request, 'upload.html', {'form': form})

@login_required
def view_files(request):
    # Sadece giriş yapmış kullanıcının dosyalarını getir
    current_user = request.user.username
    user_files = FirestoreModel.get_user_uploads(current_user)
    
    # Debug için konsola yazdır
    print(f"Current user: {current_user}")
    print(f"Files found: {len(user_files)}")
    for file in user_files:
        print(f"File: {file['zip_name']} - Uploaded by: {file['user']}")
    
    return render(request, 'view_files.html', {
        'files': user_files,
        'current_user': current_user
    })

@login_required
def test_firebase(request):
    try:
        # Firestore bağlantısını test et
        test_data = FirestoreModel.create_upload(
            user='test_user',
            zip_name='test.zip'
        )
        return JsonResponse({'status': 'success', 'message': 'Firebase connection successful!', 'data': test_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
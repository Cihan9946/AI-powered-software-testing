from django.shortcuts import render, redirect
from django.contrib.auth import logout

def home(request):
    if request.user.is_authenticated:
        # Kullanıcı giriş yaptıysa, security uygulamasına yönlendir
        return redirect('upload_file')  # 'upload_file' URL adını kontrol edin
    return render(request, 'home.html')  # Giriş sayfası


def logout_view(request):
    logout(request)
    return redirect('/')
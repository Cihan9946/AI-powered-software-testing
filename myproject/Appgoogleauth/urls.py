from django.urls import path
from . import views
from allauth.socialaccount.providers.google.views import oauth2_login

urlpatterns = [
    path('', views.home, name='home'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_file, name='upload_file'),
    path('files/', views.view_files, name='view_files'),
    path('test-firebase/', views.test_firebase, name='test_firebase'),
    path('google/login/', oauth2_login, name='google_login'),
]
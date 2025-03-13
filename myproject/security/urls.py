from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('report/<str:file_id>/', views.view_report, name='view_report'),
] 
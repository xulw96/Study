from django.urls import path

from Three import views

urlpatterns = [
    path(r'index/', views.index),
    path(r'get_grade', views.get_grade),
    path(r'get_students', views.get_students)
]
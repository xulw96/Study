from django.urls import path

from Two import views

urlpatterns = [
    path(r'index/', views.index),
    path(r'add_student/', views.add_student),
    path(r'get_students/', views.get_students),
    path(r'update_student/', views.update_student),
    path(r'delete_student/', views.delete_student),
]
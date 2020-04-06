from django.urls import path

from App import views

urlpatterns = [
    path(r'hello/', views.hello, name='hello'),
    path('uploadfile/', views.upload_file, name='upload_file'),
    path('imagefiled/', views.image_field, name='image_filed'),
    path('mine/', views.mine, name='mine')
]
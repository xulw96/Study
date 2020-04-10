from django.conf.urls import url
from django.urls import path, re_path

from Two import views

urlpatterns = [
    re_path(r'^students/(\d+)/', views.students),
    path(r'grades/', views.grades),
    re_path(r'gettime/(\d+)/(\d+)/(\d+)/', views.get_time, name='get_time'),
    re_path(r'getdate/(?P<year>\d+)/(?P<month>\d+)/(?P<day>\d+)/', views.get_date, name='get_date'),
    path('learn/', views.learn, name='learn'),
    path('havearequest/', views.have_request),
    path('createstudent/', views.create_student),
    path('docreatestudent/', views.do_create_student, name='do_create_student')
]

app_name = 'Two'

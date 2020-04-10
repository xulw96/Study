from django.urls import path

from Two import views

urlpatterns = [
    path('hello/', views.hello, name='hello'),
    path('login/', views.login, name='login'),
    path('mine/', views.mine, name='mine'),
    path('logout/', views.logout, name='logout'),
    path('register/', views.register, name='register'),
    path('studentlogin/', views.student_login, name='student_login'),
    path('studentmine/', views.student_mine, name='student_mine')
]
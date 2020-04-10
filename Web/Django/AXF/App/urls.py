from django.urls import path

from App import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path('market/', views.market, name='home'),
    path('cart/', views.cart, name='home'),
    path('mine/', views.mine, name='home'),

]
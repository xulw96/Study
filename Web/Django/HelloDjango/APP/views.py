from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
def hello(request):
    return HttpResponse('double click 666')


def index(request):
    return render(request, 'index.html')
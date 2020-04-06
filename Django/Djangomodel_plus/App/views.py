from django.http import HttpResponse
from django.shortcuts import render

from App.models import UserModel


def hello(request):
    return HttpResponse('hello')


def upload_file(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    elif request.method == "POST":
        icon = request.FILES.get('icon')
        with open(r'C:\Users\Administrator\PycharmProjects'
                  r'\Study\Django\Djangomodel_plus\static\img\icon.img') as save_file:
            for part in icon.chunks():
                save_file.write(part)
                save_file.flush()
        return HttpResponse('file upload successfully')


def image_field(request):
    if request.method == 'GET':
        return render(request, 'image_filed.html')
    elif request.method == "POST":
        username = request.POST.get('username')
        icon = request.FILES.get('icon')
        user = UserModel()
        user.u_name = username
        user.u_icon = icon
        user.save()
        return HttpResponse('upload successfully %d' % user.id)


def mine(request):
    username = request.GET.get('username')
    user = UserModel.objects.get(u_name=username)

    data = {
        "username": username,
        'icon_url': '/static/upload' + user.u_icon.url
    }
    return HttpResponse('personal info')
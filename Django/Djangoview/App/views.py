import random
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse


def hello(request):
    response = HttpResponse()
    response.content = 'lol'
    response.status_code = 404
    response.write('it\'s stucked')
    return response


def get_ticket(request):
    if random.randrange(10) > 5:
        url = reverse('app:hello')
        return redirect(url)
    return HttpResponse('cons, you get it')


def get_info(request):
    data = {
        'status': 200,
        'msg': 'ok',
    }
    return JsonResponse(data=data)


def set_cookie(request):
    response = HttpResponse('set cookie')
    response.set_cookie('username', 'Rock')
    return response


def get_cookie(request):
    username = request.COOKIES.get('username')
    return HttpResponse(username)


def login(request):
    return render(request, 'login.html')


def do_login(request):
    uname = request.POST.get('uname')
    response = redirect(reverse('app:mine'))
    # response.set_cookie('uname', uname, max_age=60)
    response.set_signed_cookie('content', uname, 'Rock')
    return response


def mine(request):
    # uname = request.COOKIES.get('uname')
    try:
        uname = request.get_signed_cookie('content', salt='Rock')
        if uname:
            return render(request, 'mine.html', context={'uname': uname})

    except Exception as e:
        print('fail to get value')

    return redirect(reverse('app:login'))


def logout(request):
    response = redirect(reverse('app:login'))
    response.delete_cookie('content')
    return response

import random
from io import BytesIO
from time import sleep

from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from django.core.cache import caches
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import csrf_exempt

from App.models import Student
from Djangocache import settings


def index(request):
    return HttpResponse('hello')


# @cache_page(timeout=30)
def news(request):
    # cache is like a database
    cache = caches('redis_backend')
    result = cache.get('news')

    if result:
        return HttpResponse(result)

    news_list = []
    for i in range(10):
        news_list.append("recently trade war %d" % i)

    sleep(5)
    data = {
        'news_list': news_list
    }
    response = render(request, 'news.html', context=locals())

    cache.set('news', response.content, timeout=30)
    return response


@cache_page(60, cache='default')
def jokes(request):
    sleep(5)
    return HttpResponse('JokeList')


def home(request):
    return HttpResponse('Home')


def get_phone(request):
    if random.randrange(100) > 95:
        return HttpResponse('success')
    else:
        return HttpResponse('fail')


def get_ticket(request):
    return HttpResponse('fail')


def search(request):
    return HttpResponse('search_result')


def calc(request):
    a = 250
    b = 250
    result = (a + b) / 0
    return HttpResponse(result)


@csrf_exempt
def login(request):
    if request.method == 'GET':
        render(request, 'login.html')
    elif request.method == 'POST':
        receive_code = request.POST.get('verify_code')
        store_code = request.session.get('verify_code')

        if receive_code.lower() == store_code.lower():
            return HttpResponse('post successfully')
        else:
            return redirect(reverse('app:login'))


def add_students(request):
    for i in range(100):
        student = Student()
        student.s_name = 'xiaoming %d' % i
        student.s_age = i

        student.save()
    return HttpResponse('create student successfully')


# paginator
def get_students(request):
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 10))

    students = Student.objects.all()[per_page * (page - 1):page * per_page]

    data = {
        'students': students
    }
    return render(request, 'students.html', context=data)


def get_students_with_page(request):
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 10))
    students = Student.objects.all()

    paginator = Paginator(students, per_page)
    page_object = paginator.page(page)

    data = {
        'page_object': page_object,
        'page_range': paginator.page_range,
    }
    return render(request, 'students_with_page.html', context=data)


def get_code(request):
    mode = 'RGD'
    size = (200, 100)

    def _get_color():
        return random.randrange(255)

    def _generate_code():
        source = 'asdjcfboiawuehrbgtfoui21345asdcasdc'
        code = ''
        for i in range(4):
            code += random.choice(source)
        return code

    red = _get_color()
    green = _get_color()
    blue = _get_color()
    color_bg = (red, green, blue)

    image = Image.new(mode=mode, size=size, color=color_bg)
    image_draw = ImageDraw(image, mode=mode)
    image_font = ImageFont.truetype(settings.FONT_PATH, 100)

    verify_code = _generate_code()

    request.session['verify_code'] = verify_code

    for i in range(len(verify_code)):
        image_draw.text(xy=(40*i, 0), text=verify_code[i], font=image_font)

    for i in range(1000):
        fill = (_get_color(), _get_color(), _get_color())
        xy = (random.randrange(201), random.randrange(100))
        image_draw.point(xy=xy, fill=fill)

    fp = BytesIO()
    image.save(fp, 'png')

    return HttpResponse(fp.getvalue(), content_type='img/png')


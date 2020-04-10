from django.http import HttpResponse
from django.shortcuts import render
from .models import Grade, Student


def students(request, g_id):
    student_list = Student.objects.filter(s_grade_id=g_id)
    return render(request, 'grade_student_list', context=locals())


def student(request, s_id):
    return HttpResponse('get student')


def grades(request):
    grade_list = Grade.objects.all()

    # locals, parse local variable into a dict
    return render(request, 'grade_list.html', context=locals())


def get_time(request, hour, minute, second):
    return HttpResponse('time %s: %s: %s' % (hour, minute, second))


def get_date(request, day, month, year):
    return HttpResponse("Date %s-%s-%s" %(year, month, day))


def learn(request):
    return HttpResponse('Love to learn')


def have_request(request):
    print(request.path)
    print(request.Get.get('hobby'))
    # multivalue dict; one key to multi-value
    print(request.Get.getlist('hobby'))
    print(request.method)
    print(request.META)
    print('Remote IP', request.META.get('REMOTE_ADDR'))
    return HttpResponse('see the request content')


def create_student(request):
    return render(request, 'student.html')



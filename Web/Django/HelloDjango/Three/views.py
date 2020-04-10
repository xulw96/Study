from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from Three.models import Student, Grade


def index(request):
    # realize render manually
    # support template language
    three_index = loader.get_template('three_index.html')
    context = {
        'student_name': 'Sunck'
    }
    result = three_index.render(context=context)
    return HttpResponse(result)


def get_grade(request):
    student = Student.objects.get(pk=1)
    grade = student.s_grade
    return HttpResponse('grade %s' % grade.g_name)


def get_students(request):
    grade = Grade.objects.get(pk=1)
    students = grade.student_set.all()
    context = {
        'students': students
    }
    return render(request, 'student_three_list.html', context=context)

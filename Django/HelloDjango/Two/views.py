import random

from django.http import HttpResponse
from django.shortcuts import render
from Two.models import Student


def index(request):
    return HttpResponse('Two index')


def add_student(request):
    # ORM management on database
    student = Student()
    student.s_name = "Jerry%d" % random.randrange(100)
    student.save()
    return HttpResponse('add success %s' % student.s_name)


def get_students(request):
    students = Student.objects.all()
    for student in students:
        print(student.s_name)

    context = {
        'hobby': 'play games',
        'eat': 'meat',
        'students': students
    }
    return render(request, 'student_three_list.html', context=context)


def update_student(request):
    student = Student.objects.get(pk=2)  # primary key
    student.s_name = 'Jack'
    student.save()
    return HttpResponse('student update success')


def delete_student():
    student = Student.objects.get(pk=3)
    student.delete()
    return HttpResponse('student delete success')


def students():
    return None
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .models import Student


def hello(request):
    return HttpResponse('hello')


def index(request):
    # return render(request, 'index.html')
    temp = loader.get_template('index.html')
    content = temp.render()
    return HttpResponse(content)


def get_students(request):
    students = Student.objects.all().filter(s_name='Sunck')
    student_dict = {
        'hobby': 'coding',
        'time': 'year',
    }
    code = """<h2>sleep</h2>
        <script type="text/javascript">
        var lis = document.getElementsByTagName("li");
        for (var i=0; i< lis.length; i++){
            var li =  lis[i];
            li.innerHTML="japan is in china";
        }
    alert('you site is attacked')
    </script>"""
    data = {
        'students': students,
        'student_dict': student_dict,
        'count': 10,
        'code': code,
    }
    return render(request, 'student_list.html', context=data)


def temp(request):
    return render(request, 'home.html', context={'title': 'home'})


def home(request):
    return render(request, 'home_mine.html')
import random
from django.http import HttpResponse
from django.shortcuts import render
from App.models import Person


def add_persons(request):
    for i in range(15):
        person = Person()
        flag = random.randrange(1000)
        person.p_name = 'Tom {}'.format(flag)
        person.p_age = flag
        person.p_sex = True
        person.save()
    return HttpResponse('create successfully')


def get_persons(request):
    # persons = Person.objects.exclude(p_age__lt=50).filter(p_age__lt=80)
    # gt, lt, gte, exact, ignore, contains, startswith/endswith, in
    persons = Person.objects.order_by('-id')
    # persons_values = persons.values()
    context = {
        'persons': persons
    }
    return render(request, 'person_list.html', context=context)


def add_person(request):
    # person = Person.objects.create(p_name='sunck', p_age=15, p_sex=True)
    # person = Person(p_age=28)
    person = Person.create('Jack')
    person.save()
    return HttpResponse('Sunck succeed')


def get_person(request):
    # person = Person.objects.get(p_age=20)
    # person = Person.objects.all().first()
    person = Person.objects.all().last()
    print(person)
    return HttpResponse('get succeed')

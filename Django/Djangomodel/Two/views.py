from django.db.models import Max, F, Q
from django.http import HttpResponse
from django.shortcuts import render
from .models import User, Order, Grade, Customer, Company


def get_user(request):
    usrname = 'Sunck'
    passwd = '120'
    users = User.objects.filter(u_name=usrname)

    # if users.exists():
    if users.count():
        user = users.first()
        if user.u_passwd == passwd:
            print('log on successfully')
        else:
            print('wrong password')
    else:
        print('user not exist')
    return HttpResponse('get successfully')


def get_users(request):
    # slice from SQL, not QuerySet. Not support negative value
    users = User.objects.all()[1: 3]
    context = {
        'users': users
    }
    return render(request, 'user_list.html', context=context)


def get_orders(request):
    # get time has UTC problem: stop timezone or create timezone in database
    # orders = Order.objects.filter(o_time__year=2018)
    orders = Order.objects.filter(o_time_month=9)
    for order in orders:
        print(order.o_num)
    return HttpResponse('get successfully')


def get_grades(request):
    # select with join
    grades = Grade.objects.filter(student__s_name='Jack')
    for grade in grades:
        print(grade.g_name)
    return HttpResponse('get successfully')


def get_customer(request):
    # aggregate
    result = Customer.objects.aggregate(Max())
    print(result)
    return HttpResponse('get successfully')


def get_company(request):
    # F object, support arithmetic
    companies = Company.objects.filter(c_boy_num__lt=F('c_girl_num')-15)
    # Q object, support logic arithmetic
    companies = Company.objects.filter(Q(c_boy_num__gt=1) & Q(c_girl_num__gt=10))
    for company in companies:
        print(company.c_name)
    return HttpResponse('get successfully')

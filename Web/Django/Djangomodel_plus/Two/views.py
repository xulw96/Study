from django.http import HttpResponse
from django.shortcuts import render

from Two.models import Person, IDCard, Customer, Goods, Cat


def hello(request):
    return HttpResponse("Two Hello")


def add_person(request):
    username = request.GET.get('username')
    person = Person()
    person.p_name = username
    person.save()
    return HttpResponse("Create person successfully %d" % person.id)


def add_id_card(request):
    id_num = request.GET.get('idnum')
    idcard = IDCard()
    idcard.id_num = id_num
    idcard.save()
    return HttpResponse('IDCard %d' % idcard.id)


def bind_card(request):
    person = Person.objects.last()
    idcard = IDCard.objects.last()
    idcard.id_person = person
    idcard.save()
    return HttpResponse("bind successfully")


def remove_person(request):
    person = Person.objects.last()
    person.delete()
    return HttpResponse('remove person successfully')


def remove_id_card(request):
    idcard = IDCard.objects.last()
    idcard.delete()
    return HttpResponse('remove idcard successfully')


def get_person(request):
    idcard = IDCard.objects.last()
    person = idcard.id_person
    return HttpResponse(person.p_name)


def get_idcard(request):
    person = Person.objects.last()
    # shadow attribute for Person
    idcard = person.idcard
    return HttpResponse(idcard.id_num)


def add_customer(request):
    c_name = request.Get.get("c_name")
    customer = Customer()
    customer.c_name = c_name
    customer.save()
    return HttpResponse('create customer successfully {}'.format(customer.id))


def add_goods(request):
    g_name = request.GET.get('gname')
    goods = Goods()
    goods.g_name = g_name
    goods.save()
    return HttpResponse("Create goods successfully {}".format(goods.id))


def add_to_cart(request):
    customer = Customer.objects.last()
    goods = Goods.objects.last()
    customer.goods_set.add(goods)
    # the same: goods.g_customer.add(customer)
    return HttpResponse('add successfully')


def get_goods_list(request, customerid):
    customer = Customer.objects.get(pk=customerid)
    goods_list = customer.goods_set.all()
    return render(request, 'goods_list.html', context=goods_list)


def add_cat(request):
    cat = Cat()
    cat.a_name = 'Tom'
    cat.c_eat = 'Fish'
    cat.save()
    return HttpResponse('create cat successfully %d' % cat.id)


def add_dog(request):
    dog = Cat()
    dog.a_name = 'Tom'
    dog.save()
    return HttpResponse('create cat successfully %d' % dog.id)
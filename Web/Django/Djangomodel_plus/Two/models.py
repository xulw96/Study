from django.db import models


class Person(models.Model):
    p_name = models.CharField(max_length=16)
    p_sex = models.BooleanField(default=False)


class IDCard(models.Model):
    id_num = models.CharField(max_length=18, unique=True)
    id_person = models.OneToOneField(Person, null=True, blank=True, on_delete=models.CASCADE)
    # on_delete=models.PROTECT: protect the row data in primary sheet
    # on_delete=models.SET_NULL


class Customer(models.Model):
    c_name = models.CharField(max_length=16)


class Goods(models.Model):
    g_name = models.CharField(max_length=16)
    g_customer = models.ManyToManyField(Customer)


class Animal(models.Model):
    a_name = models.CharField(max_length=16)

    class Meta:
        abstract = True  # no relation in database; won't use foreign key


class Cat(Animal):
    c_eat = models.CharField(max_length=32)


class Dog(Animal):
    d_legs = models.IntegerField(default=4)

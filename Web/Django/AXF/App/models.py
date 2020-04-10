from django.db import models


class Main(models.Model):
    img = models.CharField(max_length=255)
    name = models.CharField(max_length=64)
    trackid = models.IntegerField(default=1)
    objects = models.Manager()

    class Meta:
        abstract = True


class MainWheel(Main):

    class Meta:
        db_table = 'axf_wheel'


class MainNav(Main):
    class Meta:
        db_table = 'axf_nav'


class MainMustBuy(Main):

    class Meta:
        db_table = 'axf_mustbuy'
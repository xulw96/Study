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


class MainShop(Main):

    class Meta:
        db_table = 'axf_shop'


class MainShow(Main):
    categoryid = models.IntegerField(default=1)
    brandname = models.CharField(max_length=64)

    img1 = models.CharField(max_length=255)
    childcid1 = models.IntegerField(default=1)
    productid1 = models.IntegerField(default=1)
    longname1 = models.CharField(max_length=128)
    price1 = models.FloatField(default=1)
    marketprice1 = models.FloatField(default=0)

    img2= models.CharField(max_length=255)
    childcid2 = models.IntegerField(default=1)
    productid2 = models.IntegerField(default=1)
    longname2 = models.CharField(max_length=128)
    price2 = models.FloatField(default=1)
    marketprice2 = models.FloatField(default=0)

    img3 = models.CharField(max_length=255)
    childcid3 = models.IntegerField(default=1)
    productid3 = models.IntegerField(default=1)
    longname3 = models.CharField(max_length=128)
    price3 = models.FloatField(default=1)
    marketprice3 = models.FloatField(default=0)

    class Meta:
        db_table = 'axf_mainshow'
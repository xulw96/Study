from django.db import models


class Student(models.Model):
    s_name = models.CharField(max_length=16)


class Grade(models.Model):
    g_name = models.CharField(max_length=16)


class School(models.Model):
    s_name = models.CharField(max_length=16)


class Book(models.Model):
    b_name = models.CharField(max_length=16, blank=True, null=True)

    class Meta:
        db_table = 'Book'


class UserModel(models.Model):
    u_name = models.CharField(max_length=16)
    # relative to media_root
    u_icon = models.ImageField(upload_to='icons')

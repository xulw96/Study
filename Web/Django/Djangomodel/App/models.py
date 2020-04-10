from django.db import models


# changing database model needs to do migration
class Person(models.Model):
    p_name = models.CharField(max_length=16, unique=True)
    p_age = models.IntegerField(default=18, db_column='age')
    # False for male
    p_sex = models.BooleanField(default=False, db_column='sex')
    p_hobby = models.CharField(max_length=32, null=True, blank=True)

    @classmethod
    def create(cls, p_name, p_age=100, p_sex=True, p_hobby='gaming'):
        return cls(p_name=p_name, p_age=p_age, p_sex=p_sex, p_hobby=p_hobby)

    class Meta:
        db_table = 'People'


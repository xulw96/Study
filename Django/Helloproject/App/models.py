from django.db import models
from tinymce.models import HTMLField


class Blog(models.Model):
    b_content = HTMLField()
from django.db import models

# Create your models here.

from django.db import models


class Img(models.Model):
    img_url = models.ImageField(upload_to='img/')  # 指定图片上传路径，即media/img/
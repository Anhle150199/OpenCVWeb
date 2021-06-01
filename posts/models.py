from django.db import models

# Create your models here.
class CV(models.Model):
    # title = models.TextField()
    image = models.ImageField(upload_to='images/')
    imageResult = models.ImageField(upload_to='images/')
    # def __str__(self):
    #     return self.title

class OjDt(models.Model):
    image = models.ImageField(upload_to='images/')
    imageTemplate = models.ImageField(upload_to='images/')
    imageResult = models.ImageField(upload_to='images/')

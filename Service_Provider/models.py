from django.db import models

# Create your models here.
class ClientRegisterModel(models.Model):
    username = models.CharField(max_length=30)  
    email = models.CharField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.IntegerField()
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

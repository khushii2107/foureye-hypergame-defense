from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class detection_type(models.Model):
    Flow_ID = models.TextField()
    Source_IP = models.TextField()
    Source_Port = models.TextField()
    Destination_IP = models.TextField()
    Destination_Port = models.TextField()
    Timestamp = models.TextField()
    Flow_Duration = models.TextField()
    Total_Fwd_Packets = models.TextField()
    Total_Backward_Packets = models.TextField()
    Total_Length_of_Fwd_Packets = models.TextField()
    Total_Length_of_Bwd_Packets = models.TextField()
    Fwd_Packet_Length_Max = models.TextField()
    Fwd_Packet_Length_Min = models.TextField()
    Bwd_Packet_Length_Max = models.TextField()
    Flow_Bytes = models.TextField()
    Flow_Packets = models.TextField()
    Fwd_Packets = models.TextField()
    Bwd_Packets = models.TextField()
    Max_Packet_Length = models.TextField()
    Prediction = models.TextField()



class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)




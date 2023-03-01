from django.db import models

class Question(models.Model):
    question = models.CharField(max_length=140)
    context = models.TextField(null=True, blank=True)
    answer = models.TextField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField("date created", auto_now_add=True)
    ask_count = models.IntegerField(default=1)
    audio_src_url = models.CharField(default="", max_length=255, null=True, blank=True)


class Person(models.Model):
    year = models.IntegerField()
    person_name = models.CharField(max_length=100)
    frequency = models.IntegerField()

    def __str__(self):
        return f'{self.year} {self.person_name} ({self.frequency})'
    
class Organization(models.Model):
    year = models.IntegerField()
    name = models.CharField(max_length=100)
    frequency = models.IntegerField()

    def __str__(self):
        return f"{self.year} - {self.name} - {self.frequency}"
    
class Location(models.Model):
    year = models.IntegerField()
    location = models.CharField(max_length=100)
    frequency = models.IntegerField()

    def __str__(self):
        return f"{self.year} - {self.location} - {self.frequency}"    
from django.db import models
from django.contrib.auth.models import User

class Tenant(models.Model):
    name = models.CharField(max_length=255)
    subdomain = models.CharField(max_length=255, unique=True)
    admins = models.ManyToManyField(User, related_name='tenants')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

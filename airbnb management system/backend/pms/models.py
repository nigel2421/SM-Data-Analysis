from django.db import models
from customers.models import Tenant

class Unit(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    address = models.TextField()
    status = models.CharField(max_length=50, choices=[
        ('GREEN', 'Occupied'),
        ('RED', 'Maintenance'),
        ('YELLOW', 'Turnover')
    ], default='GREEN')
    price_per_night = models.DecimalField(max_digits=10, decimal_places=2)
    dynamic_pricing_enabled = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} - {self.tenant.name}"

class Booking(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE, related_name='bookings')
    guest_name = models.CharField(max_length=255)
    check_in = models.DateField()
    check_out = models.DateField()
    total_price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.guest_name} at {self.unit.name}"

class MaintenanceRequest(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE, related_name='maintenance_requests')
    description = models.TextField()
    photo = models.ImageField(upload_to='maintenance/', null=True, blank=True)
    status = models.CharField(max_length=50, choices=[
        ('PENDING', 'Pending'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed')
    ], default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

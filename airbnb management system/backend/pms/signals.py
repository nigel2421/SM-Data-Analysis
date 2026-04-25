from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Booking, Unit

@receiver(post_save, sender=Booking)
def trigger_turnover_logistics(sender, instance, created, **kwargs):
    """
    Logistics Agent: Triggers turnover tasks when a booking is created or updated.
    In a real scenario, this would check if today is check_out day.
    """
    # Example logic: if the booking just ended or is confirmed for today
    # we trigger a notification to the cleaning crew.
    print(f"Logistics Agent: Checkout confirmed for {instance.unit.name}. Notifying cleaning crew...")
    
    # Update unit status to YELLOW (Turnover)
    unit = instance.unit
    if unit.status != 'YELLOW':
        unit.status = 'YELLOW'
        unit.save()

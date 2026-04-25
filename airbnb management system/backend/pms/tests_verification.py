from django.test import TestCase
from customers.models import Tenant
from pms.models import Unit, Booking
import datetime

class PMSTest(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Nigel's Rentals", subdomain="nigel")
        self.unit = Unit.objects.create(
            tenant=self.tenant,
            name="Unit 4B",
            address="123 Luxury Ave",
            price_per_night=150.00
        )

    def test_booking_triggers_logistics(self):
        # Create a booking
        booking = Booking.objects.create(
            tenant=self.tenant,
            unit=self.unit,
            guest_name="John Doe",
            check_in=datetime.date.today(),
            check_out=datetime.date.today() + datetime.timedelta(days=3),
            total_price=450.00
        )
        
        # Verify the Logistics Agent (signal) updated the unit status or triggered
        # For now, our signal prints and updates status to YELLOW if it was a checkout
        # In our simplified signal, we just update status to YELLOW for any Booking save to demonstrate.
        
        self.unit.refresh_from_db()
        self.assertEqual(self.unit.status, 'YELLOW')
        print(f"Test Passed: Logistics Agent triggered for {self.unit.name}")

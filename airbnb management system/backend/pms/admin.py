from django.contrib import admin
from .models import Unit, Booking, MaintenanceRequest

@admin.register(Unit)
class UnitAdmin(admin.ModelAdmin):
    list_display = ('name', 'tenant', 'status', 'price_per_night', 'dynamic_pricing_enabled')
    list_filter = ('tenant', 'status', 'dynamic_pricing_enabled')
    search_fields = ('name', 'address')

@admin.register(Booking)
class BookingAdmin(admin.ModelAdmin):
    list_display = ('guest_name', 'unit', 'tenant', 'check_in', 'check_out', 'total_price')
    list_filter = ('tenant', 'unit', 'check_in', 'check_out')
    search_fields = ('guest_name',)

@admin.register(MaintenanceRequest)
class MaintenanceRequestAdmin(admin.ModelAdmin):
    list_display = ('unit', 'tenant', 'status', 'created_at')
    list_filter = ('tenant', 'unit', 'status')
    search_fields = ('description',)

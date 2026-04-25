from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UnitViewSet, BookingViewSet, MaintenanceRequestViewSet

router = DefaultRouter()
router.register(r'units', UnitViewSet)
router.register(r'bookings', BookingViewSet)
router.register(r'maintenance', MaintenanceRequestViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

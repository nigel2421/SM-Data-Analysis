from rest_framework import viewsets, permissions
from .models import Unit, Booking, MaintenanceRequest
from .serializers import UnitSerializer, BookingSerializer, MaintenanceRequestSerializer

class TenantFilteredViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Filter objects to only those belonging to tenants the user is an admin of
        return self.queryset.filter(tenant__admins=self.request.user)

    def perform_create(self, serializer):
        # Automatically assign the first tenant the user is an admin of
        tenant = self.request.user.tenants.first()
        serializer.save(tenant=tenant)

class UnitViewSet(TenantFilteredViewSet):
    queryset = Unit.objects.all()
    serializer_class = UnitSerializer

class BookingViewSet(TenantFilteredViewSet):
    queryset = Booking.objects.all()
    serializer_class = BookingSerializer

class MaintenanceRequestViewSet(TenantFilteredViewSet):
    queryset = MaintenanceRequest.objects.all()
    serializer_class = MaintenanceRequestSerializer

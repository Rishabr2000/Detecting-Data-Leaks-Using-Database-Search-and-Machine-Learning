from django.urls import path
from .views import hello_world, new_view, check_leak_view

urlpatterns = [
    path('', hello_world, name='hello_world'),
    path('new/', new_view, name='new_view'),
    path('check-leak/', check_leak_view, name='check_leak'),
]

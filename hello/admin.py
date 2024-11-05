from django.contrib import admin
from .models import Person, UserData

class UserDataAdmin(admin.ModelAdmin):
    list_display = ('email', 'password')

admin.site.register(Person)
admin.site.register(UserData, UserDataAdmin)

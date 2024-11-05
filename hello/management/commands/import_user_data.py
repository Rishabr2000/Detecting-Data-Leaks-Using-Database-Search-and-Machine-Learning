import os
from django.core.management.base import BaseCommand
from hello.models import UserData

class Command(BaseCommand):
    help = 'Import user data from all files in a directory'

    def add_arguments(self, parser):
        parser.add_argument('directory', type=str, help='Directory containing the data files')

    def handle(self, *args, **kwargs):
        directory = kwargs['directory']
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        line = line.strip()
                        if line:  # Ensure line is not empty
                            try:
                                email, password = line.split(':')
                                if not UserData.objects.filter(email=email).exists():
                                    UserData.objects.create(email=email, password=password)
                                    self.stdout.write(self.style.SUCCESS(f"Successfully imported {email} from {filename}"))
                                else:
                                    self.stdout.write(self.style.WARNING(f"Skipping duplicate email in {filename}: {email}"))
                            except ValueError:
                                self.stdout.write(self.style.WARNING(f"Skipping invalid line in {filename}: {line}"))
                            except Exception as e:
                                self.stdout.write(self.style.ERROR(f"Error processing line in {filename}: {line} - {str(e)}"))
        self.stdout.write(self.style.SUCCESS('Successfully imported user data from all files in the directory'))

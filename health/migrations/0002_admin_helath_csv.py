
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('health', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Admin_Helath_CSV',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, null=True)),
                ('csv_file', models.FileField(blank=True, null=True, upload_to='')),
            ],
        ),
    ]

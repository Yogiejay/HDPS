

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('health', '0002_admin_helath_csv'),
    ]

    operations = [
        migrations.CreateModel(
            name='Search_Data',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prediction_accuracy', models.CharField(blank=True, max_length=100, null=True)),
                ('result', models.CharField(blank=True, max_length=100, null=True)),
                ('values_list', models.CharField(blank=True, max_length=100, null=True)),
                ('patient', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='health.patient')),
            ],
        ),
    ]

# Generated by Django 3.0 on 2025-07-12 13:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Remote_User', '0002_auto_20250712_1906'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClientRegister_Model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=30)),
                ('email', models.EmailField(max_length=30)),
                ('password', models.CharField(max_length=10)),
                ('phoneno', models.CharField(max_length=10)),
                ('country', models.CharField(max_length=30)),
                ('state', models.CharField(max_length=30)),
                ('city', models.CharField(max_length=30)),
            ],
        ),
    ]

# Generated by Django 4.1.7 on 2023-02-26 23:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0012_remove_question_real_answer'),
    ]

    operations = [
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField()),
                ('person_name', models.CharField(max_length=100)),
                ('frequency', models.IntegerField()),
            ],
        ),
    ]
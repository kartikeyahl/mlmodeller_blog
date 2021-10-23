from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from django.contrib.auth.models import User
from .models import Project, Profile

from django.core.mail import send_mail
from django.conf import settings

# @receiver(post_save, sender=Profile)


def sendmail(sender, instance, created, **kwargs):
    if created:
        subject = 'new blog added'
        message = 'Hi!\nNew blog is added, go checkout!!\n\nhttps://www.mlmodeller.in/blogs/'

        send_mail(
            subject,
            message,
            settings.EMAIL_HOST_USER,
            list(Profile.objects.values_list('email',flat=True)),
            fail_silently=False,
        )

post_save.connect(sendmail, sender=Project)
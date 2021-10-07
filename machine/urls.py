from django.contrib import admin
from django.shortcuts import render ,redirect
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.auth import views as auth_views

def Index(request):
    template_name='index.html'
    context = {
        'homePage':True,
        'title':'ML MODELLDER | Home'
    }
    return render(request, template_name,context)

def About(request):
    template_name='about us.html'
    context = {
        'aboutusPage':True,
        'title':'ML MODELLDER | About Us'
    }
    return render(request, template_name,context)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',Index),
    path('about-us/',About),
    path('classification/', include('classification.urls')),
    path('blog/', include('blog.urls')),
    #path('deeplearning/', include('deeplearning.urls')),
    path('regression/', include('regression.urls')),
    path('projects/', include('projects.urls')),
    path('mainuser', include('users.urls')),
    path('api/', include('api.urls')),

    path('reset_password/', auth_views.PasswordResetView.as_view(template_name="reset_password.html"),
         name="reset_password"),

    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name="reset_password_sent.html"),
         name="password_reset_done"),

    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="reset.html"),
         name="password_reset_confirm"),

    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name="reset_password_complete.html"),
         name="password_reset_complete"),

]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
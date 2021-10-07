from django.shortcuts import render
from .models import Blog
# Create your views here.

def Index(request):
    template = 'blog/index.html'
    context = {'title':"ML Modeller : Blog"}
    return render(request,template,context)
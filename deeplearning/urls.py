from django.urls import path
from django.shortcuts import render
from .views import ANNSigmoiod , ANNSoftmax

def Index(request):
    template_name="Deep Learning/Deep Learning.html"
    context = {
        "deeplearningPage":True,
        'title':'ML MODELLDER | Deep Learning',
    }
    return render(request, template_name,context)


urlpatterns = [
    #path('',Index,name="Deeplearning"),
    #path('ann-sigmoid/',ANNSigmoiod,name="ann-sigmoid"),
    #path('ann-softmax/',ANNSoftmax,name="ann-softmax"),
]
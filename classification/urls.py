from django.urls import path
from django.shortcuts import render
from django.views.generic import TemplateView
from .views import DecisionTree, KNearestNeighbors, KernelSVM, LogisticRegression, NaiveBayes, RandomForest, SupportVector

def Index(request):
    template_name="Classification/Classification.html"
    context = {
        "classificationPage":True,
        'title':'ML MODELLDER | Classification',
    }
    return render(request, template_name,context)

urlpatterns = [
    path('',Index,name="classfication"),
    path('decision-tree/', DecisionTree,name="decision-tree"),
    path('k-nearest-neighbors/', KNearestNeighbors,name="k-nearest-neighbors"),
    path('kernel-svm/', KernelSVM,name="kernel-svm"),
    #path('logistic-regression/', LogisticRegression,name="logistic-regression"),
    path('naive-bayes/', NaiveBayes,name="naive-bayes"),
    path('support-vector/', SupportVector,name="support-vector"),
    path('random-forest/', RandomForest,name="random-forest"),
]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.projects, name="projects"),
    path('blog/<str:pk>/', views.project, name="project"),

    path('create-blog/', views.createProject, name="create-project"),

    path('update-blog/<str:pk>/', views.updateProject, name="update-project"),

    path('delete-blog/<str:pk>/', views.deleteProject, name="delete-project"),
]

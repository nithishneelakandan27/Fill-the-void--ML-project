from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload'),
    path('analysis/', views.analysis, name='analysis'),
    path('run-prediction/', views.run_prediction, name='run_prediction'),
    path('results/', views.results, name='results'),
    path('download/', views.download_cleaned, name='download'),
    path('clear-session/', views.clear_session, name='clear_session'),
]

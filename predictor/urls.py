from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('results/', views.ResultsListView.as_view(), name='results'),
    path('predictions/', views.PredictionsView.as_view(), name='predictions'),
    path('simulation/', views.simulation_view, name='simulation'),
    path('api/simulate-bet/', views.simulate_bet, name='simulate_bet'),
] 
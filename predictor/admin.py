from django.contrib import admin
from .models import LotteryResult, Prediction

@admin.register(LotteryResult)
class LotteryResultAdmin(admin.ModelAdmin):
    list_display = ('draw_number', 'draw_date', 'numbers', 'created_at')
    list_filter = ('draw_date',)
    search_fields = ('draw_number', 'numbers')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('prediction_date', 'model_type', 'numbers', 'confidence_score')
    list_filter = ('model_type', 'prediction_date')
    search_fields = ('numbers',)

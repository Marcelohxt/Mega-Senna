from django.db import models

# Create your models here.

class LotteryResult(models.Model):
    draw_number = models.IntegerField(unique=True)
    draw_date = models.DateField()
    numbers = models.CharField(max_length=50)  # Números sorteados separados por vírgula
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-draw_date']

    def __str__(self):
        return f"Concurso {self.draw_number} - {self.draw_date}"

class Prediction(models.Model):
    prediction_date = models.DateTimeField(auto_now_add=True)
    numbers = models.CharField(max_length=50)
    model_type = models.CharField(max_length=50)  # Tipo de modelo usado
    confidence_score = models.FloatField()

    class Meta:
        ordering = ['-prediction_date']

    def __str__(self):
        return f"Previsão {self.model_type} - {self.prediction_date}"

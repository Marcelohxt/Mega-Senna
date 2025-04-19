from django.shortcuts import render
from django.views.generic import ListView, TemplateView
from .models import LotteryResult, Prediction
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import random
from datetime import datetime
from itertools import combinations

class HomeView(TemplateView):
    template_name = 'predictor/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        latest_results = LotteryResult.objects.all().order_by('-draw_date')[:10]

        # Prepara os números para cada resultado
        for result in latest_results:
            result.number_list = result.numbers.split(',')

        latest_predictions = Prediction.objects.all().order_by('-prediction_date')[:5]
        
        # Prepara os números para cada previsão
        for prediction in latest_predictions:
            prediction.number_list = prediction.numbers.split(',')

        context['latest_results'] = latest_results
        context['latest_predictions'] = latest_predictions
        return context

class ResultsListView(ListView):
    model = LotteryResult
    template_name = 'predictor/results.html'
    context_object_name = 'results'
    paginate_by = 20

class PredictionsView(TemplateView):
    template_name = 'predictor/predictions.html'

    def analyze_sequences(self, numbers_data):
        """Analisa sequências de números que aparecem juntos frequentemente"""
        sequence_counts = defaultdict(int)
        for numbers in numbers_data:
            # Analisa pares de números
            for pair in combinations(sorted(numbers), 2):
                sequence_counts[pair] += 1
        return sequence_counts

    def analyze_parity(self, numbers_data):
        """Analisa distribuição de números pares e ímpares"""
        parity_stats = []
        for numbers in numbers_data:
            even = sum(1 for n in numbers if n % 2 == 0)
            odd = 6 - even
            parity_stats.append((even, odd))
        return parity_stats

    def analyze_ranges(self, numbers_data):
        """Analisa distribuição de números por faixa"""
        range_stats = []
        for numbers in numbers_data:
            low = sum(1 for n in numbers if n <= 30)
            high = 6 - low
            range_stats.append((low, high))
        return range_stats

    def calculate_confidence(self, predictions, historical_data):
        """Calcula score de confiança para cada previsão"""
        scores = {}
        for method, numbers in predictions.items():
            if method == 'final':
                continue
            score = 0
            # Verifica frequência dos números no histórico
            for num in numbers:
                score += sum(1 for data in historical_data if num in data)
            scores[method] = score / (len(historical_data) * 6)
        return scores

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Coletar dados para análise
        results = LotteryResult.objects.all().order_by('draw_date')
        if results.exists():
            # Preparar dados para o modelo
            numbers_data = []
            for result in results:
                numbers = list(map(int, result.numbers.split(',')))
                numbers_data.append(numbers)
            
            df = pd.DataFrame(numbers_data)
            
            # Análise de Frequência
            all_numbers = [num for sublist in numbers_data for num in sublist]
            number_frequency = Counter(all_numbers)
            most_common = number_frequency.most_common(6)
            frequency_prediction = [num for num, _ in most_common]
            
            # Análise de Padrões
            def analyze_patterns(numbers_list):
                patterns = []
                for i in range(len(numbers_list) - 1):
                    current = numbers_list[i]
                    next_num = numbers_list[i + 1]
                    pattern = [next_num[j] - current[j] for j in range(6)]
                    patterns.append(pattern)
                return patterns

            patterns = analyze_patterns(numbers_data)
            last_pattern = patterns[-1]
            pattern_prediction = [numbers_data[-1][i] + last_pattern[i] for i in range(6)]
            pattern_prediction = [max(1, min(60, num)) for num in pattern_prediction]
            
            # Modelo de Regressão Linear
            X = df.iloc[:-1]
            y = df.iloc[1:]
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            linear_prediction = linear_model.predict(df.iloc[-1:].values)
            linear_prediction = np.clip(linear_prediction.round(), 1, 60).astype(int)[0]
            
            # Modelo Random Forest
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            rf_prediction = rf_model.predict(df.iloc[-1:].values)
            rf_prediction = np.clip(rf_prediction.round(), 1, 60).astype(int)[0]
            
            # Modelo de Clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(scaled_data)
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_prediction = np.clip(cluster_centers[-1].round(), 1, 60).astype(int)
            
            # Novas análises
            sequences = self.analyze_sequences(numbers_data)
            parity_stats = self.analyze_parity(numbers_data)
            range_stats = self.analyze_ranges(numbers_data)
            
            # Combinação de Previsões com pesos
            def combine_predictions(predictions, weights):
                combined = []
                for i in range(6):
                    numbers = []
                    for pred, weight in zip(predictions, weights):
                        numbers.extend([pred[i]] * weight)
                    combined.append(int(np.median(numbers)))
                return combined

            all_predictions = [
                frequency_prediction,
                pattern_prediction,
                linear_prediction,
                rf_prediction,
                cluster_prediction
            ]
            
            # Pesos baseados na confiança de cada método
            weights = [3, 2, 2, 3, 1]  # Pesos arbitrários, podem ser ajustados
            
            final_prediction = combine_predictions(all_predictions, weights)
            
            # Calcula scores de confiança
            confidence_scores = self.calculate_confidence({
                'frequency': frequency_prediction,
                'pattern': pattern_prediction,
                'linear': linear_prediction,
                'random_forest': rf_prediction,
                'cluster': cluster_prediction,
                'final': final_prediction
            }, numbers_data)
            
            # Salvar as previsões no banco de dados
            prediction = Prediction.objects.create(
                numbers=','.join(map(str, final_prediction)),
                model_type='Combinado',
                confidence_score=sum(confidence_scores.values()) / len(confidence_scores)
            )
            
            # Preparar contexto para o template
            context['predictions'] = {
                'frequency': frequency_prediction,
                'pattern': pattern_prediction,
                'linear': linear_prediction,
                'random_forest': rf_prediction,
                'cluster': cluster_prediction,
                'final': final_prediction
            }
            
            context['confidence_scores'] = confidence_scores
            context['number_frequency'] = dict(number_frequency)
            context['patterns'] = patterns[-5:]
            context['sequences'] = dict(sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:10])
            context['parity_stats'] = parity_stats[-5:]
            context['range_stats'] = range_stats[-5:]
            
        return context

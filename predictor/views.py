from django.shortcuts import render
from django.views.generic import ListView, TemplateView
from .models import LotteryResult, Prediction
from .analytics import LotteryAnalytics
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
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy import stats
from django.db.models import Q
from django.utils import timezone
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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
    ordering = ['-draw_date']

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filtros de data
        start_date = self.request.GET.get('start_date')
        end_date = self.request.GET.get('end_date')
        
        if start_date:
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                queryset = queryset.filter(draw_date__gte=start_date)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
                queryset = queryset.filter(draw_date__lte=end_date)
            except ValueError:
                pass
        
        # Filtro por número
        number = self.request.GET.get('number')
        if number:
            try:
                number = int(number)
                queryset = queryset.filter(numbers__contains=str(number))
            except ValueError:
                pass
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Adiciona os valores dos filtros ao contexto
        context['start_date'] = self.request.GET.get('start_date', '')
        context['end_date'] = self.request.GET.get('end_date', '')
        context['number'] = self.request.GET.get('number', '')
        
        # Prepara os dados para análise
        queryset = self.get_queryset()
        if queryset.exists():
            # Converte os números para lista de inteiros
            numbers_list = []
            dates = []
            for result in queryset:
                numbers = list(map(int, result.numbers.split(',')))
                numbers_list.append(numbers)
                dates.append(result.draw_date)
            
            # Realiza todas as análises
            analytics = LotteryAnalytics()
            
            # Análise de Frequência
            frequency_data = analytics.analyze_frequency_distribution(numbers_list)
            context['number_frequency'] = frequency_data['distribution']
            context['most_common'] = frequency_data['most_common']
            context['least_common'] = frequency_data['least_common']
            
            # Análise de Paridade
            parity_stats = analytics.analyze_parity(numbers_list)
            context['parity_stats'] = parity_stats[-10:]  # Últimos 10 sorteios
            
            # Análise de Sequências
            sequences = analytics.analyze_sequences(numbers_list)
            context['sequences'] = dict(sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Análise de Gaps
            gaps = analytics.analyze_gaps(numbers_list)
            context['gaps'] = {k: np.mean(v) for k, v in gaps.items()}
            
            # Análise de Somas
            sums_data = analytics.analyze_sums(numbers_list)
            context['sums'] = sums_data
            
            # Números Atrasados
            delayed = analytics.analyze_delayed_numbers(numbers_list)
            context['delayed_numbers'] = [{'number': num, 'delay': gaps.get(num, [0])[-1]} for num in delayed]
            
            # Análise de Faixas
            range_stats = analytics.analyze_ranges(numbers_list)
            context['range_stats'] = range_stats[-10:]  # Últimos 10 sorteios
            
            # Análise de Tendências
            trends = analytics.analyze_trends(numbers_list)
            context['trends'] = trends
            
            # Análise de Números Consecutivos
            consecutive = analytics.analyze_consecutive(numbers_list)
            context['consecutive'] = dict(sorted(consecutive.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Análise de Padrões Temporais
            time_patterns = analytics.analyze_time_patterns(numbers_list, dates)
            context['weekday_patterns'] = time_patterns['weekday']
            context['month_patterns'] = time_patterns['month']
            
            # Prepara dados para os gráficos
            context['chart_data'] = {
                'frequency': {
                    'labels': list(map(str, range(1, 61))),
                    'data': [frequency_data['distribution'].get(i, 0) for i in range(1, 61)]
                },
                'parity': {
                    'labels': ['Pares', 'Ímpares'],
                    'data': [
                        sum(1 for even, _ in parity_stats),
                        sum(1 for _, odd in parity_stats)
                    ]
                },
                'sums': {
                    'labels': list(map(str, sums_data['distribution'].keys())),
                    'data': list(sums_data['distribution'].values())
                }
            }
            
            # Converte os dados para JSON para uso no JavaScript
            context['chart_data_json'] = json.dumps(context['chart_data'])
        
        return context

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

    def analyze_trends(self, numbers_data):
        """Analisa tendências de crescimento/decrescimento dos números"""
        trends = []
        for i in range(6):
            numbers = [draw[i] for draw in numbers_data]
            slope, _, _, _, _ = stats.linregress(range(len(numbers)), numbers)
            trends.append(slope)
        return trends

    def analyze_gaps(self, numbers_data):
        """Analisa quanto tempo cada número leva para reaparecer"""
        last_seen = defaultdict(int)
        gaps = defaultdict(list)
        for i, draw in enumerate(numbers_data):
            for num in draw:
                if num in last_seen:
                    gaps[num].append(i - last_seen[num])
                last_seen[num] = i
        return gaps

    def analyze_sum_distribution(self, numbers_data):
        """Analisa a distribuição das somas dos números"""
        sums = [sum(draw) for draw in numbers_data]
        mean_sum = np.mean(sums)
        std_sum = np.std(sums)
        return mean_sum, std_sum

    def analyze_differences(self, numbers_data):
        """Analisa as diferenças entre números consecutivos"""
        differences = []
        for draw in numbers_data:
            sorted_draw = sorted(draw)
            diff = [sorted_draw[i+1] - sorted_draw[i] for i in range(5)]
            differences.append(diff)
        return differences

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

    def train_model_for_each_number(self, X, y, model_class, **kwargs):
        """Treina um modelo separado para cada número"""
        models = []
        predictions = []
        for i in range(6):
            model = model_class(**kwargs)
            model.fit(X, y[:, i])
            models.append(model)
            predictions.append(model.predict(X[-1:])[0])
        return np.array(predictions)

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
            
            # Preparar dados para modelos
            X = df.iloc[:-1].values
            y = df.iloc[1:].values
            
            # Modelo de Regressão Linear
            linear_prediction = self.train_model_for_each_number(X, y, LinearRegression)
            linear_prediction = np.clip(linear_prediction.round(), 1, 60).astype(int)
            
            # Modelo Random Forest
            rf_prediction = self.train_model_for_each_number(X, y, RandomForestRegressor, n_estimators=100)
            rf_prediction = np.clip(rf_prediction.round(), 1, 60).astype(int)
            
            # Modelo de Clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(scaled_data)
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_prediction = np.clip(cluster_centers[-1].round(), 1, 60).astype(int)
            
            # Modelo de Rede Neural
            nn_prediction = self.train_model_for_each_number(
                X, y, MLPRegressor, 
                hidden_layer_sizes=(100, 50), 
                max_iter=1000
            )
            nn_prediction = np.clip(nn_prediction.round(), 1, 60).astype(int)
            
            # Modelo SVM
            svm_prediction = self.train_model_for_each_number(X, y, SVR, kernel='rbf')
            svm_prediction = np.clip(svm_prediction.round(), 1, 60).astype(int)
            
            # Análise de Componentes Principais
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            pca_prediction = np.clip(pca.inverse_transform(pca_data[-1:]).round(), 1, 60).astype(int)[0]
            
            # Novas análises
            sequences = self.analyze_sequences(numbers_data)
            parity_stats = self.analyze_parity(numbers_data)
            range_stats = self.analyze_ranges(numbers_data)
            trends = self.analyze_trends(numbers_data)
            gaps = self.analyze_gaps(numbers_data)
            mean_sum, std_sum = self.analyze_sum_distribution(numbers_data)
            differences = self.analyze_differences(numbers_data)
            
            # Previsão baseada em tendências
            trend_prediction = []
            for i in range(6):
                last_number = numbers_data[-1][i]
                trend = trends[i]
                predicted = last_number + trend
                trend_prediction.append(max(1, min(60, int(predicted))))
            
            # Previsão baseada em gaps
            gap_prediction = []
            for num in range(1, 61):
                if num in gaps:
                    avg_gap = np.mean(gaps[num])
                    if avg_gap > 0:
                        gap_prediction.append((num, avg_gap))
            gap_prediction.sort(key=lambda x: x[1])
            gap_prediction = [num for num, _ in gap_prediction[:6]]
            
            # Previsão baseada em soma
            target_sum = int(np.random.normal(mean_sum, std_sum))
            sum_prediction = []
            while len(sum_prediction) < 6:
                num = random.randint(1, 60)
                if num not in sum_prediction:
                    sum_prediction.append(num)
            sum_prediction = sorted(sum_prediction)
            
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
                cluster_prediction,
                nn_prediction,
                svm_prediction,
                pca_prediction,
                trend_prediction,
                gap_prediction,
                sum_prediction
            ]
            
            # Pesos baseados na confiança de cada método
            weights = [3, 2, 2, 3, 1, 2, 2, 1, 2, 2, 2]  # Pesos ajustados para os novos modelos
            
            final_prediction = combine_predictions(all_predictions, weights)
            
            # Calcula scores de confiança
            confidence_scores = self.calculate_confidence({
                'frequency': frequency_prediction,
                'pattern': pattern_prediction,
                'linear': linear_prediction,
                'random_forest': rf_prediction,
                'cluster': cluster_prediction,
                'neural_network': nn_prediction,
                'svm': svm_prediction,
                'pca': pca_prediction,
                'trend': trend_prediction,
                'gap': gap_prediction,
                'sum': sum_prediction,
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
                'neural_network': nn_prediction,
                'svm': svm_prediction,
                'pca': pca_prediction,
                'trend': trend_prediction,
                'gap': gap_prediction,
                'sum': sum_prediction,
                'final': final_prediction
            }
            
            context['confidence_scores'] = confidence_scores
            context['number_frequency'] = dict(number_frequency)
            context['patterns'] = patterns[-5:]
            context['sequences'] = dict(sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:10])
            context['parity_stats'] = parity_stats[-5:]
            context['range_stats'] = range_stats[-5:]
            context['trends'] = trends
            context['gaps'] = dict(gaps)
            context['mean_sum'] = mean_sum
            context['std_sum'] = std_sum
            context['differences'] = differences[-5:]
            
        return context

def simulation_view(request):
    return render(request, 'predictor/simulation.html')

@csrf_exempt
def simulate_bet(request):
    if request.method == 'POST':
        try:
            print("Recebendo requisição de simulação...")
            
            # Verifica se o corpo da requisição está vazio
            if not request.body:
                return JsonResponse({'error': 'Corpo da requisição vazio'}, status=400)
            
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {str(e)}")
                print("Corpo recebido:", request.body)
                return JsonResponse({'error': 'Formato JSON inválido'}, status=400)
            
            print("Dados recebidos:", data)
            
            numbers = data.get('numbers', [])
            bet_type = data.get('betType', 6)
            
            # Validação dos dados
            if not numbers:
                return JsonResponse({'error': 'Nenhum número foi selecionado'}, status=400)
            
            if not isinstance(numbers, list):
                return JsonResponse({'error': 'Formato inválido para os números'}, status=400)
            
            if not all(isinstance(n, int) for n in numbers):
                return JsonResponse({'error': 'Os números devem ser inteiros'}, status=400)
            
            if not all(1 <= n <= 60 for n in numbers):
                return JsonResponse({'error': 'Os números devem estar entre 1 e 60'}, status=400)
            
            if len(numbers) != bet_type:
                return JsonResponse({'error': f'Selecione exatamente {bet_type} números'}, status=400)
            
            print(f"Simulando aposta com {len(numbers)} números, tipo {bet_type}")
            
            # Obtém os últimos 100 sorteios para simulação
            try:
                last_draws = LotteryResult.objects.all().order_by('-draw_date')[:100]
            except Exception as e:
                print(f"Erro ao buscar sorteios: {str(e)}")
                return JsonResponse({'error': 'Erro ao buscar sorteios do banco de dados'}, status=500)
            
            print(f"Encontrados {len(last_draws)} sorteios para simulação")
            
            if not last_draws:
                return JsonResponse({'error': 'Não há sorteios disponíveis para simulação'}, status=400)
            
            hits_history = []
            total_hits = 0
            max_hits = 0
            total_prizes = 0
            
            # Tabela de premiação (exemplo)
            prize_table = {
                6: {6: 1000000, 5: 1000, 4: 50},
                7: {6: 1000000, 5: 1000, 4: 50, 3: 10},
                8: {6: 1000000, 5: 1000, 4: 50, 3: 10},
                9: {6: 1000000, 5: 1000, 4: 50, 3: 10},
                10: {6: 1000000, 5: 1000, 4: 50, 3: 10}
            }
            
            for draw in last_draws:
                try:
                    draw_numbers = list(map(int, draw.numbers.split(',')))
                    hits = len(set(numbers) & set(draw_numbers))
                    
                    hits_history.append({
                        'draw_number': draw.draw_number,
                        'hits': hits
                    })
                    
                    total_hits += hits
                    
                    if hits > max_hits:
                        max_hits = hits
                    
                    # Calcula prêmio se houver
                    if hits >= 3 and bet_type in prize_table and hits in prize_table[bet_type]:
                        total_prizes += prize_table[bet_type][hits]
                except Exception as e:
                    print(f"Erro ao processar sorteio {draw.draw_number}: {str(e)}")
                    continue
            
            average_hits = total_hits / len(last_draws)
            
            response_data = {
                'average_hits': average_hits,
                'max_hits': max_hits,
                'total_prizes': total_prizes,
                'hits_history': hits_history
            }
            
            print("Simulação concluída. Enviando resposta:", response_data)
            return JsonResponse(response_data)
            
        except Exception as e:
            print("Erro na simulação:", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Método não permitido'}, status=405)

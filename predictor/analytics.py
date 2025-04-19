from collections import Counter, defaultdict
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta

class LotteryAnalytics:
    @staticmethod
    def analyze_parity(numbers_list):
        """Analisa distribuição de números pares e ímpares"""
        parity_stats = []
        for numbers in numbers_list:
            even = sum(1 for n in numbers if n % 2 == 0)
            odd = 6 - even
            parity_stats.append((even, odd))
        return parity_stats

    @staticmethod
    def analyze_sequences(numbers_list):
        """Analisa sequências de números que aparecem juntos frequentemente"""
        sequence_counts = defaultdict(int)
        for numbers in numbers_list:
            # Analisa pares de números
            for pair in combinations(sorted(numbers), 2):
                sequence_counts[pair] += 1
        return sequence_counts

    @staticmethod
    def analyze_gaps(numbers_list):
        """Analisa quanto tempo cada número leva para reaparecer"""
        last_seen = defaultdict(int)
        gaps = defaultdict(list)
        for i, draw in enumerate(numbers_list):
            for num in draw:
                if num in last_seen:
                    gaps[num].append(i - last_seen[num])
                last_seen[num] = i
        return gaps

    @staticmethod
    def analyze_sums(numbers_list):
        """Analisa a distribuição das somas dos números"""
        sums = [sum(draw) for draw in numbers_list]
        return {
            'mean': np.mean(sums),
            'std': np.std(sums),
            'min': min(sums),
            'max': max(sums),
            'distribution': Counter(sums)
        }

    @staticmethod
    def analyze_delayed_numbers(numbers_list, last_draws=10):
        """Identifica números que estão atrasados"""
        all_numbers = set(range(1, 61))
        recent_numbers = set()
        for draw in numbers_list[-last_draws:]:
            recent_numbers.update(draw)
        delayed = all_numbers - recent_numbers
        return sorted(delayed)

    @staticmethod
    def analyze_ranges(numbers_list):
        """Analisa distribuição de números por faixa"""
        range_stats = []
        for numbers in numbers_list:
            low = sum(1 for n in numbers if n <= 30)
            high = 6 - low
            range_stats.append((low, high))
        return range_stats

    @staticmethod
    def analyze_trends(numbers_list):
        """Analisa tendências de crescimento/decrescimento dos números"""
        trends = []
        for i in range(6):
            numbers = [draw[i] for draw in numbers_list]
            slope = np.polyfit(range(len(numbers)), numbers, 1)[0]
            trends.append(slope)
        return trends

    @staticmethod
    def analyze_consecutive(numbers_list):
        """Analisa números consecutivos nos sorteios"""
        consecutive_counts = defaultdict(int)
        for numbers in numbers_list:
            sorted_numbers = sorted(numbers)
            for i in range(5):
                if sorted_numbers[i+1] - sorted_numbers[i] == 1:
                    consecutive_counts[(sorted_numbers[i], sorted_numbers[i+1])] += 1
        return consecutive_counts

    @staticmethod
    def analyze_frequency_distribution(numbers_list):
        """Analisa a distribuição de frequência dos números"""
        all_numbers = [num for draw in numbers_list for num in draw]
        frequency = Counter(all_numbers)
        return {
            'most_common': frequency.most_common(10),
            'least_common': frequency.most_common()[:-11:-1],
            'distribution': dict(frequency)
        }

    @staticmethod
    def analyze_time_patterns(numbers_list, dates):
        """Analisa padrões temporais nos sorteios"""
        weekday_patterns = defaultdict(list)
        month_patterns = defaultdict(list)
        
        for numbers, date in zip(numbers_list, dates):
            weekday_patterns[date.weekday()].extend(numbers)
            month_patterns[date.month].extend(numbers)
            
        return {
            'weekday': {k: Counter(v) for k, v in weekday_patterns.items()},
            'month': {k: Counter(v) for k, v in month_patterns.items()}
        } 
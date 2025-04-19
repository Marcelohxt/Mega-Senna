from django import template
import numpy as np

register = template.Library()

@register.filter
def mean(value):
    """Calcula a média de uma lista de números"""
    if not value:
        return 0
    return np.mean(value)

@register.filter
def median(value):
    """Calcula a mediana de uma lista de números"""
    if not value:
        return 0
    return np.median(value)

@register.filter
def std(value):
    """Calcula o desvio padrão de uma lista de números"""
    if not value:
        return 0
    return np.std(value) 
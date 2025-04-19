import requests
from datetime import datetime, timedelta
import time
from .models import LotteryResult
import json
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MegaSenaScraper:
    def __init__(self):
        self.base_url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Referer': 'https://loterias.caixa.gov.br/Paginas/Mega-Sena.aspx'
        }

    def get_result_by_number(self, draw_number):
        """Obtém o resultado de um concurso específico"""
        url = f"{self.base_url}/{draw_number}"
        logger.info(f"Buscando concurso {draw_number}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, dict):
                raise Exception(f"Formato de dados inesperado: {type(data)}")

            # Processa os números
            numbers = data.get('listaDezenas', [])
            if not numbers:
                raise Exception("Nenhum número encontrado nos dados")

            # Converte os números para string separada por vírgulas
            numbers_str = ','.join(numbers)
            
            # Obtém a data do sorteio
            draw_date = data.get('dataApuracao')
            if not draw_date:
                raise Exception("Data de apuração não encontrada")

            # Obtém o número do concurso
            draw_number = data.get('numero')
            if not draw_number:
                raise Exception("Número do concurso não encontrado")

            return {
                'draw_number': draw_number,
                'draw_date': datetime.strptime(draw_date, '%d/%m/%Y').date(),
                'numbers': numbers_str
            }

        except requests.RequestException as e:
            logger.error(f"Erro ao buscar concurso {draw_number}: {e}")
            return None

    def get_results(self, start_date=None, end_date=None):
        """
        Obtém os resultados da Mega Sena dentro do período especificado.
        Se nenhuma data for fornecida, busca os últimos 40 anos.
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=40*365)
        if not end_date:
            end_date = datetime.now()

        # Primeiro, obtém o último concurso
        url = f"{self.base_url}/?dataInicio={start_date.strftime('%d/%m/%Y')}&dataFim={end_date.strftime('%d/%m/%Y')}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict):
                raise Exception(f"Formato de dados inesperado: {type(data)}")

            # Obtém o número do último concurso
            last_draw = data.get('numero')
            if not last_draw:
                raise Exception("Número do último concurso não encontrado")

            # Obtém o número do primeiro concurso que queremos
            first_draw = 1  # Primeiro concurso da Mega Sena

            results = []
            # Busca cada concurso individualmente
            for draw_number in range(first_draw, last_draw + 1):
                result = self.get_result_by_number(draw_number)
                if result:
                    results.append(result)
                time.sleep(0.5)  # Evita sobrecarregar a API

            return results

        except requests.RequestException as e:
            logger.error(f"Erro na requisição HTTP: {e}")
            raise Exception(f"Erro na requisição HTTP: {e}")

    def save_results(self, results):
        """
        Salva os resultados no banco de dados
        """
        saved_count = 0
        for result in results:
            try:
                # Verifica se o resultado já existe
                if not LotteryResult.objects.filter(draw_number=result['draw_number']).exists():
                    LotteryResult.objects.create(
                        draw_number=result['draw_number'],
                        draw_date=result['draw_date'],
                        numbers=result['numbers']
                    )
                    saved_count += 1
                    logger.info(f"Salvo resultado {result['draw_number']}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultado {result['draw_number']}: {e}")
                continue
        
        return saved_count

def run_scraper():
    """
    Função para executar o scraper e salvar os resultados
    """
    scraper = MegaSenaScraper()
    try:
        logger.info("Iniciando coleta de dados da Mega Sena...")
        results = scraper.get_results()
        logger.info(f"Encontrados {len(results)} resultados")
        
        logger.info("Salvando resultados no banco de dados...")
        saved_count = scraper.save_results(results)
        logger.info(f"Salvos {saved_count} novos resultados")
        logger.info("Processo concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o processo: {e}")
        raise

if __name__ == "__main__":
    run_scraper() 
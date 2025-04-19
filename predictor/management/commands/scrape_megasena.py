from django.core.management.base import BaseCommand
from predictor.scraper import run_scraper

class Command(BaseCommand):
    help = 'Coleta os resultados históricos da Mega Sena'

    def handle(self, *args, **options):
        self.stdout.write('Iniciando coleta de dados da Mega Sena...')
        try:
            run_scraper()
            self.stdout.write(self.style.SUCCESS('Processo concluído com sucesso!'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Erro durante o processo: {e}')) 
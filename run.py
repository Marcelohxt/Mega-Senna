import os
import subprocess
import time
import webbrowser
from threading import Thread

def run_django_server():
    os.system('python manage.py runserver')

def open_chrome():
    # Aguarda 2 segundos para o servidor iniciar
    time.sleep(2)
    # Abre o Chrome com a URL do servidor Django
    webbrowser.get('chrome').open('http://127.0.0.1:8000')

if __name__ == '__main__':
    # Inicia o servidor Django em uma thread separada
    django_thread = Thread(target=run_django_server)
    django_thread.daemon = True
    django_thread.start()
    
    # Abre o Chrome
    open_chrome()
    
    # Mantém o script rodando
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServidor encerrado.") 
# Simulador de Apostas da Mega-Sena

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Django](https://img.shields.io/badge/django-4.2-green)
![Bootstrap](https://img.shields.io/badge/bootstrap-5.0-purple)


![image](https://github.com/user-attachments/assets/b89f095d-9d26-4658-866f-3d11dbabef2b)


Um simulador de apostas da Mega-Sena desenvolvido em Django que permite aos usuários simular apostas e analisar resultados históricos.

## 🎯 Funcionalidades

- **Simulação de Apostas**
  - Seleção de 6 a 15 números
  - Geração de números aleatórios
  - Cálculo de probabilidades
  - Análise de resultados históricos

- **Histórico de Apostas**
  - Registro de todas as apostas simuladas
  - Visualização de estatísticas
  - Análise de desempenho

- **Interface Intuitiva**
  - Design moderno e responsivo
  - Tema escuro para melhor visualização
  - Grid interativo para seleção de números

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute as migrações:
```bash
python manage.py migrate
```

5. Inicie o servidor:
```bash
python manage.py runserver
```

## 📋 Requisitos

- Python 3.8 ou superior
- Django 4.2
- Bootstrap 5
- SQLite3

## 🛠️ Estrutura do Projeto

```
├── megasena/              # Configurações do projeto
├── predictor/             # Aplicação principal
│   ├── migrations/        # Migrações do banco de dados
│   ├── static/           # Arquivos estáticos
│   ├── templates/        # Templates HTML
│   ├── admin.py         # Configuração do admin
│   ├── models.py        # Modelos de dados
│   ├── views.py         # Views da aplicação
│   └── urls.py          # URLs da aplicação
├── manage.py             # Script de gerenciamento
├── requirements.txt      # Dependências do projeto
└── README.md            # Documentação
```

## 💡 Como Usar

1. Acesse a página inicial do simulador
2. Selecione os números desejados (6 a 15)
3. Escolha o tipo de aposta
4. Clique em "Simular Aposta"
5. Analise os resultados e estatísticas

## 📊 Recursos de Análise

- Média de acertos
- Maior número de acertos
- Total de prêmios conquistados
- Histórico de acertos por concurso

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📞 Suporte

Para suporte, envie um email para marcelo_hxt@hotmail.com ou abra uma issue no GitHub.

---

Desenvolvido com ❤️ por Marcelo Henrique

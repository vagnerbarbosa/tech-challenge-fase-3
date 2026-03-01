# 🕷️ Módulo de Web Scraping - Dados Médicos

## 📖 Visão Geral

Este módulo contém scrapers para coleta de dados médicos de três fontes:

| Fonte | Tipo de Dados | Arquivo Gerado |
|-------|---------------|----------------|
| HCPA | Protocolos assistenciais | `protocolos_medicos.csv` |
| TelessaúdeRS | Perguntas frequentes e condutas | `perguntas_frequentes.csv` |
| RadReport | Modelos de laudos radiológicos | `modelos_laudos.csv` |

## 🚀 Como Executar

### Execução Completa (Todos os Scrapers)

```bash
# A partir do diretório raiz do projeto
python -m src.scraping.run_scrapers
```

### Execução Individual

```bash
# HCPA - Protocolos Médicos
python -m src.scraping.hcpa_scraper

# TelessaúdeRS - Perguntas Frequentes
python -m src.scraping.telessaude_scraper

# RadReport - Modelos de Laudos
python -m src.scraping.radreport_scraper
```

## 📂 Estrutura dos Arquivos Gerados

### protocolos_medicos.csv

| Coluna | Descrição |
|--------|----------|
| titulo | Nome do protocolo |
| especialidade | Área médica |
| descricao | Breve descrição do protocolo |
| link | URL do documento |
| fonte | Origem dos dados (HCPA) |

### perguntas_frequentes.csv

| Coluna | Descrição |
|--------|----------|
| pergunta | Questão clínica |
| resposta | Resposta/conduta médica |
| especialidade | Área médica |
| categoria | Tipo de conteúdo |
| fonte | Origem dos dados (TelessaúdeRS) |

### modelos_laudos.csv

| Coluna | Descrição |
|--------|----------|
| nome | Nome do exame/procedimento |
| modalidade | Tipo de imagem (TC, RM, US, etc.) |
| indicacoes | Indicações clínicas |
| estrutura_laudo | Template do laudo |
| especialidade | Subespecialidade radiológica |
| fonte | Origem dos dados (RadReport) |

## 🛠️ Arquitetura

```
src/scraping/
├── __init__.py              # Exports do módulo
├── base_scraper.py          # Classe base com funcionalidades comuns
├── hcpa_scraper.py          # Scraper do HCPA
├── telessaude_scraper.py    # Scraper do TelessaúdeRS
├── radreport_scraper.py     # Scraper do RadReport
└── run_scrapers.py          # Script principal
```

### BaseScraper

Classe abstrata que fornece:
- Gerenciamento de sessões HTTP
- Headers realistas (User-Agent)
- Delays entre requisições (1-3s)
- Sistema de retry com backoff
- Salvamento em CSV

## ⚙️ Configurações

### Variáveis de Ambiente

```bash
# Caminho para salvar os dados (default: ./data)
DATA_PATH=./data

# Nível de log (default: INFO)
LOG_LEVEL=INFO
```

### Boas Práticas Implementadas

- ✅ **User-Agent** realista para simular navegador
- ✅ **Delays aleatórios** entre requisições (1-3s)
- ✅ **Retry com backoff** em caso de falhas
- ✅ **Timeout** para evitar travamentos
- ✅ **Logging estruturado** para monitoramento
- ✅ **Tratamento de erros** robusto

## 📊 Exemplo de Uso em Código

```python
from src.scraping import HCPAScraper, TelessaudeScraper, RadReportScraper
from src.utils.logging_config import setup_logging

# Configura logging
setup_logging()

# Executa scraper individual
scraper = HCPAScraper(output_path="./data")
protocolos = scraper.scrape()  # Retorna lista de dicionários
filepath = scraper.run()       # Executa e salva CSV

print(f"Coletados {len(protocolos)} protocolos")
print(f"Arquivo salvo em: {filepath}")
```

## 📈 Fontes de Dados

### HCPA - Hospital de Clínicas de Porto Alegre

- **URL**: https://www.hcpa.edu.br
- **Conteúdo**: Protocolos assistenciais para diversas especialidades
- **Atualização**: Protocolos passam por consulta pública antes da publicação

### TelessaúdeRS - UFRGS

- **URL**: https://www.ufrgs.br/telessauders
- **Conteúdo**: Telecondutas e perguntas frequentes para atenção primária
- **Parceria**: Ministério da Saúde e SES/RS

### RadReport - RSNA

- **URL**: https://radreport.org
- **Conteúdo**: Templates de laudos radiológicos padronizados
- **Nota**: RSNA não publica novos templates desde dezembro de 2022

## ⚠️ Considerações

1. **Uso responsável**: Os scrapers implementam delays para não sobrecarregar os servidores.

2. **Dados complementares**: Além do scraping web, os scrapers incluem dados estruturados conhecidos para garantir quantidade adequada para estudos.

3. **Atualização**: Execute os scrapers periodicamente para obter dados atualizados.

4. **Limitações**: Algumas páginas podem requerer JavaScript para renderização completa. Os dados estruturados complementam o conteúdo dinâmico.

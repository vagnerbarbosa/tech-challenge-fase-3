# 🕷️ Módulo de Web Scraping - Dados Médicos

## 📖 Visão Geral

Este módulo contém scrapers para coleta de dados médicos de três fontes:

| Fonte | Tipo de Dados | Arquivo Gerado |
|-------|---------------|----------------|
| CONITEC/MS | Protocolos Clínicos e Diretrizes Terapêuticas | `protocolos_medicos.csv` |
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
# CONITEC/MS - Protocolos Clínicos
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
| titulo | Nome do protocolo clínico |
| especialidade | Área médica (mapeamento automático) |
| descricao | Breve descrição do protocolo |
| link | URL do documento no portal CONITEC |
| fonte | Origem dos dados (CONITEC/MS) |

**Estatísticas da Coleta:**
- 📊 **182 registros** de alta qualidade
- ✅ **0% valores nulos** (dados completos)
- ✅ **100% taxa de validação**
- 🔗 **84.6% links únicos**

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
├── hcpa_scraper.py          # Scraper CONITEC/MS (nome mantido por compatibilidade)
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

### Parâmetro `max_items` - Limite de Itens

Cada scraper suporta o parâmetro `max_items` para limitar a quantidade de dados coletados:

| Scraper | Parâmetro | Default | Descrição |
|---------|-----------|---------|-----------|
| CONITEC/MS | `max_items` | 50 | Limite de protocolos médicos |
| TelessaúdeRS | `max_items` | 30 | Limite de FAQs/perguntas |
| RadReport | `max_items` | 20 | Limite de modelos de laudos |

**Nota:** Use `None` para coletar todos os dados disponíveis (sem limite).

#### Modificando os Limites

Os limites padrão estão configurados em `run_scrapers.py`:

```python
SCRAPER_CONFIG = {
    "hcpa": {"max_items": 50},        # Protocolos médicos (CONITEC/MS)
    "telessaude": {"max_items": 30},  # FAQs/perguntas
    "radreport": {"max_items": 20},   # Modelos de laudos
}
```

Para alterar os limites, edite os valores no dicionário `SCRAPER_CONFIG` ou passe parâmetros diretamente:

```python
# Via função
from src.scraping.run_scrapers import run_all_scrapers

results = run_all_scrapers(
    max_items_hcpa=100,      # Coletar até 100 protocolos
    max_items_telessaude=50, # Coletar até 50 FAQs
    max_items_radreport=None # Sem limite para laudos
)
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

# Executa scraper individual com limite de itens
scraper = HCPAScraper(output_path="./data", max_items=10)
protocolos = scraper.scrape()  # Retorna lista de dicionários (máximo 10)
filepath = scraper.run()       # Executa e salva CSV

print(f"Coletados {len(protocolos)} protocolos")
print(f"Arquivo salvo em: {filepath}")

# Exemplo sem limite (coleta todos os dados)
scraper_full = TelessaudeScraper(max_items=None)
faqs = scraper_full.scrape()
print(f"Coletadas {len(faqs)} FAQs")
```

## 📈 Fontes de Dados

### CONITEC/MS - Comissão Nacional de Incorporação de Tecnologias no SUS

- **URL**: https://www.gov.br/conitec/pt-br/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas
- **Conteúdo**: Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) oficiais do Ministério da Saúde
- **Qualidade**: Documentos revisados e aprovados pelo Ministério da Saúde do Brasil
- **Cobertura**: 50+ especialidades médicas mapeadas automaticamente

> **Nota histórica**: Anteriormente este scraper coletava dados do HCPA (Hospital de Clínicas de Porto Alegre), mas a fonte foi migrada para CONITEC/MS devido a melhor disponibilidade e qualidade dos dados. O nome do arquivo (`hcpa_scraper.py`) foi mantido por compatibilidade.

#### Mapeamento Automático de Especialidades

O scraper implementa um sistema inteligente de mapeamento que identifica automaticamente a especialidade médica baseada em palavras-chave no título do protocolo:

| Palavras-chave | Especialidade Mapeada |
|----------------|----------------------|
| diabetes, insulina, glicemia | Endocrinologia |
| câncer, tumor, neoplasia, oncologia | Oncologia |
| coração, cardíaco, infarto, arritmia | Cardiologia |
| renal, rim, diálise, nefropatia | Nefrologia |
| hepatite, fígado, hepático, cirrose | Hepatologia |
| ... | (50+ especialidades) |

### TelessaúdeRS - UFRGS

- **URL**: https://www.ufrgs.br/telessauders
- **Conteúdo**: Telecondutas e perguntas frequentes para atenção primária
- **Parceria**: Ministério da Saúde e SES/RS

### RadReport - RSNA

- **URL**: https://radreport.org
- **Conteúdo**: Templates de laudos radiológicos padronizados
- **Nota**: RSNA não publica novos templates desde dezembro de 2022

## 📊 Comparativo de Qualidade

| Métrica | CONITEC/MS (Atual) | HCPA (Anterior) |
|---------|-------------------|-----------------|
| Total de registros | 182 | ~35 |
| Valores nulos | 0% | ~57% |
| Taxa de validação | 100% | ~60% |
| Links únicos | 84.6% | ~70% |
| Especialidades mapeadas | 50+ | ~15 |

## ⚠️ Considerações

1. **Uso responsável**: Os scrapers implementam delays para não sobrecarregar os servidores.

2. **Dados complementares**: Além do scraping web, os scrapers incluem dados estruturados conhecidos para garantir quantidade adequada para estudos.

3. **Atualização**: Execute os scrapers periodicamente para obter dados atualizados.

4. **Limitações**: Algumas páginas podem requerer JavaScript para renderização completa. Os dados estruturados complementam o conteúdo dinâmico.

5. **Validação rigorosa**: O scraper CONITEC/MS implementa validação para garantir que apenas conteúdo médico relevante seja coletado (filtros regex, palavras-chave médicas obrigatórias, remoção de entradas duplicadas).

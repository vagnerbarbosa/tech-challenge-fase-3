# 🕷️ Módulo de Web Scraping - Dados Médicos

## 📖 Visão Geral

Este módulo contém scrapers para coleta de dados médicos de três fontes. Todos os scrapers geram arquivos **JSONL** diretamente no formato para fine-tuning.

| Fonte | Tipo de Dados | Arquivo Gerado |
|-------|---------------|----------------|
| CONITEC/MS | Protocolos Clínicos e Diretrizes | `protocolos_medicos.jsonl` |
| TelessaúdeRS | Perguntas frequentes e condutas | `perguntas_frequentes.jsonl` |
| RadReport | Modelos de laudos radiológicos | `modelos_laudos.jsonl` |

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

## 📂 Formato JSONL Gerado

Todos os scrapers geram arquivos JSONL em `data/raw/` com a seguinte estrutura:

```json
{
  "instruction": "Pergunta ou instrução para o modelo",
  "input": "Contexto adicional (opcional)",
  "output": "Resposta esperada",
  "source": "Fonte dos dados"
}
```

### Exemplo de Registro

```json
{"instruction": "Quando está indicado o uso de antibióticos em IVAS?", "input": "Especialidade: Infectologia | Categoria: Uso racional de medicamentos", "output": "A maioria das infecções de vias aéreas superiores é causada por vírus...", "source": "TelessaúdeRS-UFRGS"}
```

## 🛠️ Arquitetura

```
src/scraping/
├── __init__.py              # Exports do módulo
├── base_scraper.py          # Classe base com _save_to_jsonl()
├── hcpa_scraper.py          # Scraper CONITEC/MS
├── telessaude_scraper.py    # Scraper do TelessaúdeRS
├── radreport_scraper.py     # Scraper do RadReport
└── run_scrapers.py          # Script principal

data/
├── raw/                     # Arquivos JSONL gerados pelos scrapers
│   ├── protocolos_medicos.jsonl
│   ├── perguntas_frequentes.jsonl
│   └── modelos_laudos.jsonl
└── processed/               # JSONL unificado para training
    └── medical_data_unified.jsonl
```

## ⚙️ Configurações

### Variáveis de Ambiente

```bash
DATA_PATH=./data        # Caminho base para dados (default: ./data)
LOG_LEVEL=INFO          # Nível de log (default: INFO)
```

### Parâmetro `max_items`

Cada scraper suporta o parâmetro `max_items` para limitar a quantidade de dados:

| Scraper | Default | Descrição |
|---------|---------|-----------|
| CONITEC/MS | 50 | Protocolos médicos |
| TelessaúdeRS | 30 | FAQs/perguntas |
| RadReport | 20 | Modelos de laudos |

Use `None` para coletar todos os dados disponíveis.

## 📊 Fontes de Dados

### CONITEC/MS - Comissão Nacional de Incorporação de Tecnologias no SUS

- **URL**: https://www.gov.br/conitec/
- **Conteúdo**: Protocolos Clínicos e Diretrizes Terapêuticas (PCDT)
- **Transformação**: `titulo` → instruction, `descricao` → output

### TelessaúdeRS - UFRGS

- **URL**: https://www.ufrgs.br/telessauders
- **Conteúdo**: Telecondutas e perguntas frequentes
- **Transformação**: `pergunta` → instruction, `resposta` → output

### RadReport - RSNA

- **URL**: https://radreport.org
- **Conteúdo**: Templates de laudos radiológicos padronizados
- **Transformação**: `nome` → instruction, `estrutura_laudo` → output

## 🔄 Fluxo de Dados

```
1. Scraping                    2. Preparação                3. Training
   ─────────────────────────────────────────────────────────────────────
   
   run_scrapers.py            data_preparation.py          training.py
         │                           │                          │
         ▼                           ▼                          ▼
   data/raw/                  data/processed/              Fine-tuning
   ├── protocolos.jsonl  ───► medical_data_unified.jsonl ───► LLM
   ├── perguntas.jsonl
   └── modelos.jsonl
```

### Exemplo de Uso

```python
# 1. Executar scraping
from src.scraping.run_scrapers import run_all_scrapers
results = run_all_scrapers()

# 2. Preparar dados para training
from src.fine_tuning.data_preparation import DataPreparation
prep = DataPreparation()
dataset = prep.prepare_dataset()
```

## ⚠️ Considerações

1. **Uso responsável**: Os scrapers implementam delays (1-3s) entre requisições.

2. **Formato direto**: Dados já saem no formato JSONL pronto para fine-tuning.

3. **Sem conversões**: Não há mais etapa de CSV → JSONL, simplificando o fluxo.

4. **Atualização**: Execute os scrapers periodicamente para dados atualizados.

## 📈 Boas Práticas Implementadas

- ✅ User-Agent realista para simular navegador
- ✅ Delays aleatórios entre requisições (1-3s)
- ✅ Retry com backoff em caso de falhas
- ✅ Timeout para evitar travamentos
- ✅ Logging estruturado para monitoramento
- ✅ Tratamento de erros robusto
- ✅ Formato JSONL nativo (sem conversões)

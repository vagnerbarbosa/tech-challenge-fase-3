# 🏥 Assistente Virtual Médico Generalista - Tech Challenge Fase 3

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://langchain.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-orange.svg)](https://huggingface.co/)

## 📋 Descrição do Projeto

Este projeto implementa um **Assistente Virtual Médico Generalista** utilizando técnicas avançadas de processamento de linguagem natural (NLP) e aprendizado de máquina. O sistema combina:

- 🔧 **Fine-tuning de LLM** (Large Language Model)
- 🔗 **LangChain** para integração e orquestração
- 🔄 **LangGraph** para fluxos automatizados
- 🔒 **Anonimização de dados** médicos sensíveis

## 🎯 Objetivo da Fase 3

Desenvolver um assistente virtual médico generalista capaz de:

1. **Responder perguntas** sobre diversas condições médicas e sintomas
2. **Fornecer orientações** gerais de saúde e bem-estar
3. **Manter conversação** contextualizada e segura
4. **Garantir privacidade** através de anonimização de dados
5. **Orientar encaminhamento** para especialistas quando necessário

## 📁 Estrutura do Repositório

```
projeto_fase3/
├── data/                          # Datasets
│   ├── raw/                       # Dados brutos (não versionados)
│   └── processed/                 # Dados processados e anonimizados
├── notebooks/                     # Jupyter notebooks para experimentação
├── src/
│   ├── fine_tuning/              # Pipeline de fine-tuning
│   │   ├── data_preparation.py   # Pré-processamento e anonimização
│   │   ├── training.py           # Treinamento do modelo
│   │   └── evaluation.py         # Avaliação do modelo
│   ├── langchain_integration/    # Integração LangChain
│   │   ├── assistant.py          # Assistente médico principal
│   │   ├── chains.py             # Chains do LangChain
│   │   └── tools.py              # Ferramentas customizadas
│   ├── langgraph_flows/          # Fluxos automatizados
│   │   └── medical_workflow.py   # Workflow médico
│   └── utils/                    # Utilitários
│       ├── logging_config.py     # Configuração de logs
│       └── validators.py         # Validadores de segurança
├── models/                        # Modelos treinados (não versionados)
├── logs/                          # Logs do sistema
├── tests/                         # Testes unitários
├── .gitignore                     # Arquivos a ignorar
├── .env.example                   # Exemplo de variáveis de ambiente
├── requirements.txt               # Dependências Python
├── README.md                      # Este arquivo
└── main.py                        # Script principal
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.10 ou superior
- CUDA 11.8+ (para GPU - recomendado)
- 16GB+ RAM
- Conta no Hugging Face (para acesso aos modelos)

### Passos

1. **Clone o repositório**
```bash
git clone https://github.com/vagnerbarbosa/tech-challenge-fase-3.git
cd tech-challenge-fase-3
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite o arquivo .env com suas credenciais
```

5. **Faça login no Hugging Face**
```bash
huggingface-cli login
```

## 💻 Como Executar

### Pipeline Completo
```bash
python main.py
```

### Etapas Individuais

```bash
# 1. Preparação de dados
python -m src.fine_tuning.data_preparation

# 2. Treinamento do modelo
python -m src.fine_tuning.training

# 3. Avaliação
python -m src.fine_tuning.evaluation

# 4. Executar assistente
python -m src.langchain_integration.assistant
```

### Testes
```bash
pytest tests/ -v
```

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Python | 3.10+ | Linguagem principal |
| Transformers | 4.36+ | Modelos de linguagem |
| LangChain | 0.1+ | Orquestração de LLM |
| LangGraph | 0.0.20+ | Fluxos automatizados |
| PyTorch | 2.1+ | Framework de DL |
| PEFT | 0.7+ | Fine-tuning eficiente (LoRA) |
| Pandas | 2.1+ | Manipulação de dados |

## 📊 Dataset

O projeto utiliza dados médicos de diversas especialidades, seguindo rigorosos padrões de:

- ✅ **LGPD** - Lei Geral de Proteção de Dados
- ✅ **Anonimização** - Remoção de dados identificáveis
- ✅ **Segurança** - Validação de inputs e outputs

## 🏥 Áreas de Atuação

O assistente pode fornecer informações educativas sobre:

- **Clínica Geral**: sintomas comuns, orientações gerais
- **Prevenção**: hábitos saudáveis, check-ups recomendados
- **Emergências**: reconhecimento de sinais de alerta
- **Medicamentos**: informações gerais (não substituindo prescrição)
- **Encaminhamentos**: orientação sobre especialidades médicas

## 📝 Licença

Este projeto é desenvolvido para fins acadêmicos como parte da pós-graduação em IA para Desenvolvedores (IADT).

## 👨‍💻 Autor

**Vagner Barbosa**
- GitHub: [@vagnerbarbosa](https://github.com/vagnerbarbosa)
- Website: [vagnerbarbosa.com](https://www.vagnerbarbosa.com)

---

*Tech Challenge - FIAP/Alura - Pós-Graduação em IA para Desenvolvedores*

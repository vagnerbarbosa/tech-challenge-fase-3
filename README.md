# 🏥 Tech Challenge Fase 3 - Assistente Virtual Médico

> **Pós-graduação em Inteligência Artificial para Desenvolvedores (IADT)**  
> Assistente virtual médico personalizado utilizando Fine-tuning de LLM e LangChain

---

## 📋 Descrição do Projeto

Este projeto implementa um **assistente virtual médico** que utiliza técnicas avançadas de Inteligência Artificial para fornecer informações e orientações de saúde. O sistema é construído com:

- **Fine-tuning de LLM**: Customização de modelos de linguagem (LLaMA ou Falcon) para o domínio médico
- **LangChain**: Framework para orquestração de prompts e integração com LLMs
- **LangGraph**: Fluxos automatizados para workflows médicos complexos
- **Dados Anonimizados**: Proteção de informações sensíveis de pacientes

---

## 📁 Estrutura do Repositório

```
tech-challenge-fase-3/
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
│   │   └── medical_workflow.py   # Workflow médico com LangGraph
│   └── utils/                    # Utilitários
│       ├── logging_config.py     # Configuração de logs
│       └── validators.py         # Validadores de segurança
├── models/                        # Modelos treinados (não versionados)
├── logs/                          # Logs do sistema
├── tests/                         # Testes unitários
├── .env.example                   # Exemplo de variáveis de ambiente
├── requirements.txt               # Dependências Python
├── main.py                        # Script principal
└── README.md                      # Este arquivo
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.10+
- CUDA (recomendado para treinamento GPU)
- Git

### Passos de Instalação

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

---

## 💻 Como Executar

### Preparação de Dados
```bash
python -m src.fine_tuning.data_preparation
```

### Fine-tuning do Modelo
```bash
python -m src.fine_tuning.training
```

### Executar o Assistente
```bash
python main.py
```

### Executar Testes
```bash
pytest tests/ -v
```

---

## 📊 Requisitos da Fase 3

| Requisito | Descrição | Status |
|-----------|-----------|--------|
| Fine-tuning LLM | Customização de LLaMA/Falcon para domínio médico | 🔄 |
| Integração LangChain | Orquestração de prompts e chains | 🔄 |
| Fluxos LangGraph | Workflows automatizados | 🔄 |
| Anonimização | Proteção de dados sensíveis (LGPD) | 🔄 |
| Logging | Sistema de logs estruturado | 🔄 |
| Validação | Verificação de segurança das respostas | 🔄 |

---

## 🔐 Segurança e Privacidade

- **Anonimização**: Todos os dados de pacientes são anonimizados antes do processamento
- **LGPD**: Conformidade com a Lei Geral de Proteção de Dados
- **Validação**: Respostas são validadas para evitar informações médicas incorretas

---

## 📝 Licença

Este projeto é desenvolvido para fins educacionais como parte do Tech Challenge da pós-graduação IADT.

---

## 👤 Autor

**Vagner Barbosa**  
- GitHub: [@vagnerbarbosa](https://github.com/vagnerbarbosa)
- Website: [vagnerbarbosa.com](https://www.vagnerbarbosa.com)

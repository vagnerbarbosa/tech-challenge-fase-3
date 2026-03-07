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

## 📊 Requisitos da Fase 3

| Requisito | Descrição | Status |
|-----------|-----------|--------|
| Fine-tuning LLM | Customização de LLaMA/Falcon para domínio médico | 🔄 |
| Integração LangChain | Orquestração de prompts e chains | 🔄 |
| Fluxos LangGraph | Workflows automatizados | 🔄 |
| Anonimização | Proteção de dados sensíveis (LGPD) | 🔄 |
| Logging | Sistema de logs estruturado | 🔄 |
| Validação | Verificação de segurança das respostas | 🔄 |

## 📁 Estrutura do Repositório

```
projeto_fase3/
├── data/                              # Datasets
│   ├── raw/                           # Dados brutos (não versionados)
│   │   └── .gitkeep
│   └── processed/                     # Dados processados e anonimizados
│       └── .gitkeep
├── logs/                              # Logs do sistema (não versionados)
│   └── .gitkeep
├── models/                            # Modelos treinados (não versionados)
│   └── .gitkeep
├── notebooks/                         # Jupyter notebooks para experimentação
│   └── .gitkeep
├── src/                               # Código fonte principal
│   ├── __init__.py                    # Inicializador do pacote src
│   ├── fine_tuning/                   # Pipeline de fine-tuning do LLM
│   │   ├── __init__.py                # Exports: DataPreparation, ModelTrainer, ModelEvaluator
│   │   ├── data_preparation.py        # Pré-processamento e anonimização de dados médicos
│   │   ├── training.py                # Treinamento do modelo com LoRA/PEFT
│   │   └── evaluation.py              # Avaliação de qualidade do modelo
│   ├── langchain_integration/         # Integração com LangChain
│   │   ├── __init__.py                # Exports: MedicalAssistant, MedicalChains, MedicalTools
│   │   ├── assistant.py               # Assistente médico principal
│   │   ├── chains.py                  # Chains de Q&A médico
│   │   └── tools.py                   # Ferramentas: emergência, temperatura, especialidades
│   ├── langgraph_flows/               # Fluxos automatizados com LangGraph
│   │   ├── __init__.py                # Exports: MedicalWorkflow
│   │   └── medical_workflow.py        # Workflow de conversação médica
│   └── utils/                         # Utilitários do projeto
│       ├── __init__.py                # Exports: setup_logging, get_logger, DataValidator, InputValidator
│       ├── logging_config.py          # Configuração centralizada de logs
│       └── validators.py              # Validadores de entrada e dados
├── tests/                             # Testes unitários
│   ├── __init__.py                    # Inicializador do pacote de testes
│   └── test_validators.py             # Testes para InputValidator e DataValidator
├── .env.example                       # Exemplo de variáveis de ambiente
├── .gitignore                         # Arquivos e pastas ignorados pelo Git
├── login_hf.py                        # Script auxiliar para login no Hugging Face
├── main.py                            # Script principal - ponto de entrada da aplicação
├── README.md                          # Documentação do projeto
└── requirements.txt                   # Dependências Python do projeto
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.10 ou superior
- CUDA 11.8+ (para GPU - recomendado)
- 16GB+ RAM
- Conta no Hugging Face (para acesso aos modelos)
- Git (recomendado: Git for Windows com Git Bash)

---

## 🪟 Instalação no Windows (Passo a Passo)

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/vagnerbarbosa/tech-challenge-fase-3.git
cd tech-challenge-fase-3
```

### 2️⃣ Crie um ambiente virtual

**Opção A - PowerShell ou CMD:**
```powershell
python -m venv venv
```

**Opção B - Git Bash:**
```bash
python -m venv venv
```

### 3️⃣ Ative o ambiente virtual

**⚠️ IMPORTANTE:** Escolha o comando correto de acordo com o terminal que você está usando:

| Terminal | Comando de Ativação |
|----------|---------------------|
| **PowerShell** | `.\venv\Scripts\Activate.ps1` |
| **CMD (Prompt de Comando)** | `venv\Scripts\activate.bat` |
| **Git Bash / MINGW64** | `source venv/Scripts/activate` |

**Exemplo no PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Exemplo no CMD:**
```cmd
venv\Scripts\activate.bat
```

**Exemplo no Git Bash:**
```bash
source venv/Scripts/activate
```

> 💡 **Dica:** Após ativar, você verá `(venv)` no início da linha do terminal, indicando que o ambiente virtual está ativo.

> ⚠️ **Problema com PowerShell?** Se receber erro de "Execution Policy", execute primeiro:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4️⃣ Instale as dependências

Com o ambiente virtual ativado, execute:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **Dica:** Se encontrar erros durante a instalação, tente instalar as dependências principais primeiro:
> ```bash
> pip install torch transformers langchain huggingface_hub
> pip install -r requirements.txt
> ```

### 5️⃣ Configure as variáveis de ambiente

**Opção A - Copiar e editar o arquivo .env:**

**No PowerShell:**
```powershell
Copy-Item .env.example .env
notepad .env
```

**No CMD:**
```cmd
copy .env.example .env
notepad .env
```

**No Git Bash:**
```bash
cp .env.example .env
notepad .env
```

**Opção B - Editar manualmente:**

Abra o arquivo `.env` em um editor de texto (VS Code, Notepad++, etc.) e configure:

```env
# Seu token do Hugging Face (obtenha em https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=hf_seu_token_aqui

# Alternativa: HF_TOKEN (usado automaticamente pelo huggingface_hub)
HF_TOKEN=hf_seu_token_aqui

# Modelo base
BASE_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf

# Configurações de caminhos
MODEL_PATH=./models
DATA_PATH=./data
LOG_PATH=./logs

# Outras configurações (ajuste conforme necessário)
MAX_SEQ_LENGTH=512
BATCH_SIZE=4
LEARNING_RATE=2e-4
NUM_EPOCHS=3
LOG_LEVEL=INFO
```

### 6️⃣ Faça login no Hugging Face

Existem **várias formas** de fazer login no Hugging Face. Escolha a que funcionar melhor no seu ambiente:

#### ✅ Método 1: Script Auxiliar (RECOMENDADO para Windows/Git Bash)

```bash
python login_hf.py
```

Este script abre um prompt interativo para você inserir seu token.

#### ✅ Método 2: Comando Python Direto

```bash
python -c "from huggingface_hub import login; login()"
```

#### ✅ Método 3: Comando `hf` (CLI oficial)

```bash
hf auth login
```

> ⚠️ **Nota:** No Windows/Git Bash, o executável `hf` pode não estar no PATH. Se receber "command not found", use o Método 1 ou 2.

#### ✅ Método 4: Variável de Ambiente (Sem login interativo)

Adicione seu token ao arquivo `.env`:
```env
HF_TOKEN=hf_seu_token_aqui
```

A biblioteca `huggingface_hub` detecta automaticamente esta variável.

#### 📝 Onde obter seu token?

1. Acesse: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Clique em **"New token"**
3. Dê um nome e escolha o tipo:
   - **Read**: Para baixar modelos (suficiente para a maioria dos casos)
   - **Write**: Para fazer upload de modelos
4. Copie o token gerado (começa com `hf_`)

---

## 🐧 Instalação no Linux/Mac

1. **Clone o repositório**
```bash
git clone https://github.com/vagnerbarbosa/tech-challenge-fase-3.git
cd tech-challenge-fase-3
```

2. **Crie e ative o ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
nano .env  # ou use seu editor preferido
```

5. **Faça login no Hugging Face**
```bash
# Opção 1: CLI oficial
hf auth login

# Opção 2: Script auxiliar
python login_hf.py

# Opção 3: Python direto
python -c "from huggingface_hub import login; login()"
```

---

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

---

## ❓ Solução de Problemas Comuns (Windows)

### Erro: "hf: command not found" ou "huggingface-cli: command not found"

**Problema:** O executável do Hugging Face CLI não está no PATH do sistema (muito comum no Git Bash/Windows).

**Soluções:**

1. **Use o script auxiliar (mais fácil):**
   ```bash
   python login_hf.py
   ```

2. **Use Python diretamente:**
   ```bash
   python -c "from huggingface_hub import login; login()"
   ```

3. **Use variável de ambiente:**
   - Adicione `HF_TOKEN=hf_seu_token` no arquivo `.env`
   - A biblioteca detectará automaticamente

4. **Encontre o executável manualmente:**
   ```bash
   # Descubra onde está instalado
   python -c "import huggingface_hub; print(huggingface_hub.__file__)"
   
   # O executável geralmente está em:
   # Windows: venv/Scripts/hf.exe
   # Linux/Mac: venv/bin/hf
   ```

### Erro: "Execution Policy" no PowerShell
**Problema:** PowerShell bloqueia a execução de scripts.
**Solução:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Erro: "python não é reconhecido"
**Problema:** Python não está no PATH do sistema.
**Solução:** 
- Reinstale o Python marcando a opção "Add Python to PATH"
- Ou use `py` em vez de `python`:
```bash
py -m venv venv
```

### Erro ao instalar torch/PyTorch
**Solução:** Instale o PyTorch separadamente antes das outras dependências:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Substitua `cu118` pela sua versão do CUDA, ou use `cpu` se não tiver GPU)

### Erro: "No module named 'src'"
**Problema:** O Python não está encontrando os módulos do projeto.
**Solução:** Execute os comandos a partir da raiz do projeto (onde está o `main.py`)

### Erro: "Token não autorizado" ou "401 Unauthorized"
**Problema:** Token do Hugging Face inválido ou sem permissões adequadas.
**Soluções:**
1. Verifique se o token está correto (começa com `hf_`)
2. Verifique se o token tem permissão de leitura
3. Para modelos privados/gated (como LLaMA), aceite os termos de uso na página do modelo
4. Gere um novo token em: https://huggingface.co/settings/tokens

---

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

## 👥 Equipe

Este projeto foi desenvolvido por:

- [Adriel Santos](https://github.com/AdrielCandido)
- [João Marcos](https://github.com/joaomarcos999)
- [Leticia Nepomucena](https://github.com/LeticiaNepomucena)
- [Lucas Silva](https://github.com/lucfsilva)
- [Vagner Barbosa](https://github.com/vagnerbarbosa)

---

*Tech Challenge - FIAP/Alura - Pós-Graduação em IA para Desenvolvedores*

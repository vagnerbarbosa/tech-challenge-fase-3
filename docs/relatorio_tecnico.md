# 📋 Relatório Técnico - Assistente Virtual Médico Generalista

**Tech Challenge IADT - Fase 3**  
**Data:** Março 2026  
**Versão:** 2.0.0

---

## 1. Descrição da Arquitetura

### 1.1 Visão Geral

O sistema implementa um **Assistente Virtual Médico Generalista** que combina técnicas de Fine-Tuning de LLMs, orquestração com LangChain/LangGraph, Retrieval-Augmented Generation (RAG) e integração com bases de dados estruturadas de prontuários médicos.

### 1.2 Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    MedicalWorkflow (LangGraph)                │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Classify  │→│   Router      │→│  Handlers              │ │
│  │ Message   │  │ (Conditional) │  │  • Greeting            │ │
│  └──────────┘  └──────────────┘  │  • Emergency            │ │
│                                    │  • Vital Signs          │ │
│                                    │  • Question (+ RAG)     │ │
│                                    │  • Farewell             │ │
│                                    └────────────────────────┘ │
│                                             │                  │
│                                    ┌────────▼────────┐        │
│                                    │ Generate Response│        │
│                                    │ + Citations      │        │
│                                    └─────────────────┘        │
└─────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐
│  MedicalRAG     │      │  PatientDatabase    │
│  (TF-IDF /      │      │  (SQLite)           │
│   Embeddings)   │      │                     │
│                 │      │  Prontuários         │
│  Protocolos:   │      │  simulados com:     │
│  • CONITEC     │      │  • Histórico        │
│  • TelessaúdeRS│      │  • Medicações       │
│  • RadReport   │      │  • Alergias         │
└─────────────────┘      │  • Exames           │
                         └─────────────────────┘
```

### 1.3 Módulos do Sistema

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/fine_tuning/` | Pipeline de fine-tuning com LoRA (BioMistral-7B/TinyLlama) |
| `src/langchain_integration/` | Assistente, Chains, Tools e **RAG** |
| `src/langgraph_flows/` | Workflow de conversação com StateGraph |
| `src/database/` | Base de dados simulada de prontuários (SQLite) |
| `src/scraping/` | Coleta de protocolos médicos (CONITEC, TelessaúdeRS, RadReport) |
| `src/utils/` | Logging, validadores de segurança |

---

## 2. Decisões Técnicas e Justificativas

### 2.1 Fine-Tuning com LoRA/QLoRA

**Decisão:** Utilizar LoRA (Low-Rank Adaptation) com quantização de 4 bits para fine-tuning.

**Justificativa:**
- Reduz drasticamente os requisitos de memória GPU (de ~28GB para ~8GB)
- Permite treinar modelos de 7B parâmetros em hardware consumer
- Mantém ~95% da performance do fine-tuning completo
- Treinamento significativamente mais rápido

**Parâmetros LoRA utilizados:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, v_proj

### 2.2 Sistema RAG (Retrieval-Augmented Generation)

**Decisão:** Implementar RAG com TF-IDF como método padrão e embeddings como opção avançada.

**Justificativa:**
- TF-IDF é leve, não requer GPU e funciona bem para vocabulário médico em português
- N-grams (1,2) capturam termos compostos médicos (ex: "pressão arterial")
- Suporte a sentence-transformers como upgrade quando disponível
- Fallback para busca por keywords em ambientes mínimos

**Configuração:**
- `max_features=5000` para controlar dimensionalidade
- `sublinear_tf=True` para melhor discriminação de termos
- `top_k=3` documentos por consulta
- `min_score=0.05` para filtrar resultados irrelevantes

### 2.3 Base de Dados de Prontuários

**Decisão:** SQLite para a base simulada de prontuários médicos.

**Justificativa:**
- Zero configuração (sem servidor de banco de dados)
- Portabilidade total (arquivo único)
- Suporte nativo a consultas SQL para buscas estruturadas
- Campos JSON armazenados como TEXT com parsing automático
- Adequado para demonstração e prototipagem

**Schema:** Tabela `pacientes` com campos para dados demográficos, histórico médico, medicações, alergias, exames recentes e consultas anteriores.

### 2.4 Explainability (Citação de Fontes)

**Decisão:** Incluir citações de fontes em todas as respostas do assistente.

**Justificativa:**
- Transparência sobre a origem das informações médicas
- Permite ao profissional de saúde verificar as referências
- Atende ao requisito de rastreabilidade (R3 do Tech Challenge)
- Formatos claros: "Segundo protocolo CONITEC...", "Base TelessaúdeRS..."

### 2.5 LangGraph para Workflow

**Decisão:** Utilizar LangGraph StateGraph para orquestração do fluxo.

**Justificativa:**
- Roteamento condicional baseado em classificação de mensagens
- Estado compartilhado entre nós do grafo
- Extensibilidade (fácil adicionar novos tipos de mensagem)
- Suporte nativo a cycles e conditional edges

---

## 3. Métricas de Avaliação

### 3.1 Métricas do Modelo Fine-Tuned

| Métrica | Descrição | Implementação |
|---------|-----------|---------------|
| **Perplexidade** | Mede a qualidade do modelo de linguagem | `ModelEvaluator.calculate_perplexity()` |
| **QA Quality** | Similaridade entre respostas geradas e esperadas | `ModelEvaluator.evaluate_qa()` usando `SequenceMatcher` |
| **Repetition Penalty** | Controle de repetições na geração | Parâmetro `repetition_penalty=1.3` |

### 3.2 Métricas do Sistema RAG

| Métrica | Descrição |
|---------|-----------|
| **Relevance Score** | Similaridade cosseno entre query e documentos (0-1) |
| **Top-K Precision** | Proporção de documentos relevantes nos top-K resultados |
| **Coverage** | Número de fontes distintas consultadas por resposta |

### 3.3 Validação de Segurança

| Verificação | Implementação |
|-------------|---------------|
| Detecção de XSS/Injection | `InputValidator.validate_query()` |
| Sanitização de texto | `InputValidator.sanitize_text()` |
| Detecção de emergências | `MedicalTools.is_emergency_question()` |
| Validação de dados médicos | `DataValidator.validate_medical_records()` |

---

## 4. Fluxo de Dados do Sistema

### 4.1 Pipeline de Dados (Offline)

```
Web Scraping → JSONL → Data Preparation → Fine-tuning → Modelo Treinado
     │                                                        │
     └────── Indexação RAG ◄──────────────────────────────────┘
                  │
           Índice TF-IDF / Embeddings
```

### 4.2 Fluxo de Consulta (Online)

```
1. Usuário envia pergunta
2. InputValidator verifica segurança da entrada
3. LangGraph classifica o tipo de mensagem
4. Router direciona para handler adequado
5. [Para perguntas] MedicalRAG busca protocolos relevantes
6. [Se paciente definido] PatientDatabase recupera prontuário
7. Contexto é injetado no prompt do LLM
8. LLM gera resposta contextualizada
9. Citações de fontes são anexadas à resposta
10. Resposta é salva na memória de conversação
```

---

## 5. Tecnologias e Dependências

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| Python | 3.10+ | Linguagem principal |
| Transformers | 4.36+ | Modelos LLM (BioMistral-7B) |
| LangChain | 0.1+ | Orquestração de chains e prompts |
| LangGraph | 0.0.20+ | Workflow com StateGraph |
| PyTorch | 2.1+ | Framework de deep learning |
| PEFT | 0.7+ | Fine-tuning eficiente (LoRA) |
| scikit-learn | 1.3+ | TF-IDF para RAG |
| SQLite3 | stdlib | Base de dados de prontuários |
| BeautifulSoup4 | 4.12+ | Web scraping |
| Graphviz | 0.21+ | Geração de diagramas |

---

## 6. Resultados e Conclusões

### 6.1 Funcionalidades Entregues

- ✅ **Fine-tuning de LLM** com LoRA/QLoRA (BioMistral-7B, TinyLlama)
- ✅ **Web Scraping** de 3 fontes médicas brasileiras (CONITEC, TelessaúdeRS, RadReport)
- ✅ **Integração LangChain** com chains de Q&A e sumarização
- ✅ **Workflow LangGraph** com classificação e roteamento de mensagens
- ✅ **Sistema RAG** para busca semântica em protocolos médicos
- ✅ **Base de Prontuários** simulada com SQLite (5 pacientes fictícios)
- ✅ **Explainability** com citação de fontes em todas as respostas
- ✅ **Segurança** com validação de entrada, sanitização e detecção de emergências
- ✅ **Anonimização** de dados sensíveis (LGPD)
- ✅ **Documentação** completa com diagrama de fluxo e relatório técnico

### 6.2 Limitações Conhecidas

- O modelo fine-tuned requer GPU para inferência em tempo real
- A base de prontuários é simulada (dados fictícios)
- O RAG com TF-IDF tem limitações em queries muito curtas ou ambíguas
- A detecção de emergências é baseada em palavras-chave (não é classificador ML)

### 6.3 Melhorias Futuras

- Implementar embeddings com modelos multilíngues para melhor busca semântica em PT-BR
- Expandir a base de prontuários com geração sintética via LLM
- Adicionar avaliação automatizada de qualidade das respostas (BLEU, ROUGE)
- Integrar com APIs reais de registros médicos (FHIR/HL7)
- Implementar interface web (Streamlit/Gradio) para demonstração interativa

---

## 7. Referências

1. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains (2024)
2. LangChain Documentation - https://python.langchain.com/
3. LangGraph Documentation - https://langchain-ai.github.io/langgraph/
4. PEFT: Parameter-Efficient Fine-Tuning - https://huggingface.co/docs/peft
5. Protocolos CONITEC/MS - https://www.gov.br/conitec/
6. TelessaúdeRS-UFRGS - https://www.ufrgs.br/telessauders/

---

*Relatório gerado como parte do Tech Challenge IADT - Fase 3*  
*FIAP/Alura - Pós-Graduação em IA para Desenvolvedores*

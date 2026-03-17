#!/usr/bin/env python3
"""
Tech Challenge Fase 3 - Assistente Virtual Médico Generalista
=============================================================

Script principal que orquestra todo o pipeline:
1. Preparação e anonimização de dados médicos
2. Fine-tuning do modelo LLM
3. Avaliação do modelo
4. Execução do assistente médico generalista com LangChain

"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Adiciona o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

# Carrega variáveis de ambiente
load_dotenv()

# Imports dos módulos do projeto
from src.utils.logging_config import setup_logging, get_logger
from src.fine_tuning.data_preparation import DataPreparation
from src.fine_tuning.training import ModelTrainer
from src.fine_tuning.evaluation import ModelEvaluator
from src.langchain_integration.assistant import MedicalAssistant
from src.langgraph_flows.medical_workflow import MedicalWorkflow


def main():
    """
    Função principal que executa o pipeline completo.
    """
    # Configura logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("Tech Challenge Fase 3 - Assistente Virtual Médico Generalista")
    logger.info("="*60)
    
    try:
        # Etapa 1: Preparação de Dados
        logger.info("\n📊 Etapa 1: Preparação e Anonimização de Dados Médicos")
        data_prep = DataPreparation()
        dataset = data_prep.prepare_dataset()
        logger.info(f"Dataset preparado com {len(dataset)} registros")
        
        # Etapa 2: Fine-tuning do Modelo
        logger.info("\n🔧 Etapa 2: Fine-tuning do Modelo LLM")
        trainer = ModelTrainer()
        model, tokenizer = trainer.train(dataset)
        logger.info("Modelo treinado com sucesso!")
        
        # Etapa 3: Avaliação
        logger.info("\n📈 Etapa 3: Avaliação do Modelo")
        evaluator = ModelEvaluator(model, tokenizer)
        metrics = evaluator.evaluate(dataset)
        logger.info(f"Métricas de avaliação: {metrics}")
        
        # Etapa 4: Configuração do Assistente
        logger.info("\n🤖 Etapa 4: Configuração do Assistente Médico Generalista")
        assistant = MedicalAssistant(model, tokenizer)
        
        # Etapa 5: Workflow com LangGraph
        logger.info("\n🔄 Etapa 5: Configuração do Workflow Médico")
        workflow = MedicalWorkflow(assistant)
        
        # Modo interativo
        logger.info("\n" + "="*60)
        logger.info("Assistente Médico Generalista pronto! Digite 'sair' para encerrar.")
        logger.info("="*60 + "\n")
        
        while True:
            user_input = input("\n👤 Você: ").strip()
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                logger.info("Encerrando assistente...")
                break
            
            if not user_input:
                continue
            
            # Processa através do workflow
            response = workflow.process(user_input)
            print(f"\n🏥 Assistente: {response}")
        
        logger.info("Sessão encerrada com sucesso!")
        
    except KeyboardInterrupt:
        logger.info("\nOperação interrompida pelo usuário.")
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
        raise


#!/usr/bin/env python3
"""
Tech Challenge Fase 3 - Assistente Virtual Médico

Script principal para execução do assistente médico.
Integra fine-tuning de LLM com LangChain e LangGraph.
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
import click

from src.utils.logging_config import setup_logging, get_logger

# Carrega variáveis de ambiente
load_dotenv()

# Configura logging
logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🏥 Assistente Virtual Médico - Tech Challenge Fase 3"""
    pass


@cli.command()
@click.option("--data-path", default="./data/raw", help="Caminho dos dados brutos")
def prepare_data(data_path: str):
    """Prepara e anonimiza os dados para treinamento."""
    logger.info(f"Preparando dados de: {data_path}")
    from src.fine_tuning.data_preparation import DataPreparation
    
    prep = DataPreparation(data_path)
    prep.run()
    logger.info("Dados preparados com sucesso!")


@cli.command()
@click.option("--model-name", default=None, help="Nome do modelo base")
@click.option("--epochs", default=3, help="Número de épocas de treinamento")
def train(model_name: str, epochs: int):
    """Executa o fine-tuning do modelo."""
    model = model_name or os.getenv("BASE_MODEL", "meta-llama/Llama-2-7b-hf")
    logger.info(f"Iniciando treinamento com modelo: {model}")
    
    from src.fine_tuning.training import ModelTrainer
    
    trainer = ModelTrainer(model_name=model, num_epochs=epochs)
    trainer.train()
    logger.info("Treinamento concluído!")


@cli.command()
def evaluate():
    """Avalia o modelo treinado."""
    logger.info("Avaliando modelo...")
    from src.fine_tuning.evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate()
    logger.info(f"Métricas: {metrics}")


@cli.command()
@click.option("--interactive", "-i", is_flag=True, help="Modo interativo")
def chat(interactive: bool):
    """Inicia o assistente médico."""
    logger.info("Iniciando assistente médico...")
    
    from src.langchain_integration.assistant import MedicalAssistant
    
    assistant = MedicalAssistant()
    
    if interactive:
        print("\n🏥 Assistente Virtual Médico")
        print("=" * 40)
        print("Digite 'sair' para encerrar.\n")
        
        while True:
            try:
                user_input = input("Você: ").strip()
                if user_input.lower() in ["sair", "exit", "quit"]:
                    print("\nAté logo! Cuide-se. 👋")
                    break
                
                if user_input:
                    response = assistant.respond(user_input)
                    print(f"\nAssistente: {response}\n")
            except KeyboardInterrupt:
                print("\n\nEncerrando...")
                break
    else:
        # Modo não-interativo (para testes)
        print("Assistente inicializado. Use --interactive para modo de chat.")


@cli.command()
def workflow():
    """Executa o workflow médico com LangGraph."""
    logger.info("Executando workflow médico...")
    
    from src.langgraph_flows.medical_workflow import MedicalWorkflow
    
    workflow = MedicalWorkflow()
    workflow.run()


if __name__ == "__main__":
    setup_logging()
    cli()

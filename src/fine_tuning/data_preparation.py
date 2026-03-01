"""
Módulo de Preparação e Anonimização de Dados
============================================

Responsável por:
- Carregar dados médicos
- Anonimizar informações sensíveis (LGPD)
- Preparar dataset para fine-tuning
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict

from src.utils.logging_config import get_logger
from src.utils.validators import DataValidator

logger = get_logger(__name__)


class DataPreparation:
    """
    Classe para preparação e anonimização de dados médicos.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Inicializa o preparador de dados.
        
        Args:
            data_path: Caminho para os dados. Se None, usa DATA_PATH do .env
        """
        self.data_path = Path(data_path or os.getenv("DATA_PATH", "./data"))
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.validator = DataValidator()
        
        # Garante que os diretórios existam
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataPreparation inicializado. Data path: {self.data_path}")
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonimiza informações sensíveis no texto.
        
        Args:
            text: Texto a ser anonimizado
            
        Returns:
            Texto anonimizado
        """
        if not text:
            return text
        
        # Padrões para anonimização
        patterns = {
            # CPF: XXX.XXX.XXX-XX
            r'\d{3}\.\d{3}\.\d{3}-\d{2}': '[CPF_ANONIMIZADO]',
            # RG: XX.XXX.XXX-X
            r'\d{2}\.\d{3}\.\d{3}-[0-9X]': '[RG_ANONIMIZADO]',
            # Telefone: (XX) XXXXX-XXXX ou (XX) XXXX-XXXX
            r'\(\d{2}\)\s*\d{4,5}-?\d{4}': '[TELEFONE_ANONIMIZADO]',
            # Email
            r'[\w\.-]+@[\w\.-]+\.\w+': '[EMAIL_ANONIMIZADO]',
            # Nomes próprios (simplificado - em produção usar NER)
            r'\b[A-Z][a-záéíóúàèìòùâêîôûãõ]+\s+[A-Z][a-záéíóúàèìòùâêîôûãõ]+\b': '[NOME_ANONIMIZADO]',
            # Datas de nascimento
            r'\d{2}/\d{2}/\d{4}': '[DATA_ANONIMIZADA]',
            # Endereços (simplificado)
            r'Rua\s+[\w\s]+,\s*\d+': '[ENDERECO_ANONIMIZADO]',
        }
        
        anonymized = text
        for pattern, replacement in patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized
    
    def load_raw_data(self, filename: str = "diabetes.csv") -> pd.DataFrame:
        """
        Carrega dados brutos do arquivo.
        
        Args:
            filename: Nome do arquivo de dados
            
        Returns:
            DataFrame com os dados carregados
        """
        file_path = self.raw_path / filename
        
        if not file_path.exists():
            logger.warning(f"Arquivo não encontrado: {file_path}")
            logger.info("Criando dataset de exemplo...")
            return self._create_sample_dataset()
        
        logger.info(f"Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
        
        return df
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """
        Cria um dataset de exemplo para demonstração.
        
        Returns:
            DataFrame com dados de exemplo
        """
        # Exemplos de perguntas e respostas médicas
        samples = [
            {
                "instruction": "O que é diabetes tipo 2?",
                "input": "",
                "output": "Diabetes tipo 2 é uma condição crônica que afeta a forma como o corpo metaboliza a glicose. Ocorre quando o corpo desenvolve resistência à insulina ou não produz insulina suficiente. Os fatores de risco incluem obesidade, sedentarismo e histórico familiar."
            },
            {
                "instruction": "Quais são os sintomas da diabetes?",
                "input": "",
                "output": "Os principais sintomas da diabetes incluem: sede excessiva (polidipsia), urinação frequente (poliúria), fome excessiva (polifagia), perda de peso inexplicada, fadiga, visão turva, cicatrização lenta de feridas e formigamento nas extremidades."
            },
            {
                "instruction": "Como controlar a glicemia?",
                "input": "Paciente com diabetes tipo 2 recém-diagnosticado",
                "output": "Para controlar a glicemia, recomenda-se: 1) Alimentação equilibrada com baixo índice glicêmico; 2) Exercícios físicos regulares (30 min/dia); 3) Monitoramento regular da glicose; 4) Medicação conforme prescrição médica; 5) Controle do peso; 6) Evitar álcool e tabaco."
            },
            {
                "instruction": "Qual a diferença entre diabetes tipo 1 e tipo 2?",
                "input": "",
                "output": "Diabetes tipo 1 é uma doença autoimune onde o sistema imunológico ataca as células beta do pâncreas, geralmente diagnosticada na infância/adolescência. Diabetes tipo 2 é caracterizada por resistência à insulina, mais comum em adultos e associada ao estilo de vida."
            },
            {
                "instruction": "O que é hipoglicemia?",
                "input": "",
                "output": "Hipoglicemia é quando os níveis de açúcar no sangue ficam muito baixos (abaixo de 70 mg/dL). Sintomas incluem tremores, sudorese, confusão mental, tontura e fome intensa. Tratamento imediato: consumir 15g de carboidrato de ação rápida."
            },
        ]
        
        df = pd.DataFrame(samples)
        
        # Salva o dataset de exemplo
        sample_path = self.processed_path / "sample_medical_qa.csv"
        df.to_csv(sample_path, index=False)
        logger.info(f"Dataset de exemplo salvo em: {sample_path}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> Dataset:
        """
        Prepara os dados para o formato de treinamento.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Dataset do Hugging Face pronto para treinamento
        """
        # Formata as instruções no formato de chat
        def format_instruction(row):
            if row.get('input', ''):
                text = f"### Instrução:\n{row['instruction']}\n\n### Contexto:\n{row['input']}\n\n### Resposta:\n{row['output']}"
            else:
                text = f"### Instrução:\n{row['instruction']}\n\n### Resposta:\n{row['output']}"
            return text
        
        df['text'] = df.apply(format_instruction, axis=1)
        
        # Anonimiza os textos
        df['text'] = df['text'].apply(self.anonymize_text)
        
        # Converte para Dataset
        dataset = Dataset.from_pandas(df[['text']])
        
        logger.info(f"Dataset preparado para treinamento: {len(dataset)} exemplos")
        
        return dataset
    
    def prepare_dataset(self) -> Dataset:
        """
        Pipeline completo de preparação de dados.
        
        Returns:
            Dataset pronto para fine-tuning
        """
        logger.info("Iniciando preparação do dataset...")
        
        # Carrega dados
        df = self.load_raw_data()
        
        # Valida dados
        if not self.validator.validate_dataframe(df):
            logger.warning("Validação falhou, usando dataset de exemplo")
            df = self._create_sample_dataset()
        
        # Prepara para treinamento
        dataset = self.prepare_for_training(df)
        
        logger.info("Preparação do dataset concluída!")
        
        return dataset


if __name__ == "__main__":
    # Teste do módulo
    prep = DataPreparation()
    dataset = prep.prepare_dataset()
    print(f"\nExemplo de dado preparado:\n{dataset[0]['text'][:500]}...")

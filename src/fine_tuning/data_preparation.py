"""
Módulo de Preparação e Anonimização de Dados

Responsável por:
- Carregar dados médicos brutos
- Anonimizar informações sensíveis (LGPD)
- Preparar dataset para fine-tuning
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from faker import Faker
from datasets import Dataset

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataAnonymizer:
    """Anonimizador de dados médicos sensíveis."""
    
    def __init__(self, locale: str = "pt_BR"):
        self.fake = Faker(locale)
        self._name_map: Dict[str, str] = {}
        self._cpf_map: Dict[str, str] = {}
        
    def anonymize_name(self, name: str) -> str:
        """Substitui nome por nome fictício."""
        if name not in self._name_map:
            self._name_map[name] = self.fake.name()
        return self._name_map[name]
    
    def anonymize_cpf(self, cpf: str) -> str:
        """Substitui CPF por CPF fictício."""
        if cpf not in self._cpf_map:
            self._cpf_map[cpf] = self.fake.cpf()
        return self._cpf_map[cpf]
    
    def anonymize_text(self, text: str) -> str:
        """Anonimiza texto completo."""
        # Padrões para identificar informações sensíveis
        patterns = {
            r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b': self._replace_cpf,  # CPF
            r'\b\d{2}/\d{2}/\d{4}\b': self._replace_date,  # Datas
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': self._replace_name,  # Nomes
        }
        
        result = text
        for pattern, replacer in patterns.items():
            result = re.sub(pattern, replacer, result)
        
        return result
    
    def _replace_cpf(self, match) -> str:
        return self.anonymize_cpf(match.group())
    
    def _replace_date(self, match) -> str:
        return self.fake.date()
    
    def _replace_name(self, match) -> str:
        return self.anonymize_name(match.group())


class DataPreparation:
    """Pipeline de preparação de dados para fine-tuning."""
    
    def __init__(self, data_path: str = "./data/raw"):
        self.data_path = Path(data_path)
        self.output_path = Path("./data/processed")
        self.anonymizer = DataAnonymizer()
        
        # Cria diretório de saída se não existir
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Carrega dados brutos de diferentes formatos."""
        logger.info(f"Carregando dados de {self.data_path}")
        
        all_data = []
        
        # Procura por arquivos CSV
        for csv_file in self.data_path.glob("*.csv"):
            logger.info(f"Lendo {csv_file.name}")
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        # Procura por arquivos JSON
        for json_file in self.data_path.glob("*.json"):
            logger.info(f"Lendo {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            all_data.append(df)
        
        if not all_data:
            logger.warning("Nenhum dado encontrado!")
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def anonymize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica anonimização em todas as colunas de texto."""
        logger.info("Anonimizando dados sensíveis...")
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            df[col] = df[col].apply(
                lambda x: self.anonymizer.anonymize_text(str(x)) if pd.notna(x) else x
            )
        
        logger.info(f"Anonimização concluída para {len(text_columns)} colunas")
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> Dataset:
        """Converte dados para formato de treinamento."""
        logger.info("Preparando dataset para fine-tuning...")
        
        # Formato esperado: prompt/response ou instruction/output
        training_data = []
        
        # TODO: Adaptar conforme estrutura real dos dados
        for _, row in df.iterrows():
            training_data.append({
                "instruction": row.get("pergunta", row.get("instruction", "")),
                "input": row.get("contexto", row.get("input", "")),
                "output": row.get("resposta", row.get("output", ""))
            })
        
        dataset = Dataset.from_list(training_data)
        logger.info(f"Dataset criado com {len(dataset)} exemplos")
        return dataset
    
    def save_processed_data(self, dataset: Dataset, name: str = "medical_dataset"):
        """Salva dataset processado."""
        output_file = self.output_path / f"{name}.json"
        
        dataset.to_json(str(output_file))
        logger.info(f"Dataset salvo em {output_file}")
    
    def run(self):
        """Executa pipeline completo de preparação."""
        logger.info("=" * 50)
        logger.info("Iniciando preparação de dados")
        logger.info("=" * 50)
        
        # 1. Carregar dados
        df = self.load_raw_data()
        if df.empty:
            logger.error("Nenhum dado para processar!")
            return
        
        # 2. Anonimizar
        df_anon = self.anonymize_data(df)
        
        # 3. Preparar para treinamento
        dataset = self.prepare_for_training(df_anon)
        
        # 4. Salvar
        self.save_processed_data(dataset)
        
        logger.info("Pipeline de preparação concluído!")
        return dataset


if __name__ == "__main__":
    prep = DataPreparation()
    prep.run()

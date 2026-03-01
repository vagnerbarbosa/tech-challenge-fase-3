"""
Validadores de Segurança
========================

Implementa validações de entrada e dados para segurança.
"""

import re
from typing import Tuple, Any
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class InputValidator:
    """
    Validador de entradas do usuário.
    """
    
    def __init__(self):
        """
        Inicializa o validador.
        """
        # Tamanhos limites
        self.max_input_length = 2000
        self.min_input_length = 2
        
        # Padrões suspeitos (injection, XSS, etc.)
        self.suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'\{\{.*\}\}',
            r'\$\{.*\}',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
        ]
        
        logger.info("InputValidator inicializado")
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Valida uma consulta do usuário.
        
        Args:
            query: Texto da consulta
            
        Returns:
            Tupla (é_válido, mensagem)
        """
        # Verifica se é string
        if not isinstance(query, str):
            return False, "A entrada deve ser um texto."
        
        # Remove espaços extras
        query = query.strip()
        
        # Verifica tamanho mínimo
        if len(query) < self.min_input_length:
            return False, "Por favor, digite uma pergunta mais completa."
        
        # Verifica tamanho máximo
        if len(query) > self.max_input_length:
            return False, f"Sua mensagem é muito longa. Limite: {self.max_input_length} caracteres."
        
        # Verifica padrões suspeitos
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Padrão suspeito detectado: {pattern}")
                return False, "Sua mensagem contém caracteres não permitidos."
        
        return True, "OK"
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitiza o texto de entrada.
        
        Args:
            text: Texto a ser sanitizado
            
        Returns:
            Texto sanitizado
        """
        if not text:
            return ""
        
        # Remove tags HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove caracteres de controle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normaliza espaços
        text = ' '.join(text.split())
        
        return text.strip()


class DataValidator:
    """
    Validador de dados para o dataset.
    """
    
    def __init__(self):
        """
        Inicializa o validador de dados.
        """
        self.required_columns = ["instruction", "output"]
        self.optional_columns = ["input", "text"]
        
        logger.info("DataValidator inicializado")
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Valida um DataFrame de dados.
        
        Args:
            df: DataFrame a ser validado
            
        Returns:
            True se válido
        """
        # Verifica se é DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.error("Entrada não é um DataFrame")
            return False
        
        # Verifica se está vazio
        if df.empty:
            logger.error("DataFrame está vazio")
            return False
        
        # Verifica colunas obrigatórias
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Colunas ausentes: {missing}")
            # Se tem coluna 'text', ainda é válido
            if 'text' not in df.columns:
                return False
        
        # Verifica valores nulos nas colunas obrigatórias
        for col in self.required_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(f"Coluna '{col}' tem {null_count} valores nulos")
        
        logger.info(f"DataFrame validado: {len(df)} linhas, {len(df.columns)} colunas")
        return True
    
    def validate_medical_record(self, record: dict) -> Tuple[bool, str]:
        """
        Valida um registro médico.
        
        Args:
            record: Dicionário com dados médicos
            
        Returns:
            Tupla (é_válido, mensagem)
        """
        required_fields = ["instruction", "output"]
        
        for field in required_fields:
            if field not in record or not record[field]:
                return False, f"Campo obrigatório ausente: {field}"
        
        # Verifica tamanho da instrução
        if len(record["instruction"]) < 5:
            return False, "Instrução muito curta"
        
        # Verifica tamanho da saída
        if len(record["output"]) < 10:
            return False, "Resposta muito curta"
        
        return True, "OK"


if __name__ == "__main__":
    # Testes
    input_val = InputValidator()
    
    # Teste de validação
    print(input_val.validate_query("O que é diabetes?"))
    print(input_val.validate_query("<script>alert('xss')</script>"))
    print(input_val.validate_query(""))
    
    # Teste de sanitização
    print(input_val.sanitize_input("  <b>Texto</b>   com   espaços  "))

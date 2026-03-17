"""
Configuração de Logging
=======================

Configura o sistema de logging para o projeto.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: Optional[str] = None,
    log_path: Optional[str] = None,
) -> None:
    """
    Configura o sistema de logging.
    
    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_path: Caminho para salvar logs
    """
    # Configurações
    level = getattr(logging, log_level or os.getenv("LOG_LEVEL", "INFO").upper())
    log_dir = Path(log_path or os.getenv("LOG_PATH", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Arquivo de log com data
    log_file = log_dir / f"medical_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Formato do log
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configura handlers
    handlers = [
        logging.StreamHandler(),  # Console
        logging.FileHandler(log_file, encoding="utf-8"),  # Arquivo
    ]
    
    # Configura logging
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
    
    # Reduz verbosidade de bibliotecas externas
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado. Arquivo: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger com o nome especificado.
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


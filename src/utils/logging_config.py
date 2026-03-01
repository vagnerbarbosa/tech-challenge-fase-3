"""
Configuração de Logging

Implementa sistema de logs estruturado para o assistente médico.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger

# Remove handler padrão
logger.remove()


def setup_logging(
    log_level: str = None,
    log_dir: str = "./logs",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> None:
    """
    Configura o sistema de logging.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        log_dir: Diretório para arquivos de log
        log_to_file: Se deve salvar logs em arquivo
        log_to_console: Se deve mostrar logs no console
    """
    # Pega nível do ambiente ou usa INFO como padrão
    level = log_level or os.getenv("LOG_LEVEL", "INFO")
    
    # Formato de log
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Formato simples para arquivo (sem cores)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Log no console
    if log_to_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True
        )
    
    # Log em arquivo
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Log geral
        logger.add(
            log_path / "app_{time:YYYY-MM-DD}.log",
            format=file_format,
            level=level,
            rotation="00:00",  # Rotação à meia-noite
            retention="30 days",  # Mantém 30 dias
            compression="zip"
        )
        
        # Log de erros separado
        logger.add(
            log_path / "errors_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="ERROR",
            rotation="00:00",
            retention="90 days",
            compression="zip"
        )
        
        # Log de auditoria médica (importante para compliance)
        logger.add(
            log_path / "audit_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation="00:00",
            retention="365 days",  # Retenção longa para auditoria
            compression="zip",
            filter=lambda record: "audit" in record["extra"]
        )


def get_logger(name: str) -> "logger":
    """
    Retorna logger configurado para um módulo.
    
    Args:
        name: Nome do módulo (geralmente __name__)
        
    Returns:
        Logger configurado
    """
    return logger.bind(name=name)


def log_audit(action: str, details: dict, user_id: str = "system") -> None:
    """
    Registra ação de auditoria.
    
    Args:
        action: Tipo de ação realizada
        details: Detalhes da ação
        user_id: ID do usuário que realizou a ação
    """
    logger.bind(audit=True).info(
        f"AUDIT | User: {user_id} | Action: {action} | Details: {details}"
    )


# Exemplo de uso
if __name__ == "__main__":
    setup_logging(log_level="DEBUG")
    
    test_logger = get_logger(__name__)
    test_logger.debug("Mensagem de debug")
    test_logger.info("Mensagem de info")
    test_logger.warning("Mensagem de aviso")
    test_logger.error("Mensagem de erro")
    
    log_audit("login", {"ip": "192.168.1.1"}, "user123")

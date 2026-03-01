"""
Base Scraper - Classe Base para Web Scraping
=============================================

Fornece funcionalidades comuns para todos os scrapers:
- Gerenciamento de sessões HTTP
- Headers e delays para boas práticas
- Tratamento de erros
- Salvamento de dados em CSV
"""

import os
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseScraper(ABC):
    """
    Classe base abstrata para scrapers de dados médicos.
    Implementa padrões comuns e boas práticas de web scraping.
    """
    
    # Headers para simular navegador real
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    # Delays para respeitar os servidores
    MIN_DELAY = 1.0  # segundos
    MAX_DELAY = 3.0  # segundos
    
    def __init__(
        self,
        output_path: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Inicializa o scraper base.
        
        Args:
            output_path: Caminho para salvar CSVs. Default: ./data/processed
            timeout: Timeout para requisições em segundos
            max_retries: Número máximo de tentativas por requisição
        """
        self.output_path = Path(output_path or os.getenv("DATA_PATH", "./data")) / "processed"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configura sessão HTTP
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        
        logger.info(f"{self.__class__.__name__} inicializado. Output: {self.output_path}")
    
    def _random_delay(self) -> None:
        """Aplica delay aleatório entre requisições para respeitar o servidor."""
        delay = random.uniform(self.MIN_DELAY, self.MAX_DELAY)
        time.sleep(delay)
    
    def _make_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> Optional[requests.Response]:
        """
        Faz requisição HTTP com retry e tratamento de erros.
        
        Args:
            url: URL para requisição
            method: Método HTTP (GET, POST, etc.)
            **kwargs: Argumentos adicionais para requests
            
        Returns:
            Response object ou None em caso de falha
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Requisição [{attempt}/{self.max_retries}]: {url}")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                
                self._random_delay()  # Respeita o servidor
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout na tentativa {attempt} para {url}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Erro HTTP {e.response.status_code} para {url}")
                if e.response.status_code == 404:
                    return None  # Página não existe
            except requests.exceptions.RequestException as e:
                logger.warning(f"Erro na requisição para {url}: {e}")
            
            if attempt < self.max_retries:
                wait_time = attempt * 2  # Backoff exponencial simplificado
                logger.info(f"Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
        
        logger.error(f"Falha após {self.max_retries} tentativas para {url}")
        return None
    
    def _parse_html(self, response: requests.Response) -> BeautifulSoup:
        """
        Parseia resposta HTML.
        
        Args:
            response: Response object
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(response.content, "html.parser")
    
    def _save_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        columns: Optional[List[str]] = None,
    ) -> Path:
        """
        Salva dados em arquivo CSV.
        
        Args:
            data: Lista de dicionários com os dados
            filename: Nome do arquivo (sem extensão)
            columns: Ordem das colunas (opcional)
            
        Returns:
            Path do arquivo salvo
        """
        if not data:
            logger.warning(f"Nenhum dado para salvar em {filename}")
            return None
        
        df = pd.DataFrame(data)
        
        if columns:
            df = df[columns]
        
        filepath = self.output_path / f"{filename}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        
        logger.info(f"Dados salvos em {filepath} ({len(data)} registros)")
        return filepath
    
    @abstractmethod
    def scrape(self) -> List[Dict[str, Any]]:
        """
        Método abstrato para executar o scraping.
        Deve ser implementado pelas subclasses.
        
        Returns:
            Lista de dicionários com os dados coletados
        """
        pass
    
    @abstractmethod
    def run(self) -> Path:
        """
        Executa o scraping completo e salva os dados.
        Deve ser implementado pelas subclasses.
        
        Returns:
            Path do arquivo CSV gerado
        """
        pass
    
    def close(self) -> None:
        """Fecha a sessão HTTP."""
        self.session.close()
        logger.debug(f"{self.__class__.__name__} sessão fechada.")

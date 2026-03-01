"""
Módulo de Web Scraping para Dados Médicos
==========================================

Contém scrapers para coletar dados de:
- HCPA: Protocolos médicos
- TelessaúdeRS: Perguntas frequentes médicas
- RadReport: Modelos de laudos radiológicos
"""

from .base_scraper import BaseScraper
from .hcpa_scraper import HCPAScraper
from .telessaude_scraper import TelessaudeScraper
from .radreport_scraper import RadReportScraper

__all__ = [
    "BaseScraper",
    "HCPAScraper",
    "TelessaudeScraper",
    "RadReportScraper",
]

#!/usr/bin/env python3
"""
Script Principal para Execução dos Web Scrapers
================================================

Executa todos os scrapers de dados médicos e gera os CSVs:
- protocolos_medicos.csv (HCPA)
- perguntas_frequentes.csv (TelessaúdeRS)
- modelos_laudos.csv (RadReport)

Uso:
    python -m src.scraping.run_scrapers
    
Ou diretamente:
    python src/scraping/run_scrapers.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_config import setup_logging, get_logger
from src.scraping.hcpa_scraper import HCPAScraper
from src.scraping.telessaude_scraper import TelessaudeScraper
from src.scraping.radreport_scraper import RadReportScraper

logger = get_logger(__name__)


def run_all_scrapers() -> dict:
    """
    Executa todos os scrapers e retorna os caminhos dos arquivos gerados.
    
    Returns:
        Dicionário com os caminhos dos arquivos gerados
    """
    setup_logging()
    
    logger.info("="*60)
    logger.info("Web Scraping de Dados Médicos")
    logger.info("="*60)
    logger.info(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    errors = []
    
    # 1. Scraper HCPA - Protocolos Médicos
    logger.info("\n" + "-"*50)
    logger.info("🏥 1/3 - Coletando Protocolos do HCPA...")
    logger.info("-"*50)
    try:
        hcpa_scraper = HCPAScraper()
        results['protocolos_medicos'] = hcpa_scraper.run()
        logger.info(f"✅ Protocolos salvos em: {results['protocolos_medicos']}")
    except Exception as e:
        logger.error(f"❌ Erro no scraping HCPA: {e}")
        errors.append(('HCPA', str(e)))
    
    # 2. Scraper TelessaúdeRS - Perguntas Frequentes
    logger.info("\n" + "-"*50)
    logger.info("📞 2/3 - Coletando FAQs do TelessaúdeRS...")
    logger.info("-"*50)
    try:
        telessaude_scraper = TelessaudeScraper()
        results['perguntas_frequentes'] = telessaude_scraper.run()
        logger.info(f"✅ FAQs salvas em: {results['perguntas_frequentes']}")
    except Exception as e:
        logger.error(f"❌ Erro no scraping TelessaúdeRS: {e}")
        errors.append(('TelessaúdeRS', str(e)))
    
    # 3. Scraper RadReport - Modelos de Laudos
    logger.info("\n" + "-"*50)
    logger.info("📋 3/3 - Coletando Modelos de Laudos do RadReport...")
    logger.info("-"*50)
    try:
        radreport_scraper = RadReportScraper()
        results['modelos_laudos'] = radreport_scraper.run()
        logger.info(f"✅ Modelos de laudos salvos em: {results['modelos_laudos']}")
    except Exception as e:
        logger.error(f"❌ Erro no scraping RadReport: {e}")
        errors.append(('RadReport', str(e)))
    
    # Sumário final
    logger.info("\n" + "="*60)
    logger.info("SUMÁRIO DA EXECUÇÃO")
    logger.info("="*60)
    
    logger.info(f"\n📁 Arquivos gerados:")
    for name, path in results.items():
        if path:
            logger.info(f"   - {name}: {path}")
    
    if errors:
        logger.warning(f"\n⚠️ Erros encontrados ({len(errors)}):")
        for source, error in errors:
            logger.warning(f"   - {source}: {error}")
    else:
        logger.info(f"\n✅ Todos os scrapers executados com sucesso!")
    
    logger.info(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    results = run_all_scrapers()
    
    print("\n" + "="*50)
    print("ARQUIVOS GERADOS:")
    print("="*50)
    for name, path in results.items():
        print(f"  {name}: {path}")

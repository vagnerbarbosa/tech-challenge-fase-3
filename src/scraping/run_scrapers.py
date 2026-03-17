#!/usr/bin/env python3
"""
Script Principal para Execução dos Web Scrapers
================================================

Executa todos os scrapers de dados médicos e gera arquivos JSONL em data/raw/:
- protocolos_medicos.jsonl (CONITEC/MS)
- perguntas_frequentes.jsonl (TelessaúdeRS)
- modelos_laudos.jsonl (RadReport)

O formato JSONL é otimizado para fine-tuning, com campos:
- instruction: pergunta/instrução
- input: contexto adicional
- output: resposta esperada
- source: fonte dos dados

Uso:
    python -m src.scraping.run_scrapers
    
Ou diretamente:
    python src/scraping/run_scrapers.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_config import setup_logging, get_logger
from src.scraping.hcpa_scraper import HCPAScraper
from src.scraping.telessaude_scraper import TelessaudeScraper
from src.scraping.radreport_scraper import RadReportScraper

logger = get_logger(__name__)


# ============================================================================
# Configurações de Limite de Itens
# ============================================================================
# Defina o número máximo de itens a coletar para cada scraper.
# Use None para coletar todos os dados disponíveis (sem limite).

SCRAPER_CONFIG = {
    "hcpa": {
        "max_items": 50,        # Limite de protocolos médicos (None = sem limite)
    },
    "telessaude": {
        "max_items": 30,        # Limite de FAQs/perguntas (None = sem limite)
    },
    "radreport": {
        "max_items": 20,        # Limite de modelos de laudos (None = sem limite)
    },
}


def run_all_scrapers(
    max_items_hcpa: Optional[int] = None,
    max_items_telessaude: Optional[int] = None,
    max_items_radreport: Optional[int] = None,
) -> dict:
    """
    Executa todos os scrapers e retorna os caminhos dos arquivos gerados.
    
    Args:
        max_items_hcpa: Limite de protocolos HCPA (None usa config padrão)
        max_items_telessaude: Limite de FAQs TelessaúdeRS (None usa config padrão)
        max_items_radreport: Limite de modelos RadReport (None usa config padrão)
    
    Returns:
        Dicionário com os caminhos dos arquivos gerados
    """
    setup_logging()
    
    # Usa configuração padrão se não especificado
    hcpa_limit = max_items_hcpa if max_items_hcpa is not None else SCRAPER_CONFIG["hcpa"]["max_items"]
    telessaude_limit = max_items_telessaude if max_items_telessaude is not None else SCRAPER_CONFIG["telessaude"]["max_items"]
    radreport_limit = max_items_radreport if max_items_radreport is not None else SCRAPER_CONFIG["radreport"]["max_items"]
    
    logger.info("="*60)
    logger.info("Web Scraping de Dados Médicos")
    logger.info("="*60)
    logger.info(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Limites configurados: HCPA={hcpa_limit or 'sem limite'}, "
                f"TelessaúdeRS={telessaude_limit or 'sem limite'}, "
                f"RadReport={radreport_limit or 'sem limite'}")
    
    results = {}
    errors = []
    
    # 1. Scraper HCPA - Protocolos Médicos
    logger.info("\n" + "-"*50)
    logger.info("🏥 1/3 - Coletando Protocolos do HCPA...")
    logger.info("-"*50)
    try:
        hcpa_scraper = HCPAScraper(max_items=hcpa_limit)
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
        telessaude_scraper = TelessaudeScraper(max_items=telessaude_limit)
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
        radreport_scraper = RadReportScraper(max_items=radreport_limit)
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


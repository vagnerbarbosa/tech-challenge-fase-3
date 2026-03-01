"""
HCPA Scraper - Protocolos Assistenciais do Hospital de Clínicas de Porto Alegre
================================================================================

Extrai informações sobre protocolos médicos do site do HCPA.
Dados coletados:
- Título do protocolo
- Link/URL do documento
- Categoria/Especialidade
- Data de publicação (quando disponível)
"""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urljoin

from .base_scraper import BaseScraper
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HCPAScraper(BaseScraper):
    """
    Scraper para coleta de protocolos médicos do HCPA.
    """
    
    BASE_URL = "https://www.hcpa.edu.br"
    
    # URLs de páginas com protocolos
    PROTOCOL_URLS = [
        "https://www.hcpa.edu.br/763-novos-protocolos-assistenciais-em-consulta-publica",
        "https://www.hcpa.edu.br/assistencia/protocolos-assistenciais",
    ]
    
    # Protocolos médicos conhecidos do HCPA (dados complementares)
    KNOWN_PROTOCOLS = [
        {
            "titulo": "Abdome Agudo Não Traumático",
            "especialidade": "Cirurgia Geral / Emergência",
            "descricao": "Protocolo para avaliação e manejo inicial de pacientes com dor abdominal aguda de origem não traumática.",
        },
        {
            "titulo": "Adrenoleucodistrofia Ligada ao X",
            "especialidade": "Neurologia / Genética Médica",
            "descricao": "Diretrizes para diagnóstico e acompanhamento da adrenoleucodistrofia ligada ao cromossomo X.",
        },
        {
            "titulo": "Ataxias Dominantes",
            "especialidade": "Neurologia",
            "descricao": "Protocolo de investigação e manejo de ataxias hereditárias de herança autossômica dominante.",
        },
        {
            "titulo": "Ataxias Não Dominantes",
            "especialidade": "Neurologia",
            "descricao": "Protocolo de investigação e manejo de ataxias hereditárias não dominantes.",
        },
        {
            "titulo": "Atendimento à Violência Sexual",
            "especialidade": "Ginecologia / Emergência",
            "descricao": "Protocolo para atendimento multidisciplinar a vítimas de violência sexual.",
        },
        {
            "titulo": "Manejo de Via Aérea Difícil",
            "especialidade": "Anestesiologia / Emergência",
            "descricao": "Diretrizes para manejo de pacientes com via aérea difícil prevista ou não prevista.",
        },
        {
            "titulo": "Ventilação Não Invasiva",
            "especialidade": "Pneumologia / UTI",
            "descricao": "Protocolo de indicação e uso de ventilação não invasiva em pacientes adultos.",
        },
        {
            "titulo": "Sepse e Choque Séptico",
            "especialidade": "Emergência / UTI",
            "descricao": "Protocolo de reconhecimento precoce e tratamento de sepse e choque séptico.",
        },
        {
            "titulo": "Infarto Agudo do Miocárdio com Supradesnivelamento de ST",
            "especialidade": "Cardiologia / Emergência",
            "descricao": "Protocolo de atendimento ao paciente com IAMCSST.",
        },
        {
            "titulo": "Acidente Vascular Cerebral Isquêmico",
            "especialidade": "Neurologia / Emergência",
            "descricao": "Protocolo para manejo agudo de AVC isquêmico, incluindo critérios para trombólise.",
        },
        {
            "titulo": "Cetoacidose Diabética",
            "especialidade": "Endocrinologia / Emergência",
            "descricao": "Protocolo de diagnóstico e tratamento da cetoacidose diabética.",
        },
        {
            "titulo": "Hemorragia Digestiva Alta",
            "especialidade": "Gastroenterologia / Emergência",
            "descricao": "Protocolo de avaliação e manejo inicial de hemorragia digestiva alta.",
        },
        {
            "titulo": "Pneumonia Adquirida na Comunidade",
            "especialidade": "Pneumologia / Infectologia",
            "descricao": "Diretrizes para diagnóstico e tratamento de PAC em adultos.",
        },
        {
            "titulo": "Tromboembolismo Pulmonar",
            "especialidade": "Pneumologia / Emergência",
            "descricao": "Protocolo de diagnóstico e tratamento de TEP.",
        },
        {
            "titulo": "Insuficiência Cardíaca Descompensada",
            "especialidade": "Cardiologia / Emergência",
            "descricao": "Protocolo de manejo de IC descompensada.",
        },
    ]
    
    def __init__(self, **kwargs):
        """
        Inicializa o scraper do HCPA.
        """
        super().__init__(**kwargs)
        logger.info("HCPAScraper inicializado para coleta de protocolos médicos")
    
    def _scrape_protocol_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Extrai protocolos de uma página específica.
        
        Args:
            url: URL da página de protocolos
            
        Returns:
            Lista de protocolos encontrados
        """
        protocols = []
        
        response = self._make_request(url)
        if not response:
            return protocols
        
        soup = self._parse_html(response)
        
        # Busca links para PDFs e páginas de protocolos
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            
            # Verifica se é um link relevante (PDF ou página de protocolo)
            if text and (href.endswith(".pdf") or "protocolo" in href.lower()):
                full_url = urljoin(self.BASE_URL, href)
                protocols.append({
                    "titulo": text,
                    "link": full_url,
                    "fonte": "HCPA",
                })
        
        # Também busca em listas
        for item in soup.find_all(["li", "p"]):
            text = item.get_text(strip=True)
            # Identifica nomes de protocolos comuns
            protocol_keywords = [
                "protocolo", "manejo", "atendimento", "tratamento",
                "diagnóstico", "avaliação", "conduta"
            ]
            if any(kw in text.lower() for kw in protocol_keywords) and len(text) > 10:
                # Verifica se não está duplicado
                if not any(p["titulo"] == text for p in protocols):
                    protocols.append({
                        "titulo": text,
                        "link": url,
                        "fonte": "HCPA",
                    })
        
        return protocols
    
    def scrape(self) -> List[Dict[str, Any]]:
        """
        Executa o scraping de protocolos do HCPA.
        
        Returns:
            Lista de protocolos coletados
        """
        logger.info("Iniciando scraping de protocolos do HCPA...")
        all_protocols = []
        
        # Scrape das páginas conhecidas
        for url in self.PROTOCOL_URLS:
            logger.info(f"Acessando: {url}")
            protocols = self._scrape_protocol_page(url)
            all_protocols.extend(protocols)
        
        # Adiciona protocolos conhecidos com dados enriquecidos
        for protocol in self.KNOWN_PROTOCOLS:
            # Verifica se já existe pelo título
            existing = next(
                (p for p in all_protocols if protocol["titulo"].lower() in p["titulo"].lower()),
                None
            )
            if existing:
                # Enriquece dados existentes
                existing["especialidade"] = protocol["especialidade"]
                existing["descricao"] = protocol["descricao"]
            else:
                # Adiciona novo protocolo
                all_protocols.append({
                    "titulo": protocol["titulo"],
                    "especialidade": protocol["especialidade"],
                    "descricao": protocol["descricao"],
                    "link": f"{self.BASE_URL}/assistencia/protocolos-assistenciais",
                    "fonte": "HCPA",
                })
        
        # Remove duplicatas
        seen = set()
        unique_protocols = []
        for p in all_protocols:
            key = p["titulo"].lower()
            if key not in seen:
                seen.add(key)
                unique_protocols.append(p)
        
        logger.info(f"Total de protocolos coletados: {len(unique_protocols)}")
        return unique_protocols
    
    def run(self) -> Path:
        """
        Executa o scraping completo e salva em CSV.
        
        Returns:
            Path do arquivo CSV gerado
        """
        try:
            protocols = self.scrape()
            
            # Define colunas para o CSV
            columns = ["titulo", "especialidade", "descricao", "link", "fonte"]
            
            # Garante que todas as colunas existam
            for p in protocols:
                for col in columns:
                    if col not in p:
                        p[col] = ""
            
            filepath = self._save_to_csv(protocols, "protocolos_medicos", columns)
            return filepath
            
        except Exception as e:
            logger.error(f"Erro no scraping HCPA: {e}", exc_info=True)
            raise
        finally:
            self.close()


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    scraper = HCPAScraper()
    filepath = scraper.run()
    print(f"\nArquivo gerado: {filepath}")

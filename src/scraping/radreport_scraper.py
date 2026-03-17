"""
RadReport Scraper - Modelos de Laudos Radiológicos
===================================================

Extrai templates de laudos radiológicos do RadReport (RSNA):
- Nomes dos procedimentos
- Modalidades de imagem (CT, MRI, X-ray, etc.)
- Estrutura dos laudos
- Especialidades associadas
"""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urljoin

from .base_scraper import BaseScraper
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RadReportScraper(BaseScraper):
    """
    Scraper para coleta de modelos de laudos do RadReport.
    """
    
    BASE_URL = "https://radreport.org"
    
    # URLs de páginas com templates
    TEMPLATE_URLS = [
        "https://radreport.org/home/50",
        "https://radreport.org/templates",
    ]
    
    # Templates de laudos radiológicos estruturados
    REPORT_TEMPLATES = [
        {
            "nome": "TC de Crânio sem Contraste",
            "modalidade": "TC (Tomografia Computadorizada)",
            "indicacoes": "AVC, TCE, cefaleia aguda, alteração do nível de consciência",
            "estrutura": """TÉCNICA: TC de crânio sem contraste intravenoso.

ACHADOS:
Parênquima cerebral: [descrever atenuação, lesões, sinais de edema]
Sistema ventricular: [descrever tamanho e simetria]
Cisternas: [descrever perviedade]
Linha média: [descrever se centrada ou desviada]
Estruturas ósseas: [descrever integridade]
Seios paranasais e mastoides: [descrever aeração]

IMPRESSÃO:
[Resumo dos achados principais e correlação clínica]""",
            "especialidade": "Neurorradiologia",
        },
        {
            "nome": "RM de Coluna Lombar",
            "modalidade": "RM (Ressonância Magnética)",
            "indicacoes": "Lombalgia crônica, ciatalgia, suspeita de hérnia discal",
            "estrutura": """TÉCNICA: RM da coluna lombar com sequências T1, T2, STIR sagitais e axiais.

ACHADOS:
Alinhamento vertebral: [descrever lordose e alinhamento]
Corpos vertebrais: [descrever altura, sinal medular]
Discos intervertebrais:
  L1-L2: [descrever altura, hidratação, protrusões]
  L2-L3: [descrever]
  L3-L4: [descrever]
  L4-L5: [descrever]
  L5-S1: [descrever]
Canal vertebral: [descrever calibre]
Forames neurais: [descrever perviedade]
Cone medular: [descrever posição e sinal]
Partes moles: [descrever]

IMPRESSÃO:
[Resumo dos achados]""",
            "especialidade": "Neurorradiologia",
        },
        {
            "nome": "TC de Tórax com Contraste",
            "modalidade": "TC (Tomografia Computadorizada)",
            "indicacoes": "Estadiamento oncológico, investigação de nódulo pulmonar, pneumonia complicada",
            "estrutura": """TÉCNICA: TC de tórax com contraste iodado intravenoso.

ACHADOS:
Pulmões:
  Parênquima: [descrever padrões, nódulos, consolidações]
  Pleura: [descrever espessamento, derrame]
Mediastino:
  Linfonodos: [descrever tamanho e localização]
  Grandes vasos: [descrever calibre e perviedade]
Coração e pericárdio: [descrever tamanho e contornos]
Parede torácica: [descrever partes moles e estruturas ósseas]
Porção superior do abdome: [descrever achados incidentais]

IMPRESSÃO:
[Resumo dos achados com correlação clínica]""",
            "especialidade": "Radiologia Torácica",
        },
        {
            "nome": "TC de Abdome Total",
            "modalidade": "TC (Tomografia Computadorizada)",
            "indicacoes": "Dor abdominal aguda, estadiamento oncológico, trauma abdominal",
            "estrutura": """TÉCNICA: TC de abdome e pelve com contraste intravenoso e oral.

ACHADOS:
Fígado: [descrever tamanho, contornos, lesões focais]
Vesícula e vias biliares: [descrever paredes, conteúdo, calibre]
Pâncreas: [descrever tamanho, contornos, lesões]
Baço: [descrever tamanho e homogeneidade]
Adrenais: [descrever]
Rins e vias urinárias: [descrever tamanho, parênquima, sistema coletor]
Retroperitoneu: [descrever linfonodos, vasos]
Trato gastrointestinal: [descrever espessamento, distensão]
Pelve: [descrever bexiga, útero/próstata, ovários/vesículas seminais]
Parede abdominal: [descrever hérnias, coleções]

IMPRESSÃO:
[Resumo dos achados]""",
            "especialidade": "Radiologia Abdominal",
        },
        {
            "nome": "Radiografia de Tórax PA e Perfil",
            "modalidade": "Radiografia (Raio-X)",
            "indicacoes": "Avaliação pulmonar, cardiomegalia, derrame pleural",
            "estrutura": """TÉCNICA: Radiografia de tórax em PA e perfil.

ACHADOS:
Pulmões: [descrever transparência, padrões intersticiais/alveolares]
Área cardíaca: [descrever tamanho - ICT]
Hilos pulmonares: [descrever tamanho e posição]
Mediastino: [descrever contornos]
Seios costofrênicos: [descrever se livres ou velados]
Espaços pleurais: [descrever]
Arcabouço ósseo: [descrever]
Partes moles: [descrever]

IMPRESSÃO:
[Resumo dos achados]""",
            "especialidade": "Radiologia Geral",
        },
        {
            "nome": "RM de Joelho",
            "modalidade": "RM (Ressonância Magnética)",
            "indicacoes": "Lesão meniscal, lesão ligamentar, dor crônica no joelho",
            "estrutura": """TÉCNICA: RM do joelho com sequências T1, T2 FS, DP sagitais, coronais e axiais.

ACHADOS:
Meniscos:
  Menisco medial: [descrever morfologia e sinal]
  Menisco lateral: [descrever morfologia e sinal]
Ligamentos:
  LCA: [descrever integridade, espessura, sinal]
  LCP: [descrever integridade]
  Ligamento colateral medial: [descrever]
  Ligamento colateral lateral: [descrever]
Cartilagem articular: [descrever espessura e sinal]
Osso subcondral: [descrever edema, lesões]
Tendões: [descrever patelar, quadricipital, poplíteo]
Derrame articular: [quantificar se presente]
Cisto de Baker: [descrever se presente]

IMPRESSÃO:
[Resumo dos achados]""",
            "especialidade": "Radiologia Musculoesquelética",
        },
        {
            "nome": "RM de Mama Bilateral",
            "modalidade": "RM (Ressonância Magnética)",
            "indicacoes": "Rastreamento alto risco, estadiamento de câncer de mama, avaliação de implantes",
            "estrutura": """TÉCNICA: RM bilateral das mamas com contraste paramagnético (gadolínio), incluindo sequências dinâmicas.

ACHADOS:
Composição do parênquima mamário: [classificar segundo ACR]
Mama direita:
  Parênquima: [descrever]
  Realces focais: [descrever localização, tamanho, morfologia, curva de captação]
Mama esquerda:
  Parênquima: [descrever]
  Realces focais: [descrever]
Linfonodos axilares: [descrever]
Implantes (se aplicável): [descrever integridade]

IMPRESSÃO:
[Classificação BI-RADS e recomendações]""",
            "especialidade": "Radiologia Mamária",
        },
        {
            "nome": "Ultrassonografia de Tireoide",
            "modalidade": "Ultrassonografia",
            "indicacoes": "Nódulo tireoidiano, bócio, avaliação de tireoide palpável",
            "estrutura": """TÉCNICA: Ultrassonografia da tireoide e região cervical.

ACHADOS:
Lobo direito: [dimensões] - [descrever ecogenicidade e nódulos]
Lobo esquerdo: [dimensões] - [descrever ecogenicidade e nódulos]
Istmo: [espessura]

NÓDULOS (se presentes):
  Nódulo 1: Localização, dimensões, ecogenicidade, margens, calcificações, vascularização
  [Classificar segundo TI-RADS]

Linfonodos cervicais: [descrever se visibilizados]

IMPRESSÃO:
[Classificação TI-RADS e recomendações]""",
            "especialidade": "Radiologia Geral",
        },
        {
            "nome": "TC Coronárias (Angio-TC)",
            "modalidade": "TC (Tomografia Computadorizada)",
            "indicacoes": "Avaliação de doença arterial coronariana, dor torácica atípica",
            "estrutura": """TÉCNICA: Angio-TC de artérias coronárias com sincronização cardíaca e contraste iodado.

ESCORE DE CÁLCIO: [valor] Agatston

ACHADOS:
Tronco de coronária esquerda (TCE): [descrever calibre e lesões]
Artéria descendente anterior (ADA):
  Segmento proximal: [descrever]
  Segmento médio: [descrever]
  Segmento distal: [descrever]
Artéria circunflexa (ACx): [descrever segmentos e marginais]
Coronária direita (ACD): [descrever segmentos]
Anatomia coronariana: [descrever variantes se presentes]
Função ventricular: [se avaliada]

IMPRESSÃO:
[Classificação CAD-RADS e resumo]""",
            "especialidade": "Radiologia Cardíaca",
        },
        {
            "nome": "RM Cardíaca",
            "modalidade": "RM (Ressonância Magnética)",
            "indicacoes": "Cardiomiopatia, viabilidade miocárdica, massas cardíacas",
            "estrutura": """TÉCNICA: RM cardíaca com sequências cine SSFP, T1, T2 STIR, perfusão em estresse e repouso, realce tardio.

FUNÇÃO VENTRICULAR:
Ventrículo esquerdo:
  Volume diastólico final: [mL]
  Volume sistólico final: [mL]
  Fração de ejeção: [%]
  Massa miocárdica: [g]
Ventrículo direito:
  Fração de ejeção: [%]

ANÁLISE SEGMENTAR:
[Descrever motilidade por segmentos]

PERFUSÃO MIOCÁRDICA:
[Descrever defeitos de perfusão em estresse e repouso]

REALCE TARDIO:
[Descrever padrão de realce - isquêmico/não isquêmico]

IMPRESSÃO:
[Resumo com diagnóstico diferencial]""",
            "especialidade": "Radiologia Cardíaca",
        },
        {
            "nome": "Ultrassonografia Obstétrica Morfológica",
            "modalidade": "Ultrassonografia",
            "indicacoes": "Avaliação morfológica fetal (18-24 semanas)",
            "estrutura": """TÉCNICA: Ultrassonografia obstétrica via transabdominal.

IDADE GESTACIONAL: [semanas e dias] por [parâmetro]

BIOMETRIA FETAL:
  DBP: [mm] - percentil []
  CC: [mm] - percentil []
  CA: [mm] - percentil []
  CF: [mm] - percentil []
  Peso estimado: [g] - percentil []

ANATOMIA FETAL:
  Crânio e sistema nervoso central: [descrever]
  Face: [descrever perfil, órbitas, lábios]
  Coluna vertebral: [descrever]
  Tórax e coração: [descrever câmaras, vias de saída]
  Abdome: [descrever estômago, rins, bexiga, inserção cordão]
  Membros: [descrever]
  Genitália: [se visualizada]

PLACENTA: Localização [anterior/posterior], grau [0-3], espessura [mm]
CORDÃO UMBILICAL: [3 vasos]
LÍQUIDO AMNIÓTICO: ILA [cm] ou BVM [cm]

IMPRESSÃO:
[Resumo dos achados e correlação com IG]""",
            "especialidade": "Radiologia Obstétrica",
        },
        {
            "nome": "TC de Seios da Face",
            "modalidade": "TC (Tomografia Computadorizada)",
            "indicacoes": "Sinusite, avaliação pré-operatória, complicações de sinusite",
            "estrutura": """TÉCNICA: TC dos seios paranasais sem contraste, com reconstruções axiais e coronais.

ACHADOS:
Seio maxilar direito: [descrever aeração, mucosa, conteúdo]
Seio maxilar esquerdo: [descrever]
Seio frontal direito: [descrever]
Seio frontal esquerdo: [descrever]
Células etmoidais: [descrever bilateral]
Seio esfenoidal: [descrever]
Complexo óstio-meatal: [descrever perviedade bilateral]
Septo nasal: [descrever desvios]
Cornetos: [descrever hipertrofia]
Órbitas: [descrever se incluídas]

IMPRESSÃO:
[Resumo dos achados]""",
            "especialidade": "Radiologia de Cabeça e Pescoço",
        },
    ]
    
    def __init__(self, max_items: int = None, **kwargs):
        """
        Inicializa o scraper do RadReport.
        
        Args:
            max_items: Número máximo de modelos de laudos a coletar. None = sem limite.
            **kwargs: Argumentos adicionais para BaseScraper
        """
        super().__init__(max_items=max_items, **kwargs)
        logger.info("RadReportScraper inicializado para coleta de modelos de laudos")
    
    def _scrape_template_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Extrai templates de uma página específica.
        
        Args:
            url: URL da página
            
        Returns:
            Lista de templates encontrados
        """
        templates = []
        
        response = self._make_request(url)
        if not response:
            return templates
        
        soup = self._parse_html(response)
        
        # Busca templates na página
        for item in soup.find_all(["tr", "div", "article"], class_=re.compile(r"template|report|item")):
            title = item.find(["a", "h2", "h3", "td"])
            if title:
                title_text = title.get_text(strip=True)
                link = title.get("href", "") if title.name == "a" else ""
                
                if title_text and len(title_text) > 3:
                    full_url = urljoin(self.BASE_URL, link) if link else url
                    templates.append({
                        "nome": title_text,
                        "link": full_url,
                        "fonte": "RadReport (RSNA)",
                    })
        
        # Busca links para templates específicos
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            
            if "/home/" in href or "template" in href.lower():
                if text and len(text) > 3:
                    templates.append({
                        "nome": text,
                        "link": urljoin(self.BASE_URL, href),
                        "fonte": "RadReport (RSNA)",
                    })
        
        return templates
    
    def scrape(self) -> List[Dict[str, Any]]:
        """
        Executa o scraping de templates do RadReport.
        
        Returns:
            Lista de templates coletados
        """
        logger.info("Iniciando scraping de modelos de laudos do RadReport...")
        all_templates = []
        
        # Scrape das páginas conhecidas
        for url in self.TEMPLATE_URLS:
            logger.info(f"Acessando: {url}")
            templates = self._scrape_template_page(url)
            all_templates.extend(templates)
        
        # Adiciona templates estruturados conhecidos
        for template in self.REPORT_TEMPLATES:
            all_templates.append({
                "nome": template["nome"],
                "modalidade": template["modalidade"],
                "indicacoes": template["indicacoes"],
                "estrutura_laudo": template["estrutura"],
                "especialidade": template["especialidade"],
                "fonte": "RadReport (RSNA)",
            })
        
        # Remove duplicatas
        seen = set()
        unique_templates = []
        for t in all_templates:
            key = t["nome"].lower()
            if key not in seen:
                seen.add(key)
                unique_templates.append(t)
        
        logger.info(f"Total de modelos de laudos coletados: {len(unique_templates)}")
        
        # Aplica limite de itens se configurado
        return self._apply_limit(unique_templates)
    
    def run(self) -> Path:
        """
        Executa o scraping completo e salva em JSONL.
        
        Returns:
            Path do arquivo JSONL gerado
        """
        try:
            templates = self.scrape()
            
            # Transforma para formato instruction/input/output
            transformed = []
            for t in templates:
                nome = t.get("nome", "")
                modalidade = t.get("modalidade", "")
                indicacoes = t.get("indicacoes", "")
                estrutura_laudo = t.get("estrutura_laudo", "")
                
                # Monta o input com contexto
                input_parts = []
                if modalidade:
                    input_parts.append(f"Modalidade: {modalidade}")
                if indicacoes:
                    input_parts.append(f"Indicações: {indicacoes}")
                
                transformed.append({
                    "instruction": f"Como estruturar um laudo de {nome}?",
                    "input": " | ".join(input_parts) if input_parts else "",
                    "output": estrutura_laudo,
                })
            
            filepath = self._save_to_jsonl(transformed, "modelos_laudos", "RadReport-RSNA")
            return filepath
            
        except Exception as e:
            logger.error(f"Erro no scraping RadReport: {e}", exc_info=True)
            raise
        finally:
            self.close()




if __name__ == "__main__":
    """
    Execução isolada do scraper de modelos de laudos radiológicos.
    
    Uso:
        python -m src.scraping.radreport_scraper
    
    Coleta templates de laudos do RadReport/RSNA e salva em data/raw/modelos_laudos.jsonl
    """
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    print("=" * 60)
    print("📋 SCRAPER RADREPORT - Modelos de Laudos Radiológicos")
    print("=" * 60)
    
    scraper = RadReportScraper(max_items=10)
    print(f"\nColetando modelos de laudos (limite: 10 para debug)...")
    
    filepath = scraper.run()
    
    if filepath:
        print(f"\n✅ Arquivo gerado: {filepath}")
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"   Total de registros: {len(lines)}")
        if lines:
            sample = json.loads(lines[0])
            print(f"\n📝 Primeiro registro:")
            print(f"   Instrução: {sample.get('instruction', '')[:80]}...")
            print(f"   Saída: {sample.get('output', '')[:80]}...")
    else:
        print("❌ Nenhum dado foi gerado.")

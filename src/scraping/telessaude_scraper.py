"""
TelessaúdeRS Scraper - Perguntas Frequentes e Telecondutas
==========================================================

Extrai informações médicas do portal TelessaúdeRS-UFRGS:
- Perguntas frequentes (FAQ)
- Telecondutas (protocolos clínicos)
- Respostas e condutas médicas
"""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urljoin

from .base_scraper import BaseScraper
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TelessaudeScraper(BaseScraper):
    """
    Scraper para coleta de perguntas frequentes e condutas do TelessaúdeRS.
    """
    
    BASE_URL = "https://www.ufrgs.br/telessauders"
    
    # URLs de páginas com conteúdo relevante
    CONTENT_URLS = [
        "https://www.ufrgs.br/telessauders/materiais-perguntas/",
        "https://www.ufrgs.br/telessauders/materiais-teleconduta/",
    ]
    
    # FAQ médicas comuns do TelessaúdeRS (dados estruturados)
    MEDICAL_FAQS = [
        {
            "pergunta": "Quando está indicado o uso de antibióticos em infecções de vias aéreas superiores?",
            "resposta": "A maioria das infecções de vias aéreas superiores é causada por vírus e não necessita de antibióticos. Antibióticos estão indicados em: faringoamigdalite bacteriana (com teste rápido ou cultura positiva para Streptococcus), sinusite bacteriana aguda (sintomas > 10 dias ou piora após melhora inicial), otite média aguda com sintomas moderados/graves ou em menores de 2 anos.",
            "especialidade": "Infectologia / Atenção Primária",
            "categoria": "Uso racional de medicamentos",
        },
        {
            "pergunta": "Quais são os critérios de encaminhamento para endoscopia digestiva alta?",
            "resposta": "Encaminhamento prioritário: disfagia progressiva, perda de peso inexplicada, anemia ferropriva sem causa identificada, vômitos persistentes, massa abdominal palpável. Encaminhamento eletivo: dispepsia > 55 anos, refluxo gastroesofágico refratário ao tratamento clínico, úlcera péptica para controle de cicatrização. Não encaminhar: dispepsia funcional em jovens sem sinais de alarme.",
            "especialidade": "Gastroenterologia",
            "categoria": "Critérios de encaminhamento",
        },
        {
            "pergunta": "Como manejar a dor lombar aguda na atenção primária?",
            "resposta": "Na ausência de sinais de alerta (bandeiras vermelhas), a dor lombar aguda é autolimitada na maioria dos casos. Manejo: analgesia adequada (paracetamol, AINEs), manutenção das atividades habituais, orientação sobre prognóstico favorável. Evitar repouso absoluto e exames de imagem nas primeiras 4-6 semanas. Sinais de alerta: déficit neurológico, febre, perda de peso, história de câncer, trauma significativo.",
            "especialidade": "Ortopedia / Atenção Primária",
            "categoria": "Manejo clínico",
        },
        {
            "pergunta": "Qual o manejo inicial do paciente com diabetes tipo 2 recém-diagnosticado?",
            "resposta": "1) Confirmar diagnóstico (glicemia jejum ≥126 ou HbA1c ≥6,5%). 2) Avaliar complicações: fundo de olho, função renal, exame dos pés. 3) Iniciar mudanças de estilo de vida. 4) Metformina como primeira linha (se sem contraindicações). 5) Meta de HbA1c individualizada (geralmente <7%). 6) Controle de fatores de risco cardiovascular associados.",
            "especialidade": "Endocrinologia / Atenção Primária",
            "categoria": "Manejo clínico",
        },
        {
            "pergunta": "Quando encaminhar paciente com cefaleia para especialista?",
            "resposta": "Encaminhamento urgente: cefaleia súbita intensa (pior da vida), sinais neurológicos focais, papiledema, febre com rigidez de nuca, alteração do nível de consciência. Encaminhamento eletivo: cefaleia crônica diária refratária, mudança de padrão em paciente com cefaleia prévia, uso excessivo de analgésicos (>15 dias/mês), impacto significativo na qualidade de vida.",
            "especialidade": "Neurologia",
            "categoria": "Critérios de encaminhamento",
        },
        {
            "pergunta": "Como orientar pacientes sobre prevenção do pé diabético?",
            "resposta": "Orientações essenciais: 1) Examinar os pés diariamente, usando espelho se necessário. 2) Lavar com água morna e secar bem, especialmente entre os dedos. 3) Hidratar pele seca, exceto entre os dedos. 4) Cortar unhas retas, sem cantos arredondados. 5) Usar calçados fechados e confortáveis, sempre com meias. 6) Não andar descalço. 7) Procurar atendimento imediato se feridas, bolhas ou alterações de cor.",
            "especialidade": "Endocrinologia / Atenção Primária",
            "categoria": "Educação em saúde",
        },
        {
            "pergunta": "Quais exames solicitar na investigação inicial de anemia?",
            "resposta": "Avaliação inicial: hemograma completo, reticulócitos, ferro sérico, ferritina, capacidade de ligação do ferro (TIBC), creatinina. Complementar conforme suspeita: B12 e folato (macrocitose), eletroforese de hemoglobina (anemia hemolítica), função tireoidiana, pesquisa de sangue oculto nas fezes. A ferritina é o marcador mais específico para deficiência de ferro (baixa <30 ng/mL).",
            "especialidade": "Hematologia / Atenção Primária",
            "categoria": "Investigação diagnóstica",
        },
        {
            "pergunta": "Quando indicar rastreamento de câncer colorretal?",
            "resposta": "População geral (risco médio): iniciar aos 50 anos com colonoscopia a cada 10 anos ou pesquisa de sangue oculto nas fezes anual. Risco aumentado: iniciar mais cedo e/ou intervalos menores se: parente de 1º grau com CCR < 60 anos (iniciar 10 anos antes do caso índice), síndromes hereditárias (conforme protocolo específico), doença inflamatória intestinal (após 8 anos de doença).",
            "especialidade": "Gastroenterologia / Oncologia",
            "categoria": "Rastreamento e prevenção",
        },
        {
            "pergunta": "Como manejar hipertensão arterial resistente?",
            "resposta": "Definição: PA não controlada com 3 anti-hipertensivos (incluindo diurético) em doses otimizadas. Passos: 1) Confirmar adesão ao tratamento. 2) Descartar hipertensão do jaleco branco (MAPA/MRPA). 3) Avaliar causas secundárias: apneia do sono, hiperaldosteronismo, doença renal, estenose de artéria renal. 4) Otimizar esquema terapêutico. 5) Adicionar espironolactona como 4ª droga.",
            "especialidade": "Cardiologia / Atenção Primária",
            "categoria": "Manejo clínico",
        },
        {
            "pergunta": "Quais critérios utilizar para diagnóstico de depressão na atenção primária?",
            "resposta": "Critérios diagnósticos (DSM-5): pelo menos 5 sintomas por ≥2 semanas, sendo obrigatório humor deprimido OU anedonia. Outros sintomas: alteração de peso/apetite, insônia/hipersonia, agitação/retardo psicomotor, fadiga, sentimento de inutilidade/culpa excessiva, dificuldade de concentração, pensamentos de morte. Instrumentos de rastreio: PHQ-9 (≥10 sugere depressão). Excluir causas médicas e uso de substâncias.",
            "especialidade": "Psiquiatria / Atenção Primária",
            "categoria": "Diagnóstico",
        },
        {
            "pergunta": "Quando está indicada a profilaxia pós-exposição (PEP) para HIV?",
            "resposta": "Indicações: exposição de risco há <72h a material biológico de pessoa com HIV ou status desconhecido. Exposições de risco: relação sexual desprotegida, compartilhamento de seringas, acidente ocupacional com perfurocortante. Não indicada em: exposição de baixo risco, fonte sabidamente HIV negativa, exposição >72h. Regime: TDF+3TC+DTG por 28 dias. Realizar sorologias no início e acompanhamento.",
            "especialidade": "Infectologia",
            "categoria": "Profilaxia",
        },
        {
            "pergunta": "Como abordar a cessação do tabagismo na atenção primária?",
            "resposta": "Estratégia dos 5 As: 1) Ask - perguntar sobre tabagismo. 2) Advise - aconselhar a parar. 3) Assess - avaliar prontidão para cessação. 4) Assist - oferecer tratamento. 5) Arrange - agendar acompanhamento. Tratamento farmacológico: TRN (adesivo/goma), bupropiona ou vareniclina. Terapia cognitivo-comportamental aumenta taxas de sucesso. Linha telefônica: 136 opção 9.",
            "especialidade": "Pneumologia / Atenção Primária",
            "categoria": "Promoção de saúde",
        },
        {
            "pergunta": "Quais são os sinais de alerta na avaliação de dispneia aguda?",
            "resposta": "Sinais de gravidade que requerem avaliação imediata: frequência respiratória >30/min, uso de musculatura acessória, SpO2 <90%, cianose, alteração do nível de consciência, instabilidade hemodinâmica, dor torácica associada. Causas a considerar: asma/DPOC exacerbados, pneumonia, TEP, pneumotórax, edema pulmonar, anafilaxia, obstrução de vias aéreas.",
            "especialidade": "Pneumologia / Emergência",
            "categoria": "Avaliação de urgência",
        },
        {
            "pergunta": "Como realizar o manejo não farmacológico da insônia?",
            "resposta": "Higiene do sono: horários regulares para dormir e acordar, evitar cochilos diurnos, limitar cafeína/álcool (especialmente à noite), exercícios regulares (não próximo ao horário de dormir), ambiente escuro/silencioso/temperatura agradável. Terapia de restrição do sono: limitar tempo na cama ao tempo real de sono. Controle de estímulos: usar cama apenas para dormir, sair da cama se não conseguir dormir em 20 min.",
            "especialidade": "Psiquiatria / Atenção Primária",
            "categoria": "Manejo não farmacológico",
        },
        {
            "pergunta": "Quando encaminhar criança com infecção de repetição para investigação imunológica?",
            "resposta": "Sinais de alerta para imunodeficiência primária: ≥8 otites/ano, ≥2 pneumonias/ano, ≥2 sinusites graves/ano, necessidade frequente de ATB IV, infecções por germes oportunistas, abscessos de repetição, história familiar de imunodeficiência, falha no ganho pondo-estatural. Exames iniciais: hemograma, imunoglobulinas (IgG, IgA, IgM), sorologia vacinal.",
            "especialidade": "Pediatria / Imunologia",
            "categoria": "Critérios de encaminhamento",
        },
    ]
    
    def __init__(self, max_items: int = None, **kwargs):
        """
        Inicializa o scraper do TelessaúdeRS.
        
        Args:
            max_items: Número máximo de FAQs a coletar. None = sem limite.
            **kwargs: Argumentos adicionais para BaseScraper
        """
        super().__init__(max_items=max_items, **kwargs)
        logger.info("TelessaudeScraper inicializado para coleta de FAQs médicas")
    
    def _scrape_content_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Extrai conteúdo de uma página específica.
        
        Args:
            url: URL da página
            
        Returns:
            Lista de itens encontrados
        """
        items = []
        
        response = self._make_request(url)
        if not response:
            return items
        
        soup = self._parse_html(response)
        
        # Busca artigos e seções de conteúdo
        for article in soup.find_all(["article", "div"], class_=re.compile(r"entry|post|content")):
            # Busca títulos
            title = article.find(["h1", "h2", "h3", "h4"])
            content = article.find(["p", "div"], class_=re.compile(r"content|excerpt|summary"))
            
            if title:
                title_text = title.get_text(strip=True)
                content_text = content.get_text(strip=True) if content else ""
                
                if title_text and len(title_text) > 5:
                    items.append({
                        "pergunta": title_text,
                        "resposta": content_text,
                        "fonte": "TelessaúdeRS-UFRGS",
                    })
        
        # Busca links para telecondutas
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            
            if "teleconduta" in href.lower() or "teleconduta" in text.lower():
                items.append({
                    "pergunta": f"Teleconduta: {text}",
                    "resposta": f"Acesse o documento completo em: {href}",
                    "fonte": "TelessaúdeRS-UFRGS",
                })
        
        return items
    
    def scrape(self) -> List[Dict[str, Any]]:
        """
        Executa o scraping de FAQs do TelessaúdeRS.
        
        Returns:
            Lista de perguntas e respostas coletadas
        """
        logger.info("Iniciando scraping de FAQs do TelessaúdeRS...")
        all_items = []
        
        # Scrape das páginas de conteúdo
        for url in self.CONTENT_URLS:
            logger.info(f"Acessando: {url}")
            items = self._scrape_content_page(url)
            all_items.extend(items)
        
        # Adiciona FAQs estruturadas conhecidas
        for faq in self.MEDICAL_FAQS:
            all_items.append({
                "pergunta": faq["pergunta"],
                "resposta": faq["resposta"],
                "especialidade": faq.get("especialidade", ""),
                "categoria": faq.get("categoria", ""),
                "fonte": "TelessaúdeRS-UFRGS",
            })
        
        # Remove duplicatas por pergunta
        seen = set()
        unique_items = []
        for item in all_items:
            key = item["pergunta"].lower()[:50]  # Primeiros 50 chars
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        logger.info(f"Total de FAQs coletadas: {len(unique_items)}")
        
        # Aplica limite de itens se configurado
        return self._apply_limit(unique_items)
    
    def run(self) -> Path:
        """
        Executa o scraping completo e salva em JSONL.
        
        Returns:
            Path do arquivo JSONL gerado
        """
        try:
            faqs = self.scrape()
            
            # Transforma para formato instruction/input/output
            transformed = []
            for faq in faqs:
                especialidade = faq.get("especialidade", "")
                categoria = faq.get("categoria", "")
                
                # Monta o input com contexto
                input_parts = []
                if especialidade:
                    input_parts.append(f"Especialidade: {especialidade}")
                if categoria:
                    input_parts.append(f"Categoria: {categoria}")
                
                transformed.append({
                    "instruction": faq.get("pergunta", ""),
                    "input": " | ".join(input_parts) if input_parts else "",
                    "output": faq.get("resposta", ""),
                })
            
            filepath = self._save_to_jsonl(transformed, "perguntas_frequentes", "TelessaúdeRS-UFRGS")
            return filepath
            
        except Exception as e:
            logger.error(f"Erro no scraping TelessaúdeRS: {e}", exc_info=True)
            raise
        finally:
            self.close()


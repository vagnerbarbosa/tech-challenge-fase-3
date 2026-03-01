"""
Ferramentas Customizadas do LangChain
=====================================

Implementa ferramentas auxiliares para o assistente médico generalista.
Inclui ferramentas para triagem, orientações e referências médicas.
"""

import re
from typing import List, Dict, Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MedicalTools:
    """
    Ferramentas auxiliares para o assistente médico generalista.
    """
    
    def __init__(self):
        """
        Inicializa as ferramentas.
        """
        # Palavras-chave de emergência (ampliadas para assistente generalista)
        self.emergency_keywords = [
            "emergência", "urgente", "desmaio", "desmaiando",
            "confuso", "confusão", "convulsão", "coma",
            "não consigo respirar", "dor no peito", "dor forte no peito",
            "muito mal", "morrendo", "inconsciente",
            "sangramento intenso", "hemorragia",
            "avc", "derrame", "infarto",
            "reação alérgica", "anafilaxia",
            "envenenamento", "overdose",
            "afogamento", "engasgando",
        ]
        
        # Referência de especialidades médicas
        self.medical_specialties = {
            "cardiologia": ["coração", "pressão", "arritmia", "colesterol"],
            "dermatologia": ["pele", "alergia", "mancha", "acne", "coceira"],
            "endocrinologia": ["hormônio", "tireoide", "diabetes", "metabolismo"],
            "gastroenterologia": ["estômago", "intestino", "diarreia", "constipação"],
            "neurologia": ["dor de cabeça", "enxaqueca", "tontura", "memória"],
            "ortopedia": ["osso", "articulação", "fratura", "coluna"],
            "pneumologia": ["pulmão", "tosse", "asma", "falta de ar"],
            "psiquiatria": ["ansiedade", "depressão", "insônia", "estresse"],
        }
        
        logger.info("MedicalTools inicializado para assistente generalista")
    
    def is_emergency_question(self, text: str) -> bool:
        """
        Verifica se a mensagem indica emergência.
        
        Args:
            text: Texto da mensagem
            
        Returns:
            True se indica emergência
        """
        text_lower = text.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                logger.warning(f"Possível emergência detectada: '{keyword}'")
                return True
        
        return False
    
    def suggest_specialty(self, text: str) -> str:
        """
        Sugere especialidade médica baseada nos sintomas descritos.
        
        Args:
            text: Texto com sintomas
            
        Returns:
            Especialidade sugerida ou None
        """
        text_lower = text.lower()
        
        for specialty, keywords in self.medical_specialties.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return specialty
        
        return None
    
    def get_vital_signs_reference(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna referências de sinais vitais normais.
        
        Returns:
            Dicionário com valores de referência
        """
        return {
            "pressao_arterial": {
                "normal": "< 120/80 mmHg",
                "elevada": "120-129/< 80 mmHg",
                "hipertensao_estagio1": "130-139/80-89 mmHg",
                "hipertensao_estagio2": "≥ 140/90 mmHg",
            },
            "frequencia_cardiaca": {
                "normal_adulto": "60-100 bpm",
                "bradicardia": "< 60 bpm",
                "taquicardia": "> 100 bpm",
            },
            "temperatura": {
                "normal": "36.1-37.2°C",
                "febricula": "37.3-37.8°C",
                "febre": "> 37.8°C",
                "febre_alta": "> 39°C",
            },
            "glicemia_jejum": {
                "normal": "70-99 mg/dL",
                "pre_diabetes": "100-125 mg/dL",
                "diabetes": "≥ 126 mg/dL",
            },
        }
    
    def extract_temperature_value(self, text: str) -> float:
        """
        Extrai valor de temperatura do texto.
        
        Args:
            text: Texto contendo valor de temperatura
            
        Returns:
            Valor extraído ou None
        """
        patterns = [
            r'temperatura[:\s]+?(\d+(?:[.,]\d+)?)',
            r'febre[:\s]+?(\d+(?:[.,]\d+)?)',
            r'(\d+(?:[.,]\d+)?)\s*(?:°c|graus)',
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1).replace(',', '.')
                return float(value)
        
        return None
    
    def interpret_temperature(self, value: float) -> Dict[str, Any]:
        """
        Interpreta valor de temperatura corporal.
        
        Args:
            value: Valor da temperatura em °C
            
        Returns:
            Interpretação do valor
        """
        result = {
            "value": value,
            "unit": "°C",
            "classification": "",
            "recommendation": "",
            "alert_level": "normal",
        }
        
        if value < 35:
            result["classification"] = "Hipotermia"
            result["recommendation"] = "Busque atendimento médico."
            result["alert_level"] = "warning"
        elif value <= 37.2:
            result["classification"] = "Normal"
            result["recommendation"] = "Temperatura dentro da faixa saudável."
        elif value <= 37.8:
            result["classification"] = "Febrícula"
            result["recommendation"] = "Monitore a temperatura. Hidrate-se bem."
            result["alert_level"] = "warning"
        elif value <= 39:
            result["classification"] = "Febre"
            result["recommendation"] = "Considere antipirético conforme orientação médica. Hidrate-se."
            result["alert_level"] = "high"
        else:
            result["classification"] = "Febre alta"
            result["recommendation"] = "Busque atendimento médico, especialmente se persistir."
            result["alert_level"] = "critical"
        
        return result
    
    def get_general_health_tips(self) -> List[str]:
        """
        Retorna dicas gerais de saúde.
        
        Returns:
            Lista de dicas
        """
        tips = [
            "Mantenha uma alimentação equilibrada e variada",
            "Pratique exercícios físicos regularmente (30 min/dia)",
            "Durma de 7 a 9 horas por noite",
            "Beba água suficiente ao longo do dia (cerca de 2 litros)",
            "Faça check-ups médicos anuais",
            "Mantenha as vacinas em dia",
            "Evite o tabagismo e modere o consumo de álcool",
            "Cuide da saúde mental - busque ajuda se necessário",
            "Lave as mãos frequentemente",
            "Use protetor solar diariamente",
        ]
        
        return tips


if __name__ == "__main__":
    tools = MedicalTools()
    
    # Teste de interpretação de temperatura
    print(tools.interpret_temperature(36.5))
    print(tools.interpret_temperature(38.5))
    
    # Teste de sugestão de especialidade
    print(tools.suggest_specialty("Estou com dor de cabeça forte"))
    
    # Teste de detecção de emergência
    print(tools.is_emergency_question("Estou com dor forte no peito"))

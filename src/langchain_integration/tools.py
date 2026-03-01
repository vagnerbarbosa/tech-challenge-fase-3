"""
Ferramentas Customizadas do LangChain
=====================================

Implementa ferramentas auxiliares para o assistente médico.
"""

import re
from typing import List, Dict, Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MedicalTools:
    """
    Ferramentas auxiliares para o assistente médico.
    """
    
    def __init__(self):
        """
        Inicializa as ferramentas.
        """
        # Palavras-chave de emergência
        self.emergency_keywords = [
            "emergência", "urgente", "desmaio", "desmaiando",
            "confuso", "confusão", "convulsão", "coma",
            "não consigo respirar", "dor no peito",
            "muito mal", "morrendo", "inconsciente",
            "hipoglicemia severa", "cetoacidose",
        ]
        
        # Referência de valores de glicemia
        self.glycemia_reference = {
            "jejum_normal": (70, 99),
            "jejum_pre_diabetes": (100, 125),
            "jejum_diabetes": (126, float('inf')),
            "pos_prandial_normal": (0, 140),
            "pos_prandial_pre_diabetes": (140, 199),
            "pos_prandial_diabetes": (200, float('inf')),
            "hipoglicemia_leve": (54, 69),
            "hipoglicemia_severa": (0, 53),
        }
        
        logger.info("MedicalTools inicializado")
    
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
    
    def interpret_glycemia(self, value: float, fasting: bool = True) -> Dict[str, Any]:
        """
        Interpreta valor de glicemia.
        
        Args:
            value: Valor da glicemia em mg/dL
            fasting: Se é glicemia de jejum
            
        Returns:
            Interpretação do valor
        """
        result = {
            "value": value,
            "unit": "mg/dL",
            "fasting": fasting,
            "classification": "",
            "recommendation": "",
            "alert_level": "normal",
        }
        
        # Verifica hipoglicemia primeiro
        if value < 54:
            result["classification"] = "Hipoglicemia severa"
            result["recommendation"] = "EMERGÊNCIA: Busque atendimento médico imediato!"
            result["alert_level"] = "critical"
        elif value < 70:
            result["classification"] = "Hipoglicemia leve"
            result["recommendation"] = "Consuma 15g de carboidrato de ação rápida."
            result["alert_level"] = "warning"
        elif fasting:
            if value <= 99:
                result["classification"] = "Normal"
                result["recommendation"] = "Valores dentro da faixa saudável."
            elif value <= 125:
                result["classification"] = "Pré-diabetes"
                result["recommendation"] = "Consulte um médico para avaliação."
                result["alert_level"] = "warning"
            else:
                result["classification"] = "Indicativo de diabetes"
                result["recommendation"] = "Consulte um médico para confirmação diagnóstica."
                result["alert_level"] = "high"
        else:  # Pós-prandial
            if value <= 140:
                result["classification"] = "Normal"
                result["recommendation"] = "Valores dentro da faixa saudável."
            elif value <= 199:
                result["classification"] = "Pré-diabetes"
                result["recommendation"] = "Consulte um médico para avaliação."
                result["alert_level"] = "warning"
            else:
                result["classification"] = "Indicativo de diabetes"
                result["recommendation"] = "Consulte um médico para confirmação diagnóstica."
                result["alert_level"] = "high"
        
        return result
    
    def extract_glycemia_value(self, text: str) -> float:
        """
        Extrai valor de glicemia do texto.
        
        Args:
            text: Texto contendo valor de glicemia
            
        Returns:
            Valor extraído ou None
        """
        # Padrões para extrair valores de glicemia
        patterns = [
            r'glicemia[:\s]+?(\d+(?:\.\d+)?)',
            r'glicose[:\s]+?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*mg/dl',
            r'(\d+(?:\.\d+)?)\s*mg',
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
        
        return None
    
    def get_diet_tips(self, condition: str = "diabetes") -> List[str]:
        """
        Retorna dicas de alimentação.
        
        Args:
            condition: Condição médica
            
        Returns:
            Lista de dicas
        """
        tips = [
            "Prefira alimentos com baixo índice glicêmico",
            "Inclua fibras em todas as refeições",
            "Evite açúcares refinados e refrigerantes",
            "Consuma proteínas magras",
            "Faça refeições em horários regulares",
            "Controle o tamanho das porções",
            "Hidrate-se adequadamente com água",
            "Limite o consumo de álcool",
        ]
        
        return tips


if __name__ == "__main__":
    tools = MedicalTools()
    
    # Teste de interpretação de glicemia
    print(tools.interpret_glycemia(85, fasting=True))
    print(tools.interpret_glycemia(180, fasting=False))
    
    # Teste de detecção de emergência
    print(tools.is_emergency_question("Estou muito confuso e suando"))

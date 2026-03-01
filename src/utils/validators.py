"""
Validadores de Segurança

Implementa validação de respostas médicas para
garantir segurança e conformidade.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.utils.logging_config import get_logger, log_audit

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Resultado da validação."""
    is_valid: bool
    issues: List[str]
    cleaned_response: str
    risk_level: str  # low, medium, high


class MedicalResponseValidator:
    """Validador de respostas médicas."""
    
    # Padrões que indicam prescrição médica indevida
    PRESCRIPTION_PATTERNS = [
        r"tome\s+\d+\s*(mg|ml|comprimido)",
        r"\d+\s*(mg|ml)\s*(a cada|de|por)\s*\d+\s*hora",
        r"prescrevo",
        r"receito",
        r"dosagem\s*:\s*\d+"
    ]
    
    # Palavras que devem ser evitadas
    FORBIDDEN_TERMS = [
        "diagnóstico definitivo",
        "você tem",
        "sua doença é",
        "não precisa de médico",
        "substitui consulta",
        "certeza que é"
    ]
    
    # Avisos obrigatórios
    REQUIRED_DISCLAIMERS = [
        "profissional de saúde",
        "médico",
        "consulta"
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compila padrões regex para eficiência."""
        self.prescription_re = [
            re.compile(p, re.IGNORECASE) 
            for p in self.PRESCRIPTION_PATTERNS
        ]
    
    def validate(self, response: str) -> ValidationResult:
        """
        Valida uma resposta médica.
        
        Args:
            response: Resposta do assistente para validar
            
        Returns:
            ValidationResult com status e problemas encontrados
        """
        issues = []
        risk_level = "low"
        
        response_lower = response.lower()
        
        # 1. Verifica prescrições indevidas
        for pattern in self.prescription_re:
            if pattern.search(response):
                issues.append("Possível prescrição médica detectada")
                risk_level = "high"
                break
        
        # 2. Verifica termos proibidos
        for term in self.FORBIDDEN_TERMS:
            if term in response_lower:
                issues.append(f"Termo inadequado detectado: '{term}'")
                if risk_level != "high":
                    risk_level = "medium"
        
        # 3. Verifica presença de avisos obrigatórios
        has_disclaimer = any(
            disclaimer in response_lower 
            for disclaimer in self.REQUIRED_DISCLAIMERS
        )
        
        if not has_disclaimer:
            issues.append("Resposta sem recomendação de consultar profissional")
            if risk_level == "low":
                risk_level = "medium"
        
        # 4. Verifica tamanho mínimo da resposta
        if len(response.strip()) < 50:
            issues.append("Resposta muito curta")
        
        is_valid = len(issues) == 0 or (risk_level == "low")
        
        # Log de validação
        if issues:
            logger.warning(f"Validação encontrou problemas: {issues}")
            log_audit("response_validation", {
                "issues": issues,
                "risk_level": risk_level,
                "is_valid": is_valid
            })
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_response=response,
            risk_level=risk_level
        )
    
    def clean_response(self, response: str) -> str:
        """
        Limpa uma resposta removendo conteúdo problemático.
        
        Args:
            response: Resposta original
            
        Returns:
            Resposta limpa
        """
        cleaned = response
        
        # Remove possíveis prescrições
        for pattern in self.prescription_re:
            cleaned = pattern.sub("[informação removida por segurança]", cleaned)
        
        return cleaned
    
    def add_disclaimer(self, response: str) -> str:
        """
        Adiciona aviso de disclaimer à resposta se necessário.
        
        Args:
            response: Resposta original
            
        Returns:
            Resposta com disclaimer
        """
        disclaimer = "\n\n⚠️ *Lembre-se: Esta informação é apenas para fins educacionais e não substitui uma consulta com profissional de saúde.*"
        
        response_lower = response.lower()
        has_disclaimer = any(
            d in response_lower 
            for d in self.REQUIRED_DISCLAIMERS
        )
        
        if not has_disclaimer:
            return response + disclaimer
        
        return response
    
    def validate_and_clean(self, response: str) -> str:
        """
        Valida, limpa e adiciona disclaimers à resposta.
        
        Args:
            response: Resposta original
            
        Returns:
            Resposta processada e segura
        """
        # Valida
        result = self.validate(response)
        
        # Se risco alto, limpa
        if result.risk_level == "high":
            response = self.clean_response(response)
        
        # Adiciona disclaimer se necessário
        response = self.add_disclaimer(response)
        
        return response


class InputValidator:
    """Validador de entrada do usuário."""
    
    MAX_INPUT_LENGTH = 2000
    
    # Padrões de injeção de prompt
    INJECTION_PATTERNS = [
        r"ignore.*previous.*instructions",
        r"esqueca.*tudo",
        r"voce.*agora.*e",
        r"novo.*papel",
        r"system\s*:",
        r"<\|.*\|>"
    ]
    
    def __init__(self):
        self.injection_re = [
            re.compile(p, re.IGNORECASE) 
            for p in self.INJECTION_PATTERNS
        ]
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Valida entrada do usuário.
        
        Args:
            user_input: Texto do usuário
            
        Returns:
            Tupla (is_valid, message)
        """
        # Verifica tamanho
        if len(user_input) > self.MAX_INPUT_LENGTH:
            return False, f"Entrada muito longa. Máximo: {self.MAX_INPUT_LENGTH} caracteres."
        
        # Verifica tentativas de injeção
        for pattern in self.injection_re:
            if pattern.search(user_input):
                logger.warning(f"Possível tentativa de injeção detectada")
                log_audit("injection_attempt", {"input_preview": user_input[:100]})
                return False, "Entrada inválida detectada."
        
        # Remove caracteres de controle
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', user_input)
        
        return True, cleaned


# Exemplo de uso
if __name__ == "__main__":
    validator = MedicalResponseValidator()
    
    # Testa validação
    test_response = """Baseado nos sintomas descritos, parece que você pode estar 
    com uma infecção viral. Tome 500mg de paracetamol a cada 8 horas."""
    
    result = validator.validate(test_response)
    print(f"Válido: {result.is_valid}")
    print(f"Problemas: {result.issues}")
    print(f"Risco: {result.risk_level}")
    
    # Limpa e adiciona disclaimer
    safe_response = validator.validate_and_clean(test_response)
    print(f"\nResposta segura:\n{safe_response}")

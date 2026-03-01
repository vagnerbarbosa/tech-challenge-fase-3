"""
Testes para os Validadores
==========================
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.validators import InputValidator, DataValidator


class TestInputValidator:
    """Testes para InputValidator."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.validator = InputValidator()
    
    def test_valid_query(self):
        """Testa query válida."""
        is_valid, msg = self.validator.validate_query("O que é diabetes?")
        assert is_valid is True
    
    def test_empty_query(self):
        """Testa query vazia."""
        is_valid, msg = self.validator.validate_query("")
        assert is_valid is False
    
    def test_too_short_query(self):
        """Testa query muito curta."""
        is_valid, msg = self.validator.validate_query("a")
        assert is_valid is False
    
    def test_script_injection(self):
        """Testa proteção contra XSS."""
        is_valid, msg = self.validator.validate_query("<script>alert('xss')</script>")
        assert is_valid is False
    
    def test_sanitize_html(self):
        """Testa sanitização de HTML."""
        result = self.validator.sanitize_input("<b>Texto</b>")
        assert "<b>" not in result
        assert "Texto" in result
    
    def test_sanitize_whitespace(self):
        """Testa normalização de espaços."""
        result = self.validator.sanitize_input("  texto   com   espaços  ")
        assert result == "texto com espaços"


class TestDataValidator:
    """Testes para DataValidator."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.validator = DataValidator()
    
    def test_valid_dataframe(self):
        """Testa DataFrame válido."""
        df = pd.DataFrame({
            "instruction": ["Pergunta 1"],
            "output": ["Resposta 1"],
        })
        assert self.validator.validate_dataframe(df) is True
    
    def test_empty_dataframe(self):
        """Testa DataFrame vazio."""
        df = pd.DataFrame()
        assert self.validator.validate_dataframe(df) is False
    
    def test_missing_columns(self):
        """Testa DataFrame sem colunas obrigatórias."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        assert self.validator.validate_dataframe(df) is False
    
    def test_valid_medical_record(self):
        """Testa registro médico válido."""
        record = {
            "instruction": "O que é diabetes?",
            "output": "Diabetes é uma condição crônica que afeta o metabolismo.",
        }
        is_valid, msg = self.validator.validate_medical_record(record)
        assert is_valid is True
    
    def test_invalid_medical_record(self):
        """Testa registro médico inválido."""
        record = {
            "instruction": "abc",
            "output": "xyz",
        }
        is_valid, msg = self.validator.validate_medical_record(record)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Módulo de Avaliação do Modelo

Implementa métricas e avaliação do modelo fine-tuned.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Avaliador do modelo fine-tuned."""
    
    def __init__(
        self,
        model_path: str = "./models/assistente-medico-final",
        test_data_path: str = "./data/processed/medical_dataset.json"
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Carrega modelo treinado para avaliação."""
        logger.info(f"Carregando modelo de: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        
        logger.info("Modelo carregado para avaliação")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Gera resposta do modelo para um prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove o prompt da resposta
        return response[len(prompt):].strip()
    
    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calcula métricas de avaliação."""
        # Métricas simples (para uma avaliação completa, usar BLEU, ROUGE, etc.)
        metrics = {
            "num_samples": len(predictions),
            "avg_response_length": sum(len(p) for p in predictions) / len(predictions),
            "non_empty_responses": sum(1 for p in predictions if p.strip()) / len(predictions)
        }
        
        # TODO: Adicionar métricas mais sofisticadas
        # - BLEU score
        # - ROUGE score
        # - Perplexity
        # - Métricas específicas médicas
        
        return metrics
    
    def evaluate_on_dataset(self, num_samples: int = 50) -> Dict[str, Any]:
        """Avalia modelo no dataset de teste."""
        logger.info("Avaliando modelo no dataset...")
        
        # Carrega dados de teste
        dataset = load_dataset("json", data_files=str(self.test_data_path), split="train")
        
        # Seleciona amostras
        if len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        predictions = []
        references = []
        results = []
        
        for example in dataset:
            instruction = example.get("instruction", "")
            context = example.get("input", "")
            expected = example.get("output", "")
            
            # Formata prompt
            if context:
                prompt = f"### Instrução:\n{instruction}\n\n### Contexto:\n{context}\n\n### Resposta:\n"
            else:
                prompt = f"### Instrução:\n{instruction}\n\n### Resposta:\n"
            
            # Gera resposta
            response = self.generate_response(prompt)
            
            predictions.append(response)
            references.append(expected)
            
            results.append({
                "instruction": instruction,
                "expected": expected,
                "predicted": response
            })
        
        # Calcula métricas
        metrics = self.calculate_metrics(predictions, references)
        
        return {
            "metrics": metrics,
            "results": results
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Executa avaliação completa."""
        logger.info("=" * 50)
        logger.info("Iniciando Avaliação do Modelo")
        logger.info("=" * 50)
        
        # Carrega modelo
        self.load_model()
        
        # Avalia
        evaluation = self.evaluate_on_dataset()
        
        # Salva resultados
        results_path = self.model_path.parent / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Resultados salvos em: {results_path}")
        logger.info(f"Métricas: {evaluation['metrics']}")
        
        return evaluation["metrics"]


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()

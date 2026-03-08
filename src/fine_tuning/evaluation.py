"""
Módulo de Avaliação do Modelo
=============================

Responsável por:
- Avaliar métricas do modelo treinado
- Gerar relatórios de performance
- Comparar com baseline
- Testar capacidade de responder perguntas médicas gerais
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import pipeline

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Classe para avaliação de modelos de linguagem médicos.
    Avalia a capacidade do assistente generalista de responder perguntas médicas.
    """
    
    def __init__(self, model: Any, tokenizer: Any):
        """
        Inicializa o avaliador.
        
        Args:
            model: Modelo treinado
            tokenizer: Tokenizer do modelo
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configura pipeline de geração sem forçar device (compatível com accelerate)
        try:
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
        except Exception as e:
            logger.warning(f"Falha ao criar pipeline com device automático: {e}")
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=None  # Força device=None para evitar conflito com accelerate
            )
        
        logger.info("ModelEvaluator inicializado para assistente médico generalista")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """
        Gera resposta para um prompt médico.
        
        Args:
            prompt: Texto de entrada (pergunta médica)
            max_length: Tamanho máximo da resposta
            
        Returns:
            Resposta gerada
        """
        result = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        return result[0]['generated_text']
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """
        Calcula perplexidade média para uma lista de textos.
        
        Args:
            texts: Lista de textos para avaliação
            
        Returns:
            Perplexidade média
        """
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                
                input_ids = encodings.input_ids.to(self.device)
                
                outputs = self.model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_qa_quality(
        self, 
        questions: List[str], 
        expected_answers: List[str]
    ) -> Dict[str, float]:
        """
        Avalia qualidade das respostas de Q&A médico.
        
        Args:
            questions: Lista de perguntas médicas
            expected_answers: Lista de respostas esperadas
            
        Returns:
            Dicionário com métricas
        """
        from difflib import SequenceMatcher
        
        similarities = []
        
        for question, expected in zip(questions, expected_answers):
            prompt = f"### Instrução:\n{question}\n\n### Resposta:\n"
            generated = self.generate_response(prompt)
            
            # Remove o prompt da resposta gerada
            response = generated.replace(prompt, "").strip()
            
            # Calcula similaridade
            similarity = SequenceMatcher(None, response.lower(), expected.lower()).ratio()
            similarities.append(similarity)
        
        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
        }
    
    def evaluate(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Executa avaliação completa do modelo médico generalista.
        
        Args:
            dataset: Dataset de avaliação
            
        Returns:
            Dicionário com todas as métricas
        """
        logger.info("Iniciando avaliação do modelo médico generalista...")
        
        metrics = {}
        
        # Extrai textos do dataset
        texts = dataset['text'][:50]  # Limita para avaliação rápida
        
        # Calcula perplexidade
        logger.info("Calculando perplexidade...")
        metrics['perplexity'] = self.calculate_perplexity(texts)
        logger.info(f"Perplexidade: {metrics['perplexity']:.2f}")
        
        # Teste de geração com perguntas médicas gerais
        logger.info("Testando geração de respostas médicas...")
        test_prompts = [
            "Quais são os sintomas de uma gripe?",
            "Quando devo procurar um médico?",
            "Como melhorar minha qualidade de sono?",
        ]
        
        for prompt in test_prompts:
            response = self.generate_response(f"### Instrução:\n{prompt}\n\n### Resposta:\n")
            logger.info(f"\nPergunta: {prompt}")
            logger.info(f"Resposta: {response[:200]}...")
        
        logger.info("\nAvaliação do assistente médico generalista concluída!")
        
        return metrics


if __name__ == "__main__":
    # Este módulo requer modelo treinado para funcionar
    logger.info("Execute este módulo através do main.py")
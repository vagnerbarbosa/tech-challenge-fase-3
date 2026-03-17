import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import pipeline, AutoTokenizer

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
        
        # Configura pipeline de geração sem forçar device (compatível com accelerate)
        try:
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",  # Importante para BioMistral
                trust_remote_code=True
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
    
    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Gera resposta para um prompt médico.
        
        Args:
            prompt: Texto de entrada (pergunta médica)
            max_new_tokens: Número máximo de tokens a gerar
            
        Returns:
            Resposta gerada
        """
        # Configuração mais segura para BioMistral
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,              # Menos criatividade
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False        # Evita repetir o prompt
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
                
                input_ids = encodings.input_ids.to(self.model.device)
                
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
            # Prompt no formato BioMistral
            prompt = self.format_biomistral_prompt(question)
            generated = self.generate_response(prompt)
            
            # Calcula similaridade
            similarity = SequenceMatcher(None, generated.lower(), expected.lower()).ratio()
            similarities.append(similarity)
        
        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
        }
    
    def format_biomistral_prompt(self, pergunta: str) -> str:
        """
        Formata prompt no estilo BioMistral (Mistral-Style).
        """
        return f"<s>[INST] Você é um assistente médico técnico. Use estritamente os protocolos de laudos e diretrizes clínicas brasileiras para responder de forma concisa. Não invente diálogos adicionais.\n\nPergunta: {pergunta} [/INST]"

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
            "Como estruturar um laudo de RM de Coluna Lombar?",
            "Quais exames devo solicitar na investigação inicial de anemia?",
            "Quais são as diretrizes do protocolo de Acidentes Escorpiônicos?",
        ]
        
        for prompt in test_prompts:
            formatted_prompt = self.format_biomistral_prompt(prompt)
            response = self.generate_response(formatted_prompt)
            logger.info(f"\nPergunta: {prompt}")
            logger.info(f"Resposta: {response}")
        
        logger.info("\nAvaliação do assistente médico generalista concluída!")
        
        return metrics




if __name__ == "__main__":
    """
    Execução isolada do módulo de avaliação.
    
    Uso:
        python -m src.fine_tuning.evaluation
    
    Exibe informações sobre o avaliador e exemplos de uso.
    Nota: requer modelo e tokenizer carregados para avaliação completa.
    """
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    print("=" * 60)
    print("📊 MÓDULO DE AVALIAÇÃO DO MODELO")
    print("=" * 60)
    
    print("\nO ModelEvaluator requer um modelo e tokenizer carregados.")
    print("Para avaliação completa, execute via pipeline principal:\n")
    print("   python main.py\n")
    print("Ou integre via código:")
    print("   from src.fine_tuning.evaluation import ModelEvaluator")
    print("   evaluator = ModelEvaluator(model, tokenizer)")
    print("   metrics = evaluator.evaluate(dataset)")
    print("\nMétricas disponíveis:")
    print("   - Perplexidade (calculate_perplexity)")
    print("   - Qualidade de Q&A (evaluate_qa_quality)")
    print("   - Geração de respostas (generate_response)")
    
    print("\n📝 Exemplo de prompt BioMistral:")
    print(ModelEvaluator.format_biomistral_prompt(None, "Quais os sintomas da gripe?"))

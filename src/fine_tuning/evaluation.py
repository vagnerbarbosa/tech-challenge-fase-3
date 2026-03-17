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
        
        # Garante que o tokenizer e o modelo aceitem o comprimento configurado
        max_seq = int(os.environ.get("MAX_SEQ_LENGTH", 1024))
        if hasattr(self.tokenizer, "model_max_length"):
            self.tokenizer.model_max_length = max_seq
        
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
        
        logger.info(f"ModelEvaluator inicializado (Max Seq: {max_seq})")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Gera resposta para um prompt médico.
        """
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        return result[0]['generated_text']
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """
        Calcula perplexidade média para uma lista de textos respeitando o MAX_SEQ_LENGTH.
        """
        total_loss = 0
        total_tokens = 0
        
        max_seq = int(os.environ.get("MAX_SEQ_LENGTH", 1024))
        
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq,
                )
                
                input_ids = encodings.input_ids.to(self.model.device)
                
                # Evita erro se o input_ids for vazio
                if input_ids.size(1) == 0:
                    continue
                    
                outputs = self.model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        if total_tokens == 0:
            return float('inf')
            
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
        """
        from difflib import SequenceMatcher
        
        similarities = []
        
        for question, expected in zip(questions, expected_answers):
            prompt = self.format_biomistral_prompt(question)
            generated = self.generate_response(prompt)
            
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
        """
        logger.info("Iniciando avaliação do modelo médico generalista...")
        
        metrics = {}
        
        # Extrai textos do dataset
        texts = dataset['text'][:50] if len(dataset) > 0 else []
        
        if not texts:
            logger.warning("Dataset vazio para avaliação.")
            return {"perplexity": 0.0}

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
            try:
                response = self.generate_response(formatted_prompt)
                logger.info(f"\nPergunta: {prompt}")
                logger.info(f"Resposta: {response}")
            except Exception as e:
                logger.error(f"Erro ao gerar resposta para o prompt '{prompt}': {e}")
        
        logger.info("\nAvaliação do assistente médico generalista concluída!")
        
        return metrics

if __name__ == "__main__":
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
"""
Assistente Virtual Médico Generalista (BioMistral Edition)
=========================================================

Implementa o assistente médico generalista usando LangChain e BioMistral-7B.
Capaz de responder perguntas sobre diversas condições médicas e orientar pacientes.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.utils.logging_config import get_logger
from src.utils.validators import InputValidator
from .chains import MedicalChains
from .tools import MedicalTools

logger = get_logger(__name__)


class MedicalAssistant:
    """
    Assistente virtual médico generalista otimizado para BioMistral.
    Fornece informações educativas sobre diversas condições de saúde.
    """
    
    def __init__(self, model: Any = None, tokenizer: Any = None):
        """
        Inicializa o assistente médico generalista.
        
        Args:
            model: Modelo LLM treinado (BioMistral-7B)
            tokenizer: Tokenizer do modelo
        """
        self.model = model
        self.tokenizer = tokenizer
        self.validator = InputValidator()
        
        # Memória de conversação
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Chains e ferramentas
        self.chains = MedicalChains(model, tokenizer)
        self.tools = MedicalTools()
        
        # Prompt do sistema (incorporado no formato BioMistral)
        self.system_instruction = self._get_system_instruction()
        
        logger.info("MedicalAssistant (BioMistral) inicializado")
    
    def _get_system_instruction(self) -> str:
        """
        Retorna as diretrizes do sistema para o assistente.
        """
        return (
            "Você é um assistente médico técnico e atencioso. "
            "Forneça informações precisas baseadas em evidências. "
            "Não forneça diagnósticos definitivos. Responda em português brasileiro "
            "de forma concisa e profissional, baseando-se em protocolos clínicos."
        )
    
    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """
        Valida a entrada do usuário.
        """
        return self.validator.validate_query(user_input)
    
    def process_message(self, user_input: str) -> str:
        is_valid, message = self.validate_input(user_input)
        if not is_valid: return message

        if self.tools.is_emergency_question(user_input):
            return self._handle_emergency(user_input)

        # 1. Prompt com instrução de parada clara
        biomistral_prompt = (
            f"<s>[INST] Você é um assistente médico técnico. Responda de forma concisa e pare após concluir a resposta técnica. "
            f"Não repita cabeçalhos.\n\nPergunta: {user_input} [/INST]"
        )

        try:
            # 2. Geração com parâmetros anti-repetição
            # Aumentamos a repetition_penalty para 1.3 e baixamos a temperatura
            response = self.chains.qa_chain.invoke({
                "question": biomistral_prompt,
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
                "repetition_penalty": 1.3, # Impede o modelo de ficar em loop
                "temperature": 0.1,        # Torna a resposta mais "seca" e técnica
                "max_new_tokens": 300      # Limita o tamanho para não divagar
            })

            # 3. Limpeza agressiva de "lixo" de repetição
            # O BioMistral às vezes repete o bloco de contexto. Vamos cortar na primeira ocorrência repetida.
            clean_response = response.replace(biomistral_prompt, "").strip()
            
            # Se ele começar a repetir "Contexto:" ou "Resposta:", cortamos ali
            if "### Contexto" in clean_response:
                clean_response = clean_response.split("### Contexto")[0].strip()
            if "### Resposta" in clean_response:
                clean_response = clean_response.split("### Resposta")[0].strip()

            source_info = "\n\nFonte: Protocolos Clínicos / Dados Internos."
            response_with_source = f"{clean_response}{source_info}"

            self.memory.save_context({"input": user_input}, {"output": response_with_source})
            return response_with_source

        except Exception as e:
            logging.error(f"Erro: {e}")
            return "Erro ao processar. Tente novamente."
    
    def _handle_emergency(self, user_input: str) -> str:
        """
        Trata casos de emergência com resposta padrão de segurança.
        """
        return """⚠️ ATENÇÃO: Sua pergunta pode indicar uma situação de emergência.

Se você está enfrentando uma emergência médica:
1. Ligue imediatamente para 192 (SAMU) ou 193 (Bombeiros)
2. Vá ao pronto-socorro mais próximo
3. Não dirija se estiver se sentindo mal

Sintomas que requerem atendimento imediato:
- Dor intensa no peito
- Dificuldade para respirar
- Perda de consciência ou confusão mental
- Sangramento intenso
- Sinais de AVC (rosto caído, fraqueza, fala arrastada)
- Reações alérgicas graves
- Febre muito alta que não cede

Este assistente não substitui atendimento médico de emergência."""
    
    def clear_history(self) -> None:
        """
        Limpa o histórico de conversação.
        """
        self.memory.clear()
        logger.info("Histórico de conversação limpo")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retorna o histórico de conversação formatado.
        """
        history = self.memory.load_memory_variables({})["chat_history"]
        return [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
            for msg in history
        ]


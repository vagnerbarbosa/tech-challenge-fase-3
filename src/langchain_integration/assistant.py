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
        """
        Processa uma mensagem do usuário usando o formato de prompt do BioMistral.
        """
        # 1. Valida entrada
        is_valid, message = self.validate_input(user_input)
        if not is_valid:
            return message
        
        # 2. Verifica se é uma pergunta sobre emergência
        if self.tools.is_emergency_question(user_input):
            return self._handle_emergency(user_input)
        
        # 3. Formatação do Prompt para BioMistral (Mistral-7B Style)
        # O uso de <s>[INST] ... [/INST] é obrigatório para este modelo.
        biomistral_prompt = (
            f"<s>[INST] {self.system_instruction}\n\n"
            f"Pergunta: {user_input} [/INST]"
        )
        
        # 4. Processa através da chain principal
        try:
            # Carrega histórico para contexto (opcional, dependendo da sua qa_chain)
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            response = self.chains.qa_chain.invoke({
                "question": biomistral_prompt,
                "chat_history": chat_history,
            })
            
            # Limpeza: Remove o prompt da resposta caso o modelo o repita
            clean_response = response.replace(biomistral_prompt, "").strip()
            
            # Adiciona a fonte conforme solicitado no seu script anterior
            source_info = "\n\nFonte: Protocolos Clínicos / Diretrizes de Laudos (Treinamento Local)"
            final_output = f"{clean_response}{source_info}"
            
            # 5. Salva na memória (salvamos a pergunta limpa para não poluir o contexto futuro)
            self.memory.save_context(
                {"input": user_input},
                {"output": final_output}
            )
            
            return final_output
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."
    
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


if __name__ == "__main__":
    # Teste básico do assistente generalista
    assistant = MedicalAssistant()
    
    test_questions = [
        "Quais são os sintomas de uma gripe?",
        "Quando devo procurar um médico?",
        "Estou com dor forte no peito e dificuldade para respirar",
    ]
    
    for question in test_questions:
        print(f"\n👤 Usuário: {question}")
        response = assistant.process_message(question)
        print(f"🏥 Assistente: {response}")

"""
Assistente Virtual Médico
=========================

Implementa o assistente médico usando LangChain.
"""

import os
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
    Assistente virtual médico especializado em diabetes.
    """
    
    def __init__(self, model: Any = None, tokenizer: Any = None):
        """
        Inicializa o assistente médico.
        
        Args:
            model: Modelo LLM treinado
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
        
        # Prompt do sistema
        self.system_prompt = self._create_system_prompt()
        
        logger.info("MedicalAssistant inicializado")
    
    def _create_system_prompt(self) -> str:
        """
        Cria o prompt do sistema para o assistente.
        
        Returns:
            Prompt do sistema
        """
        return """Você é um assistente virtual médico especializado em diabetes.

Diretrizes:
1. Forneça informações precisas e baseadas em evidências científicas
2. Sempre recomende consultar um médico para diagnósticos e tratamentos
3. Não forneça diagnósticos - apenas informações educativas
4. Seja empático e acolhedor nas respostas
5. Use linguagem clara e acessível
6. Respeite a privacidade do paciente

Aviso importante: Este assistente fornece apenas informações educativas.
Para diagnósticos e tratamentos, sempre consulte um profissional de saúde.
"""
    
    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """
        Valida a entrada do usuário.
        
        Args:
            user_input: Texto de entrada do usuário
            
        Returns:
            Tupla (é_válido, mensagem)
        """
        return self.validator.validate_query(user_input)
    
    def process_message(self, user_input: str) -> str:
        """
        Processa uma mensagem do usuário.
        
        Args:
            user_input: Mensagem do usuário
            
        Returns:
            Resposta do assistente
        """
        # Valida entrada
        is_valid, message = self.validate_input(user_input)
        if not is_valid:
            return message
        
        # Verifica se é uma pergunta sobre emergência
        if self.tools.is_emergency_question(user_input):
            return self._handle_emergency(user_input)
        
        # Processa através da chain principal
        try:
            response = self.chains.qa_chain.invoke({
                "question": user_input,
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
            })
            
            # Salva na memória
            self.memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."
    
    def _handle_emergency(self, user_input: str) -> str:
        """
        Trata casos de emergência.
        
        Args:
            user_input: Mensagem do usuário
            
        Returns:
            Resposta de emergência
        """
        emergency_response = """⚠️ ATENÇÃO: Sua pergunta pode indicar uma situação de emergência.

Se você está enfrentando uma emergência médica:
1. Ligue imediatamente para 192 (SAMU) ou 193 (Bombeiros)
2. Vá ao pronto-socorro mais próximo
3. Não dirija se estiver se sentindo mal

Sintomas que requerem atendimento imediato:
- Confusão mental ou perda de consciência
- Glicemia muito alta (>400 mg/dL) ou muito baixa (<54 mg/dL)
- Dificuldade para respirar
- Dor no peito
- Vômitos persistentes

Este assistente não substitui atendimento médico de emergência."""
        
        return emergency_response
    
    def clear_history(self) -> None:
        """
        Limpa o histórico de conversação.
        """
        self.memory.clear()
        logger.info("Histórico de conversação limpo")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retorna o histórico de conversação.
        
        Returns:
            Lista de mensagens do histórico
        """
        history = self.memory.load_memory_variables({})["chat_history"]
        return [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
            for msg in history
        ]


if __name__ == "__main__":
    # Teste básico do assistente
    assistant = MedicalAssistant()
    
    test_questions = [
        "O que é diabetes?",
        "Quais são os sintomas?",
        "Estou me sentindo muito mal e confuso",
    ]
    
    for question in test_questions:
        print(f"\n👤 Usuário: {question}")
        response = assistant.process_message(question)
        print(f"🏥 Assistente: {response}")

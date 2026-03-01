"""
Assistente Médico com LangChain

Implementa o assistente virtual médico principal
utilizando LangChain para orquestração.
"""

import os
from typing import Optional, List, Dict, Any

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

from src.utils.logging_config import get_logger
from src.utils.validators import MedicalResponseValidator
from .chains import MedicalChain
from .tools import MedicalTools

logger = get_logger(__name__)


class MedicalAssistant:
    """Assistente virtual médico baseado em LangChain."""
    
    SYSTEM_PROMPT = """Você é um assistente médico virtual especializado em fornecer 
    informações gerais de saúde. Você NÃO substitui consultas médicas profissionais.
    
    Diretrizes:
    1. Sempre recomende buscar um profissional de saúde para diagnósticos
    2. Forneça informações baseadas em evidências científicas
    3. Seja empático e acolhedor nas respostas
    4. Nunca prescreva medicamentos ou tratamentos específicos
    5. Em caso de emergência, oriente a buscar atendimento imediato
    
    Histórico da conversa:
    {history}
    
    Paciente: {input}
    Assistente: """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o assistente médico.
        
        Args:
            model_path: Caminho para o modelo fine-tuned (opcional)
        """
        self.model_path = model_path or "./models/assistente-medico-final"
        self.validator = MedicalResponseValidator()
        self.tools = MedicalTools()
        
        # Inicializa memória de conversação
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Mantém últimas 5 interações
            return_messages=True
        )
        
        # Inicializa chain
        self._setup_chain()
        
        logger.info("Assistente médico inicializado")
    
    def _setup_chain(self):
        """Configura a chain do LangChain."""
        # Template de prompt
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=self.SYSTEM_PROMPT
        )
        
        # Inicializa a chain médica
        self.medical_chain = MedicalChain(model_path=self.model_path)
        
        # Chain de conversação com memória
        self.conversation = ConversationChain(
            llm=self.medical_chain.get_llm(),
            memory=self.memory,
            prompt=self.prompt,
            verbose=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def respond(self, user_input: str) -> str:
        """
        Gera resposta para entrada do usuário.
        
        Args:
            user_input: Mensagem do usuário
            
        Returns:
            Resposta do assistente
        """
        logger.info(f"Processando entrada: {user_input[:50]}...")
        
        try:
            # Verifica se é emergência
            if self._is_emergency(user_input):
                return self._emergency_response()
            
            # Gera resposta via chain
            response = self.conversation.predict(input=user_input)
            
            # Valida resposta
            validated_response = self.validator.validate_and_clean(response)
            
            logger.info("Resposta gerada com sucesso")
            return validated_response
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return self._fallback_response()
    
    def _is_emergency(self, text: str) -> bool:
        """Verifica se a mensagem indica emergência médica."""
        emergency_keywords = [
            "infarto", "avc", "derrame", "não consigo respirar",
            "desmaio", "convulsão", "sangramento intenso",
            "overdose", "suicídio", "acidente grave"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in emergency_keywords)
    
    def _emergency_response(self) -> str:
        """Resposta para situações de emergência."""
        return """🚨 ATENÇÃO: Se você ou alguém está em uma situação de emergência médica, 
        por favor ligue IMEDIATAMENTE para o SAMU (192) ou dirija-se ao pronto-socorro mais próximo.
        
        Em casos de emergência, cada segundo conta. Não espere - busque ajuda profissional agora!"""
    
    def _fallback_response(self) -> str:
        """Resposta padrão em caso de erro."""
        return """Desculpe, não consegui processar sua mensagem no momento. 
        Por favor, tente reformular sua pergunta ou, se for urgente, 
        consulte um profissional de saúde."""
    
    def clear_history(self):
        """Limpa histórico de conversação."""
        self.memory.clear()
        logger.info("Histórico de conversação limpo")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Retorna histórico da conversa."""
        return self.memory.chat_memory.messages


if __name__ == "__main__":
    assistant = MedicalAssistant()
    
    # Teste simples
    response = assistant.respond("Quais são os sintomas da gripe?")
    print(response)

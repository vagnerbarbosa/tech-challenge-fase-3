"""
Assistente Virtual Médico Generalista (BioMistral Edition)
=========================================================

Implementa o assistente médico generalista usando LangChain e BioMistral-7B.
Capaz de responder perguntas sobre diversas condições médicas e orientar pacientes.

Integra:
- RAG (Retrieval-Augmented Generation) para contexto de protocolos médicos
- Base de dados simulada de prontuários para contextualização do paciente
- Citação de fontes (Explainability) em todas as respostas
"""

import os
import logging
from typing import Any, Dict, List, Optional

try:
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.utils.logging_config import get_logger
from src.utils.validators import InputValidator
from .chains import MedicalChains
from .tools import MedicalTools
from .rag import MedicalRAG

logger = get_logger(__name__)


class MedicalAssistant:
    """
    Assistente virtual médico generalista otimizado para BioMistral.
    Fornece informações educativas sobre diversas condições de saúde.
    
    Integra:
    - RAG para busca em protocolos médicos
    - Base de prontuários para contextualização do paciente
    - Citação de fontes nas respostas (explainability)
    """
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        patient_id: Optional[int] = None,
        enable_rag: bool = True,
    ):
        """
        Inicializa o assistente médico generalista.
        
        Args:
            model: Modelo LLM treinado (BioMistral-7B)
            tokenizer: Tokenizer do modelo
            patient_id: ID do paciente para contextualização (opcional)
            enable_rag: Se True, habilita busca RAG nos protocolos
        """
        self.model = model
        self.tokenizer = tokenizer
        self.validator = InputValidator()
        self.patient_id = patient_id
        
        # Memória de conversação
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Chains e ferramentas
        self.chains = MedicalChains(model, tokenizer)
        self.tools = MedicalTools()
        
        # Sistema RAG
        self.rag = None
        if enable_rag:
            try:
                self.rag = MedicalRAG()
                logger.info(f"RAG inicializado: {self.rag.get_stats()}")
            except Exception as e:
                logger.warning(f"RAG não pôde ser inicializado: {e}")
        
        # Base de prontuários
        self.patient_db = None
        try:
            from src.database.patient_records import PatientDatabase
            self.patient_db = PatientDatabase()
            logger.info("Base de prontuários inicializada")
        except Exception as e:
            logger.warning(f"Base de prontuários não disponível: {e}")
        
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
            "de forma concisa e profissional, baseando-se em protocolos clínicos. "
            "Sempre cite as fontes das informações fornecidas."
        )
    
    def set_patient(self, patient_id: int) -> Optional[str]:
        """
        Define o paciente ativo para contextualização.
        
        Args:
            patient_id: ID do paciente na base de dados
            
        Returns:
            Resumo do prontuário ou None se não encontrado
        """
        self.patient_id = patient_id
        if self.patient_db:
            summary = self.patient_db.get_patient_summary(patient_id)
            if summary:
                logger.info(f"Paciente {patient_id} carregado para contextualização")
                return summary
            logger.warning(f"Paciente {patient_id} não encontrado na base")
        return None
    
    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """
        Valida a entrada do usuário.
        """
        return self.validator.validate_query(user_input)
    
    def process_message(self, user_input: str) -> str:
        is_valid, message = self.validate_input(user_input)
        if not is_valid:
            return message

        if self.tools.is_emergency_question(user_input):
            return self._handle_emergency(user_input)

        # 1. Recuperar contexto RAG e prontuário do paciente
        rag_context = ""
        sources = []
        patient_context = None

        # Contexto do prontuário
        if self.patient_id and self.patient_db:
            try:
                patient_context = self.patient_db.get_patient_context_for_query(
                    self.patient_id, user_input
                )
            except Exception as e:
                logger.warning(f"Erro ao recuperar contexto do paciente: {e}")

        # Busca RAG nos protocolos médicos
        if self.rag:
            try:
                rag_context, sources = self.rag.get_context_for_query(
                    query=user_input,
                    top_k=3,
                    patient_context=patient_context,
                )
            except Exception as e:
                logger.warning(f"Erro na busca RAG: {e}")

        # 2. Construir prompt com contexto
        context_block = ""
        if rag_context:
            context_block = f"\n\n[CONTEXTO RELEVANTE]\n{rag_context}\n"

        biomistral_prompt = (
            f"<s>[INST] Você é um assistente médico técnico. "
            f"Responda de forma concisa e pare após concluir a resposta técnica. "
            f"Não repita cabeçalhos. "
            f"Baseie sua resposta no contexto fornecido quando disponível."
            f"{context_block}"
            f"\nPergunta: {user_input} [/INST]"
        )

        try:
            # 3. Geração com parâmetros anti-repetição
            response = self.chains.qa_chain.invoke({
                "question": biomistral_prompt,
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
                "repetition_penalty": 1.3,
                "temperature": 0.1,
                "max_new_tokens": 300
            })

            # 4. Limpeza de repetição
            clean_response = response.replace(biomistral_prompt, "").strip()
            
            if "### Contexto" in clean_response:
                clean_response = clean_response.split("### Contexto")[0].strip()
            if "### Resposta" in clean_response:
                clean_response = clean_response.split("### Resposta")[0].strip()

            # 5. Adicionar citações de fontes (Explainability)
            citations = self._format_source_citations(sources)
            response_with_source = f"{clean_response}{citations}"

            self.memory.save_context({"input": user_input}, {"output": response_with_source})
            return response_with_source

        except Exception as e:
            logging.error(f"Erro: {e}")
            return "Erro ao processar. Tente novamente."
    
    def _format_source_citations(self, sources: List[Dict[str, str]]) -> str:
        """
        Formata citações de fontes para inclusão na resposta.
        Implementa o requisito de Explainability (R3).
        
        Args:
            sources: Lista de fontes recuperadas pelo RAG
            
        Returns:
            String formatada com citações
        """
        if not sources:
            return "\n\n📚 **Fonte:** Conhecimento médico geral do modelo."
        
        if self.rag:
            return self.rag.format_citations(sources)
        
        return "\n\n📚 **Fonte:** Protocolos Clínicos / Dados Internos."
    
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

Este assistente não substitui atendimento médico de emergência.

📚 **Fonte:** Protocolo de Emergências Médicas - SAMU/MS."""
    
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

    def list_patients(self) -> List[Dict[str, Any]]:
        """
        Lista pacientes disponíveis na base de dados.
        
        Returns:
            Lista resumida de pacientes
        """
        if self.patient_db:
            return self.patient_db.list_patients_brief()
        return []

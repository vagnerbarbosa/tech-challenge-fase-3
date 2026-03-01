"""
Workflow Médico com LangGraph
=============================

Implementa fluxos automatizados para o assistente médico.
"""

from typing import Any, Dict, TypedDict, Annotated, Sequence
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.utils.logging_config import get_logger
from src.langchain_integration.tools import MedicalTools

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Tipos de mensagem no workflow."""
    GREETING = "greeting"
    QUESTION = "question"
    EMERGENCY = "emergency"
    GLYCEMIA = "glycemia"
    GENERAL = "general"
    FAREWELL = "farewell"


class WorkflowState(TypedDict):
    """Estado do workflow."""
    messages: Sequence[BaseMessage]
    message_type: str
    user_input: str
    response: str
    glycemia_value: float
    requires_followup: bool


class MedicalWorkflow:
    """
    Workflow médico usando LangGraph.
    """
    
    def __init__(self, assistant: Any = None):
        """
        Inicializa o workflow.
        
        Args:
            assistant: Instância do MedicalAssistant
        """
        self.assistant = assistant
        self.tools = MedicalTools()
        self.graph = self._build_graph()
        
        logger.info("MedicalWorkflow inicializado")
    
    def _build_graph(self) -> StateGraph:
        """
        Constrói o grafo de workflow.
        
        Returns:
            StateGraph configurado
        """
        # Cria o grafo
        workflow = StateGraph(WorkflowState)
        
        # Adiciona nós
        workflow.add_node("classify", self._classify_message)
        workflow.add_node("handle_greeting", self._handle_greeting)
        workflow.add_node("handle_emergency", self._handle_emergency)
        workflow.add_node("handle_glycemia", self._handle_glycemia)
        workflow.add_node("handle_question", self._handle_question)
        workflow.add_node("handle_farewell", self._handle_farewell)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define entrada
        workflow.set_entry_point("classify")
        
        # Adiciona arestas condicionais
        workflow.add_conditional_edges(
            "classify",
            self._route_message,
            {
                MessageType.GREETING: "handle_greeting",
                MessageType.EMERGENCY: "handle_emergency",
                MessageType.GLYCEMIA: "handle_glycemia",
                MessageType.QUESTION: "handle_question",
                MessageType.FAREWELL: "handle_farewell",
                MessageType.GENERAL: "handle_question",
            }
        )
        
        # Conecta handlers ao gerador de resposta
        for node in ["handle_greeting", "handle_emergency", "handle_glycemia", 
                     "handle_question", "handle_farewell"]:
            workflow.add_edge(node, "generate_response")
        
        # Finaliza
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _classify_message(self, state: WorkflowState) -> WorkflowState:
        """
        Classifica o tipo de mensagem.
        
        Args:
            state: Estado atual
            
        Returns:
            Estado atualizado
        """
        user_input = state["user_input"].lower()
        
        # Verifica tipo de mensagem
        if any(word in user_input for word in ["olá", "oi", "bom dia", "boa tarde", "boa noite"]):
            state["message_type"] = MessageType.GREETING
        elif any(word in user_input for word in ["tchau", "adeus", "até logo", "obrigado"]):
            state["message_type"] = MessageType.FAREWELL
        elif self.tools.is_emergency_question(user_input):
            state["message_type"] = MessageType.EMERGENCY
        elif self.tools.extract_glycemia_value(user_input) is not None:
            state["message_type"] = MessageType.GLYCEMIA
            state["glycemia_value"] = self.tools.extract_glycemia_value(user_input)
        else:
            state["message_type"] = MessageType.QUESTION
        
        logger.debug(f"Mensagem classificada como: {state['message_type']}")
        
        return state
    
    def _route_message(self, state: WorkflowState) -> MessageType:
        """
        Roteia a mensagem para o handler apropriado.
        
        Args:
            state: Estado atual
            
        Returns:
            Tipo de mensagem para roteamento
        """
        return state["message_type"]
    
    def _handle_greeting(self, state: WorkflowState) -> WorkflowState:
        """
        Trata saudações.
        """
        state["response"] = """Olá! 👋 Sou seu assistente virtual médico especializado em diabetes.

Posso ajudá-lo com:
• Informações sobre diabetes tipo 1 e tipo 2
• Dúvidas sobre sintomas e prevenção
• Orientações sobre alimentação e estilo de vida
• Interpretação de valores de glicemia

Como posso ajudá-lo hoje?

⚠️ Lembre-se: não substituo uma consulta médica profissional."""
        
        return state
    
    def _handle_emergency(self, state: WorkflowState) -> WorkflowState:
        """
        Trata emergências.
        """
        state["response"] = """🚨 ALERTA DE EMERGÊNCIA

Identifiquei que sua mensagem pode indicar uma situação urgente.

AÇÕES IMEDIATAS:
1. 📞 Ligue 192 (SAMU) ou 193 (Bombeiros)
2. 🏥 Vá ao pronto-socorro mais próximo
3. ❌ NÃO dirija se estiver se sentindo mal

Sintomas que requerem atendimento imediato:
• Confusão mental ou desmaio
• Glicemia < 54 mg/dL ou > 400 mg/dL
• Dificuldade respiratória
• Vômitos persistentes

Este assistente NÃO substitui atendimento de emergência!"""
        
        return state
    
    def _handle_glycemia(self, state: WorkflowState) -> WorkflowState:
        """
        Trata valores de glicemia.
        """
        value = state.get("glycemia_value", 0)
        result = self.tools.interpret_glycemia(value)
        
        alert_emoji = {
            "normal": "✅",
            "warning": "⚠️",
            "high": "🔴",
            "critical": "🚨",
        }
        
        emoji = alert_emoji.get(result["alert_level"], "ℹ️")
        
        state["response"] = f"""📊 ANÁLISE DE GLICEMIA

Valor informado: {value} mg/dL

{emoji} Classificação: {result['classification']}

💡 Recomendação: {result['recommendation']}

📌 Valores de referência (jejum):
• Normal: 70-99 mg/dL
• Pré-diabetes: 100-125 mg/dL
• Indicativo de diabetes: ≥126 mg/dL

⚠️ Esta análise é apenas informativa.
Consulte seu médico para avaliação completa."""
        
        return state
    
    def _handle_question(self, state: WorkflowState) -> WorkflowState:
        """
        Trata perguntas gerais.
        """
        if self.assistant:
            state["response"] = self.assistant.process_message(state["user_input"])
        else:
            state["response"] = f"""Obrigado pela sua pergunta!

Você perguntou: "{state['user_input']}"

Como assistente virtual médico especializado em diabetes, posso fornecer informações educativas sobre a condição.

Para uma resposta mais completa e personalizada, recomendo:
1. Consultar um médico endocrinologista
2. Levar suas dúvidas anotadas para a consulta
3. Compartilhar seus exames recentes com o profissional

Posso ajudar com mais alguma dúvida?"""
        
        return state
    
    def _handle_farewell(self, state: WorkflowState) -> WorkflowState:
        """
        Trata despedidas.
        """
        state["response"] = """Obrigado por conversar comigo! 👋

Lembre-se:
✅ Mantenha suas consultas médicas em dia
✅ Monitore sua glicemia regularmente
✅ Siga uma alimentação equilibrada
✅ Pratique exercícios físicos

Cuide-se bem! Estarei aqui sempre que precisar. 💙"""
        
        return state
    
    def _generate_response(self, state: WorkflowState) -> WorkflowState:
        """
        Gera a resposta final.
        """
        # Adiciona mensagem ao histórico
        state["messages"] = list(state.get("messages", [])) + [
            HumanMessage(content=state["user_input"]),
            AIMessage(content=state["response"]),
        ]
        
        return state
    
    def process(self, user_input: str) -> str:
        """
        Processa uma mensagem através do workflow.
        
        Args:
            user_input: Mensagem do usuário
            
        Returns:
            Resposta do assistente
        """
        initial_state: WorkflowState = {
            "messages": [],
            "message_type": MessageType.GENERAL,
            "user_input": user_input,
            "response": "",
            "glycemia_value": 0.0,
            "requires_followup": False,
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return result["response"]
        except Exception as e:
            logger.error(f"Erro no workflow: {e}")
            return "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente."


if __name__ == "__main__":
    # Teste do workflow
    workflow = MedicalWorkflow()
    
    test_messages = [
        "Olá!",
        "O que é diabetes?",
        "Minha glicemia está em 180 mg/dL",
        "Estou me sentindo muito confuso",
        "Tchau!",
    ]
    
    for msg in test_messages:
        print(f"\n👤 Usuário: {msg}")
        response = workflow.process(msg)
        print(f"🏥 Assistente: {response}")

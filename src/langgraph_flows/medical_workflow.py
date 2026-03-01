"""
Workflow Médico com LangGraph
=============================

Implementa fluxos automatizados para o assistente médico generalista.
Suporta diversos tipos de consultas médicas e orientações de saúde.
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
    VITAL_SIGNS = "vital_signs"
    GENERAL = "general"
    FAREWELL = "farewell"


class WorkflowState(TypedDict):
    """Estado do workflow."""
    messages: Sequence[BaseMessage]
    message_type: str
    user_input: str
    response: str
    temperature_value: float
    requires_followup: bool


class MedicalWorkflow:
    """
    Workflow médico generalista usando LangGraph.
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
        
        logger.info("MedicalWorkflow (Generalista) inicializado")
    
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
        workflow.add_node("handle_vital_signs", self._handle_vital_signs)
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
                MessageType.VITAL_SIGNS: "handle_vital_signs",
                MessageType.QUESTION: "handle_question",
                MessageType.FAREWELL: "handle_farewell",
                MessageType.GENERAL: "handle_question",
            }
        )
        
        # Conecta handlers ao gerador de resposta
        for node in ["handle_greeting", "handle_emergency", "handle_vital_signs", 
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
        elif self.tools.extract_temperature_value(user_input) is not None:
            state["message_type"] = MessageType.VITAL_SIGNS
            state["temperature_value"] = self.tools.extract_temperature_value(user_input)
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
        state["response"] = """Olá! 👋 Sou seu assistente virtual médico generalista.

Posso ajudá-lo com:
• Informações sobre sintomas e condições de saúde
• Orientações gerais de prevenção e bem-estar
• Dicas de quando procurar atendimento médico
• Esclarecimentos sobre exames e procedimentos
• Orientação sobre especialidades médicas

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
• Dor intensa no peito
• Dificuldade respiratória severa
• Perda de consciência ou confusão mental
• Sangramento intenso que não para
• Sinais de AVC (rosto caído, fraqueza em um lado, fala arrastada)
• Reações alérgicas graves (inchaço, dificuldade de respirar)
• Febre muito alta (> 39.5°C) que não cede

Este assistente NÃO substitui atendimento de emergência!"""
        
        return state
    
    def _handle_vital_signs(self, state: WorkflowState) -> WorkflowState:
        """
        Trata valores de sinais vitais (como temperatura).
        """
        value = state.get("temperature_value", 0)
        result = self.tools.interpret_temperature(value)
        
        alert_emoji = {
            "normal": "✅",
            "warning": "⚠️",
            "high": "🔴",
            "critical": "🚨",
        }
        
        emoji = alert_emoji.get(result["alert_level"], "ℹ️")
        
        state["response"] = f"""📊 ANÁLISE DE TEMPERATURA CORPORAL

Valor informado: {value}°C

{emoji} Classificação: {result['classification']}

💡 Recomendação: {result['recommendation']}

📌 Valores de referência:
• Normal: 36.1-37.2°C
• Febrícula: 37.3-37.8°C
• Febre: 37.9-39°C
• Febre alta: > 39°C

⚠️ Esta análise é apenas informativa.
Consulte seu médico para avaliação completa, especialmente se houver outros sintomas."""
        
        return state
    
    def _handle_question(self, state: WorkflowState) -> WorkflowState:
        """
        Trata perguntas gerais de saúde.
        """
        # Verifica se pode sugerir especialidade
        specialty = self.tools.suggest_specialty(state["user_input"])
        specialty_hint = ""
        if specialty:
            specialty_hint = f"\n\n💡 Com base nos sintomas descritos, um(a) {specialty} pode ser indicado(a) para avaliação."
        
        if self.assistant:
            state["response"] = self.assistant.process_message(state["user_input"]) + specialty_hint
        else:
            state["response"] = f"""Obrigado pela sua pergunta!

Você perguntou: "{state['user_input']}"

Como assistente virtual médico generalista, posso fornecer informações educativas sobre saúde e bem-estar.

Para uma resposta mais completa e personalizada, recomendo:
1. Consultar um médico adequado para sua necessidade
2. Levar suas dúvidas anotadas para a consulta
3. Compartilhar seus exames recentes com o profissional{specialty_hint}

Posso ajudar com mais alguma dúvida?"""
        
        return state
    
    def _handle_farewell(self, state: WorkflowState) -> WorkflowState:
        """
        Trata despedidas.
        """
        state["response"] = """Obrigado por conversar comigo! 👋

Lembre-se das dicas de saúde:
✅ Mantenha suas consultas médicas em dia
✅ Pratique exercícios físicos regularmente
✅ Alimente-se de forma equilibrada
✅ Durma bem (7-9 horas por noite)
✅ Cuide da sua saúde mental
✅ Mantenha suas vacinas atualizadas

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
            "temperature_value": 0.0,
            "requires_followup": False,
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return result["response"]
        except Exception as e:
            logger.error(f"Erro no workflow: {e}")
            return "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente."


if __name__ == "__main__":
    # Teste do workflow generalista
    workflow = MedicalWorkflow()
    
    test_messages = [
        "Olá!",
        "Quais são os sintomas de uma gripe?",
        "Minha temperatura está em 38.5 graus",
        "Estou com dor forte no peito",
        "Tchau!",
    ]
    
    for msg in test_messages:
        print(f"\n👤 Usuário: {msg}")
        response = workflow.process(msg)
        print(f"🏥 Assistente: {response}")

"""
Workflow Médico com LangGraph

Implementa fluxos automatizados para triagem e
orientação médica usando LangGraph.
"""

import os
from typing import Dict, TypedDict, Annotated, List, Optional
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate

from src.utils.logging_config import get_logger
from src.langchain_integration.chains import MedicalChain

logger = get_logger(__name__)


class TriageLevel(str, Enum):
    """Níveis de triagem médica."""
    EMERGENCY = "emergency"      # Vermelho - Emergência
    URGENT = "urgent"            # Laranja - Muito urgente
    LESS_URGENT = "less_urgent"  # Amarelo - Urgente
    NOT_URGENT = "not_urgent"    # Verde - Pouco urgente
    NON_URGENT = "non_urgent"    # Azul - Não urgente


class PatientState(TypedDict):
    """Estado do paciente no workflow."""
    patient_id: str
    symptoms: str
    symptom_duration: Optional[str]
    pain_level: Optional[int]
    vital_signs: Optional[Dict[str, float]]
    triage_level: Optional[TriageLevel]
    recommendations: List[str]
    medical_history: Optional[str]
    requires_emergency: bool
    response: str


class MedicalWorkflow:
    """Workflow de triagem médica com LangGraph."""
    
    def __init__(self):
        self.medical_chain = MedicalChain()
        self.graph = self._build_graph()
        
        logger.info("Workflow médico inicializado")
    
    def _build_graph(self) -> StateGraph:
        """Constrói o grafo de workflow."""
        # Cria o grafo
        workflow = StateGraph(PatientState)
        
        # Adiciona nós
        workflow.add_node("collect_symptoms", self._collect_symptoms)
        workflow.add_node("check_emergency", self._check_emergency)
        workflow.add_node("classify_triage", self._classify_triage)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("emergency_response", self._emergency_response)
        workflow.add_node("final_response", self._final_response)
        
        # Define ponto de entrada
        workflow.set_entry_point("collect_symptoms")
        
        # Adiciona arestas
        workflow.add_edge("collect_symptoms", "check_emergency")
        
        # Condicional: emergência ou triagem normal
        workflow.add_conditional_edges(
            "check_emergency",
            self._route_after_emergency_check,
            {
                "emergency": "emergency_response",
                "continue": "classify_triage"
            }
        )
        
        workflow.add_edge("classify_triage", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "final_response")
        workflow.add_edge("emergency_response", END)
        workflow.add_edge("final_response", END)
        
        return workflow.compile()
    
    def _collect_symptoms(self, state: PatientState) -> PatientState:
        """Coleta e processa sintomas."""
        logger.info("Coletando sintomas do paciente")
        
        # Processa sintomas com o LLM
        symptoms = state.get("symptoms", "")
        
        # Aqui poderia usar NER para extrair sintomas estruturados
        # Por enquanto, apenas passa os sintomas adiante
        
        return {
            **state,
            "symptoms": symptoms,
            "recommendations": []
        }
    
    def _check_emergency(self, state: PatientState) -> PatientState:
        """Verifica se é emergência."""
        logger.info("Verificando sinais de emergência")
        
        emergency_keywords = [
            "não consigo respirar", "dor no peito forte", "desmaio",
            "perda de consciência", "convulsão", "sangramento intenso",
            "acidente grave", "overdose", "envenenamento"
        ]
        
        symptoms_lower = state.get("symptoms", "").lower()
        is_emergency = any(kw in symptoms_lower for kw in emergency_keywords)
        
        # Verifica nível de dor
        pain_level = state.get("pain_level", 0)
        if pain_level and pain_level >= 9:
            is_emergency = True
        
        return {
            **state,
            "requires_emergency": is_emergency
        }
    
    def _route_after_emergency_check(self, state: PatientState) -> str:
        """Decide a rota após verificação de emergência."""
        if state.get("requires_emergency", False):
            return "emergency"
        return "continue"
    
    def _classify_triage(self, state: PatientState) -> PatientState:
        """Classifica nível de triagem."""
        logger.info("Classificando nível de triagem")
        
        pain_level = state.get("pain_level", 0) or 0
        duration = state.get("symptom_duration", "")
        
        # Lógica simplificada de triagem
        if pain_level >= 7:
            triage = TriageLevel.URGENT
        elif pain_level >= 5 or "dias" not in duration.lower():
            triage = TriageLevel.LESS_URGENT
        elif pain_level >= 3:
            triage = TriageLevel.NOT_URGENT
        else:
            triage = TriageLevel.NON_URGENT
        
        return {
            **state,
            "triage_level": triage
        }
    
    def _generate_recommendations(self, state: PatientState) -> PatientState:
        """Gera recomendações baseadas na triagem."""
        logger.info("Gerando recomendações")
        
        triage = state.get("triage_level", TriageLevel.NON_URGENT)
        recommendations = []
        
        triage_recommendations = {
            TriageLevel.URGENT: [
                "Procure um pronto-socorro em até 1 hora",
                "Não dirija sozinho se estiver com muita dor",
                "Leve documentos e lista de medicamentos em uso"
            ],
            TriageLevel.LESS_URGENT: [
                "Agende uma consulta médica para hoje ou amanhã",
                "Monitore seus sintomas e anote mudanças",
                "Se os sintomas piorarem, procure atendimento imediato"
            ],
            TriageLevel.NOT_URGENT: [
                "Agende uma consulta médica nos próximos dias",
                "Mantenha repouso e hidratação adequada",
                "Observe a evolução dos sintomas"
            ],
            TriageLevel.NON_URGENT: [
                "Você pode agendar uma consulta de rotina",
                "Mantenha hábitos saudáveis de vida",
                "Se novos sintomas surgirem, reavalie"
            ]
        }
        
        recommendations = triage_recommendations.get(triage, [])
        
        return {
            **state,
            "recommendations": recommendations
        }
    
    def _emergency_response(self, state: PatientState) -> PatientState:
        """Gera resposta de emergência."""
        logger.warning("EMERGÊNCIA DETECTADA - Gerando resposta urgente")
        
        response = """🚨 ATENÇÃO - POSSÍVEL EMERGÊNCIA MÉDICA!
        
        Baseado nos sintomas descritos, recomendamos:
        
        1. LIGUE IMEDIATAMENTE PARA O SAMU: 192
        2. Ou vá ao pronto-socorro mais próximo
        3. Se possível, peça para alguém te acompanhar
        4. Não dirija se estiver com sintomas graves
        
        Em caso de dúvida, é sempre melhor buscar atendimento.
        Sua saúde é prioridade!"""
        
        return {
            **state,
            "triage_level": TriageLevel.EMERGENCY,
            "response": response
        }
    
    def _final_response(self, state: PatientState) -> PatientState:
        """Gera resposta final com recomendações."""
        logger.info("Gerando resposta final")
        
        triage = state.get("triage_level", TriageLevel.NON_URGENT)
        recommendations = state.get("recommendations", [])
        
        triage_labels = {
            TriageLevel.EMERGENCY: "🔴 Emergência",
            TriageLevel.URGENT: "🟠 Urgente",
            TriageLevel.LESS_URGENT: "🟡 Menos Urgente",
            TriageLevel.NOT_URGENT: "🟢 Pouco Urgente",
            TriageLevel.NON_URGENT: "🔵 Não Urgente"
        }
        
        response = f"""📋 Resultado da Triagem Virtual
        
        Classificação: {triage_labels.get(triage, 'N/A')}
        
        Recomendações:
        """
        
        for i, rec in enumerate(recommendations, 1):
            response += f"\n        {i}. {rec}"
        
        response += """\n        
        ⚠️ Lembre-se: Esta é uma orientação inicial automatizada.
        Ela não substitui a avaliação de um profissional de saúde.
        Em caso de dúvida, sempre procure atendimento médico."""
        
        return {
            **state,
            "response": response
        }
    
    def run(self, patient_data: Optional[Dict] = None) -> Dict:
        """Executa o workflow completo."""
        logger.info("Executando workflow médico")
        
        # Estado inicial
        initial_state: PatientState = {
            "patient_id": patient_data.get("patient_id", "anonymous") if patient_data else "anonymous",
            "symptoms": patient_data.get("symptoms", "") if patient_data else "",
            "symptom_duration": patient_data.get("duration") if patient_data else None,
            "pain_level": patient_data.get("pain_level") if patient_data else None,
            "vital_signs": patient_data.get("vital_signs") if patient_data else None,
            "triage_level": None,
            "recommendations": [],
            "medical_history": patient_data.get("history") if patient_data else None,
            "requires_emergency": False,
            "response": ""
        }
        
        # Executa o grafo
        final_state = self.graph.invoke(initial_state)
        
        logger.info(f"Workflow concluído - Triagem: {final_state.get('triage_level')}")
        
        return final_state


if __name__ == "__main__":
    # Teste do workflow
    workflow = MedicalWorkflow()
    
    # Caso de teste
    patient = {
        "patient_id": "test123",
        "symptoms": "Dor de cabeça forte há 2 dias, com náusea",
        "duration": "2 dias",
        "pain_level": 6
    }
    
    result = workflow.run(patient)
    print(result["response"])

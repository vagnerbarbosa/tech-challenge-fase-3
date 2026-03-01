"""
Ferramentas Customizadas para o Assistente Médico

Implementa tools do LangChain para funcionalidades específicas.
"""

from typing import Optional, Type, Dict, Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SymptomCheckerInput(BaseModel):
    """Input para checagem de sintomas."""
    symptoms: str = Field(description="Lista de sintomas separados por vírgula")


class SymptomCheckerTool(BaseTool):
    """Ferramenta para análise inicial de sintomas."""
    
    name: str = "symptom_checker"
    description: str = """Útil para quando o usuário descreve sintomas e quer entender 
    possíveis condições associadas. NÃO fornece diagnósticos, apenas informações educacionais."""
    args_schema: Type[BaseModel] = SymptomCheckerInput
    
    def _run(self, symptoms: str) -> str:
        """Executa análise de sintomas."""
        logger.info(f"Analisando sintomas: {symptoms}")
        
        # TODO: Integrar com base de conhecimento médico
        # Por enquanto, retorna orientação genérica
        
        return f"""Você mencionou os seguintes sintomas: {symptoms}
        
        É importante que você consulte um profissional de saúde para uma avaliação adequada.
        Esses sintomas podem estar relacionados a diversas condições e apenas um médico 
        pode fazer um diagnóstico correto após exame clínico.
        
        Recomendações:
        - Anote quando os sintomas começaram
        - Observe se há fatores que melhoram ou pioram
        - Procure atendimento médico em breve"""
    
    async def _arun(self, symptoms: str) -> str:
        """Versão assíncrona."""
        return self._run(symptoms)


class MedicationInfoInput(BaseModel):
    """Input para informações de medicamentos."""
    medication_name: str = Field(description="Nome do medicamento")


class MedicationInfoTool(BaseTool):
    """Ferramenta para informações sobre medicamentos."""
    
    name: str = "medication_info"
    description: str = """Fornece informações gerais sobre medicamentos. 
    NÃO prescreve ou recomenda doses."""
    args_schema: Type[BaseModel] = MedicationInfoInput
    
    def _run(self, medication_name: str) -> str:
        """Busca informações do medicamento."""
        logger.info(f"Buscando info de: {medication_name}")
        
        # TODO: Integrar com base de dados de medicamentos (ex: ANVISA)
        
        return f"""Informações sobre {medication_name}:
        
        ⚠️ IMPORTANTE: Estas são informações gerais para fins educacionais.
        Sempre siga a prescrição do seu médico.
        
        Para informações detalhadas sobre indicações, contraindicações e posologia,
        consulte a bula do medicamento ou um profissional de saúde."""
    
    async def _arun(self, medication_name: str) -> str:
        return self._run(medication_name)


class EmergencyCheckInput(BaseModel):
    """Input para verificação de emergência."""
    situation: str = Field(description="Descrição da situação")


class EmergencyCheckTool(BaseTool):
    """Ferramenta para identificar emergências médicas."""
    
    name: str = "emergency_check"
    description: str = "Verifica se uma situação requer atendimento de emergência."
    args_schema: Type[BaseModel] = EmergencyCheckInput
    
    EMERGENCY_KEYWORDS = [
        "infarto", "avc", "derrame", "não respira", "inconsciente",
        "convulsão", "sangramento", "engasgo", "overdose", "envenenamento"
    ]
    
    def _run(self, situation: str) -> str:
        """Verifica emergência."""
        logger.info(f"Verificando emergência: {situation}")
        
        situation_lower = situation.lower()
        is_emergency = any(kw in situation_lower for kw in self.EMERGENCY_KEYWORDS)
        
        if is_emergency:
            return """🚨 EMERGÊNCIA IDENTIFICADA!
            
            LIGUE IMEDIATAMENTE:
            - SAMU: 192
            - Bombeiros: 193
            - Polícia: 190
            
            Ou dirija-se ao pronto-socorro mais próximo.
            Cada segundo conta em uma emergência!"""
        
        return """Esta situação não foi identificada como emergência imediata.
        No entanto, se você sentir que precisa de ajuda urgente, 
        não hesite em ligar para o SAMU (192) ou ir ao pronto-socorro."""
    
    async def _arun(self, situation: str) -> str:
        return self._run(situation)


class MedicalTools:
    """Coleção de ferramentas médicas."""
    
    def __init__(self):
        self.tools = [
            SymptomCheckerTool(),
            MedicationInfoTool(),
            EmergencyCheckTool()
        ]
    
    def get_tools(self):
        """Retorna lista de ferramentas."""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Retorna ferramenta pelo nome."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


if __name__ == "__main__":
    tools = MedicalTools()
    
    # Testa ferramenta de emergência
    emergency_tool = tools.get_tool_by_name("emergency_check")
    result = emergency_tool.run("Estou com dor no peito forte")
    print(result)

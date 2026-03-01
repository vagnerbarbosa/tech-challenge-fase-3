"""
Chains do LangChain para o Assistente Médico

Implementa chains especializadas para diferentes
funcionalidades do assistente.
"""

import os
from typing import Optional, Dict, Any, List

import torch
from langchain.llms.base import LLM
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LocalLLM(LLM):
    """Wrapper para LLM local fine-tuned."""
    
    model_path: str = "./models/assistente-medico-final"
    tokenizer: Any = None
    model: Any = None
    pipeline: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        if model_path:
            self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo e tokenizer."""
        try:
            logger.info(f"Carregando modelo de: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            logger.info("Modelo carregado com sucesso")
            
        except Exception as e:
            logger.warning(f"Não foi possível carregar modelo local: {e}")
            logger.info("Usando modelo simulado para desenvolvimento")
            self.pipeline = None
    
    @property
    def _llm_type(self) -> str:
        return "local_medical_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Gera resposta do modelo."""
        if self.pipeline is None:
            # Modo de desenvolvimento sem modelo
            return self._mock_response(prompt)
        
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            return outputs[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Resposta simulada para desenvolvimento."""
        return """Como assistente médico virtual, posso fornecer informações gerais de saúde. 
        Por favor, lembre-se de que minhas respostas não substituem uma consulta médica profissional. 
        Recomendo que você busque orientação de um profissional de saúde para avaliação adequada."""


class MedicalChain:
    """Chain principal para consultas médicas."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "./models/assistente-medico-final"
        self._setup_chains()
    
    def _setup_chains(self):
        """Configura as chains."""
        # LLM local
        self.llm = LocalLLM(model_path=self.model_path)
        
        # Chain de classificação de sintomas
        self.symptom_prompt = PromptTemplate(
            input_variables=["symptoms"],
            template="""Analise os seguintes sintomas e forneça informações gerais:
            
            Sintomas relatados: {symptoms}
            
            Responda de forma clara e empática, sempre recomendando buscar um médico.
            
            Análise:"""
        )
        self.symptom_chain = LLMChain(llm=self.llm, prompt=self.symptom_prompt)
        
        # Chain de informações sobre medicamentos
        self.medication_prompt = PromptTemplate(
            input_variables=["medication"],
            template="""Forneça informações gerais sobre o medicamento:
            
            Medicamento: {medication}
            
            IMPORTANTE: Nunca prescreva ou recomende doses. Apenas informações educacionais.
            
            Informações:"""
        )
        self.medication_chain = LLMChain(llm=self.llm, prompt=self.medication_prompt)
        
        # Chain de orientações de saúde
        self.health_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""Forneça orientações de saúde sobre:
            
            Tema: {topic}
            
            Orientações baseadas em evidências:"""
        )
        self.health_chain = LLMChain(llm=self.llm, prompt=self.health_prompt)
    
    def get_llm(self) -> LLM:
        """Retorna a instância do LLM."""
        return self.llm
    
    def analyze_symptoms(self, symptoms: str) -> str:
        """Analisa sintomas do paciente."""
        logger.info("Analisando sintomas")
        return self.symptom_chain.run(symptoms=symptoms)
    
    def get_medication_info(self, medication: str) -> str:
        """Retorna informações sobre medicamento."""
        logger.info(f"Buscando informações sobre: {medication}")
        return self.medication_chain.run(medication=medication)
    
    def get_health_guidance(self, topic: str) -> str:
        """Retorna orientações de saúde sobre um tema."""
        logger.info(f"Gerando orientações sobre: {topic}")
        return self.health_chain.run(topic=topic)


if __name__ == "__main__":
    chain = MedicalChain()
    result = chain.analyze_symptoms("dor de cabeça e febre")
    print(result)

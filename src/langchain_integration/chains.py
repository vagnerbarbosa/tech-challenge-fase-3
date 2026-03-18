"""
Chains do LangChain
===================

Implementa as chains para o assistente médico generalista.
"""

import os
from typing import Any, Optional

try:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MedicalChains:
    """
    Classe que define as chains do LangChain para o assistente médico generalista.
    """
    
    def __init__(self, model: Any = None, tokenizer: Any = None):
        """
        Inicializa as chains.
        
        Args:
            model: Modelo LLM
            tokenizer: Tokenizer do modelo
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Inicializa chains
        self.qa_chain = self._create_qa_chain()
        self.summary_chain = self._create_summary_chain()
        
        logger.info("MedicalChains inicializado para assistente generalista")
    
    def _create_qa_chain(self):
        """
        Cria a chain de perguntas e respostas médicas.
        
        Returns:
            Chain de Q&A
        """
        # Template de prompt para Q&A médico generalista
        qa_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente médico generalista.
Responda de forma clara, precisa e empática.
Sempre recomende consultar um médico para diagnósticos.
Não forneça diagnósticos - apenas informações educativas.
Oriente sobre quando buscar atendimento especializado."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        # Se modelo disponível, usa ele; senão, retorna resposta padrão
        if self.model and self.tokenizer:
            def generate_response(inputs):
                # O prompt já vem formatado do assistant.py (formato BioMistral)
                prompt = inputs['question']

                input_ids = self.tokenizer.encode(
                    prompt,
                    return_tensors="pt"
                ).to(self.model.device)

                max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", 2048))

                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=300,  # Gera apenas novos tokens, não sobrescreve o input
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )

                # Decodifica apenas os tokens gerados (não o input)
                response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                return response.strip()
            
            chain = RunnableLambda(generate_response)
        else:
            # Resposta padrão quando modelo não está disponível
            def default_response(inputs):
                return f"""Obrigado pela sua pergunta sobre: {inputs['question']}

Como assistente médico generalista, posso fornecer informações educativas sobre saúde.
Para uma resposta completa, o modelo de fine-tuning precisa estar carregado.

Lembre-se: sempre consulte um profissional de saúde para orientações específicas."""
            
            chain = RunnableLambda(default_response)
        
        return chain
    
    def _create_summary_chain(self):
        """
        Cria a chain de sumarização de conversas.
        
        Returns:
            Chain de sumarização
        """
        summary_template = """Resuma a seguinte conversa médica, destacando:
1. Principais preocupações do paciente
2. Informações fornecidas
3. Recomendações dadas

Conversa:
{conversation}

Resumo:"""
        
        def summarize(inputs):
            conversation = inputs.get("conversation", "")
            return f"Resumo da conversa:\n{conversation[:500]}..."
        
        return RunnableLambda(summarize)
    
    def get_qa_response(self, question: str, chat_history: list = None) -> str:
        """
        Obtém resposta para uma pergunta médica.
        
        Args:
            question: Pergunta do usuário
            chat_history: Histórico de conversa
            
        Returns:
            Resposta do assistente
        """
        return self.qa_chain.invoke({
            "question": question,
            "chat_history": chat_history or [],
        })




if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()

    print("=" * 60)
    print("  MedicalChains - Demonstração")
    print("=" * 60)
    print()

    chains = MedicalChains()

    print("[INFO] MedicalChains instanciado (sem modelo LLM).")
    print()

    # Demonstra obtenção de resposta (fallback sem modelo)
    test_questions = [
        "Quais são os sintomas da gripe?",
        "Como tratar dor de cabeça?",
    ]

    for q in test_questions:
        print(f"Pergunta: {q}")
        try:
            resp = chains.get_qa_response(q)
            print(f"Resposta: {resp}")
        except Exception as e:
            print(f"  (Esperado sem LLM) Erro: {type(e).__name__}: {e}")
        print()

    print("[OK] Demonstração concluída.")

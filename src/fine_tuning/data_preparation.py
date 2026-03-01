"""
Módulo de Preparação e Anonimização de Dados
============================================

Responsável por:
- Carregar dados médicos gerais
- Anonimizar informações sensíveis (LGPD)
- Preparar dataset para fine-tuning do assistente generalista
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict

from src.utils.logging_config import get_logger
from src.utils.validators import DataValidator

logger = get_logger(__name__)


class DataPreparation:
    """
    Classe para preparação e anonimização de dados médicos.
    Suporta dados de diversas especialidades médicas para o assistente generalista.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Inicializa o preparador de dados médicos.
        
        Args:
            data_path: Caminho para os dados. Se None, usa DATA_PATH do .env
        """
        self.data_path = Path(data_path or os.getenv("DATA_PATH", "./data"))
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.validator = DataValidator()
        
        # Garante que os diretórios existam
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataPreparation inicializado. Data path: {self.data_path}")
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonimiza informações sensíveis no texto.
        
        Args:
            text: Texto a ser anonimizado
            
        Returns:
            Texto anonimizado
        """
        if not text:
            return text
        
        # Padrões para anonimização
        patterns = {
            # CPF: XXX.XXX.XXX-XX
            r'\d{3}\.\d{3}\.\d{3}-\d{2}': '[CPF_ANONIMIZADO]',
            # RG: XX.XXX.XXX-X
            r'\d{2}\.\d{3}\.\d{3}-[0-9X]': '[RG_ANONIMIZADO]',
            # Telefone: (XX) XXXXX-XXXX ou (XX) XXXX-XXXX
            r'\(\d{2}\)\s*\d{4,5}-?\d{4}': '[TELEFONE_ANONIMIZADO]',
            # Email
            r'[\w\.-]+@[\w\.-]+\.\w+': '[EMAIL_ANONIMIZADO]',
            # Nomes próprios (simplificado - em produção usar NER)
            r'\b[A-Z][a-záéíóúàèìòùâêîôûãõ]+\s+[A-Z][a-záéíóúàèìòùâêîôûãõ]+\b': '[NOME_ANONIMIZADO]',
            # Datas de nascimento
            r'\d{2}/\d{2}/\d{4}': '[DATA_ANONIMIZADA]',
            # Endereços (simplificado)
            r'Rua\s+[\w\s]+,\s*\d+': '[ENDERECO_ANONIMIZADO]',
        }
        
        anonymized = text
        for pattern, replacement in patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized
    
    def load_raw_data(self, filename: str = "medical_data.csv") -> pd.DataFrame:
        """
        Carrega dados brutos do arquivo.
        
        Args:
            filename: Nome do arquivo de dados médicos
            
        Returns:
            DataFrame com os dados carregados
        """
        file_path = self.raw_path / filename
        
        if not file_path.exists():
            logger.warning(f"Arquivo não encontrado: {file_path}")
            logger.info("Criando dataset de exemplo com dados médicos gerais...")
            return self._create_sample_dataset()
        
        logger.info(f"Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
        
        return df
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """
        Cria um dataset de exemplo para demonstração.
        Inclui perguntas e respostas médicas gerais de diversas especialidades.
        
        Returns:
            DataFrame com dados de exemplo
        """
        # Exemplos de perguntas e respostas médicas gerais
        samples = [
            {
                "instruction": "Quais são os sintomas de uma gripe comum?",
                "input": "",
                "output": "Os principais sintomas da gripe comum incluem: febre (geralmente acima de 38°C), dor de cabeça, dores musculares e nas articulações, fadiga intensa, tosse seca, coriza, dor de garganta e calafrios. Os sintomas costumam aparecer de forma súbita e durar de 7 a 10 dias. Recomenda-se repouso, hidratação e, se necessário, consultar um médico."
            },
            {
                "instruction": "Quando devo procurar um pronto-socorro?",
                "input": "",
                "output": "Deve procurar um pronto-socorro em situações de emergência como: dor no peito ou dificuldade respiratória, perda de consciência ou desmaio, sangramento intenso que não para, febre muito alta (acima de 39.5°C) que não cede com medicamentos, confusão mental súbita, suspeita de AVC (rosto caído, fraqueza em um lado do corpo, fala arrastada), trauma grave ou fraturas, reações alérgicas severas."
            },
            {
                "instruction": "O que é pressão arterial alta e como controlar?",
                "input": "Paciente com histórico familiar de hipertensão",
                "output": "A pressão arterial alta (hipertensão) ocorre quando a força do sangue contra as paredes das artérias é consistentemente elevada (acima de 140/90 mmHg). Para controlar: 1) Reduza o consumo de sal; 2) Mantenha peso saudável; 3) Pratique exercícios regularmente; 4) Evite álcool e tabaco; 5) Gerencie o estresse; 6) Tome medicamentos conforme prescrição médica; 7) Monitore a pressão regularmente."
            },
            {
                "instruction": "Quais exames de rotina são recomendados anualmente?",
                "input": "",
                "output": "Os exames de rotina recomendados variam por idade e sexo, mas geralmente incluem: hemograma completo, glicemia de jejum, perfil lipídico (colesterol e triglicerídeos), função renal e hepática, exame de urina, medição de pressão arterial. Para mulheres: papanicolau e mamografia (após 40 anos). Para homens acima de 50: PSA. Sempre consulte seu médico para um plano personalizado."
            },
            {
                "instruction": "Como aliviar dor de cabeça?",
                "input": "",
                "output": "Para aliviar dores de cabeça comuns: 1) Descanse em ambiente escuro e silencioso; 2) Aplique compressas frias ou quentes na testa/nuca; 3) Mantenha-se hidratado; 4) Evite telas de computador e celular; 5) Analgésicos como paracetamol ou ibuprofeno podem ajudar (conforme orientação médica). Procure um médico se a dor for intensa, súbita, acompanhada de febre, rigidez no pescoço ou alterações visuais."
            },
            {
                "instruction": "O que fazer em caso de queimadura leve?",
                "input": "",
                "output": "Em caso de queimadura leve (1º grau): 1) Resfrie a área com água corrente fria por 10-20 minutos; 2) Não aplique gelo diretamente; 3) Não use manteiga, pasta de dente ou outros produtos caseiros; 4) Aplique pomada para queimaduras se disponível; 5) Cubra com gaze limpa; 6) Tome analgésico se necessário. Procure atendimento médico se a queimadura for extensa, no rosto, mãos, pés ou genitália."
            },
            {
                "instruction": "Quais são os sinais de um infarto?",
                "input": "",
                "output": "Os sinais de alerta de infarto incluem: dor ou desconforto no peito (sensação de pressão, aperto ou queimação), dor que se irradia para braço esquerdo, mandíbula, costas ou estômago, falta de ar, sudorese fria, náuseas ou vômitos, tontura. Mulheres podem ter sintomas atípicos como fadiga extrema. Em caso de suspeita, ligue 192 (SAMU) imediatamente - cada minuto conta!"
            },
            {
                "instruction": "Como melhorar a qualidade do sono?",
                "input": "",
                "output": "Para melhorar o sono: 1) Mantenha horários regulares para dormir e acordar; 2) Evite cafeína e álcool à noite; 3) Crie um ambiente escuro, silencioso e fresco; 4) Evite telas 1 hora antes de dormir; 5) Pratique exercícios (mas não perto da hora de dormir); 6) Evite refeições pesadas à noite; 7) Considere técnicas de relaxamento. Se a insônia persistir por mais de 3 semanas, consulte um médico."
            },
        ]
        
        df = pd.DataFrame(samples)
        
        # Salva o dataset de exemplo
        sample_path = self.processed_path / "sample_medical_qa.csv"
        df.to_csv(sample_path, index=False)
        logger.info(f"Dataset de exemplo (dados médicos gerais) salvo em: {sample_path}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> Dataset:
        """
        Prepara os dados para o formato de treinamento.
        
        Args:
            df: DataFrame com os dados médicos
            
        Returns:
            Dataset do Hugging Face pronto para treinamento
        """
        # Formata as instruções no formato de chat
        def format_instruction(row):
            if row.get('input', ''):
                text = f"### Instrução:\n{row['instruction']}\n\n### Contexto:\n{row['input']}\n\n### Resposta:\n{row['output']}"
            else:
                text = f"### Instrução:\n{row['instruction']}\n\n### Resposta:\n{row['output']}"
            return text
        
        df['text'] = df.apply(format_instruction, axis=1)
        
        # Anonimiza os textos
        df['text'] = df['text'].apply(self.anonymize_text)
        
        # Converte para Dataset
        dataset = Dataset.from_pandas(df[['text']])
        
        logger.info(f"Dataset preparado para treinamento: {len(dataset)} exemplos")
        
        return dataset
    
    def prepare_dataset(self) -> Dataset:
        """
        Pipeline completo de preparação de dados médicos.
        
        Returns:
            Dataset pronto para fine-tuning
        """
        logger.info("Iniciando preparação do dataset médico...")
        
        # Carrega dados
        df = self.load_raw_data()
        
        # Valida dados
        if not self.validator.validate_dataframe(df):
            logger.warning("Validação falhou, usando dataset de exemplo")
            df = self._create_sample_dataset()
        
        # Prepara para treinamento
        dataset = self.prepare_for_training(df)
        
        logger.info("Preparação do dataset médico concluída!")
        
        return dataset


if __name__ == "__main__":
    # Teste do módulo
    prep = DataPreparation()
    dataset = prep.prepare_dataset()
    print(f"\nExemplo de dado preparado:\n{dataset[0]['text'][:500]}...")

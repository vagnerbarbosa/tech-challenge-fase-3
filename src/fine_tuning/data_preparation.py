"""
Módulo de Preparação e Anonimização de Dados
============================================

Responsável por:
- Carregar dados médicos gerais (CSV ou JSONL)
- Anonimizar informações sensíveis (LGPD)
- Preparar dataset para fine-tuning do assistente generalista

Formatos suportados:
- CSV: arquivos com colunas instruction, input, output
- JSONL: arquivos JSON Lines com campos instruction, input, output

Uso:
    Via main.py (pipeline completo):
        python main.py
    
    Execução isolada (requer estar na raiz do projeto):
        python -m src.fine_tuning.data_preparation
"""

import os
import re
import json
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict

# Determina a raiz do projeto (onde está main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Adiciona a raiz ao path para imports funcionarem corretamente
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
            data_path: Caminho para os dados. Se None, usa DATA_PATH do .env ou 
                      o diretório 'data' na raiz do projeto
        """
        if data_path:
            self.data_path = Path(data_path).resolve()
        elif os.getenv("DATA_PATH"):
            self.data_path = Path(os.getenv("DATA_PATH")).resolve()
        else:
            # Usa o diretório data relativo à raiz do projeto
            self.data_path = PROJECT_ROOT / "data"
        
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.validator = DataValidator()
        
        # Garante que os diretórios existam
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataPreparation inicializado")
        logger.info(f"  Raiz do projeto: {PROJECT_ROOT}")
        logger.info(f"  Diretório de dados: {self.data_path}")
        logger.info(f"  Dados brutos: {self.raw_path}")
        logger.info(f"  Dados processados: {self.processed_path}")
    
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
    
    def load_jsonl(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Carrega dados de um arquivo JSONL.
        
        Args:
            file_path: Caminho para o arquivo JSONL
            
        Returns:
            DataFrame com os dados carregados
        """
        file_path = Path(file_path)
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Linha {line_num} inválida em {file_path.name}: {e}")
        
        if not records:
            logger.warning(f"Nenhum registro válido em {file_path}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        logger.info(f"JSONL carregado: {len(df)} registros de {file_path.name}")
        return df
    
    def load_raw_data(self, filename: str = "medical_data_unified.jsonl") -> pd.DataFrame:
        """
        Carrega dados brutos do arquivo (CSV ou JSONL).
        
        Prioridade de busca:
        1. JSONL unificado já existente (medical_data_unified.jsonl)
        2. Todos os CSVs em data/raw/ (busca automática)
        3. Todos os JSONLs em data/raw/ (busca automática)
        4. Dataset de exemplo (se nada encontrado)
        
        Se encontrar CSVs/JSONLs em raw/, unifica-os automaticamente
        em data/processed/medical_data_unified.jsonl
        
        Args:
            filename: Nome do arquivo de dados (para compatibilidade)
            
        Returns:
            DataFrame com os dados carregados
        """
        # 1. Tenta carregar JSONL unificado primeiro (se já foi processado antes)
        jsonl_unified = self.processed_path / "medical_data_unified.jsonl"
        if jsonl_unified.exists():
            logger.info(f"✓ Carregando JSONL unificado existente: {jsonl_unified}")
            return self.load_jsonl(jsonl_unified)
        
        # 2. Busca TODOS os CSVs em data/raw/
        csv_files = list(self.raw_path.glob("*.csv"))
        jsonl_files = list(self.raw_path.glob("*.jsonl"))
        
        if csv_files or jsonl_files:
            logger.info("=" * 60)
            logger.info("PROCESSANDO ARQUIVOS DE DATA/RAW")
            logger.info("=" * 60)
            
            dfs = []
            
            # Processa CSVs
            for csv_file in csv_files:
                logger.info(f"  → Carregando CSV: {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    # Normaliza nomes de colunas
                    df = self._normalize_columns(df)
                    dfs.append(df)
                    logger.info(f"    ✓ {len(df)} registros carregados")
                except Exception as e:
                    logger.warning(f"    ✗ Erro ao carregar {csv_file.name}: {e}")
            
            # Processa JSONLs
            for jsonl_file in jsonl_files:
                logger.info(f"  → Carregando JSONL: {jsonl_file.name}")
                try:
                    df = self.load_jsonl(jsonl_file)
                    df = self._normalize_columns(df)
                    if not df.empty:
                        dfs.append(df)
                        logger.info(f"    ✓ {len(df)} registros carregados")
                except Exception as e:
                    logger.warning(f"    ✗ Erro ao carregar {jsonl_file.name}: {e}")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                logger.info("-" * 60)
                logger.info(f"✓ Total unificado: {len(combined)} registros de {len(dfs)} arquivo(s)")
                
                # Salva o JSONL unificado
                self._save_unified_jsonl(combined, jsonl_unified)
                
                return combined
        
        # 3. Tenta CSVs já processados (fallback)
        processed_csvs = list(self.processed_path.glob("*.csv"))
        if processed_csvs:
            logger.info("Carregando CSVs já processados...")
            dfs = []
            for csv_file in processed_csvs:
                if csv_file.name != "sample_medical_qa.csv":  # Ignora sample
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    dfs.append(df)
                    logger.info(f"  → {csv_file.name}: {len(df)} registros")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                return combined
        
        # 4. Nenhum dado encontrado
        logger.warning("=" * 60)
        logger.warning("NENHUM ARQUIVO DE DADOS ENCONTRADO")
        logger.warning("=" * 60)
        logger.warning("")
        logger.warning("O script buscou em:")
        logger.warning(f"  1. {self.processed_path / 'medical_data_unified.jsonl'}")
        logger.warning(f"  2. {self.raw_path}/*.csv")
        logger.warning(f"  3. {self.raw_path}/*.jsonl")
        logger.warning("")
        logger.warning("Para usar seus próprios dados:")
        logger.warning("  1. Coloque arquivos CSV ou JSONL em: data/raw/")
        logger.warning("  2. Formato esperado: colunas 'instruction', 'input', 'output'")
        logger.warning("  3. Execute novamente: python -m src.fine_tuning.data_preparation")
        logger.warning("")
        logger.info("Criando dataset de exemplo para demonstração...")
        return self._create_sample_dataset()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza nomes de colunas para o formato padrão.
        Aceita variações comuns como 'pergunta', 'resposta', 'question', etc.
        
        Args:
            df: DataFrame com colunas possivelmente não padronizadas
            
        Returns:
            DataFrame com colunas normalizadas
        """
        # Mapeamento de variações para nomes padrão
        column_mapping = {
            # instruction
            'instruction': 'instruction',
            'instrucao': 'instruction',
            'pergunta': 'instruction',
            'question': 'instruction',
            'prompt': 'instruction',
            # input
            'input': 'input',
            'entrada': 'input',
            'context': 'input',
            'contexto': 'input',
            # output
            'output': 'output',
            'saida': 'input',
            'resposta': 'output',
            'response': 'output',
            'answer': 'output',
        }
        
        # Aplica mapeamento (case insensitive)
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in column_mapping:
                new_columns[col] = column_mapping[col_lower]
        
        if new_columns:
            df = df.rename(columns=new_columns)
        
        # Garante que as colunas necessárias existam
        for col in ['instruction', 'input', 'output']:
            if col not in df.columns:
                df[col] = ''
        
        return df[['instruction', 'input', 'output']]
    
    def _save_unified_jsonl(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Salva o DataFrame unificado como JSONL.
        
        Args:
            df: DataFrame a ser salvo
            output_path: Caminho do arquivo JSONL de saída
        """
        logger.info(f"Salvando JSONL unificado: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Trata NaN e valores vazios
                instruction = row.get('instruction', '')
                input_val = row.get('input', '')
                output_val = row.get('output', '')
                
                # Converte NaN para string vazia
                instruction = '' if pd.isna(instruction) else str(instruction)
                input_val = '' if pd.isna(input_val) else str(input_val)
                output_val = '' if pd.isna(output_val) else str(output_val)
                
                record = {
                    'instruction': instruction,
                    'input': input_val,
                    'output': output_val
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ JSONL unificado salvo com {len(df)} registros")
    
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
            input_val = row.get('input', '')
            # Trata NaN e strings "nan"
            if pd.isna(input_val) or str(input_val).lower() == 'nan' or not str(input_val).strip():
                text = f"### Instrução:\n{row['instruction']}\n\n### Resposta:\n{row['output']}"
            else:
                text = f"### Instrução:\n{row['instruction']}\n\n### Contexto:\n{input_val}\n\n### Resposta:\n{row['output']}"
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


def run():
    """
    Função principal para execução isolada do módulo.
    Executa a preparação de dados e exibe um exemplo.
    """
    from dotenv import load_dotenv
    
    # Carrega variáveis de ambiente do .env na raiz do projeto
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Variáveis de ambiente carregadas de: {env_path}")
    
    print("\n" + "=" * 60)
    print("PREPARAÇÃO DE DADOS - Execução Isolada")
    print("=" * 60 + "\n")
    
    prep = DataPreparation()
    dataset = prep.prepare_dataset()
    
    print("\n" + "-" * 60)
    print(f"Dataset preparado com sucesso: {len(dataset)} exemplos")
    print("-" * 60)
    print("\nExemplo de dado preparado:")
    print(dataset[0]['text'][:500] + "...")
    print("\n✓ Preparação concluída!")
    
    return dataset


if __name__ == "__main__":
    run()

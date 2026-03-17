"""
Módulo de Preparação de Dados para Fine-Tuning
==============================================

Responsável por:
- Validar e preparar o diretório data/raw/
- Invocar scrapers automaticamente se necessário
- Carregar arquivos JSONL de data/raw/
- Unificar em um único JSONL em data/processed/
- Validar e limpar os dados
- Anonimizar informações sensíveis (LGPD)
- Preparar dataset para fine-tuning

Formato JSONL esperado:
- instruction: pergunta ou instrução para o modelo
- input: contexto adicional (opcional)
- output: resposta esperada
- source: fonte dos dados (opcional)

Uso:
    Via main.py (pipeline completo):
        python main.py
    
    Execução isolada:
        python -m src.fine_tuning.data_preparation
"""

import os
import re
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset

# Determina a raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger
from src.utils.validators import DataValidator

logger = get_logger(__name__)


class DataPreparation:
    """
    Classe para preparação de dados médicos para fine-tuning.
    Trabalha exclusivamente com arquivos JSONL.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Inicializa o preparador de dados.
        
        Args:
            data_path: Caminho para os dados. Se None, usa DATA_PATH do .env 
                      ou o diretório 'data' na raiz do projeto
        """
        if data_path:
            self.data_path = Path(data_path).resolve()
        elif os.getenv("DATA_PATH"):
            self.data_path = Path(os.getenv("DATA_PATH")).resolve()
        else:
            self.data_path = PROJECT_ROOT / "data"
        
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self.validator = DataValidator()
        
        # Garante que os diretórios existam
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataPreparation inicializado")
        logger.info(f"  Dados brutos: {self.raw_path}")
        logger.info(f"  Dados processados: {self.processed_path}")
    
    def validate_raw_directory(self) -> bool:
        """
        Valida se o diretório data/raw/ contém arquivos JSONL válidos.
        
        Returns:
            True se há arquivos JSONL válidos, False caso contrário
        """
        logger.info("=" * 60)
        logger.info("VALIDAÇÃO DO DIRETÓRIO DE DADOS")
        logger.info("=" * 60)
        
        # Lista todos os arquivos no diretório
        all_files = list(self.raw_path.iterdir()) if self.raw_path.exists() else []
        
        # Ignora arquivos ocultos e .gitkeep
        visible_files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
        
        if not visible_files:
            logger.warning(f"Diretório vazio: {self.raw_path}")
            return False
        
        # Verifica arquivos JSONL
        jsonl_files = [f for f in visible_files if f.suffix.lower() == '.jsonl']
        non_jsonl_files = [f for f in visible_files if f.suffix.lower() != '.jsonl']
        
        if non_jsonl_files:
            logger.warning(f"Arquivos não-JSONL encontrados ({len(non_jsonl_files)}):")
            for f in non_jsonl_files:
                logger.warning(f"  - {f.name}")
        
        if not jsonl_files:
            logger.warning("Nenhum arquivo JSONL encontrado!")
            return False
        
        # Verifica se os arquivos JSONL são válidos (não vazios e com JSON válido)
        valid_jsonl_count = 0
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        # Tenta parsear a primeira linha para verificar se é JSON válido
                        json.loads(lines[0])
                        valid_jsonl_count += 1
                        logger.info(f"  ✓ {jsonl_file.name}: {len(lines)} linhas")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"  ✗ {jsonl_file.name}: arquivo inválido - {e}")
        
        if valid_jsonl_count == 0:
            logger.warning("Nenhum arquivo JSONL válido encontrado!")
            return False
        
        logger.info(f"Validação concluída: {valid_jsonl_count} arquivo(s) JSONL válido(s)")
        return True
    
    def clean_raw_directory(self):
        """
        Limpa o diretório data/raw/ removendo arquivos não-JSONL.
        Mantém .gitkeep se existir.
        """
        logger.info("Limpando diretório data/raw/...")
        
        all_files = list(self.raw_path.iterdir()) if self.raw_path.exists() else []
        
        for f in all_files:
            if f.is_file() and f.name != '.gitkeep':
                try:
                    f.unlink()
                    logger.info(f"  Removido: {f.name}")
                except Exception as e:
                    logger.error(f"  Erro ao remover {f.name}: {e}")
        
        logger.info("Limpeza concluída")
    
    def invoke_scrapers(self) -> bool:
        """
        Tenta invocar os scrapers para gerar dados em data/raw/.
        
        Returns:
            True se os scrapers executaram com sucesso, False caso contrário
        """
        logger.info("=" * 60)
        logger.info("INVOCANDO SCRAPERS")
        logger.info("=" * 60)
        
        try:
            from src.scraping.run_scrapers import run_all_scrapers
            
            logger.info("Executando scrapers de dados médicos...")
            results = run_all_scrapers()
            
            # Verifica se algum arquivo foi gerado
            generated = [path for path in results.values() if path]
            
            if generated:
                logger.info(f"✓ Scrapers executados com sucesso! {len(generated)} arquivo(s) gerado(s)")
                return True
            else:
                logger.warning("Scrapers executados, mas nenhum arquivo foi gerado")
                return False
                
        except ImportError as e:
            logger.error(f"Erro ao importar scrapers: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro ao executar scrapers: {e}")
            return False
    
    def create_example_file(self) -> Path:
        """
        Cria um arquivo de exemplo em data/raw/ com 5 registros.
        
        Returns:
            Path do arquivo criado
        """
        logger.info("=" * 60)
        logger.info("CRIANDO ARQUIVO DE EXEMPLO")
        logger.info("=" * 60)
        
        samples = [
            {
                "instruction": "Quais são os sintomas de uma gripe comum?",
                "input": "",
                "output": "Os principais sintomas da gripe comum incluem: febre (geralmente acima de 38°C), dor de cabeça, dores musculares e nas articulações, fadiga intensa, tosse seca, coriza, dor de garganta e calafrios. Os sintomas costumam aparecer de forma súbita e durar de 7 a 10 dias.",
                "source": "Exemplo",
            },
            {
                "instruction": "Quando devo procurar um pronto-socorro?",
                "input": "",
                "output": "Deve procurar um pronto-socorro em situações de emergência como: dor no peito ou dificuldade respiratória, perda de consciência, sangramento intenso que não para, febre muito alta (acima de 39.5°C) que não cede com medicamentos, suspeita de AVC (rosto caído, fraqueza em um lado do corpo, fala arrastada), trauma grave.",
                "source": "Exemplo",
            },
            {
                "instruction": "O que é pressão arterial alta e como controlar?",
                "input": "Paciente com histórico familiar de hipertensão",
                "output": "A pressão arterial alta (hipertensão) ocorre quando a força do sangue contra as paredes das artérias é consistentemente elevada (acima de 140/90 mmHg). Para controlar: reduza o consumo de sal, mantenha peso saudável, pratique exercícios regularmente, evite álcool e tabaco, tome medicamentos conforme prescrição médica.",
                "source": "Exemplo",
            },
            {
                "instruction": "Como identificar sinais de desidratação?",
                "input": "",
                "output": "Os principais sinais de desidratação incluem: sede intensa, urina escura e em pouca quantidade, boca e lábios secos, tontura ou vertigem, cansaço excessivo, dor de cabeça, pele seca e com pouca elasticidade. Em casos graves: confusão mental, batimentos cardíacos acelerados e pressão baixa. Crianças e idosos são mais vulneráveis.",
                "source": "Exemplo",
            },
            {
                "instruction": "Qual a diferença entre gripe e resfriado?",
                "input": "",
                "output": "A gripe é causada pelo vírus Influenza e apresenta sintomas mais intensos: febre alta, dores no corpo, fadiga severa. Já o resfriado é causado por diversos vírus (como rinovírus) e tem sintomas mais leves: coriza, espirros, dor de garganta leve. A gripe pode evoluir para complicações graves como pneumonia, enquanto o resfriado geralmente se resolve em poucos dias.",
                "source": "Exemplo",
            },
        ]
        
        output_file = self.raw_path / "example_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ Arquivo de exemplo criado: {output_file}")
        logger.info(f"  Total de registros: {len(samples)}")
        
        return output_file
    
    def ensure_data_available(self) -> bool:
        """
        Garante que há dados disponíveis para a pipeline.
        Valida, limpa, invoca scrapers ou cria exemplo conforme necessário.
        
        Returns:
            True se dados estão disponíveis, False caso contrário
        """
        # 1. Valida diretório atual
        if self.validate_raw_directory():
            logger.info("✓ Dados válidos encontrados em data/raw/")
            return True
        
        # 2. Limpa diretório se há arquivos inválidos
        logger.info("Dados inválidos ou ausentes. Iniciando processo de recuperação...")
        self.clean_raw_directory()
        
        # 3. Tenta invocar scrapers
        if self.invoke_scrapers():
            # Revalida após scrapers
            if self.validate_raw_directory():
                return True
        
        # 4. Fallback: cria arquivo de exemplo
        logger.warning("Scrapers falharam ou não geraram dados. Usando dados de exemplo...")
        self.create_example_file()
        
        return True
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonimiza informações sensíveis no texto (LGPD).
        
        Args:
            text: Texto a ser anonimizado
            
        Returns:
            Texto anonimizado
        """
        if not text:
            return text
        
        patterns = {
            r'\d{3}\.\d{3}\.\d{3}-\d{2}': '[CPF_ANONIMIZADO]',
            r'\d{2}\.\d{3}\.\d{3}-[0-9X]': '[RG_ANONIMIZADO]',
            r'\(\d{2}\)\s*\d{4,5}-?\d{4}': '[TELEFONE_ANONIMIZADO]',
            r'[\w\.-]+@[\w\.-]+\.\w+': '[EMAIL_ANONIMIZADO]',
            r'\d{2}/\d{2}/\d{4}': '[DATA_ANONIMIZADA]',
            r'Rua\s+[\w\s]+,\s*\d+': '[ENDERECO_ANONIMIZADO]',
        }
        
        anonymized = text
        for pattern, replacement in patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized
    
    def load_jsonl(self, file_path: Path) -> List[Dict]:
        """
        Carrega dados de um arquivo JSONL.
        
        Args:
            file_path: Caminho para o arquivo JSONL
            
        Returns:
            Lista de registros
        """
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Valida campos obrigatórios
                    if 'instruction' in record and 'output' in record:
                        records.append(record)
                    else:
                        logger.warning(f"Linha {line_num} em {file_path.name}: campos obrigatórios ausentes")
                except json.JSONDecodeError as e:
                    logger.warning(f"Linha {line_num} em {file_path.name}: JSON inválido - {e}")
        
        logger.info(f"  ✓ {file_path.name}: {len(records)} registros carregados")
        return records
    
    def load_raw_data(self) -> List[Dict]:
        """
        Carrega todos os arquivos JSONL de data/raw/.
        
        Returns:
            Lista unificada de registros
        """
        jsonl_files = list(self.raw_path.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.warning("Nenhum arquivo JSONL encontrado após validação!")
            return []
        
        logger.info("=" * 60)
        logger.info("CARREGANDO ARQUIVOS JSONL")
        logger.info("=" * 60)
        logger.info(f"Diretório: {self.raw_path}")
        logger.info(f"Arquivos encontrados: {len(jsonl_files)}")
        
        all_records = []
        for jsonl_file in sorted(jsonl_files):
            records = self.load_jsonl(jsonl_file)
            all_records.extend(records)
        
        logger.info("-" * 60)
        logger.info(f"Total unificado: {len(all_records)} registros")
        
        return all_records
    
    def validate_and_clean(self, records: List[Dict]) -> List[Dict]:
        """
        Valida e limpa os registros.
        
        Args:
            records: Lista de registros brutos
            
        Returns:
            Lista de registros validados e limpos
        """
        cleaned = []
        
        for record in records:
            instruction = str(record.get('instruction', '')).strip()
            input_val = str(record.get('input', '')).strip()
            output = str(record.get('output', '')).strip()
            
            # Validações básicas
            if len(instruction) < 5:
                continue
            if len(output) < 10:
                continue
            
            cleaned.append({
                'instruction': instruction,
                'input': input_val,
                'output': output,
                'source': record.get('source', ''),
            })
        
        logger.info(f"Registros após validação: {len(cleaned)} (removidos: {len(records) - len(cleaned)})")
        return cleaned
    
    def save_unified_jsonl(self, records: List[Dict]) -> Path:
        """
        Salva os registros unificados em data/processed/.
        
        Args:
            records: Lista de registros
            
        Returns:
            Path do arquivo salvo
        """
        output_file = self.processed_path / "medical_data_unified.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ JSONL unificado salvo: {output_file}")
        logger.info(f"  Total de registros: {len(records)}")
        
        return output_file
    
    def prepare_for_training(self, records: List[Dict]) -> Dataset:
        """
        Prepara os dados para o formato de treinamento.
        
        Args:
            records: Lista de registros validados
            
        Returns:
            Dataset do Hugging Face pronto para treinamento
        """
        formatted_records = []
        
        for record in records:
            instruction = record['instruction']
            input_val = record.get('input', '')
            output = record['output']
            
            # Formata no padrão de chat
            if input_val:
                text = f"### Instrução:\n{instruction}\n\n### Contexto:\n{input_val}\n\n### Resposta:\n{output}"
            else:
                text = f"### Instrução:\n{instruction}\n\n### Resposta:\n{output}"
            
            # Anonimiza
            text = self.anonymize_text(text)
            formatted_records.append({'text': text})
        
        dataset = Dataset.from_list(formatted_records)
        logger.info(f"Dataset preparado para treinamento: {len(dataset)} exemplos")
        
        return dataset
    
    def prepare_dataset(self) -> Dataset:
        """
        Pipeline completo de preparação de dados.
        
        Returns:
            Dataset pronto para fine-tuning
        """
        logger.info("Iniciando preparação do dataset...")
        
        # 0. Garante que há dados disponíveis
        self.ensure_data_available()
        
        # 1. Carrega dados brutos
        records = self.load_raw_data()
        
        if not records:
            logger.error("Nenhum registro encontrado! Criando dataset mínimo de exemplo.")
            records = self._create_minimal_sample()
        
        # 2. Valida e limpa
        records = self.validate_and_clean(records)
        
        # 3. Salva JSONL unificado
        self.save_unified_jsonl(records)
        
        # 4. Prepara para treinamento
        dataset = self.prepare_for_training(records)
        
        logger.info("Preparação do dataset concluída!")
        
        return dataset
    
    def _create_minimal_sample(self) -> List[Dict]:
        """
        Cria uma lista mínima de exemplos para evitar falhas na pipeline.
        
        Returns:
            Lista com registros de exemplo
        """
        return [
            {
                "instruction": "Quais são os sintomas de uma gripe comum?",
                "input": "",
                "output": "Os principais sintomas da gripe incluem febre, dor de cabeça, dores no corpo, tosse e fadiga.",
                "source": "Exemplo",
            },
            {
                "instruction": "Quando devo procurar um médico?",
                "input": "",
                "output": "Procure um médico se os sintomas forem graves, persistentes ou se houver sinais de emergência.",
                "source": "Exemplo",
            },
        ]


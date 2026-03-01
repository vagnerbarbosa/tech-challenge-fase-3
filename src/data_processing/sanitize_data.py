"""
Módulo de sanitização de dados para conversão de CSVs para JSONL.

Este módulo implementa funções para:
- Limpar texto (remover HTML, caracteres especiais, espaços múltiplos)
- Validar tamanho mínimo de campos
- Converter CSVs médicos para formato instruction/input/output
- Gerar arquivos JSONL individuais e unificados
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SanitizationStats:
    """Estatísticas de sanitização."""
    total_input: int = 0
    total_output: int = 0
    removed_html: int = 0
    removed_short: int = 0
    removed_empty: int = 0
    removed_duplicate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstructionRecord:
    """Registro no formato instruction/input/output."""
    instruction: str
    output: str
    input: str = ""
    source: str = ""
    category: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source": self.source,
            "category": self.category
        }


class DataSanitizer:
    """Classe para sanitização e conversão de dados médicos."""
    
    # Configurações de validação
    MIN_INSTRUCTION_LENGTH = 5
    MIN_OUTPUT_LENGTH = 10
    
    # Padrões para limpeza
    HTML_PATTERN = re.compile(r'<[^>]+>')
    MULTIPLE_SPACES = re.compile(r'\s+')
    SPECIAL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    BOM_PATTERN = re.compile(r'^\ufeff')
    
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/processed"):
        """
        Inicializa o sanitizador.
        
        Args:
            input_dir: Diretório com os CSVs de entrada
            output_dir: Diretório para os arquivos JSONL de saída
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats: Dict[str, SanitizationStats] = {}
        
    def clean_text(self, text: str) -> str:
        """
        Limpa texto removendo HTML, caracteres especiais e espaços múltiplos.
        
        Args:
            text: Texto para limpar
            
        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""
        
        # Remove BOM
        text = self.BOM_PATTERN.sub('', text)
        
        # Remove tags HTML
        text = self.HTML_PATTERN.sub(' ', text)
        
        # Remove caracteres de controle
        text = self.SPECIAL_CHARS.sub('', text)
        
        # Normaliza espaços múltiplos
        text = self.MULTIPLE_SPACES.sub(' ', text)
        
        # Remove espaços no início e fim
        text = text.strip()
        
        return text
    
    def validate_record(self, record: InstructionRecord) -> bool:
        """
        Valida se um registro atende aos requisitos mínimos.
        
        Args:
            record: Registro para validar
            
        Returns:
            True se válido, False caso contrário
        """
        instruction_valid = len(record.instruction) >= self.MIN_INSTRUCTION_LENGTH
        output_valid = len(record.output) >= self.MIN_OUTPUT_LENGTH
        
        return instruction_valid and output_valid
    
    def process_perguntas_frequentes(self, df: pd.DataFrame) -> List[InstructionRecord]:
        """
        Processa o CSV de perguntas frequentes.
        
        Transformação:
        - pergunta → instruction
        - resposta → output
        - especialidade + categoria → input
        """
        records = []
        stats = SanitizationStats(total_input=len(df))
        
        for _, row in df.iterrows():
            # Limpa campos
            pergunta = self.clean_text(str(row.get('pergunta', '')))
            resposta = self.clean_text(str(row.get('resposta', '')))
            especialidade = self.clean_text(str(row.get('especialidade', '')))
            categoria = self.clean_text(str(row.get('categoria', '')))
            fonte = self.clean_text(str(row.get('fonte', '')))
            
            # Verifica se teve HTML removido
            if '<' in str(row.get('resposta', '')):
                stats.removed_html += 1
            
            # Cria input combinando especialidade e categoria
            input_parts = []
            if especialidade and especialidade != 'nan':
                input_parts.append(f"Especialidade: {especialidade}")
            if categoria and categoria != 'nan':
                input_parts.append(f"Categoria: {categoria}")
            input_text = ". ".join(input_parts)
            
            # Cria registro
            record = InstructionRecord(
                instruction=pergunta,
                output=resposta,
                input=input_text,
                source=fonte or "TelessaúdeRS",
                category="perguntas_frequentes"
            )
            
            # Valida
            if not record.instruction or not record.output:
                stats.removed_empty += 1
                continue
                
            if not self.validate_record(record):
                stats.removed_short += 1
                continue
            
            records.append(record)
        
        stats.total_output = len(records)
        self.stats['perguntas_frequentes'] = stats
        
        logger.info(f"perguntas_frequentes: {stats.total_input} → {stats.total_output} registros")
        return records
    
    def process_modelos_laudos(self, df: pd.DataFrame) -> List[InstructionRecord]:
        """
        Processa o CSV de modelos de laudos.
        
        Transformação:
        - Criar instruction sintética: "Como estruturar um laudo de {nome}?"
        - estrutura_laudo → output
        - modalidade + indicacoes → input
        """
        records = []
        stats = SanitizationStats(total_input=len(df))
        
        for _, row in df.iterrows():
            # Limpa campos
            nome = self.clean_text(str(row.get('nome', '')))
            modalidade = self.clean_text(str(row.get('modalidade', '')))
            indicacoes = self.clean_text(str(row.get('indicacoes', '')))
            estrutura = self.clean_text(str(row.get('estrutura_laudo', '')))
            especialidade = self.clean_text(str(row.get('especialidade', '')))
            fonte = self.clean_text(str(row.get('fonte', '')))
            
            # Verifica se teve HTML removido
            if '<' in str(row.get('estrutura_laudo', '')):
                stats.removed_html += 1
            
            # Cria instruction sintética
            if nome and nome != 'nan':
                instruction = f"Como estruturar um laudo de {nome}?"
            else:
                stats.removed_empty += 1
                continue
            
            # Cria input combinando modalidade e indicações
            input_parts = []
            if modalidade and modalidade != 'nan':
                input_parts.append(f"Modalidade: {modalidade}")
            if indicacoes and indicacoes != 'nan':
                input_parts.append(f"Indicações: {indicacoes}")
            if especialidade and especialidade != 'nan':
                input_parts.append(f"Especialidade: {especialidade}")
            input_text = ". ".join(input_parts)
            
            # Cria registro
            record = InstructionRecord(
                instruction=instruction,
                output=estrutura,
                input=input_text,
                source=fonte or "RadReport",
                category="modelos_laudos"
            )
            
            # Valida
            if not record.output:
                stats.removed_empty += 1
                continue
                
            if not self.validate_record(record):
                stats.removed_short += 1
                continue
            
            records.append(record)
        
        stats.total_output = len(records)
        self.stats['modelos_laudos'] = stats
        
        logger.info(f"modelos_laudos: {stats.total_input} → {stats.total_output} registros")
        return records
    
    def process_protocolos_medicos(self, df: pd.DataFrame) -> List[InstructionRecord]:
        """
        Processa o CSV de protocolos médicos.
        
        Transformação:
        - titulo → instruction (formatada como pergunta)
        - descricao → output
        - especialidade → input
        """
        records = []
        stats = SanitizationStats(total_input=len(df))
        seen_titles = set()
        
        for _, row in df.iterrows():
            # Limpa campos
            titulo = self.clean_text(str(row.get('titulo', '')))
            descricao = self.clean_text(str(row.get('descricao', '')))
            especialidade = self.clean_text(str(row.get('especialidade', '')))
            fonte = self.clean_text(str(row.get('fonte', '')))
            
            # Verifica se teve HTML removido
            if '<' in str(row.get('descricao', '')):
                stats.removed_html += 1
            
            # Cria instruction a partir do título
            if titulo and titulo != 'nan':
                # Formata como pergunta se não for
                if not titulo.endswith('?'):
                    instruction = f"Quais são as diretrizes do protocolo de {titulo}?"
                else:
                    instruction = titulo
            else:
                stats.removed_empty += 1
                continue
            
            # Remove duplicatas baseado no título normalizado
            title_key = titulo.lower().strip()
            if title_key in seen_titles:
                stats.removed_duplicate += 1
                continue
            seen_titles.add(title_key)
            
            # Cria input
            input_text = f"Especialidade: {especialidade}" if especialidade and especialidade != 'nan' else ""
            
            # Cria registro
            record = InstructionRecord(
                instruction=instruction,
                output=descricao,
                input=input_text,
                source=fonte or "CONITEC/MS",
                category="protocolos_medicos"
            )
            
            # Valida
            if not record.output:
                stats.removed_empty += 1
                continue
                
            if not self.validate_record(record):
                stats.removed_short += 1
                continue
            
            records.append(record)
        
        stats.total_output = len(records)
        self.stats['protocolos_medicos'] = stats
        
        logger.info(f"protocolos_medicos: {stats.total_input} → {stats.total_output} registros")
        return records
    
    def save_jsonl(self, records: List[InstructionRecord], filename: str) -> Path:
        """
        Salva registros em formato JSONL.
        
        Args:
            records: Lista de registros
            filename: Nome do arquivo (sem extensão)
            
        Returns:
            Caminho do arquivo salvo
        """
        output_path = self.output_dir / f"{filename}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Salvo: {output_path} ({len(records)} registros)")
        return output_path
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Gera relatório de sanitização.
        
        Returns:
            Dicionário com estatísticas
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "totals": {
                "total_input": 0,
                "total_output": 0,
                "removed_html": 0,
                "removed_short": 0,
                "removed_empty": 0,
                "removed_duplicate": 0
            }
        }
        
        for name, stats in self.stats.items():
            report["files"][name] = stats.to_dict()
            for key in report["totals"]:
                report["totals"][key] += getattr(stats, key, 0)
        
        return report
    
    def sanitize_all(self) -> Dict[str, List[InstructionRecord]]:
        """
        Processa todos os CSVs e gera arquivos JSONL.
        
        Returns:
            Dicionário com registros processados por tipo
        """
        all_records = {}
        
        # Processa perguntas frequentes
        perguntas_path = self.input_dir / "perguntas_frequentes.csv"
        if perguntas_path.exists():
            df = pd.read_csv(perguntas_path, encoding='utf-8-sig')
            all_records['perguntas_frequentes'] = self.process_perguntas_frequentes(df)
            self.save_jsonl(all_records['perguntas_frequentes'], 'perguntas_frequentes')
        else:
            logger.warning(f"Arquivo não encontrado: {perguntas_path}")
        
        # Processa modelos de laudos
        laudos_path = self.input_dir / "modelos_laudos.csv"
        if laudos_path.exists():
            df = pd.read_csv(laudos_path, encoding='utf-8-sig')
            all_records['modelos_laudos'] = self.process_modelos_laudos(df)
            self.save_jsonl(all_records['modelos_laudos'], 'modelos_laudos')
        else:
            logger.warning(f"Arquivo não encontrado: {laudos_path}")
        
        # Processa protocolos médicos
        protocolos_path = self.input_dir / "protocolos_medicos.csv"
        if protocolos_path.exists():
            df = pd.read_csv(protocolos_path, encoding='utf-8-sig')
            all_records['protocolos_medicos'] = self.process_protocolos_medicos(df)
            self.save_jsonl(all_records['protocolos_medicos'], 'protocolos_medicos')
        else:
            logger.warning(f"Arquivo não encontrado: {protocolos_path}")
        
        # Cria arquivo unificado
        unified_records = []
        for records in all_records.values():
            unified_records.extend(records)
        
        if unified_records:
            self.save_jsonl(unified_records, 'medical_data_unified')
        
        return all_records


def sanitize_all_csvs(input_dir: str = "data/processed", output_dir: str = "data/processed") -> Dict[str, Any]:
    """
    Função de conveniência para sanitizar todos os CSVs.
    
    Args:
        input_dir: Diretório com os CSVs
        output_dir: Diretório para saída
        
    Returns:
        Relatório de sanitização
    """
    sanitizer = DataSanitizer(input_dir, output_dir)
    sanitizer.sanitize_all()
    return sanitizer.generate_report()


if __name__ == "__main__":
    # Execução direta para testes
    report = sanitize_all_csvs()
    print(json.dumps(report, indent=2, ensure_ascii=False))

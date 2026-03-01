#!/usr/bin/env python3
"""
Script executável para sanitização de dados médicos.

Este script:
1. Lê os CSVs de dados médicos
2. Aplica sanitização (limpeza de texto, validação)
3. Converte para formato JSONL
4. Gera relatório de sanitização

Uso:
    python -m src.data_processing.run_sanitization
    
    ou diretamente:
    
    python src/data_processing/run_sanitization.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.sanitize_data import DataSanitizer

# Configuração de logging detalhado
def setup_logging(log_file: Path = None):
    """Configura logging detalhado."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def print_banner():
    """Imprime banner do script."""
    print("=" * 60)
    print("   SANITIZAÇÃO DE DADOS MÉDICOS - CSV → JSONL")
    print("=" * 60)
    print()


def print_stats_table(report: dict):
    """Imprime tabela de estatísticas."""
    print("\n" + "=" * 60)
    print("   RELATÓRIO DE SANITIZAÇÃO")
    print("=" * 60)
    
    # Cabeçalho
    print(f"\n{'Arquivo':<25} {'Entrada':>10} {'Saída':>10} {'Removidos':>12}")
    print("-" * 60)
    
    # Linhas por arquivo
    for name, stats in report.get('files', {}).items():
        entrada = stats.get('total_input', 0)
        saida = stats.get('total_output', 0)
        removidos = entrada - saida
        print(f"{name:<25} {entrada:>10} {saida:>10} {removidos:>12}")
    
    # Totais
    totals = report.get('totals', {})
    print("-" * 60)
    total_in = totals.get('total_input', 0)
    total_out = totals.get('total_output', 0)
    total_removed = total_in - total_out
    print(f"{'TOTAL':<25} {total_in:>10} {total_out:>10} {total_removed:>12}")
    
    # Detalhes de remoção
    print("\n" + "-" * 40)
    print("Detalhes dos registros removidos:")
    print(f"  - HTML/tags:    {totals.get('removed_html', 0):>5}")
    print(f"  - Muito curtos: {totals.get('removed_short', 0):>5}")
    print(f"  - Vazios:       {totals.get('removed_empty', 0):>5}")
    print(f"  - Duplicados:   {totals.get('removed_duplicate', 0):>5}")
    
    # Taxa de aproveitamento
    if total_in > 0:
        taxa = (total_out / total_in) * 100
        print(f"\n  Taxa de aproveitamento: {taxa:.1f}%")
    
    print("=" * 60)


def save_report(report: dict, output_dir: Path):
    """Salva relatório em arquivo JSON."""
    report_path = output_dir / "sanitization_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dumps(report, f, indent=2, ensure_ascii=False)
    return report_path


def main():
    """Função principal."""
    print_banner()
    
    # Define diretórios
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configura logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sanitization_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("Iniciando sanitização de dados médicos")
    logger.info(f"Diretório de dados: {data_dir}")
    logger.info(f"Log sendo salvo em: {log_file}")
    
    # Verifica se os arquivos existem
    csv_files = ['protocolos_medicos.csv', 'perguntas_frequentes.csv', 'modelos_laudos.csv']
    print("\nVerificando arquivos de entrada:")
    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            print(f"  ✓ {csv_file}")
            logger.info(f"Arquivo encontrado: {csv_file}")
        else:
            print(f"  ✗ {csv_file} (não encontrado)")
            logger.warning(f"Arquivo não encontrado: {csv_file}")
    
    # Executa sanitização
    print("\nExecutando sanitização...")
    sanitizer = DataSanitizer(str(data_dir), str(data_dir))
    
    try:
        all_records = sanitizer.sanitize_all()
        report = sanitizer.generate_report()
        
        # Exibe estatísticas
        print_stats_table(report)
        
        # Lista arquivos gerados
        print("\nArquivos JSONL gerados:")
        jsonl_files = list(data_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            size_kb = jsonl_file.stat().st_size / 1024
            print(f"  ✓ {jsonl_file.name} ({size_kb:.1f} KB)")
            logger.info(f"Arquivo gerado: {jsonl_file.name} ({size_kb:.1f} KB)")
        
        # Salva relatório
        report_path = data_dir / "sanitization_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Relatório salvo: {report_path.name}")
        
        logger.info("Sanitização concluída com sucesso")
        print("\n✓ Sanitização concluída com sucesso!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante sanitização: {e}", exc_info=True)
        print(f"\n✗ Erro durante sanitização: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

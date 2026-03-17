"""
Base de Dados Simulada de Prontuários Médicos
==============================================

Módulo que simula uma base de dados de prontuários médicos usando SQLite.
Contém dados fictícios de pacientes para demonstração do sistema RAG
e contextualização de consultas médicas.

Todos os dados são fictícios e gerados para fins educacionais.
"""

import sqlite3
import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Dados fictícios de prontuários
SAMPLE_PATIENTS = [
    {
        "id": 1,
        "nome": "Maria Silva Santos",
        "idade": 45,
        "sexo": "F",
        "peso_kg": 68.5,
        "altura_cm": 162,
        "tipo_sanguineo": "O+",
        "historico_medico": json.dumps([
            "Hipertensão arterial sistêmica (diagnosticada em 2018)",
            "Diabetes mellitus tipo 2 (diagnosticada em 2020)",
            "Dislipidemia"
        ], ensure_ascii=False),
        "medicacoes": json.dumps([
            {"nome": "Losartana 50mg", "posologia": "1x ao dia", "horario": "manhã"},
            {"nome": "Metformina 850mg", "posologia": "2x ao dia", "horario": "café e jantar"},
            {"nome": "Sinvastatina 20mg", "posologia": "1x ao dia", "horario": "noite"}
        ], ensure_ascii=False),
        "alergias": json.dumps(["Dipirona", "Sulfonamidas"], ensure_ascii=False),
        "exames_recentes": json.dumps([
            {"tipo": "Hemoglobina glicada (HbA1c)", "resultado": "7.2%", "data": "2026-01-15", "referencia": "< 7.0%"},
            {"tipo": "Glicemia em jejum", "resultado": "132 mg/dL", "data": "2026-01-15", "referencia": "70-99 mg/dL"},
            {"tipo": "Colesterol total", "resultado": "210 mg/dL", "data": "2026-01-15", "referencia": "< 200 mg/dL"},
            {"tipo": "Pressão arterial", "resultado": "140/90 mmHg", "data": "2026-02-20", "referencia": "< 130/80 mmHg"}
        ], ensure_ascii=False),
        "consultas_anteriores": json.dumps([
            {"data": "2026-02-20", "especialidade": "Clínica Geral", "queixa": "Controle de pressão e glicemia", "conduta": "Ajuste de Losartana para 100mg. Solicitados exames de controle."},
            {"data": "2025-11-10", "especialidade": "Endocrinologia", "queixa": "Revisão de diabetes", "conduta": "Manter Metformina. Orientação sobre dieta e exercícios."}
        ], ensure_ascii=False)
    },
    {
        "id": 2,
        "nome": "João Pedro Oliveira",
        "idade": 62,
        "sexo": "M",
        "peso_kg": 82.0,
        "altura_cm": 175,
        "tipo_sanguineo": "A+",
        "historico_medico": json.dumps([
            "Infarto agudo do miocárdio (2022)",
            "Hipertensão arterial sistêmica",
            "Insuficiência cardíaca classe funcional II (NYHA)",
            "Ex-tabagista (parou em 2022)"
        ], ensure_ascii=False),
        "medicacoes": json.dumps([
            {"nome": "AAS 100mg", "posologia": "1x ao dia", "horario": "almoço"},
            {"nome": "Enalapril 20mg", "posologia": "2x ao dia", "horario": "manhã e noite"},
            {"nome": "Carvedilol 25mg", "posologia": "2x ao dia", "horario": "manhã e noite"},
            {"nome": "Atorvastatina 40mg", "posologia": "1x ao dia", "horario": "noite"},
            {"nome": "Furosemida 40mg", "posologia": "1x ao dia", "horario": "manhã"}
        ], ensure_ascii=False),
        "alergias": json.dumps(["Penicilina"], ensure_ascii=False),
        "exames_recentes": json.dumps([
            {"tipo": "Ecocardiograma", "resultado": "FE 45%, hipocinesia anterior", "data": "2026-01-20", "referencia": "FE > 55%"},
            {"tipo": "BNP", "resultado": "320 pg/mL", "data": "2026-01-20", "referencia": "< 100 pg/mL"},
            {"tipo": "Creatinina", "resultado": "1.3 mg/dL", "data": "2026-02-10", "referencia": "0.7-1.2 mg/dL"},
            {"tipo": "Potássio sérico", "resultado": "4.8 mEq/L", "data": "2026-02-10", "referencia": "3.5-5.0 mEq/L"}
        ], ensure_ascii=False),
        "consultas_anteriores": json.dumps([
            {"data": "2026-02-10", "especialidade": "Cardiologia", "queixa": "Dispneia aos esforços moderados", "conduta": "Manter medicações. Ecocardiograma de controle em 6 meses."},
            {"data": "2025-12-05", "especialidade": "Cardiologia", "queixa": "Consulta de rotina pós-IAM", "conduta": "Estável. Reforço de adesão à medicação e atividade física leve."}
        ], ensure_ascii=False)
    },
    {
        "id": 3,
        "nome": "Ana Beatriz Costa",
        "idade": 28,
        "sexo": "F",
        "peso_kg": 55.0,
        "altura_cm": 165,
        "tipo_sanguineo": "B-",
        "historico_medico": json.dumps([
            "Asma brônquica moderada persistente (desde infância)",
            "Rinite alérgica",
            "Ansiedade generalizada"
        ], ensure_ascii=False),
        "medicacoes": json.dumps([
            {"nome": "Budesonida/Formoterol 200/6mcg", "posologia": "2x ao dia", "horario": "manhã e noite"},
            {"nome": "Salbutamol spray 100mcg", "posologia": "SOS", "horario": "conforme necessidade"},
            {"nome": "Fexofenadina 180mg", "posologia": "1x ao dia", "horario": "noite"},
            {"nome": "Sertralina 50mg", "posologia": "1x ao dia", "horario": "manhã"}
        ], ensure_ascii=False),
        "alergias": json.dumps(["AINEs (ibuprofeno, naproxeno)", "Ácaros", "Pólen"], ensure_ascii=False),
        "exames_recentes": json.dumps([
            {"tipo": "Espirometria", "resultado": "VEF1 78% do previsto, resposta ao broncodilatador positiva", "data": "2026-02-01", "referencia": "VEF1 > 80%"},
            {"tipo": "IgE total", "resultado": "450 UI/mL", "data": "2026-02-01", "referencia": "< 100 UI/mL"},
            {"tipo": "Hemograma", "resultado": "Eosinofilia leve (8%)", "data": "2026-02-01", "referencia": "1-5%"}
        ], ensure_ascii=False),
        "consultas_anteriores": json.dumps([
            {"data": "2026-02-01", "especialidade": "Pneumologia", "queixa": "Crises de falta de ar noturnas", "conduta": "Ajuste de Budesonida/Formoterol. Solicitada espirometria de controle em 3 meses."},
            {"data": "2025-10-15", "especialidade": "Psiquiatria", "queixa": "Ansiedade e insônia", "conduta": "Início de Sertralina 50mg. Encaminhamento para psicoterapia."}
        ], ensure_ascii=False)
    },
    {
        "id": 4,
        "nome": "Carlos Eduardo Ferreira",
        "idade": 55,
        "sexo": "M",
        "peso_kg": 95.0,
        "altura_cm": 178,
        "tipo_sanguineo": "AB+",
        "historico_medico": json.dumps([
            "Diabetes mellitus tipo 2 (diagnosticada em 2015)",
            "Obesidade grau I (IMC 30)",
            "Esteatose hepática não alcoólica",
            "Gota",
            "Apneia obstrutiva do sono"
        ], ensure_ascii=False),
        "medicacoes": json.dumps([
            {"nome": "Metformina 1000mg", "posologia": "2x ao dia", "horario": "café e jantar"},
            {"nome": "Glicazida 60mg MR", "posologia": "1x ao dia", "horario": "café"},
            {"nome": "Alopurinol 300mg", "posologia": "1x ao dia", "horario": "almoço"},
            {"nome": "Omeprazol 20mg", "posologia": "1x ao dia", "horario": "jejum"}
        ], ensure_ascii=False),
        "alergias": json.dumps(["Nenhuma alergia conhecida"], ensure_ascii=False),
        "exames_recentes": json.dumps([
            {"tipo": "HbA1c", "resultado": "8.5%", "data": "2026-02-15", "referencia": "< 7.0%"},
            {"tipo": "Glicemia em jejum", "resultado": "168 mg/dL", "data": "2026-02-15", "referencia": "70-99 mg/dL"},
            {"tipo": "TGO/TGP", "resultado": "58/72 U/L", "data": "2026-02-15", "referencia": "< 40 U/L"},
            {"tipo": "Ácido úrico", "resultado": "7.8 mg/dL", "data": "2026-02-15", "referencia": "2.5-7.0 mg/dL"},
            {"tipo": "Ultrassom abdome", "resultado": "Esteatose hepática moderada", "data": "2026-01-10", "referencia": "Normal"}
        ], ensure_ascii=False),
        "consultas_anteriores": json.dumps([
            {"data": "2026-02-15", "especialidade": "Endocrinologia", "queixa": "Descontrole glicêmico", "conduta": "Adicionar Glicazida 60mg. Reforço dietético. Encaminhamento para nutricionista."},
            {"data": "2025-12-20", "especialidade": "Hepatologia", "queixa": "Alteração de enzimas hepáticas", "conduta": "USG abdome. Perda de peso é fundamental. Evitar álcool."}
        ], ensure_ascii=False)
    },
    {
        "id": 5,
        "nome": "Lucia Helena Rodrigues",
        "idade": 72,
        "sexo": "F",
        "peso_kg": 58.0,
        "altura_cm": 155,
        "tipo_sanguineo": "O-",
        "historico_medico": json.dumps([
            "Osteoporose (diagnosticada em 2019)",
            "Hipotireoidismo",
            "Artrose de joelhos bilateral",
            "Depressão (em remissão)",
            "Fratura de rádio distal (2024)"
        ], ensure_ascii=False),
        "medicacoes": json.dumps([
            {"nome": "Alendronato 70mg", "posologia": "1x por semana", "horario": "domingo em jejum"},
            {"nome": "Cálcio 500mg + Vitamina D 1000UI", "posologia": "2x ao dia", "horario": "almoço e jantar"},
            {"nome": "Levotiroxina 75mcg", "posologia": "1x ao dia", "horario": "jejum (30min antes do café)"},
            {"nome": "Paracetamol 750mg", "posologia": "até 3x ao dia", "horario": "SOS para dor"}
        ], ensure_ascii=False),
        "alergias": json.dumps(["Contraste iodado", "Frutos do mar"], ensure_ascii=False),
        "exames_recentes": json.dumps([
            {"tipo": "Densitometria óssea", "resultado": "T-score fêmur -2.8, coluna -3.1", "data": "2026-01-05", "referencia": "T-score > -2.5 normal"},
            {"tipo": "TSH", "resultado": "3.2 mUI/L", "data": "2026-02-01", "referencia": "0.4-4.0 mUI/L"},
            {"tipo": "Vitamina D (25-OH)", "resultado": "28 ng/mL", "data": "2026-02-01", "referencia": "30-60 ng/mL"},
            {"tipo": "Cálcio sérico", "resultado": "9.0 mg/dL", "data": "2026-02-01", "referencia": "8.5-10.5 mg/dL"}
        ], ensure_ascii=False),
        "consultas_anteriores": json.dumps([
            {"data": "2026-02-01", "especialidade": "Reumatologia", "queixa": "Dor nos joelhos e controle de osteoporose", "conduta": "Manter Alendronato. Aumentar vitamina D para 2000UI/dia. Fisioterapia para joelhos."},
            {"data": "2025-11-20", "especialidade": "Endocrinologia", "queixa": "Controle de tireoide", "conduta": "TSH dentro da faixa. Manter Levotiroxina 75mcg."}
        ], ensure_ascii=False)
    }
]


class PatientDatabase:
    """
    Gerencia a base de dados simulada de prontuários médicos.
    Utiliza SQLite como backend para persistência e consultas estruturadas.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa a base de dados de prontuários.

        Args:
            db_path: Caminho para o arquivo SQLite. Se None, usa in-memory.
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "prontuarios.db"
        )
        self._init_database()
        logger.info(f"PatientDatabase inicializado em: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Cria e retorna uma conexão ao banco SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Inicializa o schema e popula com dados de exemplo se vazio."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pacientes (
                id INTEGER PRIMARY KEY,
                nome TEXT NOT NULL,
                idade INTEGER NOT NULL,
                sexo TEXT NOT NULL,
                peso_kg REAL,
                altura_cm REAL,
                tipo_sanguineo TEXT,
                historico_medico TEXT,
                medicacoes TEXT,
                alergias TEXT,
                exames_recentes TEXT,
                consultas_anteriores TEXT
            )
        """)

        # Popula se estiver vazio
        cursor.execute("SELECT COUNT(*) FROM pacientes")
        if cursor.fetchone()[0] == 0:
            for patient in SAMPLE_PATIENTS:
                cursor.execute("""
                    INSERT INTO pacientes 
                    (id, nome, idade, sexo, peso_kg, altura_cm, tipo_sanguineo,
                     historico_medico, medicacoes, alergias, exames_recentes, consultas_anteriores)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient["id"], patient["nome"], patient["idade"],
                    patient["sexo"], patient["peso_kg"], patient["altura_cm"],
                    patient["tipo_sanguineo"], patient["historico_medico"],
                    patient["medicacoes"], patient["alergias"],
                    patient["exames_recentes"], patient["consultas_anteriores"]
                ))
            conn.commit()
            logger.info(f"Base populada com {len(SAMPLE_PATIENTS)} prontuários fictícios")

        conn.close()

    def get_patient_by_id(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Recupera prontuário completo de um paciente pelo ID.

        Args:
            patient_id: ID do paciente

        Returns:
            Dicionário com dados do paciente ou None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pacientes WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_dict(row)
        return None

    def search_patient_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Busca pacientes pelo nome (busca parcial, case-insensitive).

        Args:
            name: Nome ou parte do nome do paciente

        Returns:
            Lista de prontuários encontrados
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM pacientes WHERE LOWER(nome) LIKE ?",
            (f"%{name.lower()}%",)
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_dict(row) for row in rows]

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """
        Retorna todos os pacientes da base.

        Returns:
            Lista com todos os prontuários
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pacientes")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_dict(row) for row in rows]

    def get_patient_summary(self, patient_id: int) -> Optional[str]:
        """
        Gera um resumo textual do prontuário para injeção no contexto do LLM.

        Args:
            patient_id: ID do paciente

        Returns:
            String com resumo formatado ou None
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return None

        historico = ", ".join(patient["historico_medico"])
        alergias = ", ".join(patient["alergias"])

        meds = []
        for med in patient["medicacoes"]:
            meds.append(f"{med['nome']} ({med['posologia']})")
        medicacoes_str = "; ".join(meds)

        exames = []
        for ex in patient["exames_recentes"]:
            exames.append(f"{ex['tipo']}: {ex['resultado']} (ref: {ex['referencia']}, data: {ex['data']})")
        exames_str = "; ".join(exames)

        summary = (
            f"PRONTUÁRIO DO PACIENTE:\n"
            f"Nome: {patient['nome']} | Idade: {patient['idade']} anos | Sexo: {patient['sexo']}\n"
            f"Peso: {patient['peso_kg']}kg | Altura: {patient['altura_cm']}cm | "
            f"Tipo sanguíneo: {patient['tipo_sanguineo']}\n"
            f"Histórico: {historico}\n"
            f"Medicações em uso: {medicacoes_str}\n"
            f"Alergias: {alergias}\n"
            f"Exames recentes: {exames_str}"
        )
        return summary

    def get_patient_context_for_query(self, patient_id: int, query: str) -> Optional[str]:
        """
        Retorna contexto relevante do paciente para uma pergunta específica.
        Filtra informações do prontuário com base em palavras-chave da pergunta.

        Args:
            patient_id: ID do paciente
            query: Pergunta do usuário

        Returns:
            Contexto relevante formatado
        """
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return None

        query_lower = query.lower()
        context_parts = [
            f"Paciente: {patient['nome']}, {patient['idade']} anos, {patient['sexo']}"
        ]

        # Sempre incluir alergias (segurança)
        alergias = ", ".join(patient["alergias"])
        context_parts.append(f"Alergias: {alergias}")

        # Incluir histórico se relevante
        keywords_historico = ["histórico", "doença", "diagnóstico", "condição", "antecedente"]
        if any(kw in query_lower for kw in keywords_historico) or True:  # sempre incluir
            historico = ", ".join(patient["historico_medico"])
            context_parts.append(f"Histórico: {historico}")

        # Incluir medicações
        keywords_med = ["medicação", "medicamento", "remédio", "droga", "tratamento", "prescrição"]
        if any(kw in query_lower for kw in keywords_med) or True:
            meds = [f"{m['nome']} ({m['posologia']})" for m in patient["medicacoes"]]
            context_parts.append(f"Medicações: {'; '.join(meds)}")

        # Incluir exames se relevante
        keywords_exame = ["exame", "resultado", "laboratório", "lab", "teste"]
        if any(kw in query_lower for kw in keywords_exame) or True:
            exames = [f"{e['tipo']}: {e['resultado']}" for e in patient["exames_recentes"]]
            context_parts.append(f"Exames recentes: {'; '.join(exames)}")

        return "\n".join(context_parts)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converte uma row do SQLite em dicionário com campos JSON parseados."""
        d = dict(row)
        json_fields = ["historico_medico", "medicacoes", "alergias", "exames_recentes", "consultas_anteriores"]
        for field in json_fields:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def list_patients_brief(self) -> List[Dict[str, Any]]:
        """
        Lista resumida de pacientes (id, nome, idade).

        Returns:
            Lista com informações básicas de cada paciente
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, nome, idade, sexo FROM pacientes")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]



if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()

    print("=" * 60)
    print("  PatientDatabase - Demonstração")
    print("=" * 60)
    print()

    db = PatientDatabase()

    # Lista todos os pacientes
    patients = db.list_patients_brief()
    print(f"Total de pacientes na base: {len(patients)}")
    print()
    for p in patients:
        print(f"  ID {p['id']:>2}: {p['nome']:<30} | Idade: {p['idade']} | Sexo: {p['sexo']}")
    print()

    # Detalhe de um paciente
    if patients:
        pid = patients[0]["id"]
        patient = db.get_patient_by_id(pid)
        if patient:
            print(f"Detalhes do paciente ID {pid}:")
            for k, v in patient.items():
                print(f"  {k}: {v}")
            print()

        # Resumo clínico
        summary = db.get_patient_summary(pid)
        if summary:
            print(f"Resumo clínico do paciente ID {pid}:")
            print(summary[:500])
            print()

    # Busca por nome
    search_name = "Maria"
    results = db.search_patient_by_name(search_name)
    print(f"Busca por nome '{search_name}': {len(results)} resultado(s)")
    for r in results:
        print(f"  - {r['nome']} (ID: {r['id']})")
    print()

    print("[OK] Demonstração concluída.")

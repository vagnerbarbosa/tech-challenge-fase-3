"""
Módulo RAG (Retrieval-Augmented Generation)
============================================

Implementa busca semântica sobre protocolos médicos coletados pelos scrapers
e prontuários de pacientes, injetando contexto relevante nas consultas ao LLM.

Utiliza:
- TF-IDF + similaridade de cosseno como fallback leve (sem GPU)
- Suporte a embeddings via sentence-transformers quando disponível
- Indexação de arquivos JSONL gerados pelos scrapers
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MedicalRAG:
    """
    Sistema RAG para busca semântica em protocolos médicos.
    Indexa os dados coletados pelos scrapers e fornece contexto
    relevante para as respostas do assistente.
    """

    def __init__(self, data_dir: Optional[str] = None, use_embeddings: bool = False):
        """
        Inicializa o sistema RAG.

        Args:
            data_dir: Diretório com arquivos JSONL dos scrapers.
                      Default: ./data/raw
            use_embeddings: Se True, tenta usar sentence-transformers para embeddings.
                           Se False ou indisponível, usa TF-IDF.
        """
        self.data_dir = Path(data_dir or os.getenv("DATA_PATH", "./data")) / "raw"
        self.documents: List[Dict[str, Any]] = []
        self.use_embeddings = use_embeddings
        self._vectorizer = None
        self._tfidf_matrix = None
        self._embedding_model = None
        self._embeddings = None

        # Carrega e indexa documentos
        self._load_documents()
        self._build_index()

        logger.info(
            f"MedicalRAG inicializado com {len(self.documents)} documentos. "
            f"Método: {'embeddings' if self._embedding_model else 'TF-IDF'}"
        )

    def _load_documents(self) -> None:
        """Carrega todos os documentos JSONL do diretório de dados."""
        self.documents = []

        if not self.data_dir.exists():
            logger.warning(f"Diretório de dados não encontrado: {self.data_dir}")
            return

        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"Nenhum arquivo JSONL encontrado em {self.data_dir}")
            return

        for filepath in jsonl_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            # Compõe texto completo para indexação
                            text_parts = [
                                record.get("instruction", ""),
                                record.get("input", ""),
                                record.get("output", ""),
                            ]
                            full_text = " ".join(p for p in text_parts if p).strip()

                            if full_text:
                                self.documents.append({
                                    "text": full_text,
                                    "instruction": record.get("instruction", ""),
                                    "output": record.get("output", ""),
                                    "source": record.get("source", filepath.stem),
                                    "file": filepath.name,
                                })
                        except json.JSONDecodeError:
                            logger.debug(f"Linha inválida em {filepath.name}:{line_num}")
            except Exception as e:
                logger.error(f"Erro ao carregar {filepath}: {e}")

        logger.info(f"Carregados {len(self.documents)} documentos de {len(jsonl_files)} arquivos")

    def _build_index(self) -> None:
        """Constrói o índice de busca (TF-IDF ou embeddings)."""
        if not self.documents:
            logger.warning("Nenhum documento para indexar")
            return

        texts = [doc["text"] for doc in self.documents]

        # Tenta usar embeddings se solicitado
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
                logger.info("Índice de embeddings criado com sucesso")
                return
            except ImportError:
                logger.info("sentence-transformers não disponível, usando TF-IDF")
            except Exception as e:
                logger.warning(f"Erro ao criar embeddings: {e}. Usando TF-IDF")

        # Fallback: TF-IDF com sklearn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # Manter termos médicos em português
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            logger.info("Índice TF-IDF criado com sucesso")
        except ImportError:
            logger.warning("scikit-learn não disponível. RAG operará sem busca semântica.")

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes para uma query.

        Args:
            query: Texto da consulta
            top_k: Número máximo de resultados
            min_score: Score mínimo para incluir resultado

        Returns:
            Lista de documentos relevantes com scores
        """
        if not self.documents:
            return []

        # Busca com embeddings
        if self._embedding_model is not None and self._embeddings is not None:
            return self._search_embeddings(query, top_k, min_score)

        # Busca com TF-IDF
        if self._vectorizer is not None and self._tfidf_matrix is not None:
            return self._search_tfidf(query, top_k, min_score)

        # Fallback: busca por palavras-chave
        return self._search_keywords(query, top_k)

    def _search_tfidf(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """Busca usando TF-IDF + cosseno."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Ordena por score descendente
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = round(score, 4)
                results.append(doc)

        return results

    def _search_embeddings(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """Busca usando embeddings + cosseno."""
        query_embedding = self._embedding_model.encode([query])

        # Similaridade de cosseno
        scores = np.dot(self._embeddings, query_embedding.T).flatten()
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1, norms)
        scores = scores / norms

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = round(score, 4)
                results.append(doc)

        return results

    def _search_keywords(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Busca simples por palavras-chave (fallback)."""
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_words = set(re.findall(r'\w+', doc["text"].lower()))
            common = query_words & doc_words
            if common:
                score = len(common) / max(len(query_words), 1)
                d = doc.copy()
                d["relevance_score"] = round(score, 4)
                scored_docs.append(d)

        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_docs[:top_k]

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 3,
        patient_context: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Gera contexto completo para uma consulta, combinando:
        - Documentos relevantes dos protocolos médicos
        - Dados do prontuário do paciente (se fornecido)

        Args:
            query: Pergunta do usuário
            top_k: Número de documentos a recuperar
            patient_context: Contexto do paciente (do PatientDatabase)

        Returns:
            Tuple com (contexto_formatado, lista_de_fontes)
        """
        context_parts = []
        sources = []

        # 1. Contexto do paciente
        if patient_context:
            context_parts.append(f"[DADOS DO PACIENTE]\n{patient_context}")
            sources.append({
                "tipo": "Prontuário do Paciente",
                "descricao": "Base de dados interna de prontuários"
            })

        # 2. Documentos relevantes dos protocolos
        relevant_docs = self.search(query, top_k=top_k)
        if relevant_docs:
            context_parts.append("[PROTOCOLOS E REFERÊNCIAS MÉDICAS]")
            for i, doc in enumerate(relevant_docs, 1):
                source_name = doc.get("source", "Fonte desconhecida")
                instruction = doc.get("instruction", "")
                output = doc.get("output", "")

                context_parts.append(
                    f"Referência {i} (Fonte: {source_name}):\n"
                    f"  Tópico: {instruction}\n"
                    f"  Informação: {output[:500]}"
                )

                sources.append({
                    "tipo": source_name,
                    "descricao": instruction[:200] if instruction else doc.get("file", ""),
                    "relevancia": doc.get("relevance_score", 0)
                })

        context = "\n\n".join(context_parts) if context_parts else ""
        return context, sources

    def format_citations(self, sources: List[Dict[str, str]]) -> str:
        """
        Formata as fontes como citações legíveis para inclusão na resposta.

        Args:
            sources: Lista de fontes retornada por get_context_for_query

        Returns:
            String formatada com citações
        """
        if not sources:
            return ""

        citation_lines = ["\n📚 **Fontes consultadas:**"]
        seen = set()

        for i, src in enumerate(sources, 1):
            tipo = src.get("tipo", "Referência")
            descricao = src.get("descricao", "")
            key = f"{tipo}:{descricao}"

            if key not in seen:
                seen.add(key)
                if descricao:
                    citation_lines.append(f"  [{i}] {tipo} - {descricao}")
                else:
                    citation_lines.append(f"  [{i}] {tipo}")

        return "\n".join(citation_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do índice RAG."""
        source_counts = {}
        for doc in self.documents:
            src = doc.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "total_documents": len(self.documents),
            "index_type": "embeddings" if self._embedding_model else (
                "tfidf" if self._vectorizer else "keywords"
            ),
            "sources": source_counts,
            "data_dir": str(self.data_dir),
        }



if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()

    print("=" * 60)
    print("  MedicalRAG - Demonstração")
    print("=" * 60)
    print()

    rag = MedicalRAG()

    # Estatísticas do índice
    stats = rag.get_stats()
    print("[INFO] Estatísticas do RAG:")
    print(f"  Total de documentos: {stats['total_documents']}")
    print(f"  Tipo de índice: {stats['index_type']}")
    print(f"  Diretório de dados: {stats['data_dir']}")
    if stats.get("sources"):
        print("  Fontes:")
        for source, count in stats["sources"].items():
            print(f"    - {source}: {count} documentos")
    print()

    # Busca de exemplo
    test_queries = [
        "diabetes tipo 2 tratamento",
        "dor de cabeça sintomas",
        "hipertensão arterial",
    ]

    for query in test_queries:
        print(f"Busca: '{query}'")
        results = rag.search(query, top_k=3)
        if results:
            for i, r in enumerate(results, 1):
                title = r.get("title", r.get("source", "N/A"))
                score = r.get("score", 0)
                print(f"  {i}. [{score:.3f}] {title}")
        else:
            print("  Nenhum resultado encontrado.")
        print()

    print("[OK] Demonstração concluída.")

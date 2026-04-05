"""
Vistaar — RAG Query Engine
Handles semantic search + LLM answer generation, scoped per company.
"""

from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

import config

# ── Initialise once at import time ──────────────────────────────────────────
Settings.embed_model = OllamaEmbedding(
    model_name=config.EMBED_MODEL,
    base_url=config.OLLAMA_BASE_URL,
)
Settings.llm = Ollama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    request_timeout=config.LLM_TIMEOUT,
    system_prompt=config.SYSTEM_PROMPT,
)

_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
_vector_store = QdrantVectorStore(client=_client, collection_name=config.COLLECTION_NAME)
_index = VectorStoreIndex.from_vector_store(_vector_store)


def _append_rag_flow_log(
    question: str,
    company_id: str,
    data_type: str | None,
    response,
) -> None:
    """Append human-readable trace: question → retrieved nodes → LLM answer."""
    if not getattr(config, "RAG_DEBUG_LOG", False):
        return

    log_dir = Path(getattr(config, "RAG_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "rag_flow.log"
    max_chars = int(getattr(config, "RAG_LOG_MAX_NODE_CHARS", 4000))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts: list[str] = []
    parts.append(f"\n{'=' * 72}\n{ts}\n")
    parts.append(f"question: {question!r}\n")
    parts.append(f"company_id: {company_id!r}\n")
    parts.append(f"data_type: {data_type!r}\n")
    parts.append(f"top_k (config): {config.TOP_K}\n")

    nodes = getattr(response, "source_nodes", None) or []
    parts.append(f"\n--- RETRIEVED SOURCE NODES ({len(nodes)}) ---\n")

    for i, node in enumerate(nodes, 1):
        score = getattr(node, "score", None)
        meta = getattr(node, "metadata", {}) or {}
        text = getattr(node, "text", "") or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated for log]"

        parts.append(f"\n--- node [{i}] score={score!r} ---\n")
        parts.append(f"metadata: {meta}\n")
        parts.append(f"text:\n{text}\n")

    parts.append("\n--- FINAL ANSWER (LLM OUTPUT) ---\n")
    parts.append(str(response))
    parts.append("\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("".join(parts))


def ask(question: str, company_id: str, data_type: str = None) -> dict:
    """
    Ask a question scoped to a specific company (and optionally data type).

    Returns
    -------
    {
        "answer": str,
        "sources": [ { "text", "data_type", "source_file", "sheet" }, ... ]
    }
    """
    # Build metadata filters
    filter_list = [
        MetadataFilter(
            key="company_id",
            value=company_id,
            operator=FilterOperator.EQ,
        )
    ]
    if data_type:
        filter_list.append(
            MetadataFilter(
                key="data_type",
                value=data_type,
                operator=FilterOperator.EQ,
            )
        )

    filters = MetadataFilters(filters=filter_list)

    query_engine = _index.as_query_engine(
        similarity_top_k=config.TOP_K,
        filters=filters,
    )

    response = query_engine.query(question)

    _append_rag_flow_log(question, company_id, data_type, response)

    sources = []
    for node in response.source_nodes:
        sources.append({
            "text":        node.text[:300],
            "data_type":   node.metadata.get("data_type", ""),
            "source_file": node.metadata.get("source_file", ""),
            "sheet":       node.metadata.get("sheet", ""),
            "score":       round(node.score, 4) if node.score else None,
        })

    return {
        "answer":  str(response),
        "sources": sources,
    }

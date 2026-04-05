"""
Vistaar — central configuration
Edit these values to match your local setup.
"""

# ── Ollama ──────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
LLM_MODEL         = "llama3.1"        # ollama pull llama3.1
EMBED_MODEL       = "nomic-embed-text" # ollama pull nomic-embed-text

# ── Qdrant ──────────────────────────────────────────────
QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
COLLECTION_NAME   = "vistaar_data"

# ── RAG tuning ──────────────────────────────────────────
CHUNK_SIZE        = 512   # tokens per chunk
CHUNK_OVERLAP     = 64    # overlap between chunks
TOP_K             = 10     # how many chunks to retrieve per query
LLM_TIMEOUT       = 200.0 # seconds

# ── RAG debug / observability ───────────────────────────
# When True, each /chat request appends retrieval + answer to logs/rag_flow.log
RAG_DEBUG_LOG         = True
RAG_LOG_DIR           = "logs"
RAG_LOG_MAX_NODE_CHARS = 4000  # per retrieved chunk in the log file

# ── System prompt ───────────────────────────────────────
SYSTEM_PROMPT = """
You are Vistaar AI — a manufacturing business intelligence assistant.

Rules:
1. Answer ONLY using the context chunks provided below.
2. If the question asks for total/sum/average, compute from EVERY matching row in the context.
   Do not drop a line item. Show the arithmetic explicitly, e.g. 1200 + 800 + 2000 = 4000,
   then state the total. Double-check addition before answering.
3. Show individual breakdown AND the total when relevant.
4. Currency: use the ₹ symbol ONLY for rupee amounts (revenue, price, cost in ₹).
   For units sold, quantities, counts, or percentages, use plain numbers — never put ₹ in front of unit counts.
5. If data is missing in the context, say so clearly. Never guess or invent figures.
6. Do not mix data from different companies.
"""

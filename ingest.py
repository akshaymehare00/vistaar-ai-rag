"""
Vistaar — Ingestion Pipeline
Loads Excel/CSV files → chunks → embeds → stores in Qdrant vector DB.

Usage:
    python ingest.py
    python ingest.py --file data/acme_corp_sales_q3_2024.xlsx --company acme_corp --type sales
"""

import argparse
import os
import pandas as pd

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config


def setup_settings():
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    Settings.transformations = [
        SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    ]


def get_vector_store():
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"  Created Qdrant collection: {config.COLLECTION_NAME}")

    return QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
    )


def ingest_file(file_path: str, company_id: str, data_type: str):
    """
    Ingest a single Excel or CSV file into the vector DB.

    Parameters
    ----------
    file_path   : path to .xlsx or .csv
    company_id  : unique identifier, e.g. "acme_corp"
    data_type   : category, e.g. "sales", "production", "finance"
    """
    print(f"\n  Ingesting: {file_path}")
    print(f"  Company  : {company_id}  |  Type: {data_type}")

    ext = os.path.splitext(file_path)[1].lower()

    # Load all sheets from Excel, or single sheet from CSV
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(file_path, sheet_name=None)
    else:
        sheets = {"Sheet1": pd.read_csv(file_path)}

    documents = []
    total_rows = 0

    for sheet_name, df in sheets.items():
        df = df.fillna("").astype(str)

        # Group rows into batches of 5 so the LLM gets
        # multiple related rows together (e.g. all July rows)
        batch_size = 5
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            # Build a multi-row text block
            text = f"[{sheet_name}]\n"
            for _, row in batch.iterrows():
                text += " | ".join(
                    [f"{col}: {val}" for col, val in row.items() if val != ""]
                ) + "\n"

            # ✅ Real metadata dict — no ellipsis
            doc = Document(
                text=text,
                metadata={
                    "company_id":  company_id,
                    "data_type":   data_type,
                    "source_file": os.path.basename(file_path),
                    "sheet":       sheet_name,
                }
            )
            documents.append(doc)

        total_rows += len(df)

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"  Done — {total_rows} rows ingested across {len(sheets)} sheet(s).")


def ingest_all_defaults():
    """Ingest the 3 bundled test files."""
    test_files = [
        {
            "file_path": "data/acme_corp_sales_q3_2024.xlsx",
            "company_id": "acme_corp",
            "data_type":  "sales",
        },
        {
            "file_path": "data/beta_ltd_production_2024.xlsx",
            "company_id": "beta_ltd",
            "data_type":  "production",
        },
        {
            "file_path": "data/gamma_inc_finance_2024.xlsx",
            "company_id": "gamma_inc",
            "data_type":  "finance",
        },
    ]
    for entry in test_files:
        ingest_file(**entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vistaar ingestion pipeline")
    parser.add_argument("--file",    help="Path to Excel/CSV file")
    parser.add_argument("--company", help="Company ID, e.g. acme_corp")
    parser.add_argument("--type",    help="Data type, e.g. sales / production / finance")
    args = parser.parse_args()

    print("\nVistaar Ingestion Pipeline")
    print("=" * 40)
    setup_settings()

    if args.file and args.company and args.type:
        ingest_file(args.file, args.company, args.type)
    else:
        print("No arguments given — ingesting all 3 default test files...\n")
        ingest_all_defaults()

    print("\nIngestion complete!")
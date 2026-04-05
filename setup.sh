#!/bin/bash
# Vistaar — Quick Setup Script
# Run: bash setup.sh

echo ""
echo "====================================="
echo "  Vistaar AI — Setup Script"
echo "====================================="
echo ""

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."

echo ""
echo "[2/4] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "      Ollama found. Pulling models (this may take a while)..."
    ollama pull llama3.1
    ollama pull nomic-embed-text
    echo "      Models ready."
else
    echo "      Ollama NOT found. Install it from https://ollama.ai/download"
    echo "      Then run: ollama pull llama3.1 && ollama pull nomic-embed-text"
fi

echo ""
echo "[3/4] Checking Qdrant..."
if docker ps | grep qdrant &> /dev/null; then
    echo "      Qdrant already running."
elif command -v docker &> /dev/null; then
    echo "      Starting Qdrant via Docker..."
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
    echo "      Qdrant started."
else
    echo "      Docker NOT found. Install from https://docs.docker.com/get-docker/"
    echo "      Then run: docker run -d -p 6333:6333 qdrant/qdrant"
fi

echo ""
echo "[4/4] Ingesting test data..."
python ingest.py
echo "      Ingestion complete."

echo ""
echo "====================================="
echo "  Setup complete!"
echo "  Start the server: uvicorn main:app --reload --port 8000"
echo "  Open browser:     http://localhost:8000"
echo "====================================="
echo ""

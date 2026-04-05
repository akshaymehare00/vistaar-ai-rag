"""
Vistaar — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os

from query import ask

app = FastAPI(
    title="Vistaar AI",
    description="AI-powered Business Intelligence for Manufacturing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML chatbot UI from /static
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    company_id: str                   # e.g. "acme_corp"
    data_type:  Optional[str] = None  # optional filter: "sales" / "production" / "finance"

class SourceItem(BaseModel):
    text:        str
    data_type:   str
    source_file: str
    sheet:       str
    score:       Optional[float]

class ChatResponse(BaseModel):
    answer:  str
    sources: List[SourceItem]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the chatbot HTML UI."""
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Ask a question about a company's data.

    Example body:
    {
        "question":   "What was the total revenue in Q3?",
        "company_id": "acme_corp",
        "data_type":  "sales"
    }
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not req.company_id.strip():
        raise HTTPException(status_code=400, detail="company_id is required.")

    try:
        result = ask(
            question=req.question,
            company_id=req.company_id,
            data_type=req.data_type,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
def list_companies():
    """Returns the available test companies."""
    return {
        "companies": [
            {"id": "acme_corp",  "name": "Acme Corp",  "data_types": ["sales"]},
            {"id": "beta_ltd",   "name": "Beta Ltd",   "data_types": ["production"]},
            {"id": "gamma_inc",  "name": "Gamma Inc",  "data_types": ["finance"]},
        ]
    }


@app.get("/health")
def health():
    return {"status": "Vistaar is running", "version": "1.0.0"}

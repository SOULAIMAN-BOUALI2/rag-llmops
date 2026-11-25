import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import AdvancedLocalRAG as LocalRAG

app = FastAPI(title="RAG + LLMOps Pipeline")

# Détecte automatiquement le premier PDF dans ./data/
pdf_files = glob.glob("data/*.pdf")
if not pdf_files:
    raise RuntimeError("Aucun fichier PDF trouvé dans le dossier data/")

# Change le nom du modèle si tu utilises llama3.1, phi3, mistral, etc.
rag = LocalRAG(pdf_path=pdf_files[0], llm_model="qwen2.5-coder:7b")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
def ask_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")
    return rag.query(question=req.question, top_k=req.top_k)
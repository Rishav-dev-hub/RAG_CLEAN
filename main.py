# ============================================================
#  RAG FROM SCRATCH  —  main.py
#  A fully working FastAPI backend with RAG
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from rag_engine import RAGEngine   # our RAG logic lives here

# ── App setup ────────────────────────────────────────────────
app = FastAPI(title="RAG From Scratch", version="1.0")

# Allow the browser frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend HTML at http://localhost:8000
app.mount("/static", StaticFiles(directory="static"), name="static")

# One shared RAG engine for the whole server
rag = RAGEngine()


# ── Request / Response shapes ────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    chunks_used: list[str]
    chunk_ids: list[int]


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the chat UI."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file.
    The server will read, chunk, and index it automatically.
    """
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(400, "Only .txt and .pdf files are supported.")

    content = await file.read()

    # Save to /documents folder
    save_path = os.path.join("documents", file.filename)
    with open(save_path, "wb") as f:
        f.write(content)

    # Index it into the RAG engine
    num_chunks = rag.index_file(save_path)

    return {
        "message": f"Indexed '{file.filename}' successfully.",
        "chunks_created": num_chunks,
        "total_chunks": rag.total_chunks()
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    The core RAG endpoint.
    1. Retrieve relevant chunks for the question
    2. Build an augmented prompt
    3. Call the LLM
    4. Return the answer + which chunks were used
    """
    if rag.total_chunks() == 0:
        raise HTTPException(400, "No documents indexed yet. Upload a file first.")

    result = rag.answer(req.question)
    return result


@app.get("/status")
def status():
    """Check what's been indexed."""
    return {
        "documents": rag.indexed_files,
        "total_chunks": rag.total_chunks(),
        "ready": rag.total_chunks() > 0
    }


@app.delete("/reset")
def reset():
    """Clear all indexed documents."""
    rag.reset()
    return {"message": "Vector store cleared."}

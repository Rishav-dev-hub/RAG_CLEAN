import os
import math
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class RAGEngine:
    def __init__(self):
        self.store = []
        self.indexed_files = []

    # ── STEP 1: Load ─────────────────────────────────────────
    def load_text(self, filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def load_pdf(self, filepath):
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")

    # ── STEP 2: Chunk ─────────────────────────────────────────
    def chunk_text(self, text, chunk_size=400, overlap=60):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ── STEP 3: Embed (word frequency vectors) ────────────────
    def get_embedding(self, text):
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        stopwords = {'the','a','an','is','are','was','were','be','been',
                     'have','has','had','do','does','did','will','would',
                     'could','should','may','might','of','in','on','at',
                     'to','for','with','by','from','up','about','into',
                     'or','and','but','if','as','it','its','this','that',
                     'they','we','you','he','she','all','any','each','no'}
        vec = {}
        for w in words:
            if w not in stopwords:
                vec[w] = vec.get(w, 0) + 1
        return vec

    def cosine_similarity(self, vec_a, vec_b):
        shared = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[w] * vec_b[w] for w in shared)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── Index a file ──────────────────────────────────────────
    def index_file(self, filepath):
        print(f"\n📄 Indexing: {filepath}")

        if filepath.endswith(".pdf"):
            text = self.load_pdf(filepath)
        else:
            text = self.load_text(filepath)

        print(f"   Loaded {len(text)} characters")

        chunks = self.chunk_text(text)
        print(f"   Created {len(chunks)} chunks")

        filename = os.path.basename(filepath)
        start_id = len(self.store)

        for i, chunk in enumerate(chunks):
            print(f"   Embedding chunk {i+1}/{len(chunks)}...", end="\r")
            embedding = self.get_embedding(chunk)
            self.store.append({
                "id":        start_id + i,
                "text":      chunk,
                "embedding": embedding,
                "source":    filename
            })

        print(f"\n   ✅ Done! {len(chunks)} chunks indexed.")

        if filename not in self.indexed_files:
            self.indexed_files.append(filename)

        return len(chunks)

    # ── STEP 4: Retrieve ──────────────────────────────────────
    def retrieve(self, question, top_k=3):
        question_vec = self.get_embedding(question)
        scored = []
        for entry in self.store:
            score = self.cosine_similarity(question_vec, entry["embedding"])
            scored.append({**entry, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ── STEP 5: Generate ──────────────────────────────────────
    def generate(self, question, chunks):
        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the provided documents."
Be concise, clear, and helpful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    # ── Full RAG pipeline ────────────────────────────────────
    def answer(self, question):
        print(f"\n❓ Question: {question}")
        top_chunks = self.retrieve(question, top_k=3)
        print(f"   Retrieved {len(top_chunks)} chunks (scores: "
              f"{[round(c['score'], 3) for c in top_chunks]})")
        answer_text = self.generate(question, top_chunks)
        print(f"   ✅ Answer generated")
        return {
            "answer":      answer_text,
            "chunks_used": [c["text"] for c in top_chunks],
            "chunk_ids":   [c["id"] for c in top_chunks]
        }

    def total_chunks(self):
        return len(self.store)

    def reset(self):
        self.store = []
        self.indexed_files = []
        print("🗑️  Vector store cleared.")
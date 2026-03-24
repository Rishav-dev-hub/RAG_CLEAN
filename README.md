# RAG From Scratch — Python + FastAPI

A fully working RAG (Retrieval-Augmented Generation) backend
you can run locally in VS Code.

## Folder Structure

```
rag_project/
│
├── main.py              ← FastAPI server (the "waiter")
├── rag_engine.py        ← All RAG logic (the "kitchen")
├── requirements.txt     ← Python packages to install
├── .env                 ← Your API key goes here
│
├── static/
│   └── index.html       ← Chat UI (opens in browser)
│
└── documents/
    └── sample_handbook.txt  ← Sample document to test with
```

---

## Setup (do this once)

### Step 1 — Open the folder in VS Code
File → Open Folder → select `rag_project`

### Step 2 — Open the VS Code terminal
Terminal → New Terminal  (or press Ctrl+`)

### Step 3 — Create a virtual environment
```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll see `(venv)` appear in the terminal. Good.

### Step 4 — Install packages
```bash
pip install -r requirements.txt
```

### Step 5 — Add your API key
Open `.env` and replace `your-api-key-here` with your real key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at: https://console.anthropic.com

---

## Running the server

```bash
uvicorn main:app --reload
```

You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open your browser at: **http://localhost:8000**

The `--reload` flag means the server restarts automatically
whenever you save a file. Great for development.

---

## Using the app

1. Click **"Click to upload"** in the sidebar
2. Select `documents/sample_handbook.txt` (or any .txt/.pdf)
3. Click **"Upload & Index"** — watch the terminal as it chunks and embeds
4. Type a question like: *"What is the vacation policy?"*
5. See the retrieved chunks + generated answer

---

## API Endpoints

| Method | URL | What it does |
|--------|-----|--------------|
| GET | `/` | Opens the chat UI |
| POST | `/upload` | Upload + index a document |
| POST | `/ask` | Ask a question, get an answer |
| GET | `/status` | See what's indexed |
| DELETE | `/reset` | Clear all indexed docs |

You can also test the API directly at: **http://localhost:8000/docs**
(FastAPI gives you a free interactive API explorer)

---

## How the RAG pipeline works

```
Your document
     ↓
[Load]  →  Read the raw text
     ↓
[Chunk]  →  Split into ~400 char pieces with overlap
     ↓
[Embed]  →  Convert each chunk to a vector (list of numbers)
     ↓
[Store]  →  Keep vectors in memory

--- When you ask a question ---

Your question
     ↓
[Embed question]  →  Same embedding model
     ↓
[Similarity search]  →  Find chunks with closest vectors
     ↓
[Augment prompt]  →  question + top 3 chunks
     ↓
[Generate]  →  Claude reads context and answers
     ↓
Answer!
```

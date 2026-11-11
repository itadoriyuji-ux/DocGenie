# 📄 DocGenie

Chat with your documents. Upload a PDF or text file, ask questions in plain
English, and get answers grounded **only** in the content of your own file —
powered by a local embedding model + Google Gemini, using a RAG
(Retrieval-Augmented Generation) pipeline.

## How it works

```
Upload ──► Extract text (PyMuPDF) ──► Chunk (800 chars, 100 overlap)
                                              │
                                              ▼
                              Embed chunks (all-MiniLM-L6-v2, local)
                                              │
Ask question ──► Embed question ──► Cosine similarity ──► Top-3 chunks
                                                              │
                                                              ▼
                                              Gemini (grounded prompt) ──► Streamed answer
```

1. **Extract** — text is pulled from the PDF/txt file.
2. **Chunk** — text is split into overlapping ~800-character chunks (snapped
   to word boundaries so words aren't cut in half).
3. **Embed** — chunks are embedded locally with `all-MiniLM-L6-v2`
   (no data leaves your machine at this stage).
4. **Retrieve** — your question is embedded and cosine similarity picks the
   3 most relevant chunks.
5. **Generate** — the chunks are injected into a prompt for Gemini, and the
   answer streams back token-by-token.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at https://aistudio.google.com/apikey

## Run

```bash
streamlit run app.py
```

## Performance notes

- The embedding model, and each document's extraction + embeddings, are
  cached (`@st.cache_resource`) — re-uploading the same file or interacting
  with widgets never recomputes them.
- Answers are memoized per (file, question) in `session_state`, so widget
  interactions (like expanding the sources panel) don't fire duplicate
  Gemini API calls.
- Responses stream from Gemini, so you start reading immediately instead of
  waiting for the full completion.

# DocGenie

DocGenie is a document question answering app. You upload a PDF or text file, ask a question about it in plain English, and it gives you an answer based only on what is in your file.

It uses RAG (Retrieval Augmented Generation). The document is not sent to the LLM in full. Instead, relevant parts are found first using a local embedding model, and only those parts are sent to Gemini to generate the answer.

## How it works

1. Text is extracted from the uploaded PDF or txt file.
2. The text is split into chunks of about 800 characters with 100 characters of overlap between chunks.
3. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`. This runs locally on your machine.
4. When you ask a question, the question is embedded the same way and compared with every chunk using cosine similarity. The top 3 chunks are picked.
5. Those 3 chunks are put into a prompt along with your question and sent to Gemini. The answer is streamed back as it is generated.

```
PDF/txt -> extract text -> chunk -> embed (local model)
                                        |
question -> embed -> cosine similarity -> top 3 chunks
                                        |
                                        v
                            Gemini -> streamed answer
```

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file in the project folder with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

You can get a free key from https://aistudio.google.com/apikey

## Run

```
streamlit run app.py
```

## Performance details

A few things I had to fix while building this, since Streamlit reruns the whole script on every interaction:

- The embedding model and document embeddings are cached with `@st.cache_resource`, so uploading the same file twice does not recompute anything.
- Answers are stored in `st.session_state`, so clicking around the page (like opening the sources panel) does not send the same question to Gemini again.
- Answers stream in token by token instead of waiting for the full response.
- Chunk boundaries snap to spaces so words are not cut in half, since cut up words make worse embeddings.

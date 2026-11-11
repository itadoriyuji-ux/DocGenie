import os
from dotenv import load_dotenv
from google import genai

from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"

# Create the client once at import time, but don't crash the whole process if
# the key is missing. app.py / main() check for None and show a friendly
# message instead of a stack trace.
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


def load_document(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content

# step 2 - splitting the contents of the file into chunks(paragraph wise)

def split_into_chunks(text):
    raw_chunks = text.split("\n\n")
    chunks = []
    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if cleaned != "":
            chunks.append(cleaned)
    return chunks


def split_into_fixed_chunks(text, chunk_size=800, overlap=100):
    """Fixed-size overlapping chunks, but boundaries snap to whitespace so
    words (and the meaning of sentences) don't get cut in half."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Prefer ending on a space/newline near the edge instead of
            # chopping a word in the middle.
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Back up by the overlap, then snap forward to a word start so the
        # next chunk doesn't begin mid-word either. Bounded by `end`, so
        # progress is always guaranteed (no infinite loop).
        start = end - overlap
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1

    return chunks


def search(chunks, keyword):
    results = []
    keyword_lower = keyword.lower()
    for index, chunk in enumerate(chunks):
        if keyword_lower in chunk.lower():
            results.append((index, chunk))
    return results


def print_results(results, keyword):
    if len(results) == 0:
        print(f"\nNo matches found for '{keyword}'.\n")
        return
    print(f"\nFound {len(results)} match(es) for '{keyword}':\n")
    for chunk_number, chunk_text in results:
        print(f"--- Chunk {chunk_number} ---")
        print(chunk_text)
        print()


def embed_chunks(model, chunks):
    # Bigger batch + no progress bar: one forward pass per batch instead of
    # per-chunk overhead, which matters a lot for large documents.
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
    return embeddings


def semantic_search(model, chunks, chunk_embeddings, question, top_n=3):
    question_embedding = model.encode(question)
    similarity_scores = model.similarity(question_embedding, chunk_embeddings)[0]

    # topk picks only the top_n scores instead of fully sorting every chunk,
    # which is cheaper on large documents and returns the same result
    # (already sorted, highest first).
    k = min(top_n, len(chunks))
    top_scores, top_indices = similarity_scores.topk(k)

    return [
        (score.item(), int(i), chunks[int(i)])
        for score, i in zip(top_scores, top_indices)
    ]


def build_prompt(question, top_chunks):
    context_text = "\n\n".join(chunk_text for score, index, chunk_text in top_chunks)

    return f"""You are a helpful, knowledgeable assistant. Answer the question naturally and conversationally, as if you're explaining it to someone directly.

Base your answer on the context provided below. If the context doesn't contain enough information to answer confidently, say so naturally (e.g. "I couldn't find that in the document") instead of a robotic disclaimer.

Context:
{context_text}

Question: {question}

Answer:"""


def generate_answer_stream(client, question, top_chunks):
    """Yield the answer token-by-token so the UI can show it as it's written
    instead of making the user stare at a spinner for the full response."""
    prompt = build_prompt(question, top_chunks)

    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    got_anything = False
    for chunk in stream:
        if chunk.text:
            got_anything = True
            yield chunk.text

    # Safety filters / empty responses would otherwise silently produce
    # nothing at all.
    if not got_anything:
        yield "I couldn't generate an answer for that question. Try rephrasing it."


def generate_answer(client, question, top_chunks):
    """Non-streaming convenience wrapper (kept for the CLI / tests)."""
    return "".join(generate_answer_stream(client, question, top_chunks))

# main function

def main():
    if client is None:
        print("GEMINI_API_KEY is not set. Add it to a .env file and try again.")
        return

    filepath = input("Enter path to your text file: ").strip()
    filepath = filepath.strip('"').strip("'")

    try:
        document_text = load_document(filepath)
    except FileNotFoundError:
        print(f"Could not find a file at '{filepath}'. Check the path and try again.")
        return

    chunks = split_into_chunks(document_text)
    print(f"Loaded document with {len(chunks)} paragraph(s).")

    print("Loading embedding model... (first time may take a moment)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_embeddings = embed_chunks(model, chunks)
    print("Document ready. You can now ask questions.\n")

    while True:
        question = input("Ask a question (or 'quit' to exit): ").strip()
        if question.lower() == "quit":
            print("Goodbye!")
            break

        top_chunks = semantic_search(model, chunks, chunk_embeddings, question)

        print("\nAnswer: ", end="", flush=True)
        for token in generate_answer_stream(client, question, top_chunks):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()

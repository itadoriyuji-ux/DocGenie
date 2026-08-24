import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

from sentence_transformers import SentenceTransformer

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
    embeddings = model.encode(chunks)
    return embeddings


def semantic_search(model, chunks, chunk_embeddings, question, top_n=3):
    question_embedding = model.encode(question)
    similarity_scores = model.similarity(question_embedding, chunk_embeddings)[0]

    scored_chunks = []
    for index in range(len(chunks)):
        score = similarity_scores[index].item()
        scored_chunks.append((score, index, chunks[index]))
    scored_chunks.sort(key = lambda item: item[0], reverse = True)
    return scored_chunks[:top_n]
    

def generate_answer(client, question, top_chunks):
    context_text = "\n\n".join(chunk_text for score, index, chunk_text in top_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

# main function

def main():
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
        answer = generate_answer(client, question, top_chunks)

        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
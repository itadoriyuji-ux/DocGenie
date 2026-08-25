import streamlit as st
from main import (
    load_document,
    split_into_chunks,
    embed_chunks,
    semantic_search,
    generate_answer,
    client,
)
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DocGenie", page_icon="📄")

st.title("DocGenie")
st.write("Ask questions about your document...We give you crisp answers without external sourcing!")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
    chunks = split_into_chunks(document_text)
    st.write(f"Document loaded with {len(chunks)} paragraph(s).")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_chunks(model, chunks)

    question = st.text_input("Ask a question about your document")

    if question:
        top_chunks = semantic_search(model, chunks, chunk_embeddings, question)
        answer = generate_answer(client, question, top_chunks)
        st.write("Answer:", answer)
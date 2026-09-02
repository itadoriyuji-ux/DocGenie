import streamlit as st
import pymupdf as fitz
from main import (
    split_into_chunks,
    split_into_fixed_chunks,
    embed_chunks,
    semantic_search,
    generate_answer,
    client,
)
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DocGenie", page_icon="📄", layout="centered")

# ---- Caching: load the model only ONCE, ever, no matter how many times the script reruns ----
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---- Caching: only re-extract + re-embed if the SAME file bytes come in again, reuse the result ----
@st.cache_data(show_spinner=False)
def process_document(file_bytes, filename):
    if filename.endswith(".pdf"):
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        document_text = ""
        for page in pdf_document:
            document_text += page.get_text()
        pdf_document.close()
        chunks = split_into_fixed_chunks(document_text)
    else:
        document_text = file_bytes.decode("utf-8")
        chunks = split_into_chunks(document_text)
    model = get_model()
    chunk_embeddings = embed_chunks(model, chunks)
    return chunks, chunk_embeddings


# ---- UI starts here ----
st.title("📄 DocGenie")
st.caption("Upload a document, ask questions in plain English, get answers grounded in your own file.")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file is not None:
    with st.spinner("Reading and understanding your document..."):
        chunks, chunk_embeddings = process_document(uploaded_file.read(), uploaded_file.name)

    st.success(f"Ready! Loaded {len(chunks)} section(s) from **{uploaded_file.name}**.")

    question = st.text_input("💬 Ask a question about your document")

    if question:
        with st.spinner("Thinking..."):
            model = get_model()
            top_chunks = semantic_search(model, chunks, chunk_embeddings, question)
            answer = generate_answer(client, question, top_chunks)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("See the source sections used for this answer"):
            for score, index, chunk_text in top_chunks:
                st.markdown(f"**Section {index}** (relevance: {score:.2f})")
                st.write(chunk_text)
                st.divider()
import streamlit as st
import pymupdf as fitz
from main import (
    split_into_chunks,
    split_into_fixed_chunks,
    embed_chunks,
    semantic_search,
    generate_answer_stream,
    client,
)
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DocGenie", page_icon="📄", layout="centered")

# ---- Caching: load the model only ONCE, ever, no matter how many times the script reruns ----
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---- Caching: only re-extract + re-embed if the SAME file bytes come in again, reuse the result ----
# cache_resource (instead of cache_data) avoids deep-copying the embeddings
# array back to the script on every rerun, which gets expensive on large docs.
@st.cache_resource(show_spinner=False)
def process_document(file_bytes, filename):
    try:
        if filename.lower().endswith(".pdf"):
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
                # join() is O(n) instead of += which is O(n^2), and "\n" keeps
                # the last word of one page from gluing onto the next page's
                # first word.
                document_text = "\n".join(page.get_text() for page in pdf_document)
            chunks = split_into_fixed_chunks(document_text)
        else:
            # errors="replace" so odd encodings don't crash the upload
            document_text = file_bytes.decode("utf-8", errors="replace")
            chunks = split_into_chunks(document_text)
    except Exception as e:
        return None, f"Couldn't read **{filename}**: {e}"

    model = get_model()
    chunk_embeddings = embed_chunks(model, chunks)
    return (chunks, chunk_embeddings), None


# ---- UI starts here ----
st.title("📄 DocGenie")
st.caption("Upload a document, ask questions in plain English, get answers grounded in your own file.")

# Fail loudly but nicely if the key is missing, instead of a stack trace
# mid-question.
if client is None:
    st.error("⚠️ No API key found. Please set `GEMINI_API_KEY` in your `.env` file and restart the app.")
    st.stop()

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file is not None:
    with st.spinner("Reading and understanding your document..."):
        result, error = process_document(uploaded_file.read(), uploaded_file.name)

    if error:
        st.error(error)
        st.stop()

    chunks, chunk_embeddings = result
    st.success(f"Ready! Loaded {len(chunks)} section(s) from **{uploaded_file.name}**.")

    question = st.text_input("💬 Ask a question about your document")

    if question:
        model = get_model()
        top_chunks = semantic_search(model, chunks, chunk_embeddings, question)

        st.markdown("### Answer")

        # --- Answer memoization ---
        # Streamlit reruns this whole script on EVERY widget interaction --
        # including opening/closing the source expander below. Without this,
        # each rerun fires a duplicate (paid) Gemini API call for the exact
        # same question and context. We cache the finished answer in
        # session_state keyed by file + question, so reruns are instant.
        answers = st.session_state.setdefault("answers", {})
        answer_key = f"{uploaded_file.name}::{len(chunks)}::{question}"
        cached_answer = answers.get(answer_key)

        if cached_answer is not None:
            st.write(cached_answer)
        else:
            try:
                answer = st.write_stream(
                    generate_answer_stream(client, question, top_chunks)
                )
                answers[answer_key] = answer
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")

        with st.expander("See the source sections used for this answer"):
            for score, index, chunk_text in top_chunks:
                st.markdown(f"**Section {index}** (relevance: {score:.2f})")
                st.write(chunk_text)
                st.divider()

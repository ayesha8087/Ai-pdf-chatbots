import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="AI Study Helper", layout="wide")

# ✅ NEW TITLE (SELLING PURPOSE)
st.title("📚 AI Study Helper (PDF Notes + Q&A + Summary)")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=-1  # CPU
    )

llm = load_llm()

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

embeddings = load_embeddings()

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload PDF Files")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if files:
        all_docs = []

        for file in files:
            with open(file.name, "wb") as f:
                f.write(file.read())

            loader = PyPDFLoader(file.name)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(all_docs)

        st.session_state.db = FAISS.from_documents(chunks, embeddings)

        st.success("✅ Documents processed!")

# ---------------- MODE SELECTOR ----------------
mode = st.selectbox("Choose Task", [
    "Ask Question",
    "Summarize Chapter",
    "Generate Notes"
])

# ---------------- CHAT UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask or choose task...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    if st.session_state.db:

        docs = st.session_state.db.max_marginal_relevance_search(
            query,
            k=5,
            fetch_k=10
        )

        context = "\n\n".join([
            f"Chunk {i+1}: {clean_text(d.page_content)}"
            for i, d in enumerate(docs)
        ])

        # ---------------- MODE LOGIC ----------------

        if mode == "Summarize Chapter":

            prompt = f"""
You are a helpful AI study assistant.

Read the content carefully and create a SHORT summary.

Rules:
- Use simple English
- Use bullet points
- Mention only important ideas
- Write 5 bullet points

Content:
{context}
"""

        elif mode == "Generate Notes":

            prompt = f"""
You are an AI note generator.

Create STUDY NOTES from the content below.

Rules:
- Use bullet points
- Include important definitions
- Include key concepts
- Keep notes useful for exams
- Write at least 5 points

Content:
{context}
"""

        else:

            prompt = f"""
Answer ONLY from the context below.

Find the exact answer from text.
Do not guess.

If not found, say: Not found in document.

Context:
{context}

Question:
{query}
"""

        result = llm(
            prompt,
            max_new_tokens=220,
            temperature=0.3,
            do_sample=False
        )

        answer = result[0]["generated_text"].strip()

        if len(answer.strip()) < 10:
            answer = "Not found in document"

    else:
        answer = "⚠️ Please upload documents first."

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)

        st.download_button(
            "📥 Download Result",
            answer
        )
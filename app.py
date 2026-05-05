import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="Smart AI PDF SaaS", layout="wide")

st.title("🚀 Smart AI Document Assistant")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

llm = load_llm()

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------- SESSION MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload Documents")
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

# ---------------- CHAT UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask anything from your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    if st.session_state.db:

        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer the question using ONLY the context below.

If the answer is not clearly in the context, say:
Not found in document.

Keep answer short (max 2 lines) and simple.

Context:
{context}

Question:
{query}
"""

        result = llm(
            prompt,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=False
        )

        answer = result[0]["generated_text"].strip()
        if len(answer) < 5:
           answer = "Not found in document"

    else:
        answer = "⚠️ Please upload documents first."

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
# 📚 AI Study Helper – PDF Chatbot (RAG System)

An AI-powered Study Assistant that lets users upload PDF documents and interact with them using **chat-based Q&A, summarization, and note generation**.

Built using **Streamlit, LangChain, FAISS, Hugging Face Transformers, and Sentence Transformers embeddings**.

---

## 🚀 Features

- 📂 Upload multiple PDF files
- 💬 Ask questions from documents (Chat Q/A)
- 🧠 AI-powered summarization of chapters
- 📝 Automatic study notes generation
- 🔍 Context-based retrieval using FAISS (RAG system)
- 💾 Download generated answers
- 🧹 Clear chat functionality
- ⚡ Fast local inference using Hugging Face pipeline


## 🏗️ Tech Stack

- Python 🐍
- Streamlit 🎈
- LangChain 🦜
- FAISS (Vector Database)
- Hugging Face Transformers 🤗
- Sentence Transformers (Embeddings)
- PyPDFLoader


## 🧠 How It Works

1. User uploads PDF files
2. PDFs are split into chunks
3. Embeddings are created using `all-mpnet-base-v2`
4. Stored in FAISS vector database
5. User asks a question
6. Relevant chunks are retrieved (RAG)
7. LLM generates response using context

---

## 📸 Screenshots

### 🏠 Home Interface
()

### 💬 Q&A Mode
()

### 📝 Notes Generation
()



## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-study-helper.git
cd ai-study-helper
pip install -r requirements.txt

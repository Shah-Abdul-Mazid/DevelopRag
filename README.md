Below is a **complete, ready-to-copy `README.md`** that fully satisfies the submission requirement:

> **"A README.md explaining setup steps and approach"**

It includes:
- Clear **setup steps** (from cloning to running both backend and frontend)
- Detailed **approach explanation** (how RAG works, data flow, fallback logic)
- All required sections (features, tech stack, API usage, etc.)
- Professional formatting with emojis, tables, and code blocks

---

### Copy & Paste This into Your `README.md`

```markdown
# HR Policies Assistant - RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that answers employee questions using **official HR policy PDFs**. Built with **Pinecone vector database**, **Sentence Transformers**, **Llama 2 via Ollama**, **FastAPI**, and **Streamlit**.

---

## Features

- **PDF Ingestion**: Automatically load and parse multiple HR policy documents  
- **Smart Chunking**: Split documents into optimal chunks for retrieval  
- **Semantic Search**: Find most relevant policy sections using vector similarity  
- **Context-Aware Answers**: Llama 2 generates accurate, policy-grounded responses  
- **Fallback Mode**: Use LLM alone if no relevant document is found  
- **REST API**: Query via `POST /ask` with tunable parameters  
- **Interactive UI**: Streamlit app with real-time chat, parameter tuning, and chat history  

---

## Technology Stack

| Component           | Technology |
|---------------------|----------|
| Backend             | Python, FastAPI, Uvicorn |
| Frontend            | Streamlit |
| Vector Database     | Pinecone |
| Embeddings          | `all-MiniLM-L6-v2` (Sentence Transformers) |
| LLM                 | Llama 2 (via Ollama) |
| PDF Processing      | PyMuPDF (`fitz`) |
| Text Splitting      | `RecursiveCharacterTextSplitter` (LangChain) |
| Environment         | `.env`, Pydantic |

---

## Project Structure

```
HR-Policies-Assistant-RAG/
├── backend.py              # FastAPI + RAG pipeline
├── streamlit_app.py        # Interactive Streamlit UI
├── embeddings.py           # Embedding generation
├── vectorstore.py          # Pinecone integration
├── rag_retriever.py        # Core RAG logic (ingest + retrieve + generate)
├── data/policies/          # Place your HR PDF files here
├── .env                    # API keys and config
├── requirements.txt
└── README.md               # This file
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/HR-Policies-Assistant-RAG.git
cd HR-Policies-Assistant-RAG
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=hr-policies
OLLAMA_HOST=http://localhost:11434
```

> Get your Pinecone key from [pinecone.io](https://www.pinecone.io/)

### 5. Start Ollama & Pull Llama 2
```bash
ollama serve &
ollama pull llama2
```

### 6. Add HR Policy Documents
Place all HR policy PDFs in:
```
data/policies/
```

> Example files: `leave_policy.pdf`, `code_of_conduct.pdf`, etc.

---

## Run the Application

### Step 1: Start the Backend (FastAPI)
```bash
uvicorn backend:app --reload --port 8000
```

**API will be available at:**  
http://localhost:8000  
**Interactive Docs (Swagger UI):** http://localhost:8000/docs

### Step 2: Launch the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

**Chat interface opens at:** http://localhost:8501

---

## How It Works (Technical Approach)

### 1. **PDF Ingestion**
- `PyMuPDF` loads text from all PDFs in `data/policies/`
- Text is extracted page-by-page and concatenated

### 2. **Document Chunking**
- `RecursiveCharacterTextSplitter` splits text into chunks:
  - `chunk_size=1000`
  - `chunk_overlap=200` (preserves context across chunks)

### 3. **Embedding Generation**
- Each chunk → vector using `all-MiniLM-L6-v2` (384-dimensional)
- Fast, lightweight, excellent for semantic similarity

### 4. **Vector Storage**
- Embeddings + metadata stored in **Pinecone**
- Index auto-created if not exists (`dimension=384`, `metric=cosine`)

### 5. **Query Processing**
```text
User Query 
  → Embed (same model) 
  → Similarity search in Pinecone (top_k, score_threshold) 
  → Retrieve matching chunks 
  → Build context 
  → Send to Llama 2 with prompt
  → Return concise answer
```

### 6. **Prompt Engineering**
```text
You are an HR assistant. Answer using ONLY the following policy context.
If unsure, say "I couldn't find this in the policies."

Context: [...]
Question: [...]
Answer:
```

### 7. **Fallback Logic**
- If no chunk scores above `score_threshold` → use LLM directly
- Prevents hallucination when policy is missing

---

## API Usage

**Endpoint**: `POST /ask`

### Request
```json
{
  "query": "How many days of sick leave do I get?",
  "top_k": 5,
  "score_threshold": 0.35,
  "fallback_to_llm": true
}
```

### Response
```json
{
  "query": "How many days of sick leave do I get?",
  "answer": "Full-time employees are entitled to 12 days of paid sick leave per year.",
  "status": "success"
}
```

---

## Streamlit UI Features

- **Live chat** with user/assistant bubbles  
- **Parameter tuning**:  
  - `Top K` (1–20)  
  - `Similarity Threshold` (0.0–1.0)  
  - `Fallback to LLM` toggle  
- **Clear Chat** button  
- **Connection status** for Pinecone & Ollama  
- **Timestamps** on messages  
- **Responsive, modern design**



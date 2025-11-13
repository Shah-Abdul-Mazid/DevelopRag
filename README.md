# 🤖 HR Policies Assistant - RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from HR policy documents using **PDF ingestion, vector embeddings, Pinecone vector database, and Llama 2 (Ollama)**. The chatbot supports both **API-based querying** and a **Streamlit frontend**.

---

## 📄 Features

- ✅ Ingest multiple PDF HR policy documents automatically.
- ✅ Split documents into chunks for efficient embedding and retrieval.
- ✅ Generate semantic embeddings with **Sentence Transformers**.
- ✅ Store and query document embeddings using **Pinecone vector database**.
- ✅ Retrieve relevant context with a **similarity search**.
- ✅ Generate concise answers with **Llama 2 (Ollama)** using retrieved context.
- ✅ Fallback to LLM for queries without relevant documents.
- ✅ REST API using **FastAPI**.
- ✅ Interactive frontend with **Streamlit**, including:
  - Connection status
  - RAG parameter tuning (Top K, similarity threshold)
  - Clear chat history
  - Real-time chat interface

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| Frontend | Streamlit |
| Vector Store | Pinecone |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | Llama 2 via Ollama |
| Document Loading | PyMuPDF |
| Document Splitting | RecursiveCharacterTextSplitter (LangChain) |
| API & Env | Python `dotenv`, Pydantic |

---

## 🏗️ Project Structure

hr-policies-assistant/
├── backend.py # FastAPI backend with RAG & LLM integration
├── streamlit_app.py # Streamlit frontend for chat interface
├── embeddings.py # EmbeddingManager for text/PDFs
├── vectorstore.py # Pinecone integration & similarity search
├── rag_retriever.py # RAGRetriever class
├── data/policies/ # HR PDF/text documents
├── .env # API keys & configuration
├── requirements.txt # Python dependencies
└── README.md


---

## ⚡ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Shah-Abdul-Mazid/HR-Policies-Assistant.git
cd HR-Policies-Assistant


2. Create Virtual Environment
python -m venv .venv
# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables

Create a .env file:

API_KEY=<your_pinecone_api_key>
OLLAMA_API_KEY=<your_ollama_key_if_needed>

5. Add HR Documents

Place PDF files in:

data/policies/

6. Run Backend
python backend.py


FastAPI will run at:

http://localhost:8000


Swagger UI (interactive API docs):

http://localhost:8000/docs

7. Run Frontend
streamlit run streamlit_app.py

💬 API Usage
POST /ask

Ask a question to the RAG chatbot.

Request Body:

{
  "query": "What is the company's leave policy?",
  "top_k": 5,
  "score_threshold": 0.4,
  "fallback_to_llm": true
}


Response:

{
  "query": "What is the company's leave policy?",
  "answer": "Employees are entitled to 20 days of paid leave per year...",
  "status": "success"
}

🧠 How It Works

PDF Ingestion: Load HR PDFs using PyMuPDFLoader.

Document Chunking: Split PDFs into smaller chunks using RecursiveCharacterTextSplitter.

Embeddings: Generate semantic embeddings using Sentence Transformers.

Vector Storage: Store embeddings in Pinecone with metadata.

Retrieval: Perform similarity search on query embeddings.

Generation: Use Llama 2 (Ollama) to generate answers using retrieved context.

Fallback: If no relevant context is found, optionally use LLM alone.

🎨 Streamlit Frontend

Real-time chat interface with user and assistant messages.

Display message timestamps.

Adjustable Top K and similarity threshold.

LLM fallback toggle.

Clear chat history and connection status indicators.

Custom CSS styling for modern UI.

📈 Customization

Change embedding model: all-MiniLM-L6-v2 → any HuggingFace transformer.

Adjust chunk_size and chunk_overlap in split_documents() for more/less context.

Tune top_k and score_threshold in API or frontend for better precision.
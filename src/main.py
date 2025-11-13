import os
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================================
# 🔧 Setup
# ==========================================================
load_dotenv()
print("✅ Environment loaded successfully.")

api_key = os.getenv("api_key")
if not api_key:
    raise RuntimeError("❌ Missing Pinecone API key! Add it to your .env or Render env variables.")
print("🔑 Pinecone API key found!")

# ==========================================================
# 📄 PDF Processing Utilities
# ==========================================================
def process_all_pdfs(pdf_directory: str):
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found in directory:", pdf_directory)
        return []

    print(f"📂 Found {len(pdf_files)} PDF files.")
    for pdf_file in pdf_files:
        print(f"📘 Loading: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
            all_documents.extend(documents)
            print(f"   → Loaded {len(documents)} pages")
        except Exception as e:
            print(f"❌ Error reading {pdf_file.name}: {e}")
    print(f"✅ Total documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    if not documents:
        print("⚠️ No documents to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", "", ",", "."],
    )
    split_docs = splitter.split_documents(documents)
    print(f"✂️ Split {len(documents)} docs into {len(split_docs)} chunks.")
    return split_docs

# ==========================================================
# 🧬 Embedding Manager
# ==========================================================
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"🧠 Generating embeddings for {len(texts)} texts...")
        return self.model.encode(texts, show_progress_bar=False)

# ==========================================================
# 📦 Pinecone Vector Store
# ==========================================================
INDEX_NAME = "hr-policies"
DIMENSION = 384
REGION = "us-east-1"
CLOUD = "aws"

pc = Pinecone(api_key=api_key)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"🚀 Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        print("⏳ Waiting for Pinecone index to initialize...")
        time.sleep(5)
    print(f"✅ Pinecone index '{INDEX_NAME}' is ready.")

index = pc.Index(INDEX_NAME)

class VectorStore:
    def __init__(self, namespace="hr-policies"):
        self.namespace = namespace
        self.index = index
        stats = self.index.describe_index_stats()
        count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        print(f"📚 Vector store initialized. Namespace='{namespace}', vectors={count}")

    def add_documents(self, documents, embeddings):
        print(f"📥 Adding {len(documents)} vectors to Pinecone...")
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vectors.append((
                f"doc_{uuid.uuid4().hex[:8]}_{i}",
                embedding.tolist(),
                {
                    "text": doc.page_content,
                    "source_file": doc.metadata.get("source_file", "unknown"),
                },
            ))
        self.index.upsert(vectors=vectors, namespace=self.namespace)
        print(f"✅ {len(vectors)} documents uploaded successfully.")

    def similarity_search(self, query_embedding, top_k=5, score_threshold=0.3):
        res = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        results = [
            {
                "text": match["metadata"].get("text", ""),
                "score": match["score"],
                "source": match["metadata"].get("source_file", ""),
            }
            for match in res.get("matches", [])
            if match["score"] >= score_threshold
        ]
        print(f"🔎 Retrieved {len(results)} similar documents.")
        return results

# ==========================================================
# 🧩 Retriever
# ==========================================================
class RAGRetriever:
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k=5, score_threshold=0.3):
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        return self.vectorstore.similarity_search(query_emb, top_k, score_threshold)

# ==========================================================
# 🦙 LLM Setup
# ==========================================================
llm = Ollama(model="llama2")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a concise and factual HR policy assistant. Use only provided context."),
    ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
])

# ==========================================================
# 🤖 RAG Pipeline
# ==========================================================
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_retriever = RAGRetriever(vectorstore, embedding_manager)

def rag_query(query: str, top_k=5, score_threshold=0.4):
    docs = rag_retriever.retrieve(query, top_k, score_threshold)
    if not docs:
        print("⚠️ No context found; fallback to LLM.")
        # FIX: Ollama returns string directly, not an object with .content
        result = llm(f"Answer this HR-related question: {query}")
        return result if isinstance(result, str) else str(result)

    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    chain = prompt_template | llm
    result = chain.invoke({"context": context, "query": query})
    # FIX: Handle both string and object responses
    return result if isinstance(result, str) else (result.content if hasattr(result, "content") else str(result))

# ==========================================================
# ⚡ FastAPI App
# ==========================================================
app = FastAPI(title="HR RAG Chatbot API", version="2.0.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.4
    fallback_to_llm: Optional[bool] = True  # Added to match frontend

class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str

# ==========================================================
# 🚀 Routes
# ==========================================================
@app.get("/")
async def root():
    return {
        "message": "✅ HR RAG Chatbot API is live!",
        "endpoints": {
            "POST /ask": "Submit a query",
            "GET /health": "Health check",
        },
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "llama2", "embedding_model": "all-MiniLM-L6-v2"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer = rag_query(req.query, req.top_k, req.score_threshold)
        return QueryResponse(query=req.query, answer=answer, status="success")
    except Exception as e:
        print(f"❌ Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# ==========================================================
# 🏁 Entry Point - FIX: Auto-load PDFs on startup (optional)
# ==========================================================
@app.on_event("startup")
async def startup_event():
    """Optional: Load PDFs on startup if they exist"""
    pdf_dir = Path("pdfs")  # Adjust this path to your PDF directory
    if pdf_dir.exists():
        existing_count = index.describe_index_stats().get("namespaces", {}).get("hr-policies", {}).get("vector_count", 0)
        if existing_count == 0:
            print("📂 No vectors in database. Loading PDFs...")
            docs = process_all_pdfs(str(pdf_dir))
            if docs:
                split_docs = split_documents(docs)
                texts = [doc.page_content for doc in split_docs]
                embeddings = embedding_manager.generate_embeddings(texts)
                vectorstore.add_documents(split_docs, embeddings)
        else:
            print(f"✅ Vector database already contains {existing_count} vectors")

if __name__ == "__main__":
    import uvicorn
    # FIX: Use 'main:app' instead of 'src.backend:app'
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
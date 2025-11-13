import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# Load environment variables
load_dotenv()
print("✅ All libraries imported successfully.")

# API Key
api_key = os.getenv("api_key")
if not api_key:
    raise RuntimeError("API keys not found. Set them in .env or your environment securely.")
print("🔑 API keys loaded successfully!")

# ==================== Document Processing ====================
def process_all_pdfs(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to Process")
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            all_documents.extend(documents)
            print(f" Loaded {len(documents)} Pages")
        except Exception as e:
            print(f" Error {e}")
    
    print(f"Total Documents Loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", "", ",", "."]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    if split_docs:
        print(f"Example Chunk:")
        print(f"Content: {split_docs[0].page_content[:250]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

# ==================== Embedding Manager ====================
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading Embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error Loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model Not Loaded...")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

# ==================== Pinecone Vector Store ====================
INDEX_NAME = "hr-policies"
DIMENSION = 384
METRIC = "dotproduct"
CLOUD = "aws"
REGION = "us-east-1"

pc = Pinecone(api_key=api_key)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        print(" Waiting for index to initialize...")
        time.sleep(5)
    print(f"Index '{INDEX_NAME}' is ready.")

index = pc.Index(INDEX_NAME)

class VectorStore:
    def __init__(self, collection_name: str = "hr-policies", persist_directory: str = "../HR Policies/vector_store"):
        self.collection_name = collection_name
        self.namespace = collection_name
        self.index = index
        
        stats = self.index.describe_index_stats()
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
        ns_stats = stats_dict.get("namespaces", {}).get(self.namespace, {})
        existing_count = ns_stats.get("vector_count", 0)
        
        print(f"Vector Store initialized. Namespace: '{self.namespace}'")
        print(f"Existing documents in namespace: {existing_count}")
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to Pinecone namespace '{self.namespace}'...")
        
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            text = getattr(doc, "page_content", str(doc))
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["doc_index"] = i
            metadata["content_length"] = len(text)
            metadata["text"] = text
            
            vectors.append((doc_id, embedding.tolist(), metadata))
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
        
        stats = self.index.describe_index_stats().to_dict()
        count = stats.get("namespaces", {}).get(self.namespace, {}).get("vector_count", 0)
        print(f"✅ Added {len(documents)} documents. Total in namespace: {count}")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, score_threshold: float = 0.0):
        if query_embedding.shape != (DIMENSION,):
            query_embedding = query_embedding.flatten()
        
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=None
        )
        
        hits = []
        for match in results["matches"]:
            if match["score"] >= score_threshold:
                hits.append({
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "metadata": match["metadata"]
                })
        
        return hits

# ==================== RAG Retriever ====================
class RAGRetriever:
    def __init__(self, vectorstore: VectorStore, embedding_manager: Any):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving Documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")
        
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vectorstore.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            retrieved_docs = []
            if results:
                for i, hit in enumerate(results):
                    retrieved_docs.append({
                        'id': hit['id'],
                        'content': hit['text'],
                        'metadata': hit['metadata'],
                        'similarity_score': hit['score'],
                        'distance': 1 - hit['score'],
                        'rank': i + 1
                    })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No Documents Found...")
            
            return retrieved_docs
        except Exception as e:
            print(f"Error During Retrieval: {e}")
            return []

# ==================== Initialize Components ====================
# Initialize only if needed (comment out if already done)
# all_pdf_documents = process_all_pdfs("../HR Policies")
# chunks = split_documents(all_pdf_documents)

embedding_manager = EmbeddingManager()
vectorstore = VectorStore()

# Add documents if needed (comment out after first run)
# if chunks:
#     chunk_texts = [chunk.page_content for chunk in chunks]
#     embeddings = embedding_manager.generate_embeddings(chunk_texts)
#     vectorstore.add_documents(chunks, embeddings)

rag_retriever = RAGRetriever(vectorstore, embedding_manager)

# ==================== LLM Setup ====================
llm = Ollama(model="llama2")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely using only the provided context."),
    ("human", """
Context: {context}

Question: {query}

Answer:""")
])

def rag_simple_api(query: str, rag_retriever, llm, top_k=5, score_threshold=0.4, fallback_to_llm=True):
    """RAG function for API"""
    results = rag_retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    
    if not results:
        if fallback_to_llm:
            # Use invoke method instead of calling directly
            response = llm.invoke(f"Answer this question: {query}")
            return response if isinstance(response, str) else str(response)
        return "No relevant context found."
    
    context = "\n\n".join(f"[Source {i+1}]: {doc['content']}" for i, doc in enumerate(results))
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "query": query})
    
    # Handle different response types
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    else:
        return str(response)

# ==================== FastAPI Application ====================
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:8501"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.4
    fallback_to_llm: Optional[bool] = True

class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str

@app.get("/")
def read_root():
    return {
        "message": "RAG Chatbot API is running",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "llama2", "embedding_model": "all-MiniLM-L6-v2"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        answer = rag_simple_api(
            query=request.query,
            rag_retriever=rag_retriever,
            llm=llm,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            fallback_to_llm=request.fallback_to_llm
        )
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
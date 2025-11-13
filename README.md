# 🤖 HR Policies RAG Chatbot

This repository contains a **Retrieval-Augmented Generation (RAG) chatbot** designed to answer questions from HR policy documents using advanced AI technologies.

**Tech Stack:**

- **Vector Store:** Pinecone (serverless)  
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)  
- **LLM:** Ollama Llama2  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  

---

## 🛠 Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shah-Abdul-Mazid/DevelopRag.git
cd DevelopRag
````

### 2. Repository Structure

```
DevelopRag/
├── src/
│   ├── backend.py          # FastAPI backend
│   ├── frontend.py         # Streamlit frontend
│   ├── requirements.txt    # Dependencies for both backend & frontend
| 
├── hr_policies/            # PDF documents
│   └── ...
|
├── README.md
├── .env 
└── .gitignore
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` in the dirrectory folder:

```
api_key=YOUR_PINECONE_API_KEY
```

### 5. Run Backend

```bash
python src/backend.py
```

The FastAPI server will run on `http://localhost:8000`.

### 6. Run Frontend

```bash
streamlit run src/frontend.py
```

The Streamlit app will open in your browser and connect to the backend.

---

## 📚 Usage

* Ask questions about HR policies directly in the chat interface.
* Adjust RAG parameters in the sidebar:

  * **Number of documents (top_k)** – how many relevant documents to retrieve
  * **Similarity threshold** – minimum similarity score for retrieved documents
  * **LLM fallback** – enable/disable fallback to general LLM if no documents are found
* Chat history is preserved throughout the session.

---

## 🖼 Example Usage

### 1. Chat Interface

![Chat Interface Screenshot](frontend_screenshot_placeholder.png)
*Streamlit chat interface for interacting with the RAG chatbot.*

### 2. Sample Queries

| User Query                                                                                  | Example Response                                                                                                           |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| "When Red Cross Society in East Pakistan was transformed into the National Red Cross Society of Bangladesh?" | "The Red Cross Society in East Pakistan was transformed into the National Red Cross Society of Bangladesh on December 20, 1971." |
| "When Renamed as Bangladesh Red Cross Society by a GoB order?"                               | "The Bangladesh Red Crescent Society was renamed as Bangladesh Red Cross Society by a GoB (Government of Bangladesh) order on January 4, 1972." |
| "When Udayankur Seba Sangstha started its journey?"                                         | "Udayankur Seba Sangstha (USS) started its journey in 1997."                                                              |

### 3. Adjustable Parameters

* **Top K documents:** Retrieve more or fewer documents for context
* **Similarity threshold:** Filter documents based on relevance
* **LLM fallback:** Enable if no exact match is found in documents

---

## 🧩 Project Structure

* **src/** – Backend (FastAPI) and frontend (Streamlit) scripts
* **hr_policies/** – Folder for HR policy PDF documents
* **README.md** – Project instructions

---

## ⚡ Notes

* Ensure your Pinecone API key is set correctly.
* Preprocessing PDFs is required on the first run to populate the vector store.
* Ollama Llama2 must be installed and configured to work with the backend.

<img width="2558" height="1598" alt="image" src="https://github.com/user-attachments/assets/d9579520-4b18-44be-9717-e472df7732ed" />
```
https://www.loom.com/share/f35e57f2d7c049f69ff9ef1129584a15
```

# import streamlit as st
# import requests
# import json
# from typing import Optional
# from datetime import datetime

# # ==================== Page Configuration ====================
# st.set_page_config(
#     page_title="RAG Chatbot - HR Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==================== Custom CSS ====================
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
#     * {
#         font-family: 'Inter', sans-serif;
#     }
    
#     .main {
#         padding: 1rem 2rem;
#     }
    
#     .stTextInput > div > div > input {
#         font-size: 16px;
#         border-radius: 10px;
#     }
    
#     /* Chat Messages */
#     .chat-message {
#         padding: 1.2rem 1.5rem;
#         border-radius: 12px;
#         margin-bottom: 1rem;
#         animation: fadeIn 0.3s ease-in;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#     }
    
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
    
#     .user-message {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         margin-left: 20%;
#     }
    
#     .assistant-message {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         margin-right: 20%;
#     }
    
#     .message-header {
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#         font-size: 14px;
#         opacity: 0.9;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     .message-content {
#         line-height: 1.6;
#         font-size: 15px;
#     }
    
#     .message-time {
#         font-size: 11px;
#         opacity: 0.7;
#         margin-top: 0.5rem;
#         text-align: right;
#     }
    
#     /* Example buttons */
#     .example-btn {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         border: none;
#         cursor: pointer;
#         transition: all 0.3s ease;
#         text-align: left;
#         width: 100%;
#         margin-bottom: 0.5rem;
#     }
    
#     .example-btn:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
#     }
    
#     /* Sidebar */
#     .css-1d391kg {
#         background-color: #f8f9fa;
#     }
    
#     /* Title styling */
#     h1 {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-weight: 700;
#     }
    
#     /* Status badges */
#     .status-badge {
#         display: inline-block;
#         padding: 0.3rem 0.8rem;
#         border-radius: 20px;
#         font-size: 12px;
#         font-weight: 600;
#     }
    
#     .status-connected {
#         background-color: #d4edda;
#         color: #155724;
#     }
    
#     .status-disconnected {
#         background-color: #f8d7da;
#         color: #721c24;
#     }
    
#     /* Welcome card */
#     .welcome-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
#     }
    
#     .welcome-title {
#         font-size: 28px;
#         font-weight: 700;
#         margin-bottom: 0.5rem;
#     }
    
#     .welcome-subtitle {
#         font-size: 16px;
#         opacity: 0.9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ==================== API Configuration ====================
# API_URL = "http://localhost:8000"

# def check_api_health():
#     """Check if the API is running"""
#     try:
#         response = requests.get(f"{API_URL}/health", timeout=5)
#         return response.status_code == 200
#     except:
#         return False

# def query_api(query: str, top_k: int = 5, score_threshold: float = 0.4, fallback_to_llm: bool = True) -> Optional[dict]:
#     """Send query to the API and get response"""
#     try:
#         response = requests.post(
#             f"{API_URL}/ask",
#             json={
#                 "query": query,
#                 "top_k": top_k,
#                 "score_threshold": score_threshold,
#                 "fallback_to_llm": fallback_to_llm
#             },
#             timeout=60
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"❌ Error: {response.status_code} - {response.text}")
#             return None
#     except requests.exceptions.ConnectionError:
#         st.error("❌ Cannot connect to the API. Make sure the backend is running on port 8000.")
#         return None
#     except requests.exceptions.Timeout:
#         st.error("⏱️ Request timed out. The model might be processing a complex query.")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error: {str(e)}")
#         return None

# # ==================== Initialize Session State ====================
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "api_connected" not in st.session_state:
#     st.session_state.api_connected = check_api_health()

# # ==================== Sidebar ====================
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/bot.png", width=80)
#     st.title("⚙️ Configuration")
    
#     # API Status
#     st.markdown("### 🔌 Connection Status")
#     if st.session_state.api_connected:
#         st.markdown('<span class="status-badge status-connected">✅ Connected</span>', unsafe_allow_html=True)
#     else:
#         st.markdown('<span class="status-badge status-disconnected">❌ Disconnected</span>', unsafe_allow_html=True)
#         if st.button("🔄 Retry Connection", use_container_width=True):
#             with st.spinner("Connecting..."):
#                 st.session_state.api_connected = check_api_health()
#             st.rerun()
    
#     st.divider()
    
#     # RAG Parameters
#     st.markdown("### 🎛️ RAG Parameters")
    
#     top_k = st.slider(
#         "📚 Number of documents",
#         min_value=1,
#         max_value=10,
#         value=5,
#         help="How many relevant documents to retrieve from the vector database"
#     )
    
#     score_threshold = st.slider(
#         "🎯 Similarity threshold",
#         min_value=0.0,
#         max_value=1.0,
#         value=0.4,
#         step=0.05,
#         help="Minimum similarity score for retrieved documents (higher = more strict)"
#     )
    
#     fallback_to_llm = st.checkbox(
#         "🧠 Enable LLM fallback",
#         value=True,
#         help="If no relevant documents are found, use the LLM's general knowledge"
#     )
    
#     st.divider()
    
#     # Statistics
#     st.markdown("### 📊 Statistics")
#     st.metric("Total Messages", len(st.session_state.messages))
#     st.metric("Conversations", len(st.session_state.messages) // 2 if len(st.session_state.messages) > 0 else 0)
    
#     st.divider()
    
#     # Clear chat button
#     if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
#         st.session_state.messages = []
#         st.rerun()
    
#     st.divider()
    
#     # Info
#     st.markdown("### ℹ️ About")
#     st.info("""
#     **Technology Stack:**
    
#     🔸 Pinecone - Vector DB  
#     🔸 Sentence Transformers  
#     🔸 Ollama Llama2  
#     🔸 FastAPI Backend  
#     🔸 Streamlit Frontend  
    
#     **Version:** 1.0.0
#     """)
    
#     st.markdown("---")
#     st.markdown("""
#         <div style='text-align: center; color: #666;'>
#             <small>Made with ❤️</small>
#         </div>
#     """, unsafe_allow_html=True)

# # ==================== Main Content ====================

# # Title and subtitle
# col1, col2 = st.columns([6, 1])
# with col1:
#     st.markdown("# 🤖 HR Policies Assistant")
#     st.markdown("### Ask me anything about HR policies!")

# with col2:
#     if st.button("ℹ️", help="About this chatbot"):
#         st.info("""
#         This intelligent assistant helps you find information from HR policy documents using 
#         advanced RAG (Retrieval-Augmented Generation) technology.
#         """)

# # Display API warning if not connected
# if not st.session_state.api_connected:
#     st.error("### ⚠️ Backend API is not running")
#     st.markdown("""
#     Please start the backend server first:
#     ```bash
#     python backend.py
#     ```
#     The API should be running on `http://localhost:8000`
#     """)
#     if st.button("🔄 Check Connection Again"):
#         st.session_state.api_connected = check_api_health()
#         st.rerun()
#     st.stop()

# # Welcome message for new users
# if len(st.session_state.messages) == 0:
#     st.markdown("""
#     <div class="welcome-card">
#         <div class="welcome-title">👋 Welcome to HR Policies Assistant!</div>
#         <div class="welcome-subtitle">
#             I'm here to help you find information from your HR policy documents. 
#             Start by asking a question below!
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# st.divider()

# # Display chat messages
# chat_container = st.container()

# with chat_container:
#     for message in st.session_state.messages:
#         timestamp = message.get("timestamp", datetime.now().strftime("%I:%M %p"))
        
#         if message["role"] == "user":
#             st.markdown(f"""
#             <div class="chat-message user-message">
#                 <div class="message-header">
#                     <span>👤 You</span>
#                 </div>
#                 <div class="message-content">{message["content"]}</div>
#                 <div class="message-time">{timestamp}</div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class="chat-message assistant-message">
#                 <div class="message-header">
#                     <span>🤖 Assistant</span>
#                 </div>
#                 <div class="message-content">{message["content"]}</div>
#                 <div class="message-time">{timestamp}</div>
#             </div>
#             """, unsafe_allow_html=True)

# # Chat input at the bottom
# st.markdown("### 💬 Ask Your Question")
# query = st.chat_input("Type your question here...", key="chat_input")

# if query:
#     # Add user message to chat history
#     current_time = datetime.now().strftime("%I:%M %p")
#     st.session_state.messages.append({
#         "role": "user",
#         "content": query,
#         "timestamp": current_time
#     })
    
#     # Get response from API
#     with st.spinner("🔍 Searching documents and generating answer..."):
#         response = query_api(
#             query=query,
#             top_k=top_k,
#             score_threshold=score_threshold,
#             fallback_to_llm=fallback_to_llm
#         )
    
#     if response:
#         answer = response.get("answer", "No answer received")
        
#         # Add assistant message to chat history
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": answer,
#             "timestamp": datetime.now().strftime("%I:%M %p")
#         })
        
#         st.rerun()

# # ==================== Footer ====================
# st.markdown("<br><br>", unsafe_allow_html=True)
# st.divider()
# st.markdown("""
#     <div style='text-align: center; color: #666; padding: 1rem;'>
#         <p><strong>RAG Chatbot v1.0</strong></p>
#         <small>Powered by FastAPI, Streamlit, Pinecone & Ollama</small><br>
#         <small>© 2024 HR Policies Assistant. All rights reserved.</small>
#     </div>
# """, unsafe_allow_html=True)


import streamlit as st
import requests
from typing import Optional
from datetime import datetime

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="RAG Chatbot - HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { padding: 1rem 2rem; }
    .stTextInput > div > div > input { font-size: 16px; border-radius: 10px; }

    /* Chat Messages */
    .chat-message { padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1rem; animation: fadeIn 0.3s ease-in; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-left: 20%; }
    .assistant-message { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; margin-right: 20%; }
    .message-header { font-weight: 600; margin-bottom: 0.5rem; font-size: 14px; opacity: 0.9; display: flex; align-items: center; gap: 0.5rem; }
    .message-content { line-height: 1.6; font-size: 15px; }
    .message-time { font-size: 11px; opacity: 0.7; margin-top: 0.5rem; text-align: right; }

    /* Sidebar */
    .css-1d391kg { background-color: #f8f9fa; }

    /* Welcome card */
    .welcome-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
    .welcome-title { font-size: 28px; font-weight: 700; margin-bottom: 0.5rem; }
    .welcome-subtitle { font-size: 16px; opacity: 0.9; }

    /* Status badges */
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .status-connected { background-color: #d4edda; color: #155724; }
    .status-disconnected { background-color: #f8d7da; color: #721c24; }
    
    /* Error box styling */
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==================== API Functions ====================
def check_api_health(api_url: str):
    """Check if the API is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception:
        return False

def query_api(query: str, api_url: str, top_k: int = 5, score_threshold: float = 0.4, fallback_to_llm: bool = True) -> Optional[dict]:
    """Send query to the API and get response"""
    try:
        response = requests.post(
            f"{api_url}/ask",
            json={
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "fallback_to_llm": fallback_to_llm
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Make sure it is running.")
        return None
    except requests.exceptions.Timeout:
        st.warning("⏱️ Request timed out. Backend may be busy.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== Sidebar ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bot.png", width=80)
    st.title("⚙️ Configuration")

    # Backend selection
    st.markdown("### 🌐 API Selection")
    api_option = st.radio(
        "Select Backend",
        ("Local (http://localhost:8000)", "Deployed (https://developrag-1.onrender.com)")
    )
    API_URL = "http://localhost:8000" if "Local" in api_option else "https://developrag-1.onrender.com"

    # API Status
    st.markdown("### 🔌 Connection Status")
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = check_api_health(API_URL)

    if st.session_state.api_connected:
        st.markdown('<span class="status-badge status-connected">✅ Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-disconnected">❌ Disconnected</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.session_state.api_connected = check_api_health(API_URL)
            st.rerun()

    st.divider()

    # RAG Parameters
    st.markdown("### 🎛️ RAG Parameters")
    top_k = st.slider("📚 Number of documents", 1, 10, 5)
    score_threshold = st.slider("🎯 Similarity threshold", 0.0, 1.0, 0.4, 0.05)
    fallback_to_llm = st.checkbox("🧠 Enable LLM fallback", value=True)

    st.divider()

    # Statistics
    st.markdown("### 📊 Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("Conversations", len(st.session_state.messages)//2 if len(st.session_state.messages) > 0 else 0)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Info
    st.markdown("### ℹ️ About")
    st.info("""
    **Technology Stack:**
    🔸 Pinecone - Vector DB  
    🔸 Sentence Transformers  
    🔸 Ollama Llama2  
    🔸 FastAPI Backend  
    🔸 Streamlit Frontend  
    **Version:** 1.0.0
    """)
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><small>Made with ❤️</small></div>", unsafe_allow_html=True)

# ==================== Main Content ====================
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("# 🤖 HR Policies Assistant")
    st.markdown("### Ask me anything about HR policies!")
with col2:
    if st.button("ℹ️", help="About this chatbot"):
        st.info("This intelligent assistant helps you find information from HR policy documents using advanced RAG technology.")

# API warning - UPDATED MESSAGE
if not st.session_state.api_connected:
    st.error("### ⚠️ Backend API is not running")
    st.markdown("""
    Please start the backend server first:
    
    ```bash
    python main.py
    ```
    
    The API should be running on `http://localhost:8000`
    
    **Quick Checklist:**
    - ✅ Ollama is installed and running (`ollama serve`)
    - ✅ Llama2 model is downloaded (`ollama pull llama2`)
    - ✅ Pinecone API key is set in `.env` file
    - ✅ All dependencies are installed (`pip install -r requirements.txt`)
    """)
    st.stop()

# Welcome message
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to HR Policies Assistant!</div>
        <div class="welcome-subtitle">
            I'm here to help you find information from your HR policy documents. 
            Start by asking a question below!
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        timestamp = message.get("timestamp", datetime.now().strftime("%I:%M %p"))
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header"><span>👤 You</span></div>
                <div class="message-content">{message["content"]}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header"><span>🤖 Assistant</span></div>
                <div class="message-content">{message["content"]}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("### 💬 Ask Your Question")
query = st.chat_input("Type your question here...", key="chat_input")

if query:
    current_time = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": current_time})

    with st.spinner("🔍 Searching documents and generating answer..."):
        response = query_api(query=query, api_url=API_URL, top_k=top_k, score_threshold=score_threshold, fallback_to_llm=fallback_to_llm)

    if response:
        answer = response.get("answer", "No answer received")
        st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%I:%M %p")})
        st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>RAG Chatbot v1.0</strong></p>
        <small>Powered by FastAPI, Streamlit, Pinecone & Ollama</small><br>
        <small>© 2024 HR Policies Assistant. All rights reserved.</small>
    </div>
""", unsafe_allow_html=True)
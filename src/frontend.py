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
    .chat-message { padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1rem; animation: fadeIn 0.3s ease-in; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-left: 20%; }
    .assistant-message { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; margin-right: 20%; }
    .message-header { font-weight: 600; margin-bottom: 0.5rem; font-size: 14px; opacity: 0.9; display: flex; align-items: center; gap: 0.5rem; }
    .message-content { line-height: 1.6; font-size: 15px; }
    .message-time { font-size: 11px; opacity: 0.7; margin-top: 0.5rem; text-align: right; }
    .example-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; border: none; cursor: pointer; transition: all 0.3s ease; text-align: left; width: 100%; margin-bottom: 0.5rem; }
    .example-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
    .css-1d391kg { background-color: #f8f9fa; }
    h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700; }
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .status-connected { background-color: #d4edda; color: #155724; }
    .status-disconnected { background-color: #f8d7da; color: #721c24; }
    .welcome-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
    .welcome-title { font-size: 28px; font-weight: 700; margin-bottom: 0.5rem; }
    .welcome-subtitle { font-size: 16px; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# ==================== Backend API ====================
backend_url = "https://developrag.onrender.com/ask"

def check_api_health():
    try:
        response = requests.get(backend_url.replace("/ask", "/health"), timeout=5)
        return response.status_code == 200
    except:
        return False

def query_api(query: str, top_k: int = 5, score_threshold: float = 0.4, fallback_to_llm: bool = True) -> Optional[dict]:
    try:
        response = requests.post(
            backend_url,
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
        st.error("❌ Cannot connect to the API. Make sure the backend is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The model might be processing a complex query.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_connected" not in st.session_state:
    st.session_state.api_connected = check_api_health()

# ==================== Sidebar ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bot.png", width=80)
    st.title("⚙️ Configuration")

    st.markdown("### 🔌 Connection Status")
    if st.session_state.api_connected:
        st.markdown('<span class="status-badge status-connected">✅ Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-disconnected">❌ Disconnected</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.session_state.api_connected = check_api_health()
            st.rerun()
    
    st.divider()
    st.markdown("### 🎛️ RAG Parameters")
    top_k = st.slider("📚 Number of documents", 1, 10, 5)
    score_threshold = st.slider("🎯 Similarity threshold", 0.0, 1.0, 0.4, 0.05)
    fallback_to_llm = st.checkbox("🧠 Enable LLM fallback", value=True)
    
    st.divider()
    st.markdown("### 📊 Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("Conversations", len(st.session_state.messages) // 2 if st.session_state.messages else 0)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.info("""
    **Technology Stack:**
    🔸 Pinecone  
    🔸 Sentence Transformers  
    🔸 Ollama Llama2  
    🔸 FastAPI Backend  
    🔸 Streamlit Frontend  
    **Version:** 1.0.0
    """)
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#666;'><small>Made with ❤️</small></div>", unsafe_allow_html=True)

# ==================== Main Content ====================
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("# 🤖 HR Policies Assistant")
    st.markdown("### Ask me anything about HR policies!")
with col2:
    if st.button("ℹ️", help="About this chatbot"):
        st.info("This assistant helps you find information from HR policy documents using RAG.")

if not st.session_state.api_connected:
    st.error("### ⚠️ Backend API is not running")
    st.stop()

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to HR Policies Assistant!</div>
        <div class="welcome-subtitle">Ask any question related to HR policies and get answers instantly.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        timestamp = message.get("timestamp", datetime.now().strftime("%I:%M %p"))
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div class="message-content">{message['content']}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">🤖 Assistant</div>
                <div class="message-content">{message['content']}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("### 💬 Ask Your Question")
query = st.chat_input("Type your question here...", key="chat_input")
if query:
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": datetime.now().strftime("%I:%M %p")})
    with st.spinner("🔍 Searching documents and generating answer..."):
        response = query_api(query, top_k, score_threshold, fallback_to_llm)
    if response:
        answer = response.get("answer", "No answer received")
        st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%I:%M %p")})
        st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; padding:1rem;'>
    <p><strong>RAG Chatbot v1.0</strong></p>
    <small>Powered by FastAPI, Streamlit, Pinecone & Ollama</small><br>
    <small>© 2024 HR Policies Assistant. All rights reserved.</small>
</div>
""", unsafe_allow_html=True)

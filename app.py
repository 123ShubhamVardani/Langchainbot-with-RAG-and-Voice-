# app.py – LangChain Chatbot with Groq, Gemini, Hugging Face, RAG, Web Search, and Voice

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
from langchain_community.llms import HuggingFaceHub

# === Load API Keys & Config ===
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

def log_debug(message):
    """Logs debug messages to sidebar and file if DEBUG_MODE is True."""
    if DEBUG_MODE:
        st.sidebar.text(f"DEBUG: {message}")
        with open("startup_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {message}\n")

api_keys = {
    "groq": os.getenv("GROQ_API_KEY"),
    "serpapi": os.getenv("SERPAPI_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACEHUB_API_TOKEN")
}

# === Page Setup ===
st.set_page_config(page_title="LangChain Chatbot with RAG & Voice", layout="wide")
st.markdown("<h1 style='text-align: center;'>💬 LangChain Chatbot with RAG & Voice</h1>", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.image("assets/bot_avatar.png", width=80)
    st.title("LangChain Chatbot")
    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)

    model_mode = st.radio("Model Selection Mode", ["Automatic", "Manual"])
    available_models = [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "gemini/gemini-1.5-pro",
        "huggingface/qwen/qwen3-32b"
    ]
    selected_model_name = st.selectbox("Choose Model", available_models) if model_mode == "Manual" else None

    st.markdown("---")
    st.header("📄 Document Q&A (RAG)")
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            st.download_button("📁 Download Chat Log", f.read(), "chat_log.txt", mime="text/plain")

# === LLM Loader ===
llm = None
selected_model = ""

def load_llm(provider, model_name):
    if provider == "groq" and api_keys.get("groq"):
        return ChatGroq(temperature=temperature, model_name=model_name, groq_api_key=api_keys["groq"])
    elif provider == "gemini" and api_keys.get("google"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_keys["google"])
    elif provider == "huggingface" and api_keys.get("huggingface"):
        return HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=api_keys["huggingface"], model_kwargs={"temperature": temperature})
    return None

if model_mode == "Automatic":
    for model_id in available_models:
        provider, model_name = model_id.split("/", 1)
        try:
            candidate_llm = load_llm(provider, model_name)
            if candidate_llm:
                llm = candidate_llm
                selected_model = model_id
                st.sidebar.success(f"✅ Loaded model: {selected_model}")
                break
        except Exception as e:
            log_debug(f"Failed to load {model_id}: {e}")
    if not llm:
        st.error("❌ No available models. Please check API keys.")
        st.stop()
else:
    provider, model_name = selected_model_name.split("/", 1)
    try:
        llm = load_llm(provider, model_name)
        if llm:
            selected_model = selected_model_name
            st.sidebar.markdown(f"🧠 Using model: **{selected_model}**")
        else:
            st.error(f"❌ Failed to load: {selected_model_name}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load: {e}")
        st.stop()

# === RAG Document Processing ===
vector_db = None
if uploaded_file:
    with st.spinner("Processing document..."):
        ext = uploaded_file.name.split(".")[-1]
        temp_file_path = f"temp.{ext}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            loader_map = {
                "pdf": PyPDFLoader,
                "docx": Docx2txtLoader,
                "txt": TextLoader,
                "csv": CSVLoader
            }
            loader = loader_map.get(ext)
            pages = loader(temp_file_path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success("📄 Document loaded successfully!")
        finally:
            os.remove(temp_file_path)

# === Agent or RAG Chain ===
if vector_db:
    retriever = vector_db.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
else:
    tools = []
    if use_web_search and api_keys["serpapi"]:
        tools.append(Tool(name="Web Search", func=SerpAPIWrapper(serpapi_api_key=api_keys["serpapi"]).run, description="Web Q&A tool"))
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True)

# === Voice Input / Output ===
stt = sr.Recognizer()

def get_voice():
    try:
        with sr.Microphone() as src:
            stt.adjust_for_ambient_noise(src)
            with st.spinner("🎤 Listening..."):
                audio = stt.listen(src, timeout=5)
        return stt.recognize_google(audio)
    except Exception as e:
        st.sidebar.error(f"❌ Voice error: {e}")
        return None

def speak_text(text):
    tts = gTTS(text, lang='en')
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Avatars ===
USER_AVATAR = "assets/user_avatar.png"
BOT_AVATAR = "assets/bot_avatar.png"

# === Chat UI ===
col1, col2 = st.columns([10, 1])
with col1:
    user_input = st.chat_input("Type your message here...")
with col2:
    if st.button("🎤"):
        voice = get_voice()
        if voice:
            st.session_state.chat_history.append({"role": "user", "content": voice})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg['content'], unsafe_allow_html=True)
        if msg["role"] == "assistant":
            audio = speak_text(msg['content'])
            st.audio(audio, format='audio/mp3')

if user_input:
    st.chat_message("user", avatar=USER_AVATAR).markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.run(user_input) if vector_db else agent.run(user_input)
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(response, unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n[{datetime.now()}] Bot: {response}\n\n")

# === Footer ===
st.markdown("""
<style>
footer {visibility: hidden;}
.custom-footer {
    position: fixed; bottom: 0; right: 20px;
    font-size: 12px; color: #ccc;
}
.custom-footer img {
    height: 24px; border-radius: 50%;
    margin-left: 8px; vertical-align: middle;
}
</style>
<div class="custom-footer">Created by Shubham Vardani <img src="https://avatars.githubusercontent.com/u/104264016?v=4"></div>
""", unsafe_allow_html=True)


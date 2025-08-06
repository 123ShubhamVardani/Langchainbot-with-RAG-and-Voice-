# app.py - LangChain Chatbot with Groq, Web Search, RAG, and Voice

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO

# Load API Keys
def load_api_keys():
    load_dotenv()
    return {
        "groq": os.getenv("GROQ_API_KEY"),
        "serpapi": os.getenv("SERPAPI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

api_keys = load_api_keys()

# Streamlit page config
st.set_page_config(page_title="LangChain Chatbot with RAG & Voice", layout="wide")
st.markdown("<h1 style='text-align: center;'>💬 LangChain Chatbot with RAG & Voice</h1>", unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.image("assets/bot_avatar.png", width=80)
    st.title("LangChain\nChatbot")
    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web_search = st.checkbox("🔍 Enable Web Search", value=True)

    model_mode = st.radio("Model Selection Mode", ["Automatic", "Manual"])
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ]
    selected_model = st.selectbox("Choose Model", available_models) if model_mode == "Manual" else None

    st.markdown("---")
    st.header("📄 Document Q&A (RAG)")
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])

    if os.path.exists("chat_log.txt"):
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            st.download_button("📁 Download Chat Log", f.read(), "chat_log.txt", mime="text/plain")

# Load LLM
llm = None
if model_mode == "Automatic":
    for model in available_models:
        try:
            llm = ChatGroq(temperature=temperature, model_name=model, groq_api_key=api_keys["groq"])
            selected_model = model
            break
        except Exception as e:
            st.warning(f"❌ Model failed: {model} → {e}")
    if not llm:
        st.error("❌ No available models. Check your GROQ API key.")
        st.stop()
else:
    try:
        llm = ChatGroq(temperature=temperature, model_name=selected_model, groq_api_key=api_keys["groq"])
    except Exception as e:
        st.error(f"❌ Failed to load selected model: {e}")
        st.stop()

st.sidebar.markdown(f"🧠 Using model: **{selected_model}**", unsafe_allow_html=True)

# Vector DB
vector_db = None
if uploaded_file:
    with st.spinner("Processing document..."):
        ext = uploaded_file.name.split(".")[-1]
        temp_file_path = f"temp.{ext}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            loader = {
                "pdf": PyPDFLoader,
                "docx": Docx2txtLoader,
                "txt": TextLoader,
                "csv": CSVLoader
            }.get(ext)
            pages = loader(temp_file_path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            if not api_keys["openai"]:
                st.error("❌ OPENAI_API_KEY is required for embeddings.")
                st.stop()
            embeddings = OpenAIEmbeddings(openai_api_key=api_keys["openai"])
            vector_db = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success("📄 Document loaded successfully!")
        finally:
            os.remove(temp_file_path)

# Agent or RAG chain
if vector_db:
    retriever = vector_db.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
else:
    tools = []
    if use_web_search and api_keys["serpapi"]:
        search = SerpAPIWrapper(serpapi_api_key=api_keys["serpapi"])
        tools.append(Tool(name="Web Search", func=search.run, description="Web Q&A tool"))
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True)

# Voice input
stt = sr.Recognizer()
def get_voice():
    st.sidebar.write("🎙️ Speak now...")
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

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Avatars
USER_AVATAR = "assets/user_avatar.png"
BOT_AVATAR = "assets/bot_avatar.png"

# Chat UI
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
        st.markdown(f"{msg['content']}", unsafe_allow_html=True)
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
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(f"<div class='chat-bubble bot'>{response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] User: {user_input}\n[{datetime.now()}] Bot: {response}\n\n")

# Footer
st.markdown("""
<style>
.chat-bubble {background-color: #1e1e1e; color: white; padding: 10px 15px; border-radius: 20px; max-width: 80%; margin: 5px 0; animation: fadeIn 0.3s ease-in-out;}
.chat-bubble.bot {background-color: #2f2f2f;}
footer {visibility: hidden;}
.custom-footer {position: fixed; bottom: 0; right: 20px; font-size: 12px; color: #ccc;}
.custom-footer img {height: 24px; border-radius: 50%; margin-left: 8px; vertical-align: middle;}
</style>
<div class="custom-footer">Created by Shubham Vardani <img src="https://avatars.githubusercontent.com/u/104264016?v=4"></div>
""", unsafe_allow_html=True)
